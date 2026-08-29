#!/usr/bin/env python3
"""
Phase-0 probe (SYMEX_INTEGRATION_SPEC.md): does stock angr waste exploration in
regions our calibrated Soft posterior marks low-confidence (junk/decoy)?

For each binary:
  1. Load the per-address Soft posterior sidecar (dissertation disassemble --mode soft).
  2. Run stock angr symbolic exploration from the entry state under a fixed budget.
  3. Classify every instruction address angr actually executes against the posterior:
       - in .text & posterior >= T_hi  -> real/high-confidence code
       - in .text & posterior <  T_lo  -> LOW-CONF (junk/decoy our guidance would prune)
       - in .text & in-between         -> medium
       - outside .text                 -> extern/stub (not our concern)
  The gate signal is the share of *executed, in-.text* addresses that are low-conf.
Read-only wrt probdisasm; consumes the posterior JSON only.
"""
import json, sys, time, argparse, logging
logging.getLogger('angr').setLevel(logging.ERROR)
logging.getLogger('cle').setLevel(logging.ERROR)
logging.getLogger('pyvex').setLevel(logging.ERROR)
logging.getLogger('claripy').setLevel(logging.ERROR)
import angr

T_HI = 0.5   # posterior >= T_HI -> confident real code
T_LO = 0.5   # posterior <  T_LO -> low-confidence (junk/decoy candidate)

def load_posterior(path):
    d = json.load(open(path))
    m = {}
    for ins in d['instructions']:
        a = int(ins['address'], 16)
        m[a] = ins['posterior']
    lo = min(m) if m else 0
    hi = max(m) if m else 0
    return m, lo, hi

def load_gt(path):
    # GT = one real-instruction-start address per line (file-offset base, like the posterior)
    return set(int(l.strip(), 16) for l in open(path) if l.strip())

def probe(binpath, postpath, step_budget, active_cap, time_cap, gtpath=None):
    post, tlo, thi = load_posterior(postpath)
    gt = load_gt(gtpath) if gtpath else None
    proj = angr.Project(binpath, auto_load_libs=False)
    # .text span from the loaded object (authoritative for "in code section")
    text_lo, text_hi = None, None
    for sec in proj.loader.main_object.sections:
        if sec.name == '.text':
            text_lo, text_hi = sec.vaddr, sec.vaddr + sec.memsize
    # dissertation emits (vaddr - image_base); anchor its addresses to angr's .text
    # vaddr by shifting every key so the superset start lines up with .text start.
    delta = (text_lo - tlo) if text_lo is not None else 0
    if delta:
        post = {a + delta: p for a, p in post.items()}
        if gt is not None:
            gt = set(a + delta for a in gt)
    st = proj.factory.entry_state()
    st.options.add(angr.options.LAZY_SOLVES)
    simgr = proj.factory.simulation_manager(st)

    executed = set()          # unique instruction addrs angr ran
    edges = set()             # (src_block, dst_block) control-flow edges taken
    indirect_src = set()      # block addrs whose exit is a computed/indirect jump
    def _is_indirect(baddr):
        # VEX: a block ending in an indirect branch has a non-constant .next
        # (computed target) — that's where obfuscated dispatch explodes.
        try:
            v = proj.factory.block(baddr).vex
            import pyvex
            # computed *jump* only (Ijk_Boring w/ non-const target) — the switch/
            # virtualized-dispatch case. Exclude Ijk_Call / Ijk_Ret (returns also
            # have a non-const target but aren't the dispatch explosion we mean).
            return v.jumpkind == 'Ijk_Boring' and not isinstance(v.next, pyvex.expr.Const)
        except Exception:
            return False
    def record(hist):
        seq = list(hist.bbl_addrs)
        for baddr in seq:
            try:
                executed.update(proj.factory.block(baddr).instruction_addrs)
            except Exception:
                executed.add(baddr)
        for src, dst in zip(seq, seq[1:]):
            edges.add((src, dst))
            if src not in indirect_src and _is_indirect(src):
                indirect_src.add(src)
    steps = 0
    states_seen = 0
    t0 = time.time()
    while simgr.active and steps < step_budget and (time.time() - t0) < time_cap:
        # cap fan-out so one path-explosion doesn't dominate the budget
        if len(simgr.active) > active_cap:
            simgr.active[:] = simgr.active[:active_cap]
        for s in simgr.active:
            states_seen += 1
            record(s.history)
        try:
            simgr.step()
        except Exception:
            # a single misbehaving SimProcedure (e.g. SimFileDescriptorDuplex) must
            # not abort the whole probe; drop the offending states and keep exploring.
            simgr.active[:] = simgr.active[1:]
            if not simgr.active:
                break
        steps += 1
    wall = time.time() - t0
    # drain final active/deadended histories
    for stash in ('active', 'deadended', 'errored'):
        for s in getattr(simgr, stash, []):
            st_obj = getattr(s, 'state', s)
            try:
                record(st_obj.history)
            except Exception:
                pass

    in_text = [a for a in executed if text_lo is not None and text_lo <= a < text_hi]
    def classify(a):
        p = post.get(a)
        if p is None:
            return 'absent'      # in .text but not a superset offset we scored
        if p >= T_HI: return 'high'
        if p <  T_LO: return 'low'
        return 'med'
    from collections import Counter
    cls = Counter(classify(a) for a in in_text)
    n_text = len(in_text)
    low = cls['low'] + cls['absent']   # both = "not confident real code"

    # THE GATE CROSS-CHECK: of the low-posterior addresses angr actually executed,
    # how many are GT-real code (posterior false negatives, pruning would LOSE real
    # coverage) vs GT-junk (genuine angr waste, guidance would help)?
    gt_block = {}
    if gt is not None:
        exec_low_addrs = [a for a in in_text if (post.get(a, 1.0) < T_LO)]
        low_real = sum(1 for a in exec_low_addrs if a in gt)
        low_junk = len(exec_low_addrs) - low_real
        exec_real = sum(1 for a in in_text if a in gt)
        # THE REFRAMED (Paper-3) SIGNAL: indirect-jump target selection under
        # obfuscation. Of the targets angr actually followed out of computed/
        # indirect branches, how many are GT-junk (fake dispatch targets a
        # "only follow high-confidence resolved targets" filter would gate)?
        # On desync (direct dispatch) this is ~empty; on Tigress Virtualize/
        # Flatten/opaque it is the whole question.
        ind_targets = {dst for (src, dst) in edges if src in indirect_src}
        ind_targets_text = [a for a in ind_targets if text_lo <= a < text_hi]
        ind_junk = sum(1 for a in ind_targets_text if a not in gt)
        ind_lowconf = sum(1 for a in ind_targets_text if post.get(a, 1.0) < T_LO)
        gt_block = {
            'gt_real_starts': len(gt),
            'exec_in_gt_real': exec_real,
            'exec_low_total': len(exec_low_addrs),
            'exec_low_that_are_GT_real': low_real,
            'exec_low_that_are_GT_junk': low_junk,
            'low_junk_share_of_exec_text': round(low_junk / max(1, n_text), 4),
            # --- reframed indirect-dispatch metric (populates on obfuscated bins) ---
            'indirect_src_blocks': len(indirect_src),
            'indirect_targets_taken': len(ind_targets_text),
            'indirect_targets_GT_junk': ind_junk,
            'indirect_targets_lowconf': ind_lowconf,
            'indirect_junk_share': round(ind_junk / max(1, len(ind_targets_text)), 4),
        }
    return {
        'binary': binpath.split('/')[-1],
        'text_span': [hex(text_lo), hex(text_hi)] if text_lo else None,
        'addr_shift': hex(delta),
        'superset_offsets': len(post),
        'superset_low_frac': round(sum(1 for p in post.values() if p < T_LO) / max(1,len(post)), 4),
        'steps': steps, 'states_seen': states_seen, 'wall_s': round(wall,1),
        'executed_unique': len(executed),
        'executed_in_text': n_text,
        'exec_high': cls['high'], 'exec_med': cls['med'],
        'exec_low': cls['low'], 'exec_absent': cls['absent'],
        'exec_extern': len(executed) - n_text,
        'low_conf_share_of_text': round(low / max(1,n_text), 4),
        **gt_block,
    }

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', nargs='+', required=True, help='bin:post pairs')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--active-cap', type=int, default=40)
    ap.add_argument('--time-cap', type=float, default=90)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows = []
    for pr in a.pairs:
        parts = pr.split('::')
        b, p = parts[0], parts[1]
        g = parts[2] if len(parts) > 2 else None
        print(f'[probe] {b.split("/")[-1]} ...', flush=True)
        try:
            r = probe(b, p, a.steps, a.active_cap, a.time_cap, g)
        except Exception as e:
            r = {'binary': b.split('/')[-1], 'error': str(e)}
        print('   ', json.dumps(r), flush=True)
        rows.append(r)
    json.dump(rows, open(a.out,'w'), indent=2)
    print('wrote', a.out)
