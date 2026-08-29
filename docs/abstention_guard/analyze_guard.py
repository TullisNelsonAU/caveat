#!/usr/bin/env python3
"""Abstention-guard analysis: reads guard.csv, prints the four-condition table (packed / desync /
benign / Tigress: always-benign vs oracle vs old-rule vs guarded-rule + selection accuracy), the
legit-VM false-positive gate, the guard's entropy separation audit, and the honest go/no-go.

The gate: the guard must (a) keep packed restore 0.366->0.000 @1.00, (b) keep desync restore
~0.074->0.025 @1.00, (c) leave benign unchanged, (d) stop harming Tigress (route benign, ECE ~0.033
not 0.242) — AND trade NO packed or desync accuracy to do it (rule and guard identical on those two).
"""
import csv, sys, os, statistics as st

CSV = os.path.join(os.path.dirname(__file__), "guard.csv")

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def fnum(r, k): return float(r[k])

def group(rows, pred):
    return [r for r in rows if pred(r)]

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

def arm_means(rows):
    return {k: mean([fnum(r, k) for r in rows]) for k in
            ("ece_always_benign","ece_oracle","ece_rule","ece_guard")}

def sel_acc(rows, col, target=None):
    """fraction where pick == true regime (target None) or == given target regime."""
    if not rows: return float("nan")
    if target is None:
        return sum(1 for r in rows if r[col]==r["regime"])/len(rows)
    return sum(1 for r in rows if r[col]==target)/len(rows)

def main():
    rows = load(CSV)
    is_tig = lambda r: r["sublabel"].startswith("tig")
    is_vm  = lambda r: r["sublabel"].startswith("vm")
    core   = lambda r: not is_tig(r) and not is_vm(r)

    print(f"# Abstention-guard analysis  (n_holdout={len(rows)})\n")

    # ── region-entropy separation audit + trained threshold ──
    packed_ent   = [fnum(r,"region_ent") for r in rows if r["regime"]=="packed"]
    nonpacked_ent= [fnum(r,"region_ent") for r in rows if r["regime"]!="packed" and core(r)]
    tig_ent      = [fnum(r,"region_ent") for r in rows if is_tig(r)]
    vm_ent       = [fnum(r,"region_ent") for r in rows if is_vm(r)]
    print("## Region-entropy separation (bits/byte)")
    def rng(xs): return f"[{min(xs):.3f}, {max(xs):.3f}]" if xs else "—"
    print(f"  packed (holdout)     n={len(packed_ent):2d}  {rng(packed_ent)}")
    print(f"  non-packed core      n={len(nonpacked_ent):2d}  {rng(nonpacked_ent)}")
    print(f"  Tigress              n={len(tig_ent):2d}  {rng(tig_ent)}")
    print(f"  legit-VM             n={len(vm_ent):2d}  {rng(vm_ent)}")
    allnp = nonpacked_ent + tig_ent + vm_ent
    if packed_ent and allnp:
        gap = min(packed_ent) - max(allnp)
        print(f"  → gap (min packed − max all-non-packed) = {min(packed_ent):.3f} − {max(allnp):.3f} = {gap:+.3f}")
    print()

    # ── four-condition table ──
    print("## Four-condition table (held-out mean ECE)")
    print(f"{'condition':<22}{'n':>3} {'always-benign':>14}{'oracle':>9}{'rule(old)':>11}{'guard(new)':>12}"
          f"{'sel rule':>10}{'sel guard':>11}")
    def line(name, rows_g, sel_target=None):
        if not rows_g:
            print(f"{name:<22}{0:>3}  (no rows)"); return None
        a = arm_means(rows_g)
        sr = sel_acc(rows_g, "rule_pick", sel_target)
        sg = sel_acc(rows_g, "guard_pick", sel_target)
        print(f"{name:<22}{len(rows_g):>3} {a['ece_always_benign']:>14.4f}{a['ece_oracle']:>9.4f}"
              f"{a['ece_rule']:>11.4f}{a['ece_guard']:>12.4f}{sr:>10.2f}{sg:>11.2f}")
        return dict(a=a, sr=sr, sg=sg)

    packed = line("packed",  group(rows, lambda r: r["regime"]=="packed" and core(r)))
    desync = line("desync",  group(rows, lambda r: r["regime"]=="obfuscated" and core(r)))
    benign = line("benign",  group(rows, lambda r: r["regime"]=="benign" and core(r)))
    # Tigress: correct action is BENIGN (well-calibrated under benign map). sel = fraction kept benign.
    tig    = line("Tigress (→benign)", group(rows, is_tig), sel_target="benign")
    vm     = line("legit-VM (→benign)", group(rows, is_vm),  sel_target="benign")
    print()

    # ── Tigress detail: how many flipped packed→benign ──
    tigrows = group(rows, is_tig)
    tig_rule_packed  = sum(1 for r in tigrows if r["rule_pick"]=="packed")
    tig_guard_packed = sum(1 for r in tigrows if r["guard_pick"]=="packed")
    tig_guard_benign = sum(1 for r in tigrows if r["guard_pick"]=="benign")
    print("## Tigress routing")
    print(f"  rule  → packed: {tig_rule_packed}/{len(tigrows)}   guard → packed: {tig_guard_packed}/{len(tigrows)}"
          f"   guard → benign: {tig_guard_benign}/{len(tigrows)}")
    print()

    # ── legit-VM per-binary detail ──
    print("## Legit-VM per-binary (rule_pick → guard_pick, region_ent, ECE)")
    for r in group(rows, is_vm):
        print(f"  {r['name']:<18} {r['sublabel']:<8} rule={r['rule_pick']:<10} guard={r['guard_pick']:<10}"
              f" ent={fnum(r,'region_ent'):.2f}  a={fnum(r,'ece_always_benign'):.4f}"
              f" rule={fnum(r,'ece_rule'):.4f} guard={fnum(r,'ece_guard'):.4f}")
    print()

    # ── honest go/no-go ──
    print("## Go / No-Go")
    checks = []
    # (a) packed restore preserved + no accuracy traded
    if packed:
        ok = packed["a"]["ece_guard"] <= packed["a"]["ece_always_benign"]*0.05 + 1e-6 and packed["sg"] >= 0.999
        traded = abs(packed["a"]["ece_guard"]-packed["a"]["ece_rule"])>1e-6 or packed["sg"]<packed["sr"]-1e-9
        checks.append(("(a) packed restored @1.00, no trade",
                       packed["sg"]>=0.999 and packed["a"]["ece_guard"]<=0.01 and not traded,
                       f"guard ECE {packed['a']['ece_guard']:.4f} sel {packed['sg']:.2f}; rule ECE {packed['a']['ece_rule']:.4f} sel {packed['sr']:.2f}"))
    if desync:
        traded = abs(desync["a"]["ece_guard"]-desync["a"]["ece_rule"])>1e-6 or desync["sg"]<desync["sr"]-1e-9
        checks.append(("(b) desync restored @1.00, no trade",
                       desync["sg"]>=0.999 and desync["a"]["ece_guard"]<=desync["a"]["ece_oracle"]*1.15+1e-4 and not traded,
                       f"guard ECE {desync['a']['ece_guard']:.4f} (oracle {desync['a']['ece_oracle']:.4f}) sel {desync['sg']:.2f}; rule ECE {desync['a']['ece_rule']:.4f} sel {desync['sr']:.2f}"))
    if benign:
        checks.append(("(c) benign unchanged",
                       abs(benign["a"]["ece_guard"]-benign["a"]["ece_always_benign"])<=1e-4,
                       f"guard ECE {benign['a']['ece_guard']:.4f} vs always {benign['a']['ece_always_benign']:.4f}"))
    if tig:
        checks.append(("(d) Tigress no longer harmed (→benign ~0.033)",
                       tig["a"]["ece_guard"]<=1.5*tig["a"]["ece_always_benign"]+1e-4 and tig["sg"]>=0.99,
                       f"guard ECE {tig['a']['ece_guard']:.4f} (was rule {tig['a']['ece_rule']:.4f}); guard→benign {tig['sg']:.2f}"))
    if vm:
        checks.append(("legit-VM abstains (→benign)",
                       vm["sg"]>=0.99 and vm["a"]["ece_guard"]<=1.5*vm["a"]["ece_always_benign"]+1e-4,
                       f"guard→benign {vm['sg']:.2f}; guard ECE {vm['a']['ece_guard']:.4f} vs always {vm['a']['ece_always_benign']:.4f}"))
    allok = all(c[1] for c in checks)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<45} {detail}")
    print(f"\n  VERDICT: {'GO — guard fixes Tigress with zero cost to packed/desync/benign' if allok else 'NO-GO — see failed checks'}")

if __name__ == "__main__":
    main()
