#!/usr/bin/env python3
"""Analysis for the Paper-2 credibility run. Turns credibility.csv into the numbers that make the
detector believable: the graded-drift gradient, the MILD-subset Spearman rho (the non-saturated
claim), the OOD-baseline comparison (S tracks ECE at least as well and localizes where they can't),
the multi-packer conservative-bias table, and the confound partials.

Leads with rho and the gradient, NOT the saturated full-range AUC. Emits a gradient figure.

usage: analyze_credibility.py credibility.csv [out_dir]
"""
import csv, sys, math
from collections import defaultdict

def spearman(x, y):
    n = len(x)
    if n < 3: return float("nan")
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0]*n; i = 0
        while i < n:
            j = i
            while j+1 < n and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i+j)/2.0 + 1.0
            for k in range(i, j+1): r[order[k]] = avg
            i = j+1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx)/n, sum(ry)/n
    cov = sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    vx = math.sqrt(sum((a-mx)**2 for a in rx)); vy = math.sqrt(sum((b-my)**2 for b in ry))
    return cov/(vx*vy) if vx>0 and vy>0 else float("nan")

def partial_spearman(x, y, controls):
    """Spearman of x,y after linearly rank-regressing out each control (Gram-Schmidt on ranks)."""
    n = len(x)
    def rank(v):
        order = sorted(range(n), key=lambda i: v[i]); r=[0.0]*n; i=0
        while i<n:
            j=i
            while j+1<n and v[order[j+1]]==v[order[i]]: j+=1
            for k in range(i,j+1): r[order[k]]=(i+j)/2.0+1.0
            i=j+1
        return r
    def resid(t, zs):
        t = rank(t)[:]
        for z in zs:
            z = rank(z); mz=sum(z)/n; mt=sum(t)/n
            den=sum((zi-mz)**2 for zi in z)
            b = sum((zi-mz)*(ti-mt) for zi,ti in zip(z,t))/den if den>0 else 0.0
            t = [ti - b*(zi-mz) for ti,zi in zip(t,z)]
        return t
    rx = resid(x, controls); ry = resid(y, controls)
    mx=sum(rx)/n; my=sum(ry)/n
    cov=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    vx=math.sqrt(sum((a-mx)**2 for a in rx)); vy=math.sqrt(sum((b-my)**2 for b in ry))
    return cov/(vx*vy) if vx>0 and vy>0 else float("nan")

def auc(scores, labels):
    """ROC AUC via rank-sum (Mann-Whitney), tie-averaged."""
    pos = [s for s,l in zip(scores,labels) if l]; neg=[s for s,l in zip(scores,labels) if not l]
    if not pos or not neg: return float("nan")
    allv = sorted(scores); n=len(scores)
    def rank(v):
        order=sorted(range(n), key=lambda i: v[i]); r=[0.0]*n; i=0
        while i<n:
            j=i
            while j+1<n and v[order[j+1]]==v[order[i]]: j+=1
            for k in range(i,j+1): r[order[k]]=(i+j)/2.0+1.0
            i=j+1
        return r
    r = rank(scores)
    rpos = sum(ri for ri,l in zip(r,labels) if l)
    np_, nn = len(pos), len(neg)
    return (rpos - np_*(np_+1)/2.0)/(np_*nn)

def main():
    # default to the CSV beside this script so the script runs bare, like the others
    path = sys.argv[1] if len(sys.argv) > 1 else str(
        __import__("pathlib").Path(__file__).parent / "credibility.csv")
    rows = list(csv.DictReader(open(path)))
    F = lambda r,k: float(r[k])
    # level ordering + junk fraction (from make_upxgt/desync_gt measurements)
    LVL_JUNK = {"": 0.0, "pilot": 0.034, "d1_med": 0.056, "d2_heavy": 0.111, "d3_max": 0.179}
    LVL_ORDER = ["clean", "pilot", "d1_med", "d2_heavy", "d3_max"]

    def lvl_of(r):
        if r["role"] in ("clean_fit","clean_holdout"): return "clean"
        return r["level"]

    stats = {  # column -> (label, higher-drift-direction sign for detection)
        "s_glob_surprise": "S_glob (surprise)",
        "s_spat_moran":    "S_spat (Moran)",
        "s_spat_clustered":"S_spat (clustered)",
        "b_mean_pi":       "baseline mean-pi",
        "b_pred_entropy":  "baseline pred-entropy",
        "b_msp":           "baseline MSP",
        "b_mean_abs_llr":  "baseline |llr| (temp)",
    }

    # ---- desync + clean rows (packed handled separately: its ECE is a different construction) ----
    dc = [r for r in rows if r["role"] in ("clean_fit","clean_holdout","desync")]
    packed = [r for r in rows if r["role"]=="packed"]

    print("="*74)
    print("GRADED-DRIFT GRADIENT — per level (mean over binaries)")
    print("="*74)
    print(f"{'level':10} {'n':>3} {'junk%':>6} {'trueECE':>8} {'S_glob':>8} {'S_moran':>8} "
          f"{'msp':>7} {'entropy':>8}")
    for lv in LVL_ORDER:
        g=[r for r in dc if lvl_of(r)==lv]
        if not g: continue
        m=lambda k: sum(F(r,k) for r in g)/len(g)
        print(f"{lv:10} {len(g):>3} {LVL_JUNK.get(g[0]['level'] if lv!='clean' else '',0)*100:>6.1f} "
              f"{m('ece_calibrated'):>8.4f} {m('s_glob_surprise'):>8.3f} {m('s_spat_moran'):>8.4f} "
              f"{m('b_msp'):>7.4f} {m('b_pred_entropy'):>8.4f}")

    # ---- mild-drift subset: the non-saturated claim ----
    # mild = clean holdout + the two lowest desync densities (pilot, d1_med). Excludes the huge
    # d2/d3 drift that makes the full-range ROC trivial.
    mild = [r for r in dc if lvl_of(r) in ("clean","pilot","d1_med")]
    full = dc
    ece = lambda rs: [F(r,"ece_calibrated") for r in rs]

    def rho_table(rs, title):
        print("\n"+"-"*74); print(title+f"  (n={len(rs)})"); print("-"*74)
        e = ece(rs)
        print(f"{'statistic':24} {'rho(S,ECE)':>12}")
        out={}
        for col,lab in stats.items():
            r = spearman([F(x,col) for x in rs], e)
            out[col]=r
            print(f"{lab:24} {r:>12.3f}")
        return out

    rho_full = rho_table(full, "FULL RANGE rho(S, true ECE)  [saturated regime]")
    rho_mild = rho_table(mild, "MILD-DRIFT SUBSET rho(S, true ECE) clean+pilot+d1  [the believable claim]")
    # The sharpest non-saturated test: among binaries that are ALL mildly obfuscated (desync only,
    # lowest densities, no clean anchor), does S still rank the fine true-ECE differences? This is the
    # test a clean-vs-severe ROC can never pass — there is no easy clean/severe gap to exploit.
    within = [r for r in dc if lvl_of(r) in ("pilot","d1_med")]
    rho_within = rho_table(within, "WITHIN-DRIFT rho(S, true ECE) desync pilot+d1 only, NO clean anchor")

    # ---- non-saturated operating point: detector on the mild subset ----
    # positive = true ECE > 0.010 (a subtle-drift threshold, far below the desync-scale 0.05)
    thr = 0.010
    labels = [F(r,"ece_calibrated")>thr for r in mild]
    print("\n"+"-"*74)
    print(f"NON-SATURATED DETECTOR on mild subset (positive = true ECE > {thr}; "
          f"{sum(labels)}/{len(labels)} pos)")
    print("-"*74)
    print(f"{'statistic':24} {'AUC(mild)':>10}")
    for col,lab in stats.items():
        print(f"{lab:24} {auc([F(r,col) for r in mild], labels):>10.3f}")

    # ---- confounds: partial rho controlling entropy + size (full desync+clean) ----
    print("\n"+"-"*74); print("CONFOUNDS — partial rho(S, ECE | region_entropy, log_size)"); print("-"*74)
    e=ece(full); ent=[F(r,"region_entropy") for r in full]; sz=[math.log(F(r,"code_bytes")+1) for r in full]
    for col in ("s_glob_surprise","s_spat_moran","b_msp","b_pred_entropy"):
        p=partial_spearman([F(r,col) for r in full], e, [ent,sz])
        print(f"  partial rho {stats[col]:24} = {p:+.3f}")

    # ---- multi-packer: the S_glob-collapse / S_spat-robust replication ----
    print("\n"+"="*74); print("MULTI-PACKER SLICE (n="+str(len(packed))+")"); print("="*74)
    print(f"{'specimen':26} {'method':6} {'trueECE':>8} {'S_glob':>8} {'S_moran':>8} {'S_clust':>8} {'msp':>7}")
    by_method=defaultdict(list)
    for r in sorted(packed, key=lambda r:(r["level"],r["name"])):
        by_method[r["level"]].append(r)
        print(f"{r['name'][:26]:26} {r['level']:6} {F(r,'ece_calibrated'):>8.3f} "
              f"{F(r,'s_glob_surprise'):>8.3f} {F(r,'s_spat_moran'):>8.4f} "
              f"{F(r,'s_spat_clustered'):>8.4f} {F(r,'b_msp'):>7.4f}")
    # clean null for reference
    clean = [r for r in dc if lvl_of(r)=="clean"]
    cn = lambda k: sum(F(r,k) for r in clean)/len(clean)
    print(f"\nclean null: S_glob={cn('s_glob_surprise'):.3f}  S_moran={cn('s_spat_moran'):.4f}  "
          f"S_clust={cn('s_spat_clustered'):.4f}")
    print("Conservative-bias check: packed true ECE is high, but S_glob should sit near the clean")
    print("null (cavity contamination) while S_spat (Moran/clustered) stays elevated.")
    for meth, g in by_method.items():
        m=lambda k: sum(F(r,k) for r in g)/len(g)
        print(f"  [{meth}] n={len(g)}  meanECE={m('ece_calibrated'):.3f}  "
              f"S_glob={m('s_glob_surprise'):.3f} (clean {cn('s_glob_surprise'):.2f})  "
              f"S_moran={m('s_spat_moran'):.4f} (clean {cn('s_spat_moran'):.4f})")

    # ---- gradient figure (matplotlib) ----
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out_dir = sys.argv[2] if len(sys.argv)>2 else "."
        fig, ax = plt.subplots(1, 2, figsize=(10,4))
        col_by_lvl={"clean":"#4c78a8","pilot":"#72b7b2","d1_med":"#e4a93c","d2_heavy":"#e45756","d3_max":"#8b0000"}
        for lv in LVL_ORDER:
            g=[r for r in dc if lvl_of(r)==lv]
            if not g: continue
            ax[0].scatter([F(r,"ece_calibrated") for r in g],[F(r,"s_glob_surprise") for r in g],
                          s=18, c=col_by_lvl[lv], label=lv, alpha=0.8)
        ax[0].set_xlabel("true post-hoc ECE"); ax[0].set_ylabel("S_glob surprise (GT-free)")
        ax[0].set_yscale("log"); ax[0].set_title("S tracks ECE across the gradient"); ax[0].legend(fontsize=7)
        # mild-subset zoom
        for lv in ("clean","pilot","d1_med"):
            g=[r for r in dc if lvl_of(r)==lv]
            ax[1].scatter([F(r,"ece_calibrated") for r in g],[F(r,"s_spat_moran") for r in g],
                          s=18, c=col_by_lvl[lv], label=lv, alpha=0.8)
        ax[1].set_xlabel("true post-hoc ECE (mild subset)"); ax[1].set_ylabel("S_spat Moran's I")
        ax[1].set_title(f"mild-drift subset: rho={rho_mild['s_spat_moran']:.2f}"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{out_dir}/fig_gradient.svg"); fig.savefig(f"{out_dir}/fig_gradient.png", dpi=110)
        print(f"\nwrote {out_dir}/fig_gradient.svg/.png")
    except Exception as ex:
        print("fig skipped:", ex)

if __name__ == "__main__":
    main()
