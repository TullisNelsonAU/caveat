#!/usr/bin/env python3
"""FU1 dual-axis figure from staircase_ak.csv (no engine calls). Two panels, one series per cell:
   (left)  entropy  U_k = mean h(q) over A_k  vs rung;
   (right) decode-leak = decoy candidates surviving in A_k  vs rung  (log-ish count, the steep axis).
Emits PNG + PDF + the plotted CSV. This is the corrected headline figure (FOLLOWUP_SPEC FU1 deliverable).
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "staircase")
RUNGS = ["R0", "R2", "R3", "R4", "R5"]
XLAB = ["E0\nraw", "E2\nconfirm", "E3\nresolve", "E4\ntrace", "E5\noracle"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def inum(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def load(name):
    p = os.path.join(D, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def main():
    rows = load("staircase_ak.csv")
    if not rows:
        print("no staircase_ak.csv"); return
    cells = {}
    for r in rows:
        cells.setdefault((r["obf"], r["struct"]), {})[r["rung"]] = r

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    csv_out = [["obf", "struct", "axis"] + RUNGS]
    for cell in sorted(cells):
        Hs, Ls = [], []
        for rk in RUNGS:
            r = cells[cell].get(rk)
            Hs.append(fnum(r["U_entropy_Ak"]) if r else None)
            Ls.append(inum(r["n_decoy_Ak"]) if r else None)
        lbl = "%s / %s" % cell
        xH = [i for i, y in enumerate(Hs) if y is not None]
        axL.plot(xH, [Hs[i] for i in xH], marker="o", label=lbl)
        xL = [i for i, y in enumerate(Ls) if y is not None]
        axR.plot(xL, [Ls[i] for i in xL], marker="s", label=lbl)
        csv_out.append([cell[0], cell[1], "U_entropy_Ak"] + ["%.4f" % h if h is not None else "" for h in Hs])
        csv_out.append([cell[0], cell[1], "decoy_leak_Ak"] + [str(l) if l is not None else "" for l in Ls])

    for ax, ttl, yl in ((axL, "Entropy axis (residual uncertainty on A_k)", "U_k = mean h(q) over A_k (bits)"),
                        (axR, "Decode-leak axis (decoys surviving the anchor)", "decoy candidates in A_k (count)")):
        ax.set_xticks(range(len(RUNGS))); ax.set_xticklabels(XLAB)
        ax.set_xlabel("evidence rung"); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.grid(alpha=0.3)
    axR.set_yscale("symlog")
    axL.legend(fontsize=7, ncol=2)
    fig.suptitle("A_k-restricted recoverability staircase (dual axis)")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "staircase_ak.png"), dpi=140)
    fig.savefig(os.path.join(D, "staircase_ak.pdf"))
    with open(os.path.join(D, "staircase_ak_plot.csv"), "w", newline="") as f:
        csv.writer(f).writerows(csv_out)
    plt.close()
    print("figure -> docs/staircase/staircase_ak.{png,pdf}")


if __name__ == "__main__":
    main()
