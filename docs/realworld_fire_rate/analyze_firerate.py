#!/usr/bin/env python3
"""Turn firerate.csv + the corpus provenance into the tables the results doc quotes.

Joins on the binary's filename (`provenance.path` == `firerate.name`) and reports the fire rate
overall and broken down by language/toolchain, the routing decision the bare rule would take against
what the guard does, and the wild (S_glob, S_spat) distribution against our coreutils-fit null.

Everything here is descriptive. Nothing is fit, tuned, or thresholded on this data — the thresholds
are the published constants, baked into the harness.
"""
import csv
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
# argv[1] optionally selects a different results CSV (stratum B), argv[2] renumbers the sections so
# the two strata can sit in one document without clashing headings.
FIRE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "firerate.csv")
SEC = sys.argv[2] if len(sys.argv) > 2 else "A"
PROV = os.environ.get(
    "PROV", str(Path(__file__).parent / "provenance.csv") if "Path" in dir() else os.path.join(os.path.dirname(os.path.abspath(__file__)), "provenance.csv")
)

# Published nulls — quoted, never recomputed here.
DET_GLOB_HI, DET_SPAT_HI = 1.006, 0.105178  # exact clean-fit 95th percentiles;
# the paper prints these rounded as 1.006 / 0.1052. Rounding the SPATIAL gate down to
# 0.105 admits exactly one extra binary (netpbm__ppmrelief, s_spat = 0.105057) and
# shifts the reported counts from 139/140 to 140/141.


def _refire(rows):
    """Recompute fire_glob / fire_spat / fire_any from the raw statistics against the
    exact gates above. The CSV's precomputed flags were written by the Rust pass at
    rounded thresholds (1.01 / 0.105); trusting them shifts the headline counts."""
    for r in rows:
        try:
            g = float(r["s_glob"]); sp = float(r["s_spat"])
        except (ValueError, KeyError):
            continue
        r["fire_glob"] = "1" if g > DET_GLOB_HI else "0"
        r["fire_spat"] = "1" if sp > DET_SPAT_HI else "0"
        r["fire_any"] = "1" if (r["fire_glob"] == "1" or r["fire_spat"] == "1") else "0"
    return rows
RULE_GLOB_HI, RULE_SPAT_HI, PACK_ENT_LO = 2.5147, 0.1052, 7.1688
OWN_CORPUS_FA = 0.12  # the clean-holdout false-alarm rate we report on our own corpus (3/25)


def pct(vals, q):
    if not vals:
        return float("nan")
    v = sorted(vals)
    return v[min(len(v) - 1, max(0, int(round((len(v) - 1) * q))))]


def fmt(x, n=4):
    return "NA" if x != x else f"{x:.{n}f}"


def main():
    prov = {}
    if os.path.exists(PROV):
        with open(PROV, newline="") as f:
            for r in csv.DictReader(f):
                prov[r["path"]] = r
    else:
        print(f"!! no provenance at {PROV}; language breakdown will be empty", file=sys.stderr)

    # Last row per binary wins: raising the harness's --max-code-bytes and resuming appends a fresh
    # verdict for a name previously recorded as `too_large`.
    latest = {}
    with open(FIRE, newline="") as f:
        for r in csv.DictReader(f):
            latest[r["name"]] = r
    rows = list(latest.values())

    total = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    skipped = [r for r in rows if r["status"] != "ok"]

    def num(r, k):
        try:
            return float(r[k])
        except (ValueError, KeyError):
            return float("nan")

    globs = [num(r, "s_glob") for r in ok]
    spats = [num(r, "s_spat") for r in ok]

    _refire(ok)
    fg = sum(1 for r in ok if r["fire_glob"] == "1")
    fs_ = sum(1 for r in ok if r["fire_spat"] == "1")
    fa = sum(1 for r in ok if r["fire_any"] == "1")
    fb = sum(1 for r in ok if r["fire_both"] == "1")
    n = len(ok)
    rate = lambda x: x / n if n else float("nan")

    out = []
    A = out.append

    A(f"## {SEC}1. Corpus and completeness\n")
    A(f"- binaries submitted to the engine: **{total}**")
    A(f"- analyzed successfully: **{n}**")
    A(f"- excluded: **{len(skipped)}**")
    if skipped:
        by = defaultdict(int)
        for r in skipped:
            by[r["status"].split(":")[0]] += 1
        for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
            A(f"  - `{k}`: {v}")
    A("")

    A(f"## {SEC}2. Fire rate at the published detection nulls\n")
    A(f"Nulls: `S_glob > {DET_GLOB_HI}`, `S_spat > {DET_SPAT_HI}`. "
      "These binaries are stock Debian packages, so every firing below is a false alarm.\n")
    A("| prong | fires | rate |")
    A("|---|---|---|")
    A(f"| `S_glob > {DET_GLOB_HI}` | {fg}/{n} | **{fmt(rate(fg))}** |")
    A(f"| `S_spat > {DET_SPAT_HI}` | {fs_}/{n} | **{fmt(rate(fs_))}** |")
    A(f"| either prong | {fa}/{n} | **{fmt(rate(fa))}** |")
    A(f"| both prongs | {fb}/{n} | **{fmt(rate(fb))}** |")
    A("")
    A(f"Our own-corpus clean-holdout false-alarm rate for comparison: **{OWN_CORPUS_FA}** "
      "(3/25, `S_glob` prong only, coreutils).\n")

    # Binaries from one source package share a build environment, a toolchain invocation and often
    # most of their code, so they are not independent draws. coreutils alone contributes ~100. The
    # per-binary rate above is what a deployer sees; the package-clustered rate below is the honest
    # one for "how often does the detector fire on a *piece of software*", and the gap between them
    # is worth stating rather than hiding.
    if prov and ok:
        bysrc = defaultdict(list)
        for r in ok:
            bysrc[(prov.get(r["name"], {}) or {}).get("source", "unknown")].append(r)
        A("### Package-clustered estimate\n")
        A(f"- distinct source packages represented: **{len(bysrc)}**")
        for lbl, key in (("S_glob", "fire_glob"), ("S_spat", "fire_spat"), ("either", "fire_any")):
            per = [sum(1 for r in g if r[key] == "1") / len(g) for g in bysrc.values()]
            clustered = sum(per) / len(per) if per else float("nan")
            anyfire = sum(1 for g in bysrc.values() if any(r[key] == "1" for r in g))
            A(f"- `{lbl}`: package-mean fire rate **{fmt(clustered)}**; "
              f"packages with ≥1 firing **{anyfire}/{len(bysrc)} = {fmt(anyfire/len(bysrc))}**")
        A("")

    # ── breakdowns ────────────────────────────────────────────────────────────
    def breakdown(keyfn, title, colname, sort=None):
        groups = defaultdict(list)
        for r in ok:
            p = prov.get(r["name"])
            groups[keyfn(p, r)].append(r)
        A(f"## {SEC}{title}\n")
        A(f"| {colname} | n | S_glob fires | S_spat fires | either | both | median S_glob | median S_spat |")
        A("|---|---|---|---|---|---|---|---|")
        for k in sorted(groups, key=sort or (lambda k: -len(groups[k]))):
            g = groups[k]
            m = len(g)
            a = sum(1 for r in g if r["fire_glob"] == "1")
            b = sum(1 for r in g if r["fire_spat"] == "1")
            c = sum(1 for r in g if r["fire_any"] == "1")
            d = sum(1 for r in g if r["fire_both"] == "1")
            gv = [num(r, "s_glob") for r in g]
            sv = [num(r, "s_spat") for r in g]
            A(f"| `{k}` | {m} | {a} ({fmt(a/m,3)}) | {b} ({fmt(b/m,3)}) | {c} ({fmt(c/m,3)}) | "
              f"{d} ({fmt(d/m,3)}) | {fmt(pct(gv,0.5),3)} | {fmt(pct(sv,0.5),3)} |")
        A("")

    if prov and ok:
        breakdown(lambda p, r: (p or {}).get("lang", "unknown"),
                  "3. Fire rate by language / toolchain", "language")
        breakdown(lambda p, r: (p or {}).get("stripped", "unknown"),
                  "4. Fire rate by stripped-ness", "stripped")

    # size bucket — the obvious confound for a per-address mean
    def bucket(r):
        cb = num(r, "code_bytes")
        for lo, lab in [(8_000, "<8 KiB"), (32_000, "8-32 KiB"), (128_000, "32-128 KiB"),
                        (512_000, "128-512 KiB")]:
            if cb < lo:
                return lab
        return ">=512 KiB"
    if ok:
        breakdown(lambda p, r: bucket(r), "5. Fire rate by .text size", "size")

    # The diagnostic that explains the spatial false alarms. S_spat is Moran's I over the
    # address-ordered residual; with few decode candidates it is estimated from very little data and
    # becomes unstable, so it fires spuriously. Bucketing by candidate count rather than by byte size
    # isolates that directly, because the candidate count *is* the sample size of the statistic.
    def cbucket(r):
        c = num(r, "n_cand")
        for lo, lab in [(250, "<250"), (1000, "250-1k"), (4000, "1k-4k"), (16000, "4k-16k")]:
            if c < lo:
                return lab
        return ">=16k"
    order = {"<250": 0, "250-1k": 1, "1k-4k": 2, "4k-16k": 3, ">=16k": 4}
    if ok:
        breakdown(lambda p, r: cbucket(r), "5b. Fire rate by decode-candidate count (the sample size of S_spat)",
                  "candidates", sort=lambda k: order.get(k, 9))

    A(f"## {SEC}6. Routing: what the bare rule would do, and what the guard does\n")
    A(f"Routing thresholds: `S_glob > {RULE_GLOB_HI}` ⇒ obfuscated; "
      f"`S_spat > {RULE_SPAT_HI}` ⇒ packed; guard additionally demands `region_ent > {PACK_ENT_LO}`.\n")
    rp = defaultdict(int)
    gp = defaultdict(int)
    for r in ok:
        rp[r["rule_pick"]] += 1
        gp[r["guard_pick"]] += 1
    A("| regime routed to | bare rule | with guard |")
    A("|---|---|---|")
    for k in ("benign", "packed", "obfuscated"):
        A(f"| `{k}` | {rp[k]} ({fmt(rate(rp[k]),4)}) | {gp[k]} ({fmt(rate(gp[k]),4)}) |")
    switched_rule = rp["packed"] + rp["obfuscated"]
    switched_guard = gp["packed"] + gp["obfuscated"]
    vetoed = sum(1 for r in ok if r["guard_vetoed"] == "1")
    A("")
    A(f"- an **unguarded** system would have switched maps on **{switched_rule}/{n} = "
      f"{fmt(rate(switched_rule))}** of ordinary third-party software")
    A(f"- with the guard: **{switched_guard}/{n} = {fmt(rate(switched_guard))}**")
    A(f"- the guard vetoed **{vetoed}/{n} = {fmt(rate(vetoed))}** of decisions "
      f"({'all' if vetoed == switched_rule - switched_guard else 'some'} of the rule's switches)\n")

    A(f"## {SEC}7. The wild distribution against our coreutils-fit null\n")
    A("| statistic | wild p50 | wild p95 | wild max | clean-fit p95 (published null) |")
    A("|---|---|---|---|---|")
    A(f"| `S_glob` | {fmt(pct(globs,0.5))} | {fmt(pct(globs,0.95))} | {fmt(max(globs) if globs else float('nan'))} | {DET_GLOB_HI} |")
    A(f"| `S_spat` | {fmt(pct(spats,0.5))} | {fmt(pct(spats,0.95))} | {fmt(max(spats) if spats else float('nan'))} | {DET_SPAT_HI} |")
    A("")

    A(f"## {SEC}8. Both-prong firings (hand-inspection list)\n")
    both = sorted((r for r in ok if r["fire_both"] == "1"),
                  key=lambda r: -num(r, "s_glob"))
    if not both:
        A("None.\n")
    else:
        A("| binary | package | lang | S_glob | S_spat | region_ent | .text |")
        A("|---|---|---|---|---|---|---|")
        for r in both[:40]:
            p = prov.get(r["name"], {})
            A(f"| `{r['name']}` | {p.get('pkg','?')} {p.get('version','')} | {p.get('lang','?')} | "
              f"{fmt(num(r,'s_glob'),3)} | {fmt(num(r,'s_spat'),3)} | {fmt(num(r,'region_ent'),3)} | "
              f"{int(num(r,'code_bytes'))} |")
        A("")

    A(f"## {SEC}9. Worst offenders by S_glob\n")
    A("| binary | package | lang | S_glob | S_spat | .text |")
    A("|---|---|---|---|---|---|")
    for r in sorted(ok, key=lambda r: -num(r, "s_glob"))[:20]:
        p = prov.get(r["name"], {})
        A(f"| `{r['name']}` | {p.get('pkg','?')} | {p.get('lang','?')} | {fmt(num(r,'s_glob'),3)} | "
          f"{fmt(num(r,'s_spat'),3)} | {int(num(r,'code_bytes'))} |")
    A("")

    print("\n".join(out))


if __name__ == "__main__":
    main()
