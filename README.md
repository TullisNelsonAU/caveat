# CAVEAT

A calibrated disassembler reports a confidence for every address it decodes. That
calibration is fitted once on a labeled corpus, and it goes stale silently when
the input drifts away from that corpus — packing, desynchronization, obfuscation.
The disassembler keeps reporting confident numbers and nothing in its output says
they've stopped meaning anything.

CAVEAT notices, using a quantity the disassembler already computed. In a loopy
belief-propagation disassembler the posterior at each address is the product of
the unary evidence attached to that address and the belief the rest of the graph
propagates to it. Divide the unary factor back out and you have a prediction the
address's own evidence never touched. Score the evidence against that prediction
and you get a per-address surprise — for free, with no labels, no second model,
and no re-running inference. Aggregate it two ways, and the shape of the result
tells you which regime you're in well enough to pick the right calibration map.

This repository is everything behind that: the engine, the statistics, the
corpora scripts, the per-binary results, and the analysis that turns them into
the numbers in the paper.

## Getting it running

You need Rust 1.85 or newer and Python 3.8 or newer. Nothing else — the
inference engine is vendored in `engine/probdisasm` and the ground-truth
generator is `crates/groundtruth`, so there's nothing to fetch from anywhere.

```
cargo build --release
```

That takes a minute or two and builds all 18 crates. The first build needs
network access for crates.io; after that you can work offline, and
`--locked` works if you prefer it.

To check it actually works, run this — it takes about five seconds and needs
nothing built at all, since it reads a committed CSV:

```
cd docs/consistency_credibility
python3 analyze_credibility.py
```

You should get a table of six drift levels, with true calibration error climbing
from 0.003 on clean code to 0.369 on packed, and the correlations printed
underneath. If you see that, you're set.

## Running CAVEAT on a binary

Everything above reproduces numbers from data we already collected. If you want
to point the actual system at a binary and watch it decide, that's `tool`:

```
cargo build --release
./target/release/tool <binary> --bank experimental/tool/banks/upd_bank.json --report
```

There's a fitted calibration bank checked in at that path with all three maps in
it, so this works out of the box. Some binaries you can try are in
`corpus_pie/seeds/` (clean) and `docs/small_packed/corpus/` (packed).

What comes back, in order: the regime it detected and how confident it is; which
calibration map it selected and whether it actually applied it; the two
statistics `S_glob` and `S_spat`; the address ranges it thinks you should not
trust, if the surprise clustered anywhere; the instruction candidates with their
mean posterior and a count of low-confidence ones; and the recovered functions
with a confidence and a reachedness score each, with anything that looks like a
decoy flagged.

Two flags worth knowing. `--out result.json` writes the whole thing as JSON
instead of just printing the summary, which is what you want if you're feeding
it into something else. `--full-insns` includes every instruction rather than
the summary counts.

If you drop `--bank`, the tool still detects the regime and applies that
regime's engine configuration, but it won't apply the fitted isotonic map — so
you get the routing half of the result without the recalibration half. The
`--bank` form is the one that corresponds to the paper's restoration numbers.

The rule is benign-by-default: a signature it doesn't recognize routes to benign
rather than being forced into a map that could make calibration worse. When it
sees drift it can't place, it says so — `regime_uncertain`, "calibration may be
stale" — instead of asserting the input is clean. That is the guard from the
paper, and it's why the worst case here is the unmodified pipeline.

Scope, stated plainly: this routes structural obfuscation, meaning packing and
anti-disassembly desynchronization, where the surprise is diagnostic. Semantic
obfuscation such as Tigress virtualization preserves clean decoding, so the
statistic is blind to it and the tool will report benign. That's the limit the
paper scopes out, not a bug.

## Running everything

Every experiment lives in its own directory under `docs/`, and every one of them
works the same way: there's an `analyze_*.py` that reads the committed CSVs next
to it and prints the tables. None of them take arguments.
So the whole thing is:

```
for d in docs/*/; do
  for s in "$d"analyze_*.py; do
    [ -f "$s" ] && (cd "$d" && python3 "$(basename "$s")")
  done
done
```

That runs in well under a minute on a laptop. Two of them will fail, and that's
expected — see the rough edges below.

Here's what's where, and which part of the paper it backs:

| directory | what it shows |
|---|---|
| `consistency_credibility` | the surprise tracks true calibration error across a graded drift ladder; also the detection thresholds and the comparison against confidence and OOD baselines |
| `packer_breadth` | it generalizes across five packer configurations from three tools, including ones absent from the fit |
| `adaptive_adversary` | fifteen constructions built specifically to evade it, over two independent substrate pairs |
| `ablations` | both statistics are needed; neither alone routes correctly |
| `realworld_fire_rate` | the detector, unchanged and unretuned, on 1095 Debian binaries we didn't build |
| `spatial_null_repair` | why the spatial threshold was wrong for short binaries, and the size-aware gate that fixes it |
| `consistency_switching` | selecting a calibration map from the surprise signature, and what that does to calibration error |
| `tigress_reconcile` | the limit: semantic obfuscation that preserves clean decoding |
| `selective_disasm` | what a stale map costs an analyst who asked for a precision guarantee |

The remaining directories under `docs/` are exploratory arms that don't back
anything in the paper. They're here for completeness.

## About the ground truth

No label anywhere in this artifact comes from a disassembler. Every one comes
from the construction that produced the binary: the symbol table and section map
for ordinary programs, the injector's own record of which bytes it inserted for
desynchronized ones, and the packer's provable data-payload window for packed
images. Where construction can't give us a label, the corresponding experiment
reports no calibration error rather than falling back on a weaker oracle.
`crates/groundtruth/GROUNDTRUTH_FORMALISM.md` defines exactly what `gen-gt`
emits and why.

Corpus construction is seeded, so the splits regenerate exactly.

## Rebuilding the corpora (you probably don't need to)

Everything above runs from committed data. If you want to rebuild the corpora
from scratch, you'll need tools we can't ship: Tigress 4.0.12 (licence-gated,
academic licence required from the University of Arizona), UPX 4.2.4,
kiteshield, the ezuri Go crypter, desync-cc, and an x86-64 cross gcc. Two
directories also reference a `packerbox` Docker image that bundled the packers —
we don't have a Dockerfile for it, so those scripts are here to document how the
corpora were made rather than as a path you can run.

The wild corpus isn't redistributed either. It's 1095 unmodified Debian
bookworm binaries, better fetched from the archive than mirrored by us;
`docs/realworld_fire_rate/build_corpus.sh` pins the release and deduplicates by
content hash, and `provenance.csv` records package, version, build ID and
language for every one. Nothing was installed or executed — we only read ELF
bytes.

## Rough edges, honestly

`docs/regime_calibration/analyze_optregime.py` needs scikit-learn, and
`docs/ollvm_staleness/analyze_ollvm.py` looks for a CSV we never committed. Both
are exploratory and back nothing in the paper. Everything else runs.

The semantic-obfuscation arm was scored three separate times over slightly
different corpora, and the numbers differ in the second decimal between them.
`analyze_semantic_table.py` defaults to the boundary corpus, which is what the
paper reports; pass `--rerun` to see the later pass instead. One binary,
`p05_vm`, has a different candidate count in each of the three, which is why two
sections of the paper quote different figures for the same program.

The wild corpus is 1095 binaries out of 1540 harvested. The other 445 were
dropped for having more than about 256 KiB of `.text`. Since the headline
spatial finding is size-dependent, that's worth knowing, though the direction is
conservative — large binaries have the lowest false-alarm rate under the flat
gate.

The corpus-build scripts default their paths to the layout we developed on. All
of them take environment variable overrides (`BINS`, `CORP`, `PROB`, `DEST`),
and none are on the path you need to reproduce results.

## License

MIT, see `LICENSE`. The vendored engine keeps its own MIT license under
`engine/probdisasm/LICENSE`.
