#!/usr/bin/env python3
"""Record which engine produced each results CSV.

Why this exists
---------------
Nothing in this repo captured the `probdisasm` commit a results CSV was produced against. The
dependency is a path dependency (`Cargo.toml:36` -> `../probdisasm`), so `Cargo.lock` pins only
`version = "0.2.2"` and never a revision; no run log records a commit either. The consequence was
observed rather than theorised: `docs/consistency_credibility/tigress_graded.csv` (pre-`c62ead9`)
sat beside `credibility.csv` (post-`c62ead9`) with identical schema, identical `n` and `code_bytes`
on the 30 binaries they share, and different statistics on all 30 -- and nothing on disk said so.

This tool writes a `manifest.json` beside each group of results CSVs. Two modes:

  stamp <dir> [dir...]    Record the engine as it is RIGHT NOW. Run this from a harness immediately
                          after producing a CSV. Reads probdisasm's HEAD, branch and dirty state
                          directly out of the engine checkout, so the record is observed, not
                          guessed, and is marked `source: "recorded"`.

  backfill <dir> [...]    For CSVs that already exist with no record. No log preserved the engine
                          commit, so nothing can be recovered as fact. What IS recoverable is a
                          bound: the engine cannot have been newer than the CSV, so the newest
                          probdisasm commit at or before the CSV's git commit date is an upper
                          bound, and everything older is a candidate. Emitted as
                          `source: "inferred_upper_bound"` with the reasoning attached, or
                          `source: "unknown"` when even that is unavailable. Never as "recorded".

`backfill` never overwrites a `recorded` entry.

usage: engine_manifest.py {stamp,backfill} DIR [DIR...] [--engine PATH]
"""
import argparse
import hashlib
import re
import json
import os
import subprocess
import sys

DEFAULT_ENGINE = os.path.expanduser("~/lab/projects/probdisasm")
SCHEMA = 1


def git(repo, *args):
    """Run git in `repo`; return stripped stdout, or None if the command fails."""
    try:
        out = subprocess.run(
            ["git", "-C", repo, *args], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_rows(path):
    """Row count excluding the header and any leading '#' banner (see tigress_graded.csv)."""
    n = 0
    seen_header = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if not seen_header:
                seen_header = True
                continue
            n += 1
    return n


def engine_state(engine):
    """probdisasm HEAD as it stands now. `dirty` matters: a dirty tree means the commit is a lie."""
    head = git(engine, "rev-parse", "HEAD")
    if head is None:
        return None
    status = git(engine, "status", "--porcelain")
    return {
        "commit": head,
        "short": head[:7],
        "branch": git(engine, "rev-parse", "--abbrev-ref", "HEAD"),
        "subject": git(engine, "log", "-1", "--format=%s"),
        "date": git(engine, "log", "-1", "--format=%cI"),
        "dirty": bool(status),
        "dirty_paths": sorted(l[3:] for l in status.splitlines()) if status else [],
    }


def engine_history(engine):
    """Every probdisasm commit reachable from any ref, newest first."""
    log = git(engine, "log", "--all", "--format=%H%x1f%cI%x1f%s")
    if not log:
        return []
    out = []
    for line in log.splitlines():
        h, date, subj = line.split("\x1f", 2)
        out.append({"commit": h, "short": h[:7], "date": date, "subject": subj})
    return out


def _data_lines(text):
    """A CSV's payload: header + rows, with '#' banner lines and blanks dropped."""
    return [l for l in text.splitlines() if l.strip() and not l.startswith("#")]


def csv_date(repo, rel):
    """When the CSV's *data* last changed, as an ISO date. Upper bound on when it was produced.

    Deliberately not "when the file was last committed". Annotating a stale CSV with a '#' banner is
    exactly the kind of curation this manifest is meant to encourage, and it must not silently
    re-date the file and destroy the bound -- which is precisely what happened to
    tigress_graded.csv the first time round. So walk the commits touching the file newest-first and
    stop at the first one whose payload differs from its parent's; comment-only commits are skipped.

    Returns (date, commit) or (None, None).
    """
    log = git(repo, "log", "--format=%H%x1f%cI", "--", rel)
    if not log:
        return None, None
    for line in log.splitlines():
        h, date = line.split("\x1f", 1)
        now_text = git(repo, "show", f"{h}:{rel}")
        if now_text is None:
            continue
        prev_text = git(repo, "show", f"{h}^:{rel}")
        if prev_text is None or _data_lines(prev_text) != _data_lines(now_text):
            return date, h
    return None, None


def infer(history, when):
    """Newest engine commit at or before `when`, plus the older candidates."""
    at_or_before = [c for c in history if c["date"] <= when]
    if not at_or_before:
        return None, []
    return at_or_before[0], at_or_before[1:4]


# An "Engine of record: probdisasm `feat/chainfwd-prior @ c62ead9`" line is the established way this
# repo already states which engine a probe ran against -- it appears in most *_RESULTS.md files and,
# as a JSON field, in docs/adaptive_adversary/run_manifest.json. That is a human record, written at
# run time, and it is far better evidence than a timestamp bound. Harvest it, but only after
# checking the hash against real probdisasm history, so a typo degrades to "unknown" rather than
# inventing a commit.
ENGINE_LINE = re.compile(r"engine", re.I)
HEXTOK = re.compile(r"\b([0-9a-f]{7,40})\b")


def harvest_declared(d, history):
    """Engine declarations written by hand in this directory's docs. Returns (commit, citation)."""
    by_prefix = {}
    for c in history:
        for n in range(7, 41):
            by_prefix.setdefault(c["commit"][:n], c)

    found = []
    names = sorted(f for f in os.listdir(d) if f.endswith((".md", ".json")))
    for name in names:
        if name == "manifest.json":
            continue
        path = os.path.join(d, name)
        try:
            text = open(path, errors="replace").read()
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if not ENGINE_LINE.search(line):
                continue
            for tok in HEXTOK.findall(line.lower()):
                c = by_prefix.get(tok)
                if c:
                    found.append((c, f"{name}:{lineno}", line.strip()))
                    break

    if not found:
        return None
    commits = {c["commit"] for c, _, _ in found}
    if len(commits) > 1:
        return {
            "source": "unknown",
            "commit": None,
            "reason": "this directory's docs declare conflicting engine commits: "
            + ", ".join(sorted(c["short"] for c in {x[0]["commit"]: x[0] for x in found}.values())),
            "citations": [{"at": at, "text": txt} for _, at, txt in found],
        }
    c, at, txt = found[0]
    return {
        "source": "recorded_in_doc",
        "commit": c["commit"],
        "short": c["short"],
        "date": c["date"],
        "subject": c["subject"],
        "citation": at,
        "citation_text": txt,
        "reason": "declared by hand in this directory's documentation and verified against "
        "probdisasm history; no run log recorded it independently",
    }


def load(path):
    if not os.path.exists(path):
        return {"schema": SCHEMA, "engine": "probdisasm", "files": {}}
    with open(path) as f:
        m = json.load(f)
    m.setdefault("files", {})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stamp", "backfill"])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    args = ap.parse_args()

    engine = os.path.expanduser(args.engine)
    now = engine_state(engine) if args.mode == "stamp" else None
    history = engine_history(engine) if args.mode == "backfill" else []
    if args.mode == "stamp" and now is None:
        sys.exit(f"engine checkout not readable at {engine}")
    if args.mode == "backfill" and not history:
        print(f"!! no engine history at {engine}; every entry will be 'unknown'", file=sys.stderr)

    for d in args.dirs:
        d = os.path.abspath(os.path.expanduser(d))
        csvs = sorted(f for f in os.listdir(d) if f.endswith(".csv"))
        if not csvs:
            continue
        repo = git(d, "rev-parse", "--show-toplevel") or d
        mpath = os.path.join(d, "manifest.json")
        man = load(mpath)
        man["schema"] = SCHEMA
        man["engine"] = "probdisasm"
        man["engine_path_dependency"] = "Cargo.toml -> ../probdisasm (no revision is pinned)"

        for name in csvs:
            path = os.path.join(d, name)
            prev = man["files"].get(name, {})
            if args.mode == "backfill" and prev.get("engine", {}).get("source") == "recorded":
                continue

            rel = os.path.relpath(path, repo)
            entry = {
                "sha256": sha256(path),
                "bytes": os.path.getsize(path),
                "data_rows": data_rows(path),
                "csv_data_last_changed": None,   # filled below
                "csv_data_last_changed_commit": None,
                "csv_tracked": git(repo, "ls-files", "--", rel) not in (None, ""),
            }
            entry["csv_data_last_changed"], entry["csv_data_last_changed_commit"] = csv_date(repo, rel)

            if args.mode == "stamp":
                entry["engine"] = dict(now, source="recorded")
                if now["dirty"]:
                    entry["engine"]["warning"] = (
                        "engine checkout was dirty when this CSV was produced; the commit alone "
                        "does not reproduce it"
                    )
            else:
                declared = harvest_declared(d, history)
                if declared is not None and declared["source"] == "recorded_in_doc":
                    entry["engine"] = declared
                    man["files"][name] = entry
                    continue
                when = entry["csv_data_last_changed"]
                if when is None:
                    entry["engine"] = {
                        "source": "unknown",
                        "commit": None,
                        "reason": "CSV is untracked, so there is no commit date to bound against",
                    }
                else:
                    best, older = infer(history, when)
                    if best is None:
                        entry["engine"] = {
                            "source": "unknown",
                            "commit": None,
                            "reason": f"no engine commit at or before {when}",
                        }
                    else:
                        entry["engine"] = {
                            "source": "inferred_upper_bound",
                            "commit": None,
                            "upper_bound": best,
                            "older_candidates": older,
                            "reason": (
                                "No run log recorded the engine commit. The engine cannot have been "
                                f"newer than the CSV, so {best['short']} is the newest it could have "
                                "been; any older commit is possible. This is a bound, not a record."
                            ),
                        }

            man["files"][name] = entry

        with open(mpath, "w") as f:
            json.dump(man, f, indent=2, sort_keys=True)
            f.write("\n")
        srcs = {v["engine"]["source"] for v in man["files"].values()}
        print(f"{os.path.relpath(mpath, os.getcwd())}: {len(man['files'])} csv  {sorted(srcs)}")


if __name__ == "__main__":
    main()
