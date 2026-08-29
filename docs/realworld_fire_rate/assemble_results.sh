#!/usr/bin/env bash
# Regenerate the results tables in REALWORLD_FIRE_RATE_RESULTS.md from the CSVs.
#
# The tables are spliced in between the <!--RESULTS--> and <!--/RESULTS--> markers, so re-running
# after more binaries land updates every number at once and none of them are ever hand-copied. The
# verdict paragraph above the marker is prose and is left alone.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

TMP=$(mktemp)
{
  echo "# Results"
  echo
  echo "## Stratum A — census (\`.text\` ≤ 256 KiB, exhaustive)"
  echo
  python3 analyze_firerate.py firerate.csv A
  if [ "$(grep -c "" firerate_large.csv 2>/dev/null || echo 0)" -gt 1 ]; then
    echo
    echo "## Stratum B — large-tail sample (256 KiB < \`.text\` ≤ 768 KiB, deterministic sample)"
    echo
    echo "Sampled, not exhaustive — read these as a probe of the large tail, not a population rate."
    echo
    python3 analyze_firerate.py firerate_large.csv B
  fi
} >"$TMP"

python3 - "$TMP" <<'PY'
import sys, pathlib, re
tables = pathlib.Path(sys.argv[1]).read_text()
doc = pathlib.Path("REALWORLD_FIRE_RATE_RESULTS.md")
if not doc.exists(): doc.write_text("# Results\n\n<!--RESULTS-->\n")
s = doc.read_text()
block = "<!--RESULTS-->\n\n" + tables.rstrip() + "\n\n<!--/RESULTS-->"
if "<!--/RESULTS-->" in s:
    s = re.sub(r"<!--RESULTS-->.*?<!--/RESULTS-->", lambda _: block, s, flags=re.S)
else:
    s = s.replace("<!--RESULTS-->", block)
doc.write_text(s)
print("spliced results into REALWORLD_FIRE_RATE_RESULTS.md")
PY
rm -f "$TMP"
