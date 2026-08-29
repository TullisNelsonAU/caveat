#!/usr/bin/env bash
# Regenerate the results tables in ADAPTIVE_ADVERSARY_RESULTS.md from adaptive_adversary.csv.
#
# The tables are spliced in between the <!--RESULTS--> and <!--/RESULTS--> markers, so re-running
# after another substrate pair lands updates every number at once and none of them are ever
# hand-copied. The verdict and discrepancy prose above and below the markers is left alone.
#
# Exit status is the analyzer's: non-zero means a cell of Table V no longer reproduces from the CSV.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

TMP=$(mktemp)
rc=0
python3 analyze_adversary.py adaptive_adversary.csv >"$TMP" || rc=$?

python3 - "$TMP" <<'PY'
import sys, pathlib, re
tables = pathlib.Path(sys.argv[1]).read_text()
doc = pathlib.Path("ADAPTIVE_ADVERSARY_RESULTS.md")
if not doc.exists(): doc.write_text("# Results\n\n<!--RESULTS-->\n")
s = doc.read_text()
block = "<!--RESULTS-->\n\n" + tables.rstrip() + "\n\n<!--/RESULTS-->"
if "<!--/RESULTS-->" in s:
    s = re.sub(r"<!--RESULTS-->.*?<!--/RESULTS-->", lambda _: block, s, flags=re.S)
else:
    s = s.replace("<!--RESULTS-->", block)
doc.write_text(s)
print("spliced results into ADAPTIVE_ADVERSARY_RESULTS.md")
PY
rm -f "$TMP"

if [ "$rc" -ne 0 ]; then
  echo "!! a printed Table V cell did not reproduce — see the verification table" >&2
fi
exit "$rc"
