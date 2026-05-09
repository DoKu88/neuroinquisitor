#!/usr/bin/env bash
set -euo pipefail

EXAMPLES_DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=() FAIL=()

for script in "$EXAMPLES_DIR"/*.py; do
    [[ "$(basename "$script")" == *_utils.py ]] && continue
    [[ "$(basename "$script")" == run_all.py  ]] && continue

    echo "━━━ $(basename "$script") ━━━"
    if python "$script"; then
        PASS+=("$(basename "$script")")
    else
        FAIL+=("$(basename "$script")")
        echo "  ✗ failed (exit $?)"
    fi
    echo
done

echo "━━━ Results ━━━"
echo "  Passed : ${#PASS[@]}  (${PASS[*]:-none})"
echo "  Failed : ${#FAIL[@]}  (${FAIL[*]:-none})"

[[ ${#FAIL[@]} -eq 0 ]]
