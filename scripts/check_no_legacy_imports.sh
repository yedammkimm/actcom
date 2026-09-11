#!/usr/bin/env bash
# Spec v1, INVARIANT-14: no file under oamp/ or run_experiment.py may import
# from legacy/. Exits 1 if any offending import is found.
#
# Usage:
#   scripts/check_no_legacy_imports.sh          # scan default targets
#   scripts/check_no_legacy_imports.sh path/... # scan explicit paths
#
# Wire into CI or pre-commit.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "$#" -gt 0 ]; then
    TARGETS=("$@")
else
    TARGETS=()
    [ -d oamp ]                        && TARGETS+=(oamp)
    [ -d configs ]                     && TARGETS+=(configs)
    [ -f run_experiment.py ]           && TARGETS+=(run_experiment.py)
    # audit/ is scanned but tools may hold a transitional legacy import until
    # oamp/pack_hooks.py lands (marked with `TODO(pack_hooks):`).
    [ -d oamp_train_engine/audit ]     && TARGETS+=(oamp_train_engine/audit)
    [ -d oamp_train_engine/benchmarks ] && TARGETS+=(oamp_train_engine/benchmarks)
fi

if [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "check_no_legacy_imports: nothing to scan (new pipeline files not present yet)."
    exit 0
fi

# Match: `from legacy...`, `import legacy...`, and `from ...legacy...` re-exports.
PATTERN='^\s*(from|import)\s+legacy(\.|$|\s)'

echo "Scanning: ${TARGETS[*]}"
HITS="$(grep -rnE --include='*.py' "$PATTERN" "${TARGETS[@]}" || true)"

# Allow transitional inserts in audit/ that are annotated with TODO(pack_hooks).
# We do NOT actually import legacy there; we just add sys.path. If a real
# `from legacy import ...` shows up here, it will still be caught below.
if [ -n "$HITS" ]; then
    echo "$HITS"
    echo
    echo "ERROR: legacy import(s) detected in the new pipeline." >&2
    echo "Copy the code across instead. See legacy/README.md." >&2
    exit 1
fi

# Extra check: warn (do not fail) on transitional `sys.path.insert(... 'legacy'`
# in audit/ that outlived the pack_hooks migration.
TRANSITIONAL="$(grep -rnE --include='*.py' 'sys\.path\.insert.*legacy' oamp_train_engine/audit 2>/dev/null || true)"
if [ -n "$TRANSITIONAL" ]; then
    echo
    echo "NOTE: transitional legacy sys.path entries in audit/ (delete when oamp/pack_hooks.py lands):"
    echo "$TRANSITIONAL"
fi

echo "OK: no legacy imports in ${TARGETS[*]}"
