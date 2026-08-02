#!/usr/bin/env bash
# Run the `-m gpu` test lane as one separate `pytest` process per test
# module, instead of one process for the whole lane.
#
# Why this exists (see DESIGN.md's V9.1 "known issue" note for the full
# writeup): the gpu lane mixes multiple libraries that each own/terminate a
# process-wide EGL context -- pyrender's session-scoped `episode_renderer`
# fixture (`tests/conftest.py`) binds to the *shared default* EGL display,
# and `mujoco.Renderer` (used directly in `tests/test_crosscheck.py`, and
# transitively any other module that renders) owns a second, independent
# EGL context of its own. Running the whole lane in one pytest process asks
# both to coexist (and, in `render_mujoco_frame0` pre-V9.1, to do so
# in-process) in the same address space -- OpenGL/EGL context state is not
# designed to be shared or safely torn down across independent owners like
# that, so the *first* time a second EGL-context-owning renderer touches
# the process, it can corrupt or tear down the first one for every test
# that runs afterward in the same session (observed: cascading
# `EGLError: EGL_NOT_INITIALIZED` failures, or in the worst case a hard
# process crash with no Python traceback at all).
#
# One pytest process per test *module* guarantees each module's renderer(s)
# get a fresh process and a fresh EGL display, so no two independent
# EGL-context owners ever have to coexist in the same address space. This
# is strictly coarser than one-process-per-*test* (a module with several
# gpu tests still shares one process, and hence one `episode_renderer`
# fixture instance, across all of them) -- deliberately: it matches the
# existing `owns_renderer`/reuse convention already used within a module
# (see `tests/conftest.py`'s `episode_renderer` fixture docstring and
# `gltfworld.eval.closed_loop.run`), and keeps per-module runtime
# reasonable for modules with several gpu tests (e.g. the training-smoke
# modules) while still fixing the actual cross-module contention.
#
# Usage: scripts/run_gpu_tests.sh [extra pytest args, applied to every module]
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

EXTRA_ARGS=("$@")

echo "== collecting gpu-marked test modules =="
COLLECTED=$(uv run pytest -m gpu --collect-only -q 2>&1)
COLLECT_STATUS=$?
if [ $COLLECT_STATUS -ne 0 ]; then
    echo "$COLLECTED"
    echo "FATAL: pytest collection itself failed (exit $COLLECT_STATUS); cannot proceed." >&2
    exit 1
fi

# Collected node ids look like "tests/test_foo.py::test_bar[param]"; take the
# file path (everything before the first "::"), dedupe, keep first-seen order.
mapfile -t MODULES < <(echo "$COLLECTED" | grep -E '^tests/.*\.py::' | cut -d: -f1 | awk '!seen[$0]++')

if [ ${#MODULES[@]} -eq 0 ]; then
    echo "No gpu-marked tests collected -- nothing to run."
    echo "$COLLECTED"
    exit 0
fi

echo "Found ${#MODULES[@]} gpu-marked module(s):"
printf '  %s\n' "${MODULES[@]}"
echo

declare -a RESULTS
OVERALL_STATUS=0
START_ALL=$(date +%s)

for module in "${MODULES[@]}"; do
    echo "== running: uv run pytest ${module} -m gpu -v ${EXTRA_ARGS[*]:-} =="
    START=$(date +%s)
    if uv run pytest "$module" -m gpu -v "${EXTRA_ARGS[@]}"; then
        STATUS="PASS"
    else
        STATUS="FAIL"
        OVERALL_STATUS=1
    fi
    END=$(date +%s)
    RESULTS+=("${module}|${STATUS}|$((END - START))s")
    echo
done

END_ALL=$(date +%s)

echo "======================================================================"
echo "per-module gpu lane results (each module = its own pytest process)"
echo "======================================================================"
printf '%-55s %-6s %s\n' "module" "result" "time"
printf '%-55s %-6s %s\n' "------" "------" "----"
for row in "${RESULTS[@]}"; do
    IFS='|' read -r module status elapsed <<< "$row"
    printf '%-55s %-6s %s\n' "$module" "$status" "$elapsed"
done
echo "----------------------------------------------------------------------"
echo "total wall time: $((END_ALL - START_ALL))s"

if [ $OVERALL_STATUS -ne 0 ]; then
    echo
    echo "RESULT: at least one gpu module FAILED in isolation (a real bug, not"
    echo "cross-module EGL contention -- isolation alone can't paper over that)."
else
    echo
    echo "RESULT: all gpu modules passed in isolation."
fi

exit $OVERALL_STATUS
