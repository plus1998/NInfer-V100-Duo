#!/usr/bin/env bash
# tp1 regression gate: single-GPU behavior must not drift from upstream.
#
# Usage: scripts/tp1-regression.sh [artifact]
#   TP1_GPU  physical GPU index to run on (default 0)
#
# Assertion notes (observed against a real run of ./build/apps/ninfer; the
# goldens, their generator and the exact recording commands live in
# tests/data/tp1-golden/):
#   - The CLI never echoes the input prompt to stdout or stderr. stdout only
#     carries generated *content* (apps/cli/main.cpp: StreamingSink routes the
#     content channel to std::cout, reasoning to std::cerr). So the original
#     `grep "Count from 1 to 20" stdout` idea can never pass — it must assert
#     on the model's generated continuation instead, i.e. the counted numbers.
#   - The load summary's identity fields print on stderr as two separate
#     `summary` rows: `target` (e.g. "qwen3_8_27b") and `weights` (e.g.
#     "nvfp4") -- there is no single combined "qwen3_8_27b/qwen3.8-27b/nvfp4"
#     line, so identity is checked as two independent patterns.
#   - Runs with --greedy (forces temperature 0.0, apps/cli/options.cpp:187) so
#     the generated continuation is deterministic. This gate re-runs on every
#     later commit and asserts on specific generated tokens, so it must not
#     be a stochastic sampler; --greedy is also what the tp1-vs-tp2 parity
#     harness (tools/tp2/parity.cpp) uses.
#
# The checks above are a smoke test: they prove the single-GPU path still runs
# and still identifies itself, but on their own they could not have caught a
# token-level drift away from upstream. The golden comparison at the end of this
# script is the actual identity gate -- three greedy cases whose token ids were
# recorded from upstream feaf4dd's own binary and this branch's, byte-identical
# on both. See tests/data/tp1-golden/MANIFEST.md for how the baseline was taken
# and exactly what it does and does not cover.
set -euo pipefail
cd "$(dirname "$0")/.."

ART="${1:-/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer}"
GPU="${TP1_GPU:-0}"
OUT="$(mktemp /tmp/tp1-regression.out.XXXXXX)"
ERR="$(mktemp /tmp/tp1-regression.err.XXXXXX)"

fail() {
    echo "TP1-REGRESSION: FAIL — $1"
    echo "--- stdout ($OUT) ---"
    cat "$OUT" 2>/dev/null || true
    echo "--- stderr ($ERR) ---"
    cat "$ERR" 2>/dev/null || true
    exit 1
}

if [[ ! -f "$ART" ]]; then
    fail "artifact not found: $ART"
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
    ./build/apps/ninfer "$ART" \
    --prompt "Count from 1 to 20, one number per line." \
    --max-context 4096 --max-new 64 --kv-dtype int8 --greedy \
    >"$OUT" 2>"$ERR" || fail "ninfer exited non-zero"

# Generated-continuation check: the model must actually produce the counting
# sequence on stdout, not just echo the prompt (it never echoes the prompt at
# all — see notes above). Require the first three counted numbers, each on
# its own line, in order.
grep -qx "1" "$OUT" && grep -qx "2" "$OUT" && grep -qx "3" "$OUT" ||
    fail "generated continuation missing expected sequential numbers 1/2/3 on stdout"

# Identity check: load-summary target + weights rows on stderr.
grep -qE "^summary +target +qwen3_8_27b" "$ERR" ||
    fail "load summary missing target=qwen3_8_27b"
grep -qE "^summary +weights +nvfp4" "$ERR" ||
    fail "load summary missing weights=nvfp4"

# --- golden-token identity gate -----------------------------------------------
#
# Re-records the three cases with THIS build and requires byte equality against
# the committed goldens: token ids, generated text, and the deterministic
# summary rows. Adds roughly 35 s (three loads plus a 29k-token prefill).
#
# Set TP1_SKIP_GOLDEN=1 to run only the smoke checks above; this is for a host
# without the NVFP4 artifact, not for a failing comparison.
GOLDEN_DIR="tests/data/tp1-golden"
if [[ "${TP1_SKIP_GOLDEN:-0}" == "1" ]]; then
    echo "TP1-REGRESSION: golden gate SKIPPED (TP1_SKIP_GOLDEN=1)"
else
    RECORDED="$(mktemp -d /tmp/tp1-golden.XXXXXX)"
    trap 'rm -rf "$RECORDED"' EXIT
    TP1_GPU="$GPU" "$GOLDEN_DIR/record.sh" ./build/apps/ninfer "$ART" "$RECORDED" >/dev/null ||
        fail "golden recording failed"
    for case_index in 1 2 3; do
        for kind in ids txt summary; do
            expected="$GOLDEN_DIR/case-$case_index.$kind"
            actual="$RECORDED/case-$case_index.$kind"
            [[ -f "$expected" ]] || fail "missing golden file: $expected"
            cmp -s "$expected" "$actual" || {
                echo "TP1-REGRESSION: FAIL — case $case_index $kind differs from the upstream-feaf4dd golden"
                diff -u "$expected" "$actual" || true
                echo "This is a tp1 identity break: --tp 1 no longer reproduces upstream."
                echo "Do not re-record the golden to make this pass; find the change that moved it."
                exit 1
            }
        done
    done
    echo "TP1-REGRESSION: golden gate PASS (3 cases, ids + text + summary byte-equal)"
fi

echo "TP1-REGRESSION: PASS"
echo "identity OK (target=qwen3_8_27b, weights=nvfp4)"
