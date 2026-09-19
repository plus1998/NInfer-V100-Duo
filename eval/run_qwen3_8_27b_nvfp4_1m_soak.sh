#!/usr/bin/env bash
# The 1,048,576-token soak under TP2 + YaRN x4.
#
# One request, greedy, non-thinking, from a ~950k-token public-domain prompt, decoded until the
# 1M window is physically full (~98.7k generated tokens), then a SECOND byte-identical pass.
#
# Gates
#   no OOM / no crash   both passes exit 0 and the CLI reports a terminal finish reason.
#   logit sanity        every 10,000-token window of the generated stream is checked for an
#                       out-of-vocabulary id, for a single token id taking more than 95% of the
#                       window, and for a long constant run. A NaN/inf-poisoned logit vector under
#                       argmax collapses top-1 onto a constant, so a window that stays diverse is
#                       the observable form of "the logits stayed finite" available from the CLI.
#                       The per-window distinct-token ratio is reported alongside.
#   VRAM flat           nvidia-smi sampled every 2 s for the life of each pass; the per-(process,
#                       GPU) series must not grow after the model is resident. The baseline is
#                       28,070 MiB = 27.41 GiB per device.
#   determinism         the two passes' generated token-id streams must be byte-identical
#                       (sha256 over the id stream), at temperature 0.
#
# Steps
#   prompt    build the prompt once, from the needle suite's distinct-text Gutenberg bundle A,
#             sized with the real tokenizer to ~950,000 tokens. Written once and reused by both
#             passes so the determinism comparison is over identical input bytes.
#   pass-a    the soak, pass 1.
#   pass-b    the soak, pass 2 -- same binary, same flags, same prompt file.
#   analyze   window table, VRAM stats, sha256s, timings ->
#             eval/results/soak-1m/soak-summary-w<window>.{md,json}.
#   all       prompt -> pass-a -> pass-b -> analyze  (~2 h of GPU time).
#
# `--max-new` is deliberately passed as the whole window (1048576). Engine clamps the output budget
# to `max_context - prompt_tokens + 1` and reports `finish reason context-capacity`, so the decode
# length is accounted from the engine's OWN prompt-token count rather than from an offline estimate.
#
# GPU discipline: a pass REFUSES to launch while any compute process holds GPU
# memory, asserts the list is empty again after exit, and never signals a process it did not start.
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cli_bin="${repo_dir}/build/apps/ninfer"
artifact="${NINFER_QWEN3_8_27B_NVFP4_ARTIFACT:-/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer}"
log_dir="${repo_dir}/eval/server-logs"
work_dir="${repo_dir}/eval/runs/soak-1m"
results_dir="${repo_dir}/eval/results/soak-1m"

# The 1M window, and the memory bound the boot gate measured against. NINFER_SOAK_WINDOW_TOKENS
# exists so the whole pipeline (prompt -> two passes -> analyze) can be validated end to end in a
# couple of minutes at a small window before ~2 h of GPU time is committed to the real one.
window_tokens="${NINFER_SOAK_WINDOW_TOKENS:-1048576}"
gate_mib=$((30 * 1024))

# Prompt target, in tokens as the tokenizer counts them before the chat template is applied.
# 950,000 leaves ~98.7k tokens of generation headroom inside the 1,048,576 window.
prompt_target_tokens="${NINFER_SOAK_PROMPT_TOKENS:-950000}"
window_size="${NINFER_SOAK_WINDOW:-10000}"
degenerate_share="${NINFER_SOAK_DEGENERATE_SHARE:-0.95}"

# The needle suite's distinct-text corpus. Bundle A is the adapter's `english` slot; the content
# is War and Peace + Great Expectations + Pride and Prejudice, not Paul Graham essays.
corpus_file="${NINFER_SOAK_CORPUS:-/home/pc/models/ninfer-38/needle-1m/PaulGraham_Essays.txt}"
tokenizer_json="${NINFER_1M_TOKENIZER_JSON:-/home/pc/models/ninfer-38/unsloth-nvfp4/tokenizer.json}"
tokenizer_python="${NINFER_1M_TOKENIZER_PYTHON:-${repo_dir}/eval/.venv/bin/python}"

vram_sample_seconds="${NINFER_VRAM_SAMPLE_SECONDS:-2}"

prompt_json="${work_dir}/soak-prompt-${prompt_target_tokens}.messages.json"
# Cache key for the prompt: it is a pure function of these three inputs, so a cached document is
# only reusable while all three still hash to what produced it. Recorded beside the prompt rather
# than baked into its name, so the file stays readable.
prompt_inputs="${prompt_json}.inputs"

# Every per-run name is keyed on the window. Without this an 8k pipeline validation and a real 1M
# soak share their `.latest` symlinks and their summary filenames, and the cheap validation silently
# overwrites the tracked 1M evidence (which is exactly what happened once, and had to be undone by
# hand with `ln -sfn`).
result_stem="soak-summary-w${window_tokens}"
latest_link() { printf '%s/qwen3_8_27b_nvfp4-1m-soak-%s-w%s.latest' "${log_dir}" "$1" "${window_tokens}"; }

# The 1M boot-gate configuration, verbatim, plus the three flags the soak needs:
#   --no-thinking     the whole budget belongs to the answer channel
#   --ignore-eos      decode is not allowed to stop on the model's end-of-turn token; the soak is
#                     a fixed-length decode to the context ceiling
#   --print-token-ids the generated id stream, which is what the sanity and determinism gates read
engine_flags=(
    --tp 2
    --devices "${NINFER_1M_DEVICES:-0,1}"
    --rope yarn
    --yarn-factor 4.0
    --yarn-origin 262144
    --max-context "${window_tokens}"
    --kv-dtype int8
    --kv-capacity "${NINFER_1M_KV_CAPACITY:-auto}"
    --greedy
    --no-thinking
    --ignore-eos
    --print-token-ids
)
if [[ -n "${NINFER_1M_PREFILL_CHUNK:-}" ]]; then
    engine_flags+=(--prefill-chunk "${NINFER_1M_PREFILL_CHUNK}")
fi
if [[ -n "${NINFER_1M_NO_CUDA_GRAPH:-}" ]]; then
    engine_flags+=(--no-cuda-graph)
fi

usage() {
    echo "usage: $0 [prompt|pass-a|pass-b|analyze|all]" >&2
    echo "       default: all" >&2
}

case "${1:-all}" in
    prompt)  steps=(prompt) ;;
    pass-a)  steps=(pass-a) ;;
    pass-b)  steps=(pass-b) ;;
    analyze) steps=(analyze) ;;
    all)     steps=(prompt pass-a pass-b analyze) ;;
    *)       usage; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then usage; exit 2; fi

for required in "${cli_bin}" "${artifact}"; do
    if [[ ! -f "${required}" ]]; then
        echo "missing required file: ${required}" >&2
        exit 1
    fi
done
if [[ ! -x "${cli_bin}" ]]; then
    echo "${cli_bin} must be executable" >&2
    exit 1
fi
mkdir -p -- "${log_dir}" "${work_dir}" "${results_dir}"

sampler_pid=""
cleanup() {
    if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
        kill -TERM "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
    fi
    sampler_pid=""
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# One CSV row per GPU and per (compute process, GPU) per tick: timestamp_utc,kind,key,used_mib.
# Identical to the boot gate's sampler: a TP2 run appears once per GPU it touches, so keying on
# the pid alone would hide an asymmetric split.
sample_vram() {
    local out="$1" interval="$2" uuid_map now
    printf 'timestamp_utc,kind,key,used_mib\n' >"${out}"
    uuid_map="$(nvidia-smi --query-gpu=uuid,index --format=csv,noheader,nounits 2>/dev/null \
        | tr -d '[:blank:]')"
    while true; do
        now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | awk -v ts="${now}" -F', *' '{print ts ",gpu," $1 "," $2}' >>"${out}" 2>/dev/null || true
        nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits \
            | awk -v ts="${now}" -v map="${uuid_map}" -F', *' '
                BEGIN {
                    rows = split(map, line, "\n")
                    for (i = 1; i <= rows; ++i) {
                        if (split(line[i], field, ",") == 2) { index_of[field[1]] = field[2] }
                    }
                }
                { gpu = ($2 in index_of) ? index_of[$2] : $2
                  print ts ",process," $1 "@gpu" gpu "," $3 }' >>"${out}" 2>/dev/null || true
        sleep "${interval}"
    done
}

compute_app_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | tr -d '[:blank:]' | grep -E '^[0-9]+$' || true
}

# Records the compute-app list AND the power condition. Every timing this soak reports is a
# power-limited measurement (these cards default to 600 W and are capped at 400 W on this host), so
# the artifact has to carry the cap it was taken under -- the same reason the needle sampler
# records `power.limit` (commit 58d6566).
record_compute_apps() {
    local label="$1" out="$2" rows power
    rows="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv 2>&1)"
    power="$(nvidia-smi --query-gpu=index,power.limit,power.max_limit,power.draw --format=csv 2>&1)"
    {
        printf '=== compute processes %s (%s) ===\n' "${label}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '%s\n' "${rows}"
        printf '=== power condition %s ===\n' "${label}"
        printf '%s\n' "${power}"
    } >>"${out}"
    printf 'compute processes %s:\n' "${label}"
    printf '%s\n' "${rows}" | sed 's/^/  /'
    printf 'power %s:\n' "${label}"
    printf '%s\n' "${power}" | sed 's/^/  /'
}

summary_value() { grep -E "^summary +$1" "$2" | head -1 | sed -E "s/^summary +$1 +//"; }

# ---------------------------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------------------------
prompt_input_key() {
    printf 'target=%s\ncorpus=%s\ntokenizer=%s\n' \
        "${prompt_target_tokens}" \
        "$(sha256sum "${corpus_file}" | cut -d' ' -f1)" \
        "$(sha256sum "${tokenizer_json}" | cut -d' ' -f1)"
}

build_prompt() {
    if [[ ! -f "${corpus_file}" ]]; then
        echo "missing needle corpus bundle: ${corpus_file}" >&2
        echo "rebuild it with tools/tp2/build_needle_1m_corpus.py" >&2
        return 1
    fi
    if [[ ! -f "${tokenizer_json}" ]]; then
        echo "missing tokenizer: ${tokenizer_json}" >&2
        return 1
    fi
    if [[ -s "${prompt_json}" && -s "${prompt_inputs}" ]] \
        && [[ "$(cat "${prompt_inputs}")" == "$(prompt_input_key)" ]]; then
        echo "prompt already built from the same inputs: ${prompt_json}"
        echo "  sha256 $(sha256sum "${prompt_json}" | cut -d' ' -f1)"
        return 0
    fi
    if [[ -s "${prompt_json}" ]]; then
        echo "prompt exists but its recorded inputs do not match the corpus/tokenizer on disk;"
        echo "rebuilding: ${prompt_json}"
    fi
    if [[ ! -x "${tokenizer_python}" || ! -f "${tokenizer_json}" ]]; then
        echo "the prompt builder needs a tokenizer:" >&2
        echo "  python:    ${tokenizer_python}" >&2
        echo "  tokenizer: ${tokenizer_json}" >&2
        return 1
    fi
    "${tokenizer_python}" - "${prompt_json}" "${prompt_target_tokens}" "${corpus_file}" \
        "${tokenizer_json}" <<'PY'
import json
import sys

from tokenizers import Tokenizer

out_json, target, corpus_path, tokenizer_json = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
tok = Tokenizer.from_file(tokenizer_json)

PREAMBLE = (
    "The following is a long excerpt from a public-domain novel. Read all of it.\n\n"
    "=== BEGIN EXCERPT ===\n"
)
CLOSING = (
    "\n=== END EXCERPT ===\n\n"
    "Continue the narrative from the point where the excerpt stops. Write in the same register "
    "as the excerpt, keep the established characters and setting, and keep going at length."
)

frame_tokens = len(tok.encode(PREAMBLE + CLOSING, add_special_tokens=False).ids)
# The chat template wraps the user turn; leave room rather than overshoot the 1,048,576 ceiling.
budget = target - 128 - frame_tokens

# One tokenizer pass over a generous character prefix, then cut on an exact token boundary using
# the encoding's own character offsets. No re-encode/decode round trip, so the excerpt is a literal
# prefix of the corpus file.
raw = open(corpus_path, encoding="utf-8").read()
approx_chars = min(len(raw), int(budget * 6))
head = raw[:approx_chars]
enc = tok.encode(head, add_special_tokens=False)
if len(enc.ids) < budget:
    raise SystemExit(
        "corpus prefix is too short: {} tokens available, {} needed".format(len(enc.ids), budget)
    )
cut_char = enc.offsets[budget][0]
excerpt = head[:cut_char]

text = PREAMBLE + excerpt + CLOSING
measured = len(tok.encode(text, add_special_tokens=False).ids)
with open(out_json, "w", encoding="utf-8") as handle:
    json.dump([{"role": "user", "content": text}], handle)

print(
    "prompt: {} raw tokens, {} chars ({} excerpt chars, {} frame tokens); target {}".format(
        measured, len(text), len(excerpt), frame_tokens, target
    )
)
PY
    prompt_input_key >"${prompt_inputs}"
    echo "  wrote  ${prompt_json}"
    echo "  sha256 $(sha256sum "${prompt_json}" | cut -d' ' -f1)"
    echo "  inputs ${prompt_inputs}"
}

# ---------------------------------------------------------------------------------------------
# pass-a / pass-b
# ---------------------------------------------------------------------------------------------
run_pass() {
    local pass="$1" run_stamp prefix err_log out_log vram_log smi_log tok_log rc=0
    local started ended

    if [[ ! -s "${prompt_json}" ]]; then
        echo "prompt file missing; run '$0 prompt' first: ${prompt_json}" >&2
        return 1
    fi

    run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    prefix="${log_dir}/qwen3_8_27b_nvfp4-1m-soak-${pass}-${run_stamp}"
    # A pass is identified by its stamp; the `.latest` symlink below is what `analyze` follows.
    err_log="${prefix}.cli.log"
    out_log="${prefix}.reply.txt"
    vram_log="${prefix}.vram.csv"
    smi_log="${prefix}.nvidia-smi.txt"
    tok_log="${prefix}.tokens.txt"

    echo
    echo "=== soak ${pass} ==="
    echo "prompt:    ${prompt_json}"
    echo "cli log:   ${err_log}"
    echo "reply:     ${out_log}"
    echo "vram log:  ${vram_log}"
    echo "smi log:   ${smi_log}"
    echo "tokens:    ${tok_log}"

    : >"${smi_log}"
    record_compute_apps "before launch" "${smi_log}"
    if [[ -n "$(compute_app_pids)" ]]; then
        echo "  gate  no compute process before launch  FAIL" | tee -a "${smi_log}"
        echo "another process is already using a GPU; this script will not start alongside it" >&2
        echo "(it never signals a process it did not start -- stop the other job first)" >&2
        return 1
    fi
    echo "  gate  no compute process before launch  PASS" | tee -a "${smi_log}"

    sample_vram "${vram_log}" "${vram_sample_seconds}" &
    sampler_pid=$!

    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set +e
    "${cli_bin}" "${artifact}" "${engine_flags[@]}" \
        --messages "${prompt_json}" --max-new "${window_tokens}" \
        >"${out_log}" 2>"${err_log}"
    rc=$?
    set -e
    ended="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
        kill -TERM "${sampler_pid}" 2>/dev/null || true
        wait "${sampler_pid}" 2>/dev/null || true
    fi
    sampler_pid=""

    echo "cli exit:  ${rc}   (${started} -> ${ended})"
    if [[ "${rc}" -ne 0 ]]; then
        echo "--- last 40 log lines ---"
        tail -40 "${err_log}"
        return "${rc}"
    fi

    # The id stream, one id per line, is what the sanity and determinism gates read.
    grep -E '^tokens +generated ids' "${err_log}" \
        | sed -E 's/^tokens +generated ids +//' | tr ' ' '\n' | grep -E '^[0-9]+$' >"${tok_log}"
    echo "  token ids captured: $(wc -l <"${tok_log}")"

    echo "--- generation ---"
    printf '  prompt tokens     %s\n' "$(summary_value 'prompt tokens' "${err_log}")"
    printf '  generated tokens  %s\n' "$(summary_value 'generated tokens' "${err_log}")"
    printf '  finish reason     %s\n' "$(summary_value 'finish reason' "${err_log}")"
    printf '  prefill speed     %s\n' "$(summary_value 'prefill speed' "${err_log}")"
    printf '  decode speed      %s\n' "$(summary_value 'decode speed' "${err_log}")"
    grep -E '^generate +text prefill' "${err_log}" | sed 's/^/  /' || true
    grep -E '^generate +decode' "${err_log}" | sed 's/^/  /' || true

    echo "--- clean exit ---"
    record_compute_apps "after exit" "${smi_log}"
    local leftover
    leftover="$(compute_app_pids)"
    if [[ -z "${leftover}" ]]; then
        echo "  gate  no compute process after exit  PASS" | tee -a "${smi_log}"
    else
        echo "  gate  compute processes still resident after exit: $(echo "${leftover}" | tr '\n' ' ') FAIL" \
            | tee -a "${smi_log}"
        echo "soak ${pass}: CLEAN-EXIT GATE FAILED"
        return 1
    fi

    # A stable name per pass AND per window, so `analyze` does not have to guess which stamp was
    # the last one and a small-window validation cannot repoint the real soak's link.
    ln -sfn "$(basename "${prefix}")" "$(latest_link "${pass}")"
    echo "soak ${pass}: OK"
}

# ---------------------------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------------------------
analyze() {
    local latest_a latest_b
    latest_a="${log_dir}/$(readlink "$(latest_link pass-a)" 2>/dev/null || true)"
    latest_b="${log_dir}/$(readlink "$(latest_link pass-b)" 2>/dev/null || true)"
    if [[ ! -s "${latest_a}.tokens.txt" || ! -s "${latest_b}.tokens.txt" ]]; then
        echo "both passes must have run before analyze" >&2
        return 1
    fi
    "${tokenizer_python:-python3}" "${repo_dir}/tools/tp2/summarize_1m_soak.py" \
        --pass-a "${latest_a}" --pass-b "${latest_b}" \
        --prompt-json "${prompt_json}" \
        --window "${window_size}" --degenerate-share "${degenerate_share}" \
        --window-tokens "${window_tokens}" \
        --gate-mib "${gate_mib}" \
        --out-dir "${results_dir}" --out-stem "${result_stem}"
}

for step in "${steps[@]}"; do
    case "${step}" in
        prompt)  build_prompt ;;
        pass-a)  run_pass "pass-a" ;;
        pass-b)  run_pass "pass-b" ;;
        analyze) analyze ;;
    esac
done
