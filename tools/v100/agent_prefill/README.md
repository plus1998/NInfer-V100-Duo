# Agent first-request prefill corpus

This Node.js harness starts Pi, Codex, and Claude Agent SDKs in an empty temporary workspace.
Their first model requests go to a **localhost-only rejecting server**; there is no external
inference, authentication, agent tool execution, or provider bill. The SDK may send setup probes
or retry after a rejection, so the harness retains the largest model request from each SDK turn.
It preserves the model request payload verbatim in `<agent>.request.json` and extracts its
system instructions, tools, and messages into `<agent>.prompt.txt` for NInfer's tokenizer.
The prompt text is a *JSON rendering of the captured content*, not the provider's hidden
chat-template serialization or a provider-reported token count.

Requires Node.js 20.20+, `npm`, a locally built `ninfer_bench` and `ninfer_v100_corpus`
(`-DNINFER_BUILD_BENCHMARKS=ON`), and the official Qwen3.8-27B NVFP4 `.ninfer` artifact.
The Pi package is the upstream 0.73.1 SDK, compatible with this machine's Node 20; it has
deprecated transitive dependencies. Do not feed untrusted extensions or archives into this
local capture harness. `npm audit` on this version reports two high-severity advisory groups
in Pi and its transitive dependencies; the script never persists Pi credentials or installs
extensions. Review/audit dependencies before using this as a general-purpose agent runner.

```bash
cd tools/v100/agent_prefill
npm ci --ignore-scripts
cd ../../..
bash tools/v100/agent_prefill/run.sh \
  /absolute/path/to/qwen3_8_27b_nvfp4.ninfer /tmp/ninfer-agent-prefill \
  "$PWD/build-v100-duo"
```

`run.sh` calls the three SDKs, tokenizes each extracted prompt with the artifact's own tokenizer,
then measures the **entire** resulting token span through the public Engine at 32K capacity,
TP2, INT8 KV, with chunks 1024, 4096, and 5120. Each point generates one token, disables
prefix reuse, and includes a discarded warmup and two measured requests. Compare the
`prefill_tok_s_mean` and `prefill_seconds_mean` fields in `<agent>-c<chunk>.json`.

Keep the output directory **outside the repo**: raw requests contain complete SDK instructions
and workspace metadata and should not be published. The captured first turn depends on SDK
version, chosen model, initial task, environment, and tool policy. It is not a measured
interactive agent session, nor a like-for-like performance comparison against a running
Pi/Codex/Claude model service. Different providers' model token counts are not interchangeable;
the `.ids` files are specifically Qwen3.8 NInfer tokenizer inputs derived from each payload.
