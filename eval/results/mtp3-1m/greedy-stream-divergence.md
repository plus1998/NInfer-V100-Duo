# The greedy stream: MTP3 vs MTP-off at a 949,885-token prompt

Both runs use the identical prompt document
`eval/runs/soak-1m/soak-prompt-950000.messages.json`
(sha256 `8ee18ba41f6c306507160f93f03e4b06dceceeb36e17d0aef15ec07fa41b5d31`, the file the soak
summary records), `--greedy --no-thinking --ignore-eos`, TP2 + YaRN ×4, `--max-context 1048576`.
The MTP-off stream is the soak's pass-a, truncated to the same length.

Note: the MTP3 greedy stream's repetition is NOT an `--ignore-eos` artifact. Its own id stream
falsifies that reading (§3).

## 1. Where they part

First differing generated index: **45** (0-based). MTP3 emits one EXTRA token there.

| | ids 40..51 |
|---|---|
| MTP off | `[1541, 1204, 1467, 488, 615, 1532, 12035, 271, 1465, 5698, 821, 5888]` |
| MTP3    | `[1541, 1204, 1467, 488, 615, 303, 1532, 12035, 271, 1465, 5698, 821]` |

The extra id is `303` = ' in':

- MTP off: ' name? And how did you get here?”\n\nHe released my hands and stepped back,'
- MTP3   : ' name? And how did you get in here?”\n\nHe released my hands and stepped back'

**In-run evidence that this is a single isolated substitution, not a derailment:** after the
one-token shift the two streams run in **exact lock-step for 18 tokens** —
`off[45:63] == on[46:64]`, id for id — before parting again at
`off[63]=21099` vs `on[64]=12395`.

That the branch is a *near-tie in the logits* is an inference, not a measurement here: it follows
from the K+1-column verify mechanism (below), and the 18-token lock-step is consistent with it,
but no top-2 logit gap was captured. Measuring it would need a teacher-forced per-position capture
at a 949,885-token prompt; that was ruled out on cost (a single prefill at this length is ~16 min
before any harness work) and is left unmeasured rather than asserted.

## 2. Why the divergence is expected, and not a regression

The same effect was measured at short context first: an MTP run and an MTP-off run do not evaluate
the target model the same way — the verify round runs it over `K+1` columns at once, the ordinary
round over one — and different column counts select different GEMM shapes and a different GQA split
policy. At short context tp1 MTP and tp1 MTP-off first differ at position 65 of a 96-token
continuation (the committed asserted fixture). **Position 45 at a 949,885-token prompt is the same
phenomenon at the same order of magnitude, and it is bounded by the same guarantee:** greedy
verification commits a draft only when it equals the target's own argmax, so every token in either
stream is a token that decode would have produced from the logits it actually saw.

The needle runs are the positive control: on 1,046k-token prompts, `needle-a-mtp3`
and `needle-a-off` produced **bit-identical 24-token id streams** (both sha256 `68f8c998…`), and
both bundles reproduced the needle campaign's MTP-off answer verbatim.

## 3. After the divergence the two streams degenerate by DIFFERENT mechanisms

Both are degenerate, but not in the same way, and the id streams say which is which. `<|im_end|>`
is id **248046**; `<|im_start|>` is 248045; every id ≥ 248000 is a special token.

| stream | tokens | ids ≥ 248000 | `<|im_end|>` (248046) | mechanism |
|---|---:|---:|---:|---|
| MTP off (soak pass-a, first 12,000) | 12,000 | 77 | **23** | EOS-suppressed **turn repetition**: it finishes an answer, `--ignore-eos` drops the stop, it answers again |
| MTP3 greedy | 12,000 | **0** | **0** | **content-level degeneration**: no turn ever ends |
| MTP3 greedy, novel prefix | 1,341 | 0 | 0 | — |
| MTP3 sampled (T 0.8, top-p 0.95, top-k 20, presence 1.50) | 12,000 | 53 | 16 | turns end and restart, but the content differs each time |

**The MTP3 greedy stream contains no special token at all across all 12,000 generated ids**, so
`--ignore-eos` never fired in it. Its 187-token loop is the model repeating a sentence frame:

> ' that. She is the one who can make you feel that. She is the one who can make you know that. She is the one who can make you understand that. She is the one who can make you accept that. She is the one who can make you embrace that. She'

The MTP-off soak stream, by contrast, carries 23 `<|im_end|>` tokens in the same first
12,000 — it really is the EOS-suppressed turn loop.

So **MTP3-greedy and MTP-off-greedy degenerate by different mechanisms**, and the single
substitution at index 45 put them on different trajectories. Neither mechanism is a property
of MTP: content-level greedy looping is ordinary greedy behaviour on a long continuation prompt.

### 3.1 Where the 1,341 cut actually comes from

The `decode-novel` budget of 1,341 was derived as `12,000 − 57 × 187`, i.e. the **suffix-period
alignment** of the loop, not the point where the loop begins. Measured directly:

| quantity | index |
|---|---:|
| first 187-block repeat (`smallest i with on[i:i+187] == on[i+187:i+374]`) | **1,201** |
| first duplicate 16-gram | 1,388 |
| maximal all-novel-16-gram prefix | **1,403** |
| the cut actually used (`12,000 − 57 × 187`) | 1,341 |

So **~140 of the 1,341 tokens sit inside the loop's first block** (1,341 − 1,201). The cut is
therefore slightly biased *toward* higher acceptance than a strictly pre-loop cut would give, and
`decode-novel`'s 58.65 % should be read as a mild upper bound on the novel-text figure rather than
an exact one. It is still by far the closest of the three streams to novel text, and the
1,341-token prefix does measure 1.0 novel 16-grams because the first duplicate 16-gram only lands
at 1,388.

## 4. Repetition statistics

| stream | tokens | distinct ids | novel 16-grams | exact loop |
|---|---:|---:|---:|---|
| MTP off (soak pass-a, first 12,000) | 12,000 | 373 | 0.1538 | none within 12,000 |
| MTP off (soak pass-a, all 98,692) | 98,692 | 373 | — | 1,903-token block × 46 |
| MTP3 greedy | 12,000 | 308 | 0.1158 | 187-token block × 57, first repeat at 1,201 |
| MTP3 sampled | 12,000 | 740 | 0.254 | none |
| MTP3 greedy, novel prefix | 1,341 | 304 | 1.0 | none |

`decode-novel`'s 1,341 ids are byte-identical to `decode-greedy`'s first 1,341, so the MTP3 greedy
path is reproducible run to run at this context.

**Consequence for the acceptance numbers.** The MTP3 greedy stream degenerates *harder* than the
MTP-off one (a 187-token content loop against a 1,903-token turn loop), so its 93.54 % acceptance
is an even weaker upper bound than "repetitive stream" implies. The number to compare against the
57.96 % measured at 250k is `decode-novel`'s **58.65 %**, with the §3.1 caveat.

## 5. Reproducing §3 from the artifacts

```bash
base=eval/server-logs
for f in $base/qwen3_8_27b_nvfp4-1m-mtp3-decode-{greedy,sampled,novel}-*.tokens.txt; do
    printf '%-72s specials>=248000: %4s  im_end(248046): %4s\n' "$(basename $f)" \
        "$(awk '$1>=248000' $f | wc -l)" "$(awk '$1==248046' $f | wc -l)"
done
head -12000 $base/qwen3_8_27b_nvfp4-1m-soak-pass-a-20260828T144651Z.tokens.txt \
    | awk '$1>=248000' | sort -n | uniq -c
```

Observed: MTP3 greedy `0 / 0`, MTP3 novel `0 / 0`, MTP3 sampled `53 / 16`;
7.2 pass-a first 12,000 → `4 248044, 38 248045, 23 248046, 6 248068, 6 248069`.
