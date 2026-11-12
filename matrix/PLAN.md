# Preset study — azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ on DGX Spark (GB10)

Goal: determine the best serving preset for this checkpoint by measuring, not guessing.
Two docker stacks, one GPU: everything runs sequentially.

## Stacks
| id | image | loader | launcher |
|---|---|---|---|
| myllmbox | `myllmbox/qwen38-flash-next-vllm:v5.4` | `--load-format mbx` + MBX patches (dynamic draft depth, RecoverSSM, NVFP4 draft head, PLE on NVMe) | `recipes/*.yaml` (+ bench/serve.py) |
| eugr-b12x | `eugr/spark-vllm-b12x:latest` | `--load-format b12x`, b12x kernels (`--moe/linear/gdn-decode-kernel b12x`) | `eugr-b12x/*.yaml` (direct docker run) |

Reference preset (myllmbox AZ recipe v5.4): K=12 dynamic draft depth
(`[4 -> 8, 6 -> 7, 12 -> 6, 14 -> 5]`), bf16 KV 23 G, `max-num-seqs 16`, block 3456, YaRN 409600,
`rejection_sample_method=block`, `draft_sample_method=probabilistic`, `MBX_DRAFT_HEAD=1`.

Published reference points for this exact checkpoint (myllmbox's own numbers, calibration only):
code c=1 86.9 / c=4 211.0 / c=8 300.7 / c=16 398.2 tok/s; mixed 60.5 / 145.1 / 226.4 / 356.2;
128k prefill 2,691 tok/s; thinking-on 56.5 tok/s; KV pool 750,933 tokens.

## Hypotheses to test
1. Draft depth: fixed depth is not optimal; dynamic depth with a high ceiling (9–12) is.
   Prediction: dynamic > fixed at c=1..4; fixed high depth loses acceptance at high concurrency.
2. A 10-expert drafter (draft-k10 folder, MTP layer trained at 10) beats the 5-expert self-draft.
3. Draft-head NVFP4 copy (`MBX_DRAFT_HEAD=1`) is net-positive; disabling it costs throughput.
4. Draft-logit sharpening (`MBX_MTP_DRAFT_SCALE=2`) raises acceptance, exact output.
5. fp8 KV requires retuned depth (spec 7): pool doubles but speed drops; only useful for >400k context.
6. Native 262144 context (no YaRN) vs 400k: prefill/decoding differences.
7. eugr b12x can serve this checkpoint and is competitive; its optimal depth differs (no dynamic depth).

## Matrix tiers
- **Tier 0 (reference)**: myllmbox AZ recipe as shipped, full suite.
- **Tier 1 (scan, quick suite, ~5 min each)**: myllmbox draft/depth variants + serving knobs.
- **Tier 2 (full suite, ~10 min each)**: top-3 from Tier 1 + 3–5 eugr b12x configs.
- **Tier 3 (finalists, quality + long-context + soak)**: top-2 configs.

Draft/depth variants (draft-lab): spec 0, fixed 5, fixed 9, dynamic ceiling 9, dynamic ceiling 12 (=ref),
draft-k10 (spec 6), MBX_DRAFT_HEAD=0, MBX_MTP_DRAFT_SCALE=2, DRAFT_FP8=1, greedy vs probabilistic.
Serving knobs (lead): kv-cache-dtype fp8 + spec 7, native 262k, max-num-seqs 8, max-num-batched-tokens 4096,
kv-cache-memory 26G, chat template medium (from azampatti INT4 repo) vs stock, PLE prewarm, cpuset off.
eugr b12x (eugr-b12x): base, spec 4/6/9, kv fp8 vs bf16, block-size 16 vs 64, max-num-seqs 8/16, gpu-mem 0.75/0.8.

## Metrics
- decode: single-stream out tok/s, TPOT p50/p90/p99, TTFT p50; 3 reps, medians.
- throughput: c=1/2/4/8 (+16 opt-in) total out tok/s and per-stream.
- speculative: acceptance rate from `/metrics` deltas
  (`spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total`), tokens/step.
- prefill: TTFT at ~1k/8k/32k (+128k opt-in); KV pool tokens and max concurrency from startup log.
- cost: boot seconds, max RSS/KV usage.
- quality: quality/ suite on finalists (math, json, code, instruction, tool-calls, needle, repetition).

## "Best" definition
Primary: median total output tok/s at c=4 and c=8 (agent-server sweet spot), then c=1 latency.
Constraint: no quality regression on finalist suite (within noise), no OOM, stable for 10 min at c=8.
Tie-breakers: lower TTFT at 8k/32k, lower boot time, larger KV pool, simpler config.

## GPU protocol
Single owner at a time. Serve → health → warmup → suite → stop. Never two servers.
Every run records exact image digest, docker command, env, speculative-config, boot log.
Caches persisted across boots: `./cache/container-cache/b12x` -> `/root/.cache/b12x`,
`./cache/container-cache/flashinfer` -> `/root/.cache/flashinfer`.
