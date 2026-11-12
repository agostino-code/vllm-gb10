# Findings log — AZ model preset study

Running notes (Lead). Final report: `BENCHMARK-REPORT.md`.

## 2026-11-12 (study day)

### Setup repaired before benchmarking
1. The HF snapshot's `model.safetensors.index.json` had been clobbered at ~05:38 by an
   early draft-folder builder (it wrote the draft-only index with 4,638 tensors into the
   model snapshot). Every server boot failed with
   `NVFP4 weight_scale for layer 'column_parallel_linear' was never loaded (still NaN)`.
   Restored the original 375,487-tensor index (backup: `bench/refs/model.safetensors.index.original.json`,
   sha256 60e0e2d8…). Root cause fixed permanently in `tools/make_draft_dir.py` v2.0 by draft-lab.
2. `config.json` had also been edited to a non-published "fix" (added `exclude_modules`
   with `column_parallel_linear`). vLLM's compressed-tensors-style modelopt reader only
   reads `ignore`, so the edit did nothing but was reverted to the published config.
3. Reference preset then boots: myllmbox v5.4, K=12 dynamic depth, bf16 KV 23 GiB,
   750,933-token KV pool, 16 seats, boot 6.4 min, quality quick 0.9667.

### M0 reference measurement (shipped myllmbox AZ recipe; contaminated by one 90 s probe)
- decode (median, temp 0, thinking off): code 51.5, json 61.2, math 52.0, QA 38.6 tok/s.
- concurrency totals: C1 45.5, C2 66.9, C4 102.5, C8 138.2, C16 192.0 tok/s.
- prefill: 1k 1108, 8k 2235, 32k 2278 tok/s; 128k TTFT 43.5 s (prefill 2215 tok/s), needle OK.
- **spec-decode acceptance 17.2 %, tokens/step 3.06 (12 drafts proposed per step).**
  Accepted tokens per position: 82 %, 65 %, 52 %, 5.5 %, 1.7 %, then ~0.
- Interpretation: per-step engine time (~20 ms TPOT at 3 tok/step ≈ 60 ms/step) is on par
  with myllmbox's published box, but tokens/step is ~1.5x lower — so the throughput gap
  is an ACCEPTANCE/draft-quality problem, not kernel speed.
- Hypothesis A: self-draft runs the MTP layer at 5 experts while it was trained at 10;
  the draft-k10 folder restores native routing (model card explicitly supports this).
- Hypothesis B: NVFP4 draft-head copy (MBX_DRAFT_HEAD=1) degrades deep-position acceptance.
- Hypothesis C: draft-logit scale and/or the conservative dynamic-depth promote thresholds
  (needs >60 % acceptance at the deepest position; we measure 52 % at position 2, so the
  ladder never deepens).
- Host caveat: 4 background agents + desktop GUI share the box; published numbers were on
  a quiet box. The M0-clean rerun in the matrix quantifies the noise (client/server token
  ratio must be 0.95-1.05 for an uncontaminated run).

### Plan
Phase A: M0 clean + draft tier-1 (draft-k10 spec6, ceiling-9, scale2, head0, fixed5, fixed9).
Phase B: serving knobs (kvfp8, medium template, native262k, seqs8, batched4096, mxfp8a16)
         + draft tier-2 (no-spec, k10 spec9/dynamic, k5 control, draft fp8, ceil-flat9).
Phase C: follow-ups from findings (aggressive depth policy, k10+aggressive combo,
         temperature 0.6 and thinking-on checks), full quality on winner, soak.

## 2026-11-12 15:45 — first matrix results

### Clean M0 reference (full suite + quality quick)
decode 53.3 median (code 54.6 / json 59.9 / math 51.6 / QA 41.4), TTFT 181 ms;
C=1 49.0, C=2 67.3, C=4 103.6, C=8 142.0, C=16 198.2; prefill 1k 1266 / 8k 2341 / 32k 2260 tok/s;
96k-token needle TTFT 42.5 s (prefill 2266), needle OK; acceptance 17.0 %, tokens/step 3.04,
per-position 4422/3514/2752/214/90; boot 356 s; quality 0.9667 (unchanged from contaminated run).
Contamination-free (client/server token ratio exactly 1.0). Noise vs contaminated run: ~+3 % decode.

### draft-k10 at K=6 dynamic (d-draftk10-spec6)
External draft folder loaded (drafter reads 3 of 3 files, 4,639 tensors, 4.6 s). Boot 350 s.
decode 56.4 median (+6 % vs M0; code 58.2, json 63.4, math 52.8, QA 45.2), TTFT 160 ms,
TPOT 17.8 ms; C=2 83.2, C=8 162.9 (+15 % vs M0); C=4 68.3 is a one-time 3.1 s engine cold-start
stall at that batch size (all four requests ~3.1 s TTFT, then normal decode; not contamination —
ratio 1.0). Acceptance 34.0 % at K=6, tokens/step 3.04 (identical to M0), positions 1033/816/677/9/2.
=> HALF the draft forwards per emitted token at the same output rate; the per-position cliff after
position 2 is unchanged, so the 10-expert routing did not fix deep-position drafting; quality identical.
Same-K attribution vs draft-k5 will be checked in phase B (b-draftk5).

### eugr b12x (final)
MTP4 47.0 tok/s, 51.1 % accept, 3.04 tok/step, C=8 99.9, KV 586k; MTP6 42.4/37.8 %/3.27, C8 89.9;
MTP7 41.0/35.4 %/3.48, C8 81.5; MTP9 impossible (b12x GDN cap: at most 8 state-index columns
=> max 7 spec tokens). Deeper depth loses on both stacks. eugr boots ~6 min with a persistent
compile cache, autotune off (tuned first boot ~55-65 min). eugr under-allocates KV at util 0.75.

### Working hypothesis for the winner
Static depth 3-5 with the k10 drafter, bf16 KV, 16 seats: minimum wasted draft compute, same
tokens/step (~3.0), best decode/C8. Phase B tests b-draftk10-fixed4 / -fixed3 and the serving
knobs (kv fp8, medium template), plus the no-spec floor.

## 2026-11-12 17:45 — finalist phase complete (winner: self-draft K=4)

### Depth curve on the shipped image (myllmbox v5.4, quick suite unless noted)
| spec K | decode tok/s | acceptance | tok/step | step ms | C4 | C8 |
|---|---|---|---|---|---|---|
| 4 (curve-k4) | **60.9** | 49 % | 2.94 | 48.3 | 116.1 | 178.4 |
| 5 | 50.7 | 39 % | 2.97 | 59.0 | 91.4 | 146.4 |
| 6 | 55.7 | 33 % | 2.98 | 54.3 | 102.8 | 155.4 |
| 9 (dynamic ceiling 9) | 50.7 | 22 % | 2.95 | 58.1 | 91.9 | 147.8 |
| 12 (shipped, dynamic) | 53.3 | 17 % | 3.04 | 57.0 | 103.6 | 142.0 |
K=4 is the only depth that gets both a faster engine step and a higher acceptance; ≥5 pays
2-8 draft forwards for ~0.05-0.1 extra tokens/step. The curve is non-monotonic (K5 < K6) which is
within the ±5-10 % repeatability of the quick profile; the K4 advantage is larger than that.
Depth 3 is not bootable (image constraint), so 4 is the practical optimum and floor.

### Winner full-suite validation (finalist-winner-full)
decode 55.83 tok/s median, TTFT 168 ms, TPOT 17.98 ms; C1 48.8 / C2 76.0 / C4 115.5 /
C8 156.4 / **C16 243.1**; acceptance 50.8 %, tokens/step 3.03; prefill 1k 1267 / 8k 2545 /
32k 2506 tok/s; 96k-token needle prefill 2446 tok/s, TTFT 39.4 s, recall OK.
Quality **full** battery 0.9714: math/json/code/tool/repetition = 1.0, instruction 0.8 (the same
two model misses as the shipped preset), **long-context needle 32k/128k/262k = 1.0 (recall OK)**.
Temperature-0.6 check: decode 53.5, acceptance 50.0 %, tokens/step 3.00 — holds under sampled decoding.
7-minute C=8 soak: 343 requests, 0 failures, 206.6 tok/s aggregate, drift −2.9 %, TTFT p50 397 ms,
acceptance 57.8 %, tokens/step 3.31. Stable.

### Stack comparison (final)
myllmbox K4 wins on raw serving: eugr b12x MTP4 = 47.0 decode / 99.9 C8 (b12x autotune off,
fp8 KV under-allocated at util 0.75, max depth 7). eugr keeps ~51 % acceptance and needs no
59 GB PLE map, so it remains the memory-constrained-host choice.

### Deliverables
- `recipes/qwen38-best.yaml` + `presets/run_qwen38_best.sh` (MTP=0, K=4, shipped settings otherwise).
- `matrix/recipes/az-kvfp8-spec7.yaml` long-context variant (1.46 M token KV pool).
- `BENCHMARK-REPORT.md`; raw data `bench/results/` (RESULTS.md, summary.csv, 20+ runs).

## 2026-11-12 19:05 — CORRECTION: the depth win was measurement noise

Controlled A/B/A experiments (same window, interleaved, sustained 60 s loads) overturned the
single-run depth ranking:
- A/B/A #1 (suite quick): shipped 55.40 / 54.93 vs K=4 53.40 decode; C8 152.3/160.4 vs 153.0.
- A/B/A #2 (sustained): shipped 52.83 / 54.56 vs K=4 56.65 decode; suite C8 156.8/155.5 vs 158.1;
  **sustained C8 = 198.5 tok/s for both**; sustained C16 = 292.4/287.1 (shipped) vs 300.7 (K=4).
Conclusion: **depth 4 and depth 12 are throughput-equivalent** on this box. Acceptance differs
hugely (17 % vs ~49 %) but tokens/step (~3) and step time (~54 vs ~55 ms) compensate almost exactly;
the fused multi-step drafter makes the extra drafts nearly free. The earlier "K=4 +14 %" was a
favorable-window artifact (quick-profile single runs vary ±10-15 % on this shared host).

What survives from the study:
- keep MBX_DRAFT_HEAD=1 (draft head off = -19 % decode, robust);
- shipped preset = best; no variant beat it beyond noise;
- eugr b12x slower (C8 ~100 vs ~157) but quality-equal, max depth 7, autotune-off caveat;
- draft-k10: acceptance up, throughput neutral-to-worse (extra 10-expert drafter cost);
- fp8 KV: 1.94x KV pool for ~-8 % decode;
- medium template: neutral; draft scale: no-op at temp 0;
- boot constraints: depth <=3 and MBX_DEPTH_MODE=off do not boot on v5.4;
- config.env MTP default fixed 3 -> 0 (duplicate spec flag + K3 = unbootable).

Recommendation: keep the shipped preset (presets/run_qwen38_best.sh); qwen38-k4.yaml is an
equal-throughput alternative with half the draft compute, not a speed win.
