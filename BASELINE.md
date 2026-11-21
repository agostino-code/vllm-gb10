# Baseline

- **run**: t-today+b-today
- **generated**: 2026-11-21 23:00:14
- **box**: DGX Spark GB10
- **image**: `myllmbox/qwen38-flash-next-vllm:v5.4` (sha256:bc42593a5bbf)
- **harness**: suite 1.0 · profile full, quick · quality full, off
- **host at report time**: vm.compaction_proactiveness=0, 117G free, 15 configs
- **sources**: bench/baseline/b-today, bench/baseline/t-today (later runs override)
  - `bench/baseline/b-today`: az-524k-single, az-standard, uncensored-524k-single, uncensored-standard
  - `bench/baseline/t-today`: 256k-quad, 256k-quad-fp8, 256k-tri, 524k-batched48k-fp8, 524k-dual, 524k-fp8kv, 524k-pmu32, 786k-single, abba-1, abba-2, standard-seats8

## Summary

| config | decode | C=1 | C=4 | C=8 | C=16 | TTFT | TPOT | prefill 8k | accept | t/step | depth | KV tokens | quality | boot s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| az-standard | 63.3 | 55.5 | 126.7 | 163.0 | — | 141 | 18.4 | 2779 | 17.3 % | 3.08 | 12 | 750,933 | 0.9714 | 393 |
| az-524k-single | 63.7 | 53.8 | — | — | — | 267 | 19.1 | 2876 | 13.9 % | 3.23 | 16 | 767,013 | 0.9643 | 377 |
| uncensored-standard | 63.3 | 57.2 | 109.7 | 136.1 | — | 157 | 18.9 | 2638 | 18.0 % | 3.16 | 12 | 750,933 | 0.9643 | 330 |
| uncensored-524k-single | 63.7 | 54.5 | — | — | — | 287 | 20.4 | 2580 | 14.7 % | 3.35 | 16 | 767,013 | 0.9714 | 348 |
| 256k-quad | 61.5 | 41.0 | 72.9 | — | — | 108 | 24.6 | 2906 | 16.3 % | 2.96 | 12 | 992,612 | — | 350 |
| 256k-quad-fp8 | 59.2 | 45.4 | 74.5 | — | — | 108 | 21.7 | 2941 | 16.3 % | 2.96 | 12 | 1,384,355 | — | 358 |
| 256k-tri | 61.1 | 46.5 | — | — | — | 124 | 21.3 | 2900 | 16.5 % | 2.98 | 12 | 807,050 | — | 362 |
| 524k-batched48k-fp8 | 58.3 | 43.4 | — | — | — | 351 | 23.2 | 2984 | 11.7 % | 2.88 | 16 | 1,038,866 | — | 397 |
| 524k-dual | 62.3 | 49.0 | — | — | — | 127 | 20.5 | 2603 | 12.3 % | 2.97 | 16 | 767,013 | — | 382 |
| 524k-fp8kv | 57.6 | 40.8 | — | — | — | 338 | 24.3 | 2934 | 11.4 % | 2.83 | 16 | 1,491,955 | — | 387 |
| 524k-pmu32 | 58.7 | 44.7 | — | — | — | 130 | 22.2 | 2892 | 12.1 % | 2.94 | 16 | 767,013 | — | 375 |
| 786k-single | 59.9 | 40.7 | — | — | — | 345 | 24.8 | 2903 | 12.3 % | 2.97 | 16 | 892,616 | — | 358 |
| abba-1 | 62.3 | 47.0 | — | 177.1 | — | 107 | 21.0 | 2890 | 16.3 % | 2.95 | 12 | 750,933 | — | 389 |
| abba-2 | 61.3 | 48.8 | — | 177.4 | — | 102 | 20.3 | 2897 | 16.4 % | 2.97 | 12 | 750,933 | — | 377 |
| standard-seats8 | 60.9 | 47.2 | 76.3 | 174.9 | — | 104 | 21.3 | 2904 | 16.5 % | 2.98 | 12 | 750,933 | — | 362 |

*decode = median of the per-category medians (code/json/math/qa), temperature 0, thinking off. TTFT/TPOT are p50 at C=1. accept/t/step/depth come from the engine `/metrics` deltas, not from the client. quality = the deterministic gate, quick or full.*

## Decode by content

| config | code | json | math | qa |
|---|---|---|---|---|
| az-standard | 64.1 | 73.1 | 62.5 | 48.5 |
| az-524k-single | 64.6 | 72.1 | 62.8 | 48.0 |
| uncensored-standard | 66.5 | 71.0 | 60.1 | 45.8 |
| uncensored-524k-single | 66.1 | 70.2 | 61.3 | 48.8 |
| 256k-quad | 65.7 | 68.4 | 57.2 | 45.4 |
| 256k-quad-fp8 | 61.7 | 69.1 | 56.7 | 45.0 |
| 256k-tri | 64.4 | 69.8 | 57.8 | 45.6 |
| 524k-batched48k-fp8 | 58.4 | 69.0 | 58.1 | 44.5 |
| 524k-dual | 63.7 | 69.1 | 60.8 | 47.0 |
| 524k-fp8kv | 59.0 | 67.1 | 56.2 | 43.7 |
| 524k-pmu32 | 61.6 | 65.5 | 55.7 | 44.1 |
| 786k-single | 62.3 | 68.1 | 57.4 | 48.1 |
| abba-1 | 65.0 | 68.4 | 59.6 | 46.0 |
| abba-2 | 64.2 | 67.0 | 58.4 | 48.5 |
| standard-seats8 | 64.0 | 69.4 | 57.8 | 49.7 |

## Concurrency ladder (total output tok/s)

| config | C=1 | C=2 | C=3 | C=4 | C=8 |
|---|---|---|---|---|---|
| az-standard | 55.5 | 83.9 | — | 126.7 | 163.0 |
| az-524k-single | 53.8 | — | — | — | — |
| uncensored-standard | 57.2 | 71.6 | — | 109.7 | 136.1 |
| uncensored-524k-single | 54.5 | — | — | — | — |
| 256k-quad | 41.0 | — | — | 72.9 | — |
| 256k-quad-fp8 | 45.4 | — | — | 74.5 | — |
| 256k-tri | 46.5 | — | 55.6 | — | — |
| 524k-batched48k-fp8 | 43.4 | — | — | — | — |
| 524k-dual | 49.0 | 79.8 | — | — | — |
| 524k-fp8kv | 40.8 | — | — | — | — |
| 524k-pmu32 | 44.7 | — | — | — | — |
| 786k-single | 40.7 | — | — | — | — |
| abba-1 | 47.0 | — | — | — | 177.1 |
| abba-2 | 48.8 | — | — | — | 177.4 |
| standard-seats8 | 47.2 | — | — | 76.3 | 174.9 |

## Accepted tokens per draft position

The depth curve. On this checkpoint it historically dies after position 2 — anything that says otherwise is noise.

| config | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 | p11 | p12 | p13 | p14 | p15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| az-standard | 40% | 32% | 25% | 3% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| az-524k-single | 37% | 31% | 25% | 5% | 2% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| uncensored-standard | 39% | 32% | 26% | 2% | 2% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| uncensored-524k-single | 37% | 31% | 26% | 4% | 3% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 256k-quad | 41% | 32% | 26% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| 256k-quad-fp8 | 43% | 32% | 25% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| 256k-tri | 41% | 33% | 26% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| 524k-batched48k-fp8 | 43% | 32% | 25% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 524k-dual | 40% | 33% | 26% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 524k-fp8kv | 43% | 32% | 24% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 524k-pmu32 | 41% | 32% | 26% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 786k-single | 41% | 33% | 26% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| abba-1 | 41% | 33% | 25% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| abba-2 | 42% | 33% | 25% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| standard-seats8 | 41% | 33% | 26% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |

*percentages are the share of accepted tokens contributed by each position.*

## Prefill (tok/s)

| config | 1024 tok | 8192 tok | 32768 tok |
|---|---|---|---|
| az-standard | 583 | 2779 | 2772 |
| az-524k-single | 341 | 2876 | 2471 |
| uncensored-standard | 1413 | 2638 | 2493 |
| uncensored-524k-single | 1365 | 2580 | 2617 |
| 256k-quad | 577 | 2906 | — |
| 256k-quad-fp8 | 579 | 2941 | — |
| 256k-tri | 586 | 2900 | — |
| 524k-batched48k-fp8 | 348 | 2984 | — |
| 524k-dual | 343 | 2603 | — |
| 524k-fp8kv | 357 | 2934 | — |
| 524k-pmu32 | 345 | 2892 | — |
| 786k-single | 349 | 2903 | — |
| abba-1 | 592 | 2890 | — |
| abba-2 | 589 | 2897 | — |
| standard-seats8 | 582 | 2904 | — |

## Long-context smoke (needle at full context)

| config | target | prompt tokens | needle | prefill tok/s | TTFT s |
|---|---|---|---|---|---|
| az-standard | 131,072 | 130,607 | OK | 2694 | 48 |
| az-524k-single | 520,000 | 518,294 | OK | 2242 | 231 |
| uncensored-standard | 131,072 | 130,607 | OK | 2422 | 54 |
| uncensored-524k-single | 520,000 | 518,293 | OK | 2326 | 223 |
| 256k-quad | — | — | — | — | — |
| 256k-quad-fp8 | — | — | — | — | — |
| 256k-tri | — | — | — | — | — |
| 524k-batched48k-fp8 | — | — | — | — | — |
| 524k-dual | — | — | — | — | — |
| 524k-fp8kv | — | — | — | — | — |
| 524k-pmu32 | — | — | — | — | — |
| 786k-single | — | — | — | — | — |
| abba-1 | — | — | — | — | — |
| abba-2 | — | — | — | — | — |
| standard-seats8 | — | — | — | — | — |

*The 524k configs were never actually exercised at their ceiling before the 2026-11-21 builder fix: the '524k' smoke was really 384k. These are the first true full-context measurements on this stack.*

## Quality gate

| config | code | instruction | json | math | needle | repetition | tool | overall |
|---|---|---|---|---|---|---|---|---|
| az-standard | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9714 |
| az-524k-single | 1.00 | 0.80 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 0.9643 |
| uncensored-standard | 1.00 | 0.80 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 0.9643 |
| uncensored-524k-single | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9714 |
| 256k-quad | — | — | — | — | — | — | — | — |
| 256k-quad-fp8 | — | — | — | — | — | — | — | — |
| 256k-tri | — | — | — | — | — | — | — | — |
| 524k-batched48k-fp8 | — | — | — | — | — | — | — | — |
| 524k-dual | — | — | — | — | — | — | — | — |
| 524k-fp8kv | — | — | — | — | — | — | — | — |
| 524k-pmu32 | — | — | — | — | — | — | — | — |
| 786k-single | — | — | — | — | — | — | — | — |
| abba-1 | — | — | — | — | — | — | — | — |
| abba-2 | — | — | — | — | — | — | — | — |
| standard-seats8 | — | — | — | — | — | — | — | — |

## Configuration actually used

| config | model | KV dtype | spec config |
|---|---|---|---|
| az-standard | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| az-524k-single | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-524k | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| uncensored-standard | myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| uncensored-524k-single | myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored-524k | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| 256k-quad | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-256k4 | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| 256k-quad-fp8 | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-256k4-fp8 | fp8 | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| 256k-tri | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-256k3 | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| 524k-batched48k-fp8 | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-524k-b48k-fp8 | fp8 | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| 524k-dual | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-524k-dual | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| 524k-fp8kv | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-524k-fp8kv | fp8 | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| 524k-pmu32 | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-524k-pmu32 | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| 786k-single | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-786k | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| abba-1 | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| abba-2 | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| standard-seats8 | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-s8 | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |

## Reproduce

```bash
./bench/baseline.py --run-id b-today   # resume (add --force to re-measure)
./bench/baseline.py --run-id t-today   # resume (add --force to re-measure)
./bench/baseline.py --report bench/baseline/b-today bench/baseline/t-today   # rebuild this file from the raw results
```
