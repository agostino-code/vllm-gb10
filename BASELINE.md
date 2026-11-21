# Baseline

- **run**: b-today+p-today
- **generated**: 2026-11-22 00:41:10
- **box**: DGX Spark GB10
- **image**: `myllmbox/qwen38-flash-next-vllm:v5.4` (sha256:bc42593a5bbf)
- **harness**: suite 1.0 · profile full, quick · quality full, off
- **host at report time**: vm.compaction_proactiveness=0, 0G free, 7 configs
- **sources**: bench/baseline/b-today, bench/baseline/p-today (later runs override)
  - `bench/baseline/b-today`: az-524k-single, az-standard, uncensored-524k-single, uncensored-standard
  - `bench/baseline/p-today`: az-256k-fleet, az-786k-single, az-standard-seats8

## Summary

| config | decode | C=1 | C=4 | C=8 | C=16 | TTFT | TPOT | prefill 8k | accept | t/step | depth | KV tokens | quality | boot s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| az-standard | 63.3 | 55.5 | 126.7 | 163.0 | — | 141 | 18.4 | 2779 | 17.3 % | 3.08 | 12 | 750,933 | 0.9714 | 393 |
| az-524k-single | 63.7 | 53.8 | — | — | — | 267 | 19.1 | 2876 | 13.9 % | 3.23 | 16 | 767,013 | 0.9643 | 377 |
| uncensored-standard | 63.3 | 57.2 | 109.7 | 136.1 | — | 157 | 18.9 | 2638 | 18.0 % | 3.16 | 12 | 750,933 | 0.9643 | 330 |
| uncensored-524k-single | 63.7 | 54.5 | — | — | — | 287 | 20.4 | 2580 | 14.7 % | 3.35 | 16 | 767,013 | 0.9714 | 348 |
| az-256k-fleet | 58.0 | 45.8 | 71.3 | — | — | 106 | 21.6 | 2933 | 16.0 % | 2.92 | 12 | 1,384,355 | — | 346 |
| az-786k-single | 60.7 | 42.4 | — | — | — | 352 | 23.8 | 2893 | 12.2 % | 2.95 | 16 | 892,616 | — | 352 |
| az-standard-seats8 | 60.9 | 47.2 | 76.3 | 174.9 | — | 104 | 21.3 | 2904 | 16.5 % | 2.98 | 12 | 750,933 | — | 362 |

*decode = median of the per-category medians (code/json/math/qa), temperature 0, thinking off. TTFT/TPOT are p50 at C=1. accept/t/step/depth come from the engine `/metrics` deltas, not from the client. quality = the deterministic gate, quick or full.*

## Decode by content

| config | code | json | math | qa |
|---|---|---|---|---|
| az-standard | 64.1 | 73.1 | 62.5 | 48.5 |
| az-524k-single | 64.6 | 72.1 | 62.8 | 48.0 |
| uncensored-standard | 66.5 | 71.0 | 60.1 | 45.8 |
| uncensored-524k-single | 66.1 | 70.2 | 61.3 | 48.8 |
| az-256k-fleet | 59.2 | 67.6 | 56.9 | 44.7 |
| az-786k-single | 63.7 | 67.5 | 57.7 | 46.0 |
| az-standard-seats8 | 64.0 | 69.4 | 57.8 | 49.7 |

## Concurrency ladder (total output tok/s)

| config | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|
| az-standard | 55.5 | 83.9 | 126.7 | 163.0 |
| az-524k-single | 53.8 | — | — | — |
| uncensored-standard | 57.2 | 71.6 | 109.7 | 136.1 |
| uncensored-524k-single | 54.5 | — | — | — |
| az-256k-fleet | 45.8 | — | 71.3 | — |
| az-786k-single | 42.4 | — | — | — |
| az-standard-seats8 | 47.2 | — | 76.3 | 174.9 |

## Accepted tokens per draft position

The depth curve. On this checkpoint it historically dies after position 2 — anything that says otherwise is noise.

| config | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 | p11 | p12 | p13 | p14 | p15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| az-standard | 40% | 32% | 25% | 3% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| az-524k-single | 37% | 31% | 25% | 5% | 2% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| uncensored-standard | 39% | 32% | 26% | 2% | 2% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| uncensored-524k-single | 37% | 31% | 26% | 4% | 3% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| az-256k-fleet | 43% | 32% | 25% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| az-786k-single | 41% | 33% | 26% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| az-standard-seats8 | 41% | 33% | 26% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |

*percentages are the share of accepted tokens contributed by each position.*

## Prefill (tok/s)

| config | 1024 tok | 8192 tok | 32768 tok |
|---|---|---|---|
| az-standard | 583 | 2779 | 2772 |
| az-524k-single | 341 | 2876 | 2471 |
| uncensored-standard | 1413 | 2638 | 2493 |
| uncensored-524k-single | 1365 | 2580 | 2617 |
| az-256k-fleet | 574 | 2933 | — |
| az-786k-single | 346 | 2893 | — |
| az-standard-seats8 | 582 | 2904 | — |

## Long-context smoke (needle at full context)

| config | target | prompt tokens | needle | prefill tok/s | TTFT s |
|---|---|---|---|---|---|
| az-standard | 131,072 | 130,607 | OK | 2694 | 48 |
| az-524k-single | 520,000 | 518,294 | OK | 2242 | 231 |
| uncensored-standard | 131,072 | 130,607 | OK | 2422 | 54 |
| uncensored-524k-single | 520,000 | 518,293 | OK | 2326 | 223 |
| az-256k-fleet | 260,000 | 259,433 | OK | 2530 | 103 |
| az-786k-single | 780,000 | 778,048 | OK | 2120 | 367 |
| az-standard-seats8 | — | — | — | — | — |

*The 524k configs were never actually exercised at their ceiling before the 2026-11-21 builder fix: the '524k' smoke was really 384k. These are the first true full-context measurements on this stack.*

## Quality gate

| config | code | instruction | json | math | needle | repetition | tool | overall |
|---|---|---|---|---|---|---|---|---|
| az-standard | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9714 |
| az-524k-single | 1.00 | 0.80 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 0.9643 |
| uncensored-standard | 1.00 | 0.80 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 0.9643 |
| uncensored-524k-single | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9714 |
| az-256k-fleet | — | — | — | — | — | — | — | — |
| az-786k-single | — | — | — | — | — | — | — | — |
| az-standard-seats8 | — | — | — | — | — | — | — | — |

## Configuration actually used

| config | model | KV dtype | spec config |
|---|---|---|---|
| az-standard | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| az-524k-single | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-524k | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| uncensored-standard | myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| uncensored-524k-single | myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored-524k | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| az-256k-fleet | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-256k4-fp8 | fp8 | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| az-786k-single | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-786k | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| az-standard-seats8 | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-s8 | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |

## Reproduce

```bash
./bench/baseline.py --run-id b-today   # resume (add --force to re-measure)
./bench/baseline.py --run-id p-today   # resume (add --force to re-measure)
./bench/baseline.py --report bench/baseline/b-today bench/baseline/p-today   # rebuild this file from the raw results
```
