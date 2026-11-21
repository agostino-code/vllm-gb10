# Baseline

- **run**: 20261121T033710Z+20261121T161012Z
- **generated**: 2026-11-21 17:07:17
- **box**: DGX Spark GB10
- **image**: `myllmbox/qwen38-flash-next-vllm:v5.4` (sha256:bc42593a5bbf)
- **harness**: suite 1.0 · profile full · quality full
- **host at report time**: vm.compaction_proactiveness=0, 2G free, 4 configs
- **sources**: bench/baseline/20261121T033710Z, bench/baseline/20261121T161012Z (later runs override)
  - `bench/baseline/20261121T033710Z`: az-standard, uncensored-standard
  - `bench/baseline/20261121T161012Z`: az-524k-single, uncensored-524k-single

## Summary

| config | decode | C=1 | C=4 | C=8 | C=16 | TTFT | TPOT | prefill 8k | accept | t/step | depth | KV tokens | quality | boot s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| az-standard | 64.1 | 56.6 | 124.7 | 172.4 | — | 130 | 18.0 | 2852 | 17.4 % | 3.09 | 12 | 750,933 | 0.9714 | 593 |
| az-524k-single | 63.4 | 55.6 | — | — | — | 266 | 18.8 | 2841 | 14.2 % | 3.27 | 16 | 767,013 | 0.9714 | 385 |
| uncensored-standard | 63.9 | 55.9 | 109.7 | 140.3 | — | 154 | 19.7 | 2579 | 18.3 % | 3.19 | 12 | 750,933 | 0.9643 | 520 |
| uncensored-524k-single | 63.3 | 54.6 | — | — | — | 287 | 20.4 | 2581 | 14.5 % | 3.32 | 16 | 767,013 | 0.9786 | 344 |

*decode = median of the per-category medians (code/json/math/qa), temperature 0, thinking off. TTFT/TPOT are p50 at C=1. accept/t/step/depth come from the engine `/metrics` deltas, not from the client. quality = the deterministic gate, quick or full.*

## Decode by content

| config | code | json | math | qa |
|---|---|---|---|---|
| az-standard | 65.6 | 73.3 | 62.6 | 48.8 |
| az-524k-single | 64.3 | 71.4 | 62.4 | 49.0 |
| uncensored-standard | 66.8 | 70.3 | 61.1 | 48.1 |
| uncensored-524k-single | 65.5 | 71.1 | 61.0 | 45.6 |

## Concurrency ladder (total output tok/s)

| config | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|
| az-standard | 56.6 | 82.0 | 124.7 | 172.4 |
| az-524k-single | 55.6 | — | — | — |
| uncensored-standard | 55.9 | 74.2 | 109.7 | 140.3 |
| uncensored-524k-single | 54.6 | — | — | — |

## Accepted tokens per draft position

The depth curve. On this checkpoint it historically dies after position 2 — anything that says otherwise is noise.

| config | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 | p11 | p12 | p13 | p14 | p15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| az-standard | 39% | 31% | 25% | 3% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| az-524k-single | 37% | 30% | 26% | 5% | 2% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| uncensored-standard | 38% | 32% | 26% | 2% | 2% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| uncensored-524k-single | 37% | 31% | 26% | 4% | 3% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

*percentages are the share of accepted tokens contributed by each position.*

## Prefill (tok/s)

| config | 1024 tok | 8192 tok | 32768 tok |
|---|---|---|---|
| az-standard | 475 | 2852 | 2830 |
| az-524k-single | 344 | 2841 | 2427 |
| uncensored-standard | 1194 | 2579 | 2479 |
| uncensored-524k-single | 1267 | 2581 | 2806 |

## Long-context smoke (needle at full context)

| config | target | prompt tokens | needle | prefill tok/s | TTFT s |
|---|---|---|---|---|---|
| az-standard | 131,072 | 96,293 | OK | 2753 | 35 |
| az-524k-single | 520,000 | 518,294 | OK | 2216 | 234 |
| uncensored-standard | 131,072 | 96,293 | OK | 2455 | 39 |
| uncensored-524k-single | 520,000 | 518,293 | OK | 2318 | 224 |

*The 524k configs were never actually exercised at their ceiling before the 2026-11-21 builder fix: the '524k' smoke was really 384k. These are the first true full-context measurements on this stack.*

## Quality gate

| config | code | instruction | json | math | needle | repetition | tool | overall |
|---|---|---|---|---|---|---|---|---|
| az-standard | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9714 |
| az-524k-single | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9714 |
| uncensored-standard | 1.00 | 0.80 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 0.9643 |
| uncensored-524k-single | 1.00 | 0.90 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 0.9786 |

## Configuration actually used

| config | model | KV dtype | spec config |
|---|---|---|---|
| az-standard | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| az-524k-single | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-524k | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| uncensored-standard | myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| uncensored-524k-single | myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored-524k | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |

## Reproduce

```bash
./bench/baseline.py --run-id 20261121T033710Z   # resume (add --force to re-measure)
./bench/baseline.py --run-id 20261121T161012Z   # resume (add --force to re-measure)
./bench/baseline.py --report bench/baseline/20261121T033710Z bench/baseline/20261121T161012Z   # rebuild this file from the raw results
```
