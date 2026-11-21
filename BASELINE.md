# Baseline

- **run**: 20261121T033710Z
- **generated**: 2026-11-21 04:42:06
- **box**: DGX Spark GB10
- **image**: `myllmbox/qwen38-flash-next-vllm:v5.4` (sha256:bc42593a5bbf)
- **harness**: suite 1.0 · profile full · quality full
- **host**: vm.compaction_proactiveness=0, 116G free at start, 4 configs
- **raw**: `bench/baseline/20261121T033710Z/<config>/summary.json`

## Summary

| config | decode | C=1 | C=4 | C=8 | C=16 | TTFT | TPOT | prefill 8k | accept | t/step | depth | KV tokens | quality | boot s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| az-standard | 64.1 | 56.6 | 124.7 | 172.4 | — | 130 | 18.0 | 2852 | 17.4 % | 3.09 | 12 | 750,933 | 0.9714 | 593 |
| az-524k-single | 61.3 | 57.9 | — | — | — | 163 | 18.4 | 2643 | 14.1 % | 3.25 | 16 | 604,947 | 0.9714 | 362 |
| uncensored-standard | 63.9 | 55.9 | 109.7 | 140.3 | — | 154 | 19.7 | 2579 | 18.3 % | 3.19 | 12 | 750,933 | 0.9643 | 520 |
| uncensored-524k-single | 61.5 | 56.8 | — | — | — | 160 | 18.7 | 2355 | 14.5 % | 3.32 | 16 | 604,947 | 0.9714 | 379 |

*decode = median of the per-category medians (code/json/math/qa), temperature 0, thinking off. TTFT/TPOT are p50 at C=1. accept/t/step/depth come from the engine `/metrics` deltas, not from the client. quality = the deterministic gate, quick or full.*

## Decode by content

| config | code | json | math | qa |
|---|---|---|---|---|
| az-standard | 65.6 | 73.3 | 62.6 | 48.8 |
| az-524k-single | 62.5 | 68.9 | 60.1 | 47.8 |
| uncensored-standard | 66.8 | 70.3 | 61.1 | 48.1 |
| uncensored-524k-single | 63.9 | 67.9 | 59.0 | 45.5 |

## Concurrency ladder (total output tok/s)

| config | C=1 | C=2 | C=4 | C=8 |
|---|---|---|---|---|
| az-standard | 56.6 | 82.0 | 124.7 | 172.4 |
| az-524k-single | 57.9 | — | — | — |
| uncensored-standard | 55.9 | 74.2 | 109.7 | 140.3 |
| uncensored-524k-single | 56.8 | — | — | — |

## Accepted tokens per draft position

The depth curve. On this checkpoint it historically dies after position 2 — anything that says otherwise is noise.

| config | p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | p9 | p10 | p11 | p12 | p13 | p14 | p15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| az-standard | 39% | 31% | 25% | 3% | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| az-524k-single | 37% | 31% | 26% | 4% | 2% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| uncensored-standard | 38% | 32% | 26% | 2% | 2% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — | — | — |
| uncensored-524k-single | 37% | 31% | 26% | 4% | 3% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

*percentages are the share of accepted tokens contributed by each position.*

## Prefill (tok/s)

| config | 1024 tok | 8192 tok | 32768 tok |
|---|---|---|---|
| az-standard | 475 | 2852 | 2830 |
| az-524k-single | 263 | 2643 | 2682 |
| uncensored-standard | 1194 | 2579 | 2479 |
| uncensored-524k-single | 918 | 2355 | 2797 |

## Quality gate

| config | code | instruction | json | math | needle | repetition | tool | overall |
|---|---|---|---|---|---|---|---|---|
| az-standard | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9714 |
| az-524k-single | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9714 |
| uncensored-standard | 1.00 | 0.80 | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 0.9643 |
| uncensored-524k-single | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.9714 |

## Configuration actually used

| config | model | KV dtype | spec config |
|---|---|---|---|
| az-standard | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| az-524k-single | azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-524k | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| uncensored-standard | myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |
| uncensored-524k-single | myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored-524k | auto | `{'draft_sample_method': 'probabilistic', 'method': 'mtp', 'num_specula` |

## Reproduce

```bash
./bench/baseline.py --run-id 20261121T033710Z
./bench/baseline.py --report          # rebuild this file from the raw results
```
