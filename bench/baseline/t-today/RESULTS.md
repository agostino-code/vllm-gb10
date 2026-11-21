# Benchmark results

Generated: 2026-11-21T22:00:14.257+00:00

Run roots: /home/pgx/Documenti/vllm-gb10/bench/baseline/t-today
Runs: 12 (0 failed)

Notes: `decode_tok_s` = median single-stream decode speed over 3 repeats (usage.completion_tokens / decode window, thinking off, temp 0, seed 42). `acceptance_rate` = accepted draft tokens / proposed draft tokens from /metrics deltas; `tokens_per_step` = emitted tokens / draft step (includes the bonus token). `conc_N` = aggregate output tok/s at N concurrent streams (2 waves). `boot_s` includes container start → /health.

### Ranked by single-stream decode throughput

| run | stack | MTP | KV | seats | decode tok/s | TTFT p50 ms | TPOT p50 ms | accept | tok/step | Δdecode % | C=4 tok/s | C=8 tok/s | ΔC8 % | C=16 tok/s | boot s | errs | ratio | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abba-1 | myllmbox-v5.4 | 12 | auto | 16 | **62.4** | **136** | 16.1 | 0.160 | 2.95 | — | — | 177.1 | — | — | 389 | 0 | 1.000 | — |
| uncensored-standard | myllmbox-v5.4 | 12 | auto | 16 | 62.1 | 170 | 16.2 | 0.170 | 3.06 | — | — | 132.0 | — | — | **328** | 0 | 1.000 | — |
| 524k-dual | myllmbox-v5.4 | 16 | auto | 2 | 61.6 | 151 | 16.3 | 0.120 | 2.97 | — | — | — | — | — | 382 | 0 | 1.000 | — |
| abba-2 | myllmbox-v5.4 | 12 | auto | 16 | 61.6 | 140 | 16.3 | 0.160 | 2.97 | — | — | **177.4** | — | — | 377 | 0 | 1.000 | — |
| 256k-quad | myllmbox-v5.4 | 12 | auto | 4 | 61.6 | 152 | 16.4 | 0.160 | 2.96 | — | 72.9 | — | — | — | 350 | 0 | 1.000 | — |
| 256k-tri | myllmbox-v5.4 | 12 | auto | 3 | 61.3 | 161 | 16.4 | 0.160 | 2.98 | — | — | — | — | — | 362 | 0 | 1.000 | — |
| standard-seats8 | myllmbox-v5.4 | 12 | auto | 8 | 60.9 | 140 | 16.6 | **0.170** | 2.98 | — | **76.3** | 174.9 | — | — | 362 | 0 | 1.000 | — |
| 786k-single | myllmbox-v5.4 | 16 | auto | 1 | 59.7 | 142 | 16.9 | 0.120 | 2.97 | — | — | — | — | — | 358 | 0 | 1.000 | — |
| 256k-quad-fp8 | myllmbox-v5.4 | 12 | fp8 | 4 | 59.2 | 150 | 17.0 | 0.160 | 2.96 | — | 74.5 | — | — | — | 358 | 0 | 1.000 | — |
| 524k-pmu32 | myllmbox-v5.4 | 16 | auto | 1 | 58.2 | 257 | 17.3 | 0.120 | 2.94 | — | — | — | — | — | 375 | 0 | 1.000 | — |
| 524k-batched48k-fp8 | myllmbox-v5.4 | 16 | fp8 | 1 | 58.1 | 143 | 17.3 | 0.120 | 2.88 | — | — | — | — | — | 397 | 0 | 1.000 | — |
| 524k-fp8kv | myllmbox-v5.4 | 16 | fp8 | 1 | 57.1 | 153 | 17.6 | 0.110 | 2.83 | — | — | — | — | — | 387 | 0 | 1.000 | — |

### Concurrency scaling (aggregate output tok/s)

| run | C=1 | C=2 | C=4 | C=8 | C=16 | KV tokens | KV max conc |
|---|---|---|---|---|---|---|---|
| standard-seats8 | 47.2 | — | 76.3 | 174.9 | — | 750933 | 1.83 |
| 256k-quad-fp8 | 45.4 | — | 74.5 | — | — | 1384355 | 5.28 |
| 256k-quad | 41.0 | — | 72.9 | — | — | 992612 | 3.79 |
| 256k-tri | 46.5 | — | — | — | — | 807050 | 3.08 |
| 524k-batched48k-fp8 | 43.4 | — | — | — | — | 1038866 | 1.98 |
| 524k-dual | 49.0 | 79.8 | — | — | — | 767013 | 1.46 |
| 524k-fp8kv | 40.8 | — | — | — | — | 1491955 | 2.85 |
| 524k-pmu32 | 44.7 | — | — | — | — | 767013 | 1.46 |
| 786k-single | 40.7 | — | — | — | — | 892616 | 1.14 |
| abba-1 | 47.0 | — | — | 177.1 | — | 750933 | 1.83 |
| abba-2 | 48.8 | — | — | 177.4 | — | 750933 | 1.83 |
| uncensored-standard | 51.3 | — | — | 132.0 | — | 750933 | 1.83 |

### Speculative decoding efficiency

| run | MTP | draft depth | acceptance | Δaccept pp | tok/step | step ms | steps/s | tok/chunk | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| standard-seats8 | 12 | 12.00 | 0.170 | — | 2.98 | 49.0 | 20.4 | 3.18 | 60.9 |
| uncensored-standard | 12 | 12.00 | 0.170 | — | 3.06 | 49.3 | 20.3 | 3.38 | 62.1 |
| 256k-quad | 12 | 12.00 | 0.160 | — | 2.96 | 48.0 | 20.8 | 3.21 | 61.6 |
| 256k-quad-fp8 | 12 | 12.00 | 0.160 | — | 2.96 | 49.9 | 20.0 | 3.10 | 59.2 |
| 256k-tri | 12 | 12.00 | 0.160 | — | 2.98 | 48.5 | 20.6 | 3.20 | 61.3 |
| abba-1 | 12 | 12.00 | 0.160 | — | 2.95 | 47.4 | 21.1 | 3.25 | 62.4 |
| abba-2 | 12 | 12.00 | 0.160 | — | 2.97 | 48.2 | 20.8 | 3.23 | 61.6 |
| 524k-batched48k-fp8 | 16 | 16.00 | 0.120 | — | 2.88 | 49.5 | 20.2 | 3.05 | 58.1 |
| 524k-dual | 16 | 16.00 | 0.120 | — | 2.97 | 48.2 | 20.8 | 3.20 | 61.6 |
| 524k-pmu32 | 16 | 16.00 | 0.120 | — | 2.94 | 50.4 | 19.9 | 3.20 | 58.2 |
| 786k-single | 16 | 16.00 | 0.120 | — | 2.97 | 49.6 | 20.1 | 3.15 | 59.7 |
| 524k-fp8kv | 16 | 16.00 | 0.110 | — | 2.83 | 49.5 | 20.2 | 3.02 | 57.1 |

### Prefill / long-context

| run | 1k tok/s | 8k tok/s | 32k tok/s | 32k TTFT ms | 128k TTFT ms | 128k needle |
|---|---|---|---|---|---|---|
| 256k-quad | 577 | 2906 | — | — | — | — |
| 256k-quad-fp8 | 579 | 2941 | — | — | — | — |
| 256k-tri | 586 | 2900 | — | — | — | — |
| 524k-batched48k-fp8 | 348 | 2984 | — | — | — | — |
| 524k-dual | 343 | 2603 | — | — | — | — |
| 524k-fp8kv | 357 | 2934 | — | — | — | — |
| 524k-pmu32 | 345 | 2892 | — | — | — | — |
| 786k-single | 349 | 2903 | — | — | — | — |
| abba-1 | 592 | 2890 | — | — | — | — |
| abba-2 | 589 | 2897 | — | — | — | — |
| standard-seats8 | 582 | 2904 | — | — | — | — |
| uncensored-standard | 1328 | 2656 | — | — | — | — |

### Anomalies / warnings (do not rank on the flagged level)

- **256k-quad**: cold-start stall: C4 TTFT p50 2693 ms (>1500 ms)
- **256k-quad-fp8**: cold-start stall: C4 TTFT p50 2702 ms (>1500 ms)
- **standard-seats8**: cold-start stall: C4 TTFT p50 2667 ms (>1500 ms)

### Boot & run health

| run | boot s | profile | suite s | errs | warns | image |
|---|---|---|---|---|---|---|
| uncensored-standard | 328 | quick | 43 | 0 | 0 | sha256:bc42593a5bbf |
| 256k-quad | 350 | quick | 45 | 0 | 0 | sha256:bc42593a5bbf |
| 786k-single | 358 | quick | 39 | 0 | 0 | sha256:bc42593a5bbf |
| 256k-quad-fp8 | 358 | quick | 44 | 0 | 0 | sha256:bc42593a5bbf |
| 256k-tri | 362 | quick | 44 | 0 | 0 | sha256:bc42593a5bbf |
| standard-seats8 | 362 | quick | 50 | 0 | 0 | sha256:bc42593a5bbf |
| 524k-pmu32 | 375 | quick | 41 | 0 | 0 | sha256:bc42593a5bbf |
| abba-2 | 377 | quick | 44 | 0 | 0 | sha256:bc42593a5bbf |
| 524k-dual | 382 | quick | 43 | 0 | 0 | sha256:bc42593a5bbf |
| 524k-fp8kv | 387 | quick | 41 | 0 | 0 | sha256:bc42593a5bbf |
| abba-1 | 389 | quick | 44 | 0 | 0 | sha256:bc42593a5bbf |
| 524k-batched48k-fp8 | 397 | quick | 39 | 0 | 0 | sha256:bc42593a5bbf |

### Raw rows

See `summary.csv` for the full machine-readable table and each run's `summary.json` / `requests.jsonl` for per-request detail.

