# Benchmark results

Generated: 2026-11-21T23:09:27.178+00:00

Run roots: /home/pgx/Documenti/vllm-gb10/bench/baseline/t-today
Runs: 12 (0 failed)

Notes: `decode_tok_s` = median single-stream decode speed over 3 repeats (usage.completion_tokens / decode window, thinking off, temp 0, seed 42). `acceptance_rate` = accepted draft tokens / proposed draft tokens from /metrics deltas; `tokens_per_step` = emitted tokens / draft step (includes the bonus token). `conc_N` = aggregate output tok/s at N concurrent streams (2 waves). `boot_s` includes container start → /health.

### Ranked by single-stream decode throughput

| run | stack | MTP | KV | seats | decode tok/s | TTFT p50 ms | TPOT p50 ms | accept | tok/step | Δdecode % | C=4 tok/s | C=8 tok/s | ΔC8 % | C=16 tok/s | boot s | errs | ratio | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256k-tri | myllmbox-v5.4 | 12 | auto | 3 | **62.8** | 161 | 16.0 | 0.170 | 2.99 | — | — | — | — | — | 338 | 0 | 1.000 | — |
| 256k-quad | myllmbox-v5.4 | 12 | auto | 4 | 62.4 | 153 | 16.1 | **0.170** | 3.01 | — | 74.5 | — | — | — | 356 | 0 | 1.000 | — |
| abba-1 | myllmbox-v5.4 | 12 | auto | 16 | 62.4 | **136** | 16.1 | 0.160 | 2.95 | — | — | 177.1 | — | — | 389 | 0 | 1.000 | — |
| uncensored-standard | myllmbox-v5.4 | 12 | auto | 16 | 62.1 | 170 | 16.2 | 0.170 | 3.06 | — | — | 132.0 | — | — | **328** | 0 | 1.000 | — |
| 524k-dual | myllmbox-v5.4 | 16 | auto | 2 | 61.6 | 151 | 16.3 | 0.120 | 2.97 | — | — | — | — | — | 382 | 0 | 1.000 | — |
| abba-2 | myllmbox-v5.4 | 12 | auto | 16 | 61.6 | 140 | 16.3 | 0.160 | 2.97 | — | — | **177.4** | — | — | 377 | 0 | 1.000 | — |
| standard-seats8 | myllmbox-v5.4 | 12 | auto | 8 | 60.9 | 140 | 16.6 | 0.170 | 2.98 | — | **76.3** | 174.9 | — | — | 362 | 0 | 1.000 | — |
| 786k-single | myllmbox-v5.4 | 16 | auto | 1 | 60.6 | 154 | 16.6 | 0.120 | 2.95 | — | — | — | — | — | 352 | 0 | 1.000 | — |
| 524k-batched48k-fp8 | myllmbox-v5.4 | 16 | fp8 | 1 | 58.4 | 149 | 17.2 | 0.120 | 2.86 | — | — | — | — | — | 409 | 0 | 1.000 | — |
| 256k-quad-fp8 | myllmbox-v5.4 | 12 | fp8 | 4 | 58.4 | 154 | 17.2 | 0.160 | 2.92 | — | 71.3 | — | — | — | 346 | 0 | 1.000 | — |
| 524k-pmu32 | myllmbox-v5.4 | 16 | auto | 1 | 58.2 | 257 | 17.3 | 0.120 | 2.94 | — | — | — | — | — | 375 | 0 | 1.000 | — |
| 524k-fp8kv | myllmbox-v5.4 | 16 | fp8 | 1 | 57.1 | 153 | 17.6 | 0.110 | 2.83 | — | — | — | — | — | 387 | 0 | 1.000 | — |

### Concurrency scaling (aggregate output tok/s)

| run | C=1 | C=2 | C=4 | C=8 | C=16 | KV tokens | KV max conc |
|---|---|---|---|---|---|---|---|
| standard-seats8 | 47.2 | — | 76.3 | 174.9 | — | 750933 | 1.83 |
| 256k-quad | 48.1 | — | 74.5 | — | — | 992612 | 3.79 |
| 256k-quad-fp8 | 45.8 | — | 71.3 | — | — | 1384355 | 5.28 |
| 256k-tri | 48.7 | — | — | — | — | 807050 | 3.08 |
| 524k-batched48k-fp8 | 35.5 | — | — | — | — | 1038866 | 1.98 |
| 524k-dual | 49.0 | 79.8 | — | — | — | 767013 | 1.46 |
| 524k-fp8kv | 40.8 | — | — | — | — | 1491955 | 2.85 |
| 524k-pmu32 | 44.7 | — | — | — | — | 767013 | 1.46 |
| 786k-single | 42.4 | — | — | — | — | 892616 | 1.14 |
| abba-1 | 47.0 | — | — | 177.1 | — | 750933 | 1.83 |
| abba-2 | 48.8 | — | — | 177.4 | — | 750933 | 1.83 |
| uncensored-standard | 51.3 | — | — | 132.0 | — | 750933 | 1.83 |

### Speculative decoding efficiency

| run | MTP | draft depth | acceptance | Δaccept pp | tok/step | step ms | steps/s | tok/chunk | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 256k-quad | 12 | 12.00 | 0.170 | — | 3.01 | 48.2 | 20.7 | 3.28 | 62.4 |
| 256k-tri | 12 | 12.00 | 0.170 | — | 2.99 | 47.7 | 21.0 | 3.25 | 62.8 |
| standard-seats8 | 12 | 12.00 | 0.170 | — | 2.98 | 49.0 | 20.4 | 3.18 | 60.9 |
| uncensored-standard | 12 | 12.00 | 0.170 | — | 3.06 | 49.3 | 20.3 | 3.38 | 62.1 |
| 256k-quad-fp8 | 12 | 12.00 | 0.160 | — | 2.92 | 49.9 | 20.0 | 3.05 | 58.4 |
| abba-1 | 12 | 12.00 | 0.160 | — | 2.95 | 47.4 | 21.1 | 3.25 | 62.4 |
| abba-2 | 12 | 12.00 | 0.160 | — | 2.97 | 48.2 | 20.8 | 3.23 | 61.6 |
| 524k-batched48k-fp8 | 16 | 16.00 | 0.120 | — | 2.86 | 49.0 | 20.4 | 3.07 | 58.4 |
| 524k-dual | 16 | 16.00 | 0.120 | — | 2.97 | 48.2 | 20.8 | 3.20 | 61.6 |
| 524k-pmu32 | 16 | 16.00 | 0.120 | — | 2.94 | 50.4 | 19.9 | 3.20 | 58.2 |
| 786k-single | 16 | 16.00 | 0.120 | — | 2.95 | 48.6 | 20.6 | 3.18 | 60.6 |
| 524k-fp8kv | 16 | 16.00 | 0.110 | — | 2.83 | 49.5 | 20.2 | 3.02 | 57.1 |

### Prefill / long-context

| run | 1k tok/s | 8k tok/s | 32k tok/s | 32k TTFT ms | 128k TTFT ms | 128k needle |
|---|---|---|---|---|---|---|
| 256k-quad | 577 | 2907 | — | — | 101101 | yes |
| 256k-quad-fp8 | 574 | 2933 | — | — | 102527 | yes |
| 256k-tri | 592 | 2870 | — | — | 101400 | yes |
| 524k-batched48k-fp8 | 347 | 2960 | — | — | 219064 | yes |
| 524k-dual | 343 | 2603 | — | — | — | — |
| 524k-fp8kv | 357 | 2934 | — | — | — | — |
| 524k-pmu32 | 345 | 2892 | — | — | — | — |
| 786k-single | 346 | 2893 | — | — | 367037 | yes |
| abba-1 | 592 | 2890 | — | — | — | — |
| abba-2 | 589 | 2897 | — | — | — | — |
| standard-seats8 | 582 | 2904 | — | — | — | — |
| uncensored-standard | 1328 | 2656 | — | — | — | — |

### Anomalies / warnings (do not rank on the flagged level)

- **256k-quad**: cold-start stall: C4 TTFT p50 2686 ms (>1500 ms)
- **256k-quad-fp8**: cold-start stall: C4 TTFT p50 2695 ms (>1500 ms)
- **standard-seats8**: cold-start stall: C4 TTFT p50 2667 ms (>1500 ms)

### Boot & run health

| run | boot s | profile | suite s | errs | warns | image |
|---|---|---|---|---|---|---|
| uncensored-standard | 328 | quick | 43 | 0 | 0 | sha256:bc42593a5bbf |
| 256k-tri | 338 | quick | 146 | 0 | 0 | sha256:bc42593a5bbf |
| 256k-quad-fp8 | 346 | quick | 148 | 0 | 0 | sha256:bc42593a5bbf |
| 786k-single | 352 | quick | 407 | 0 | 0 | sha256:bc42593a5bbf |
| 256k-quad | 356 | quick | 146 | 0 | 0 | sha256:bc42593a5bbf |
| standard-seats8 | 362 | quick | 50 | 0 | 0 | sha256:bc42593a5bbf |
| 524k-pmu32 | 375 | quick | 41 | 0 | 0 | sha256:bc42593a5bbf |
| abba-2 | 377 | quick | 44 | 0 | 0 | sha256:bc42593a5bbf |
| 524k-dual | 382 | quick | 43 | 0 | 0 | sha256:bc42593a5bbf |
| 524k-fp8kv | 387 | quick | 41 | 0 | 0 | sha256:bc42593a5bbf |
| abba-1 | 389 | quick | 44 | 0 | 0 | sha256:bc42593a5bbf |
| 524k-batched48k-fp8 | 409 | quick | 259 | 0 | 0 | sha256:bc42593a5bbf |

### Raw rows

See `summary.csv` for the full machine-readable table and each run's `summary.json` / `requests.jsonl` for per-request detail.

