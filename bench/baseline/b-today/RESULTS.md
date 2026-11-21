# Benchmark results

Generated: 2026-11-21T19:45:17.332+00:00

Run roots: /home/pgx/Documenti/vllm-gb10/bench/baseline/b-today
Runs: 4 (0 failed)

Notes: `decode_tok_s` = median single-stream decode speed over 3 repeats (usage.completion_tokens / decode window, thinking off, temp 0, seed 42). `acceptance_rate` = accepted draft tokens / proposed draft tokens from /metrics deltas; `tokens_per_step` = emitted tokens / draft step (includes the bonus token). `conc_N` = aggregate output tok/s at N concurrent streams (2 waves). `boot_s` includes container start → /health.

### Ranked by single-stream decode throughput

| run | stack | MTP | KV | seats | decode tok/s | TTFT p50 ms | TPOT p50 ms | accept | tok/step | Δdecode % | C=4 tok/s | C=8 tok/s | ΔC8 % | C=16 tok/s | boot s | errs | ratio | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| az-standard | myllmbox-v5.4 | 12 | auto | 16 | **63.1** | **139** | 15.9 | 0.170 | 3.08 | — | **126.7** | **163.0** | — | — | 393 | 0 | 1.000 | 0.970 |
| az-524k-single | myllmbox-v5.4 | 16 | auto | 1 | 62.8 | 156 | 16.0 | 0.140 | 3.23 | — | — | — | — | — | 377 | 0 | 1.000 | 0.960 |
| uncensored-524k-single | myllmbox-v5.4 | 16 | auto | 1 | 62.7 | 188 | 16.0 | 0.150 | 3.35 | — | — | — | — | — | 348 | 0 | 1.000 | 0.970 |
| uncensored-standard | myllmbox-v5.4 | 12 | auto | 16 | 62.0 | 183 | 16.2 | **0.180** | 3.16 | — | 109.7 | 136.1 | — | — | **330** | 0 | 1.000 | 0.960 |

### Concurrency scaling (aggregate output tok/s)

| run | C=1 | C=2 | C=4 | C=8 | C=16 | KV tokens | KV max conc |
|---|---|---|---|---|---|---|---|
| az-standard | 55.5 | 83.9 | 126.7 | 163.0 | — | 750933 | 1.83 |
| uncensored-standard | 57.2 | 71.6 | 109.7 | 136.1 | — | 750933 | 1.83 |
| az-524k-single | 53.8 | — | — | — | — | 767013 | 1.46 |
| uncensored-524k-single | 54.5 | — | — | — | — | 767013 | 1.46 |

### Speculative decoding efficiency

| run | MTP | draft depth | acceptance | Δaccept pp | tok/step | step ms | steps/s | tok/chunk | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| uncensored-standard | 12 | 12.00 | 0.180 | — | 3.16 | 51.0 | 19.6 | 3.48 | 62.0 |
| az-standard | 12 | 12.00 | 0.170 | — | 3.08 | 48.8 | 20.5 | 3.35 | 63.1 |
| uncensored-524k-single | 16 | 16.00 | 0.150 | — | 3.35 | 53.4 | 18.7 | 3.58 | 62.7 |
| az-524k-single | 16 | 16.00 | 0.140 | — | 3.23 | 51.4 | 19.4 | 3.39 | 62.8 |

### Prefill / long-context

| run | 1k tok/s | 8k tok/s | 32k tok/s | 32k TTFT ms | 128k TTFT ms | 128k needle |
|---|---|---|---|---|---|---|
| az-standard | 583 | 2779 | 2772 | 11778 | 48480 | yes |
| uncensored-524k-single | 1365 | 2580 | 2617 | 12479 | 222787 | yes |
| uncensored-standard | 1413 | 2638 | 2493 | 13099 | 53920 | yes |
| az-524k-single | 341 | 2876 | 2471 | 13214 | 231159 | yes |

### Boot & run health

| run | boot s | profile | suite s | errs | warns | image |
|---|---|---|---|---|---|---|
| uncensored-standard | 330 | full | 472 | 0 | 0 | sha256:bc42593a5bbf |
| uncensored-524k-single | 348 | full | 635 | 0 | 0 | sha256:bc42593a5bbf |
| az-524k-single | 377 | full | 643 | 0 | 0 | sha256:bc42593a5bbf |
| az-standard | 393 | full | 430 | 0 | 0 | sha256:bc42593a5bbf |

### Raw rows

See `summary.csv` for the full machine-readable table and each run's `summary.json` / `requests.jsonl` for per-request detail.

