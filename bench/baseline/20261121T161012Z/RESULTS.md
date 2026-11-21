# Benchmark results

Generated: 2026-11-21T15:43:50.415+00:00

Run roots: /home/pgx/Documenti/vllm-gb10/bench/baseline/20261121T161012Z
Runs: 2 (0 failed)

Notes: `decode_tok_s` = median single-stream decode speed over 3 repeats (usage.completion_tokens / decode window, thinking off, temp 0, seed 42). `acceptance_rate` = accepted draft tokens / proposed draft tokens from /metrics deltas; `tokens_per_step` = emitted tokens / draft step (includes the bonus token). `conc_N` = aggregate output tok/s at N concurrent streams (2 waves). `boot_s` includes container start → /health.

### Ranked by single-stream decode throughput

| run | stack | MTP | KV | seats | decode tok/s | TTFT p50 ms | TPOT p50 ms | accept | tok/step | Δdecode % | C=4 tok/s | C=8 tok/s | ΔC8 % | C=16 tok/s | boot s | errs | ratio | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| az-524k-single | myllmbox-v5.4 | 16 | auto | 1 | **62.7** | **152** | 16.0 | 0.140 | 3.27 | — | — | — | — | — | 385 | 0 | 1.000 | 0.970 |
| uncensored-524k-single | myllmbox-v5.4 | 16 | auto | 1 | 62.3 | 186 | 16.1 | **0.150** | 3.32 | — | — | — | — | — | **344** | 0 | 1.000 | 0.980 |

### Concurrency scaling (aggregate output tok/s)

| run | C=1 | C=2 | C=4 | C=8 | C=16 | KV tokens | KV max conc |
|---|---|---|---|---|---|---|---|
| az-524k-single | 55.6 | — | — | — | — | 767013 | 1.46 |
| uncensored-524k-single | 54.6 | — | — | — | — | 767013 | 1.46 |

### Speculative decoding efficiency

| run | MTP | draft depth | acceptance | Δaccept pp | tok/step | step ms | steps/s | tok/chunk | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| uncensored-524k-single | 16 | 16.00 | 0.150 | — | 3.32 | 53.4 | 18.7 | 3.53 | 62.3 |
| az-524k-single | 16 | 16.00 | 0.140 | — | 3.27 | 52.2 | 19.1 | 3.37 | 62.7 |

### Prefill / long-context

| run | 1k tok/s | 8k tok/s | 32k tok/s | 32k TTFT ms | 128k TTFT ms | 128k needle |
|---|---|---|---|---|---|---|
| uncensored-524k-single | 1267 | 2581 | 2806 | 11639 | 223622 | yes |
| az-524k-single | 344 | 2841 | 2427 | 13457 | 233877 | yes |

### Boot & run health

| run | boot s | profile | suite s | errs | warns | image |
|---|---|---|---|---|---|---|
| uncensored-524k-single | 344 | full | 637 | 0 | 0 | sha256:bc42593a5bbf |
| az-524k-single | 385 | full | 640 | 0 | 0 | sha256:bc42593a5bbf |

### Raw rows

See `summary.csv` for the full machine-readable table and each run's `summary.json` / `requests.jsonl` for per-request detail.

