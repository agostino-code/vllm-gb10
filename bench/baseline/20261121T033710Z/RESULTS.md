# Benchmark results

Generated: 2026-11-21T03:42:05.621+00:00

Run roots: /home/pgx/Documenti/vllm-gb10/bench/baseline/20261121T033710Z
Runs: 4 (0 failed)

Notes: `decode_tok_s` = median single-stream decode speed over 3 repeats (usage.completion_tokens / decode window, thinking off, temp 0, seed 42). `acceptance_rate` = accepted draft tokens / proposed draft tokens from /metrics deltas; `tokens_per_step` = emitted tokens / draft step (includes the bonus token). `conc_N` = aggregate output tok/s at N concurrent streams (2 waves). `boot_s` includes container start → /health.

### Ranked by single-stream decode throughput

| run | stack | MTP | KV | seats | decode tok/s | TTFT p50 ms | TPOT p50 ms | accept | tok/step | Δdecode % | C=4 tok/s | C=8 tok/s | ΔC8 % | C=16 tok/s | boot s | errs | ratio | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| az-standard | myllmbox-v5.4 | 12 | auto | 16 | **63.5** | **144** | 15.8 | 0.170 | 3.09 | — | **124.7** | **172.4** | — | — | 593 | 0 | 1.000 | 0.970 |
| uncensored-standard | myllmbox-v5.4 | 12 | auto | 16 | 62.7 | 185 | 16.0 | **0.180** | 3.19 | — | 109.7 | 140.3 | — | — | 520 | 0 | 1.000 | 0.960 |
| az-524k-single | myllmbox-v5.4 | 16 | auto | 1 | 60.7 | 293 | 16.5 | 0.140 | 3.25 | — | — | — | — | — | **362** | 0 | 1.000 | 0.970 |
| uncensored-524k-single | myllmbox-v5.4 | 16 | auto | 1 | 59.8 | 323 | 16.8 | 0.140 | 3.32 | — | — | — | — | — | 379 | 0 | 1.000 | 0.970 |

### Concurrency scaling (aggregate output tok/s)

| run | C=1 | C=2 | C=4 | C=8 | C=16 | KV tokens | KV max conc |
|---|---|---|---|---|---|---|---|
| az-standard | 56.6 | 82.0 | 124.7 | 172.4 | — | 750933 | 1.83 |
| uncensored-standard | 55.9 | 74.2 | 109.7 | 140.3 | — | 750933 | 1.83 |
| az-524k-single | 57.9 | — | — | — | — | 604947 | 1.15 |
| uncensored-524k-single | 56.8 | — | — | — | — | 604947 | 1.15 |

### Speculative decoding efficiency

| run | MTP | draft depth | acceptance | Δaccept pp | tok/step | step ms | steps/s | tok/chunk | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| uncensored-standard | 12 | 12.00 | 0.180 | — | 3.19 | 51.0 | 19.6 | 3.58 | 62.7 |
| az-standard | 12 | 12.00 | 0.170 | — | 3.09 | 48.6 | 20.6 | 3.37 | 63.5 |
| az-524k-single | 16 | 16.00 | 0.140 | — | 3.25 | 53.6 | 18.7 | 3.37 | 60.7 |
| uncensored-524k-single | 16 | 16.00 | 0.140 | — | 3.32 | 55.5 | 18.0 | 3.53 | 59.8 |

### Prefill / long-context

| run | 1k tok/s | 8k tok/s | 32k tok/s | 32k TTFT ms | 128k TTFT ms | 128k needle |
|---|---|---|---|---|---|---|
| az-standard | 475 | 2852 | 2830 | 8470 | 34983 | yes |
| uncensored-524k-single | 918 | 2355 | 2797 | 8572 | 158267 | yes |
| az-524k-single | 263 | 2643 | 2682 | 8939 | 168150 | yes |
| uncensored-standard | 1194 | 2579 | 2479 | 9672 | 39224 | yes |

### Boot & run health

| run | boot s | profile | suite s | errs | warns | image |
|---|---|---|---|---|---|---|
| az-524k-single | 362 | full | 582 | 0 | 0 | sha256:bc42593a5bbf |
| uncensored-524k-single | 379 | full | 576 | 0 | 0 | sha256:bc42593a5bbf |
| uncensored-standard | 520 | full | 449 | 0 | 0 | sha256:bc42593a5bbf |
| az-standard | 593 | full | 413 | 0 | 0 | sha256:bc42593a5bbf |

### Raw rows

See `summary.csv` for the full machine-readable table and each run's `summary.json` / `requests.jsonl` for per-request detail.

