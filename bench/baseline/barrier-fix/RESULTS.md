# Benchmark results

Generated: 2026-11-22T00:52:04.061+00:00

Run roots: bench/baseline/barrier-fix
Runs: 3 (0 failed)

Notes: `decode_tok_s` = median single-stream decode speed over 3 repeats (usage.completion_tokens / decode window, thinking off, temp 0, seed 42). `acceptance_rate` = accepted draft tokens / proposed draft tokens from /metrics deltas; `tokens_per_step` = emitted tokens / draft step (includes the bonus token). `conc_N` = aggregate output tok/s at N concurrent streams (2 waves). `boot_s` includes container start → /health.

### Ranked by single-stream decode throughput

| run | stack | MTP | KV | seats | decode tok/s | TTFT p50 ms | TPOT p50 ms | accept | tok/step | Δdecode % | C=4 tok/s | C=8 tok/s | ΔC8 % | C=16 tok/s | boot s | errs | ratio | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256k-fleet-mml | myllmbox-v5.4 | 12 | fp8 | 4 | **59.6** | 157 | 16.9 | 0.160 | 2.94 | — | 72.1 | — | — | — | **340** | 0 | 1.000 | — |
| 256k-fleet-control | myllmbox-v5.4 | 12 | fp8 | 4 | 58.2 | 153 | 17.3 | **0.160** | 2.93 | — | 72.1 | — | — | — | 358 | 0 | 1.000 | — |
| 256k-fleet-yarn | myllmbox-v5.4 | 12 | fp8 | 4 | 58.1 | **142** | 17.3 | 0.160 | 2.93 | — | **72.8** | — | — | — | 362 | 0 | 1.000 | — |

### Concurrency scaling (aggregate output tok/s)

| run | C=1 | C=2 | C=4 | C=8 | C=16 | KV tokens | KV max conc |
|---|---|---|---|---|---|---|---|
| 256k-fleet-yarn | 45.0 | — | 72.8 | — | — | 1384355 | 5.28 |
| 256k-fleet-mml | 46.1 | — | 72.1 | — | — | 1458424 | 3.56 |
| 256k-fleet-control | 46.4 | — | 72.1 | — | — | 1384355 | 5.28 |

### Speculative decoding efficiency

| run | MTP | draft depth | acceptance | Δaccept pp | tok/step | step ms | steps/s | tok/chunk | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| 256k-fleet-control | 12 | 12.00 | 0.160 | — | 2.93 | 50.5 | 19.8 | 3.05 | 58.2 |
| 256k-fleet-mml | 12 | 12.00 | 0.160 | — | 2.94 | 49.4 | 20.2 | 3.12 | 59.6 |
| 256k-fleet-yarn | 12 | 12.00 | 0.160 | — | 2.93 | 50.3 | 19.9 | 3.03 | 58.1 |

### Prefill / long-context

| run | 1k tok/s | 8k tok/s | 32k tok/s | 32k TTFT ms | 128k TTFT ms | 128k needle |
|---|---|---|---|---|---|---|
| 256k-fleet-control | 593 | 2964 | — | — | — | — |
| 256k-fleet-mml | 579 | 2961 | — | — | — | — |
| 256k-fleet-yarn | 577 | 2959 | — | — | — | — |

### Anomalies / warnings (do not rank on the flagged level)

- **256k-fleet-control**: cold-start stall: C4 TTFT p50 2591 ms (>1500 ms)
- **256k-fleet-mml**: cold-start stall: C4 TTFT p50 2668 ms (>1500 ms)
- **256k-fleet-yarn**: cold-start stall: C4 TTFT p50 2698 ms (>1500 ms)

### Boot & run health

| run | boot s | profile | suite s | errs | warns | image |
|---|---|---|---|---|---|---|
| 256k-fleet-mml | 340 | quick | 45 | 0 | 0 | sha256:bc42593a5bbf |
| 256k-fleet-control | 358 | quick | 45 | 0 | 0 | sha256:bc42593a5bbf |
| 256k-fleet-yarn | 362 | quick | 45 | 0 | 0 | sha256:bc42593a5bbf |

### Raw rows

See `summary.csv` for the full machine-readable table and each run's `summary.json` / `requests.jsonl` for per-request detail.

