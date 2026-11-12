# Benchmark results

Generated: 2026-11-12T18:01:33.486+00:00

Run roots: bench/results
Runs: 25 (0 failed)

Notes: `decode_tok_s` = median single-stream decode speed over 3 repeats (usage.completion_tokens / decode window, thinking off, temp 0, seed 42). `acceptance_rate` = accepted draft tokens / proposed draft tokens from /metrics deltas; `tokens_per_step` = emitted tokens / draft step (includes the bonus token). `conc_N` = aggregate output tok/s at N concurrent streams (2 waves). `boot_s` includes container start → /health.

### Ranked by single-stream decode throughput

| run | stack | MTP | KV | seats | decode tok/s | TTFT p50 ms | TPOT p50 ms | accept | tok/step | Δdecode % | C=4 tok/s | C=8 tok/s | ΔC8 % | C=16 tok/s | boot s | errs | ratio | quality |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d-curve-k4 | myllmbox-v5.4 | 4 | auto | 16 | **60.9** | 150 | 16.6 | 0.490 | 2.94 | 14.2 | **116.1** | **178.4** | 25.6 | — | **330** | 0 | 1.000 | — |
| ab2-k4 | myllmbox-v5.4 | 4 | auto | 16 | 56.6 | 158 | 17.8 | 0.490 | 2.94 | 6.3 | 103.1 | 158.1 | 11.3 | — | 338 | 0 | 1.000 | — |
| d-draftk10-spec6 | myllmbox-v5.4 | 6 | auto | 16 | 56.4 | 160 | 17.9 | 0.340 | 3.04 | 5.8 | 68.3 | 162.9 | 14.7 | — | 350 | 0 | 1.000 | 0.970 |
| finalist-winner-full | — | — | — | — | 55.8 | 168 | 18.0 | 0.510 | 3.03 | 4.8 | 115.5 | 156.4 | 10.1 | **243.1** | — | 0 | 1.000 | 0.970 |
| finalist-curve-k6 | myllmbox-v5.4 | 6 | auto | 16 | 55.7 | 166 | 18.1 | 0.330 | 2.98 | 4.5 | 102.8 | 155.4 | 9.4 | — | 348 | 0 | 1.000 | — |
| ab-m0-a | myllmbox-v5.4 | 12 | auto | 16 | 55.4 | 163 | 18.2 | 0.160 | 2.97 | 4.0 | 101.3 | 152.3 | 7.2 | — | 352 | 0 | 1.000 | — |
| ab-m0-b | myllmbox-v5.4 | 12 | auto | 16 | 54.9 | 160 | 18.3 | 0.170 | 3.04 | 3.1 | 113.5 | 160.4 | 12.9 | — | 348 | 0 | 1.000 | — |
| ab2-m0-b | myllmbox-v5.4 | 12 | auto | 16 | 54.6 | 159 | 18.5 | 0.170 | 3.00 | 2.4 | 104.9 | 155.6 | 9.5 | — | 352 | 0 | 1.000 | — |
| finalist-winner-temp06 | — | — | — | — | 53.5 | 170 | 18.8 | 0.500 | 3.00 | 0.4 | 114.2 | 152.6 | 7.4 | — | — | 0 | 1.000 | — |
| ab-k4 | myllmbox-v5.4 | 4 | auto | 16 | 53.4 | 161 | 18.8 | 0.480 | 2.91 | 0.2 | 106.2 | 153.0 | 7.7 | — | 346 | 0 | 1.000 | — |
| m0-reference | myllmbox-v5.4 | 12 | auto | 16 | 53.3 | 181 | 18.8 | 0.170 | 3.04 | 0.0 | 103.6 | 142.0 | 0.0 | 198.2 | 356 | 0 | 1.000 | 0.970 |
| ab2-m0-a | myllmbox-v5.4 | 12 | auto | 16 | 52.8 | **148** | 19.1 | 0.160 | 2.93 | -0.8 | 101.1 | 156.8 | 10.4 | — | 356 | 0 | 1.000 | — |
| d-self-scale2 | myllmbox-v5.4 | 12 | auto | 16 | 51.8 | 172 | 19.5 | 0.170 | 3.00 | -2.8 | 97.7 | 142.9 | 0.6 | — | 348 | 0 | 1.000 | — |
| m0-reference-full | myllmbox-v5.4 | 12 | auto | 16 | 51.6 | 189 | 19.4 | 0.170 | 3.06 | -3.1 | 102.5 | 138.2 | -2.7 | 192.0 | — | 0 | — | — |
| b-k10-curve-k4 | myllmbox-v5.4 | 4 | auto | 16 | 51.1 | 194 | 19.8 | **0.520** | 3.09 | -4.1 | 107.4 | 148.3 | 4.5 | — | 340 | 0 | 1.000 | 0.960 |
| b-k10-curve-k4-medium | myllmbox-v5.4 | 4 | auto | 16 | 50.8 | 195 | 19.8 | 0.500 | 2.99 | -4.6 | 96.0 | 138.3 | -2.6 | — | 342 | 0 | 1.000 | 0.970 |
| d-dynamic-ceiling-9 | myllmbox-v5.4 | 9 | auto | 16 | 50.7 | 185 | 19.8 | 0.220 | 2.95 | -4.8 | 91.9 | 147.8 | 4.0 | — | 358 | 0 | 1.000 | — |
| finalist-curve-k5 | myllmbox-v5.4 | 5 | auto | 16 | 50.7 | 162 | 19.9 | 0.390 | 2.97 | -4.8 | 91.3 | 146.4 | 3.1 | — | 342 | 0 | 1.000 | — |
| b-medium-template | myllmbox-v5.4 | 12 | auto | 16 | 49.2 | 179 | 20.4 | 0.160 | 2.97 | -7.6 | 95.2 | 139.0 | -2.1 | — | 354 | 0 | 1.000 | 0.970 |
| b-kvfp8-spec7 | myllmbox-v5.4 | 7 | fp8 | 16 | 48.8 | 172 | 20.6 | 0.270 | 2.88 | -8.3 | 90.8 | 140.2 | -1.3 | — | 362 | 0 | 1.000 | 0.960 |
| eugr-b12x-base | eugr-b12x | 4 | fp8 | 8 | 47.0 | 319 | 21.4 | 0.510 | 3.04 | -11.7 | 78.2 | 99.9 | -29.6 | — | — | 0 | 1.000 | — |
| m0-reference-clean-full | myllmbox-v5.4 | 12 | auto | 16 | 46.9 | 193 | 21.4 | 0.170 | 3.07 | -12.1 | 89.8 | 131.8 | -7.2 | 202.6 | 356 | 0 | 0.700 | — |
| d-self-head0 | myllmbox-v5.4 | 12 | auto | 16 | 43.2 | 207 | 23.3 | 0.160 | 2.94 | -18.9 | 85.7 | 127.1 | -10.5 | — | 362 | 0 | 1.000 | — |
| eugr-b12x-spec6 | eugr-b12x | 6 | fp8 | 8 | 42.4 | 310 | 23.7 | 0.380 | 3.27 | -20.4 | 64.2 | 89.9 | -36.7 | — | — | 0 | 1.000 | — |
| eugr-b12x-spec7 | eugr-b12x | 7 | fp8 | 8 | 41.0 | 325 | 24.7 | 0.350 | 3.48 | -23.0 | 64.5 | 81.5 | -42.6 | — | — | 0 | 1.000 | — |

### Concurrency scaling (aggregate output tok/s)

| run | C=1 | C=2 | C=4 | C=8 | C=16 | KV tokens | KV max conc |
|---|---|---|---|---|---|---|---|
| d-curve-k4 | 47.2 | 75.7 | 116.1 | 178.4 | — | 750933 | 1.83 |
| finalist-winner-full | 48.8 | 76.0 | 115.5 | 156.4 | 243.1 | 750933 | 1.83 |
| finalist-winner-temp06 | 41.2 | 86.4 | 114.2 | 152.6 | — | 750933 | 1.83 |
| ab-m0-b | 43.3 | 75.4 | 113.5 | 160.4 | — | 750933 | 1.83 |
| b-k10-curve-k4 | 40.5 | 72.7 | 107.4 | 148.3 | — | 750933 | 1.83 |
| ab-k4 | 41.2 | 73.6 | 106.2 | 153.0 | — | 750933 | 1.83 |
| ab2-m0-b | 40.5 | 71.0 | 104.9 | 155.6 | — | 750933 | 1.83 |
| m0-reference | 49.0 | 67.3 | 103.6 | 142.0 | 198.2 | 750933 | 1.83 |
| ab2-k4 | 40.3 | 76.2 | 103.1 | 158.1 | — | 750933 | 1.83 |
| finalist-curve-k6 | 43.8 | 75.1 | 102.8 | 155.4 | — | 750933 | 1.83 |
| m0-reference-full | 45.5 | 66.9 | 102.5 | 138.2 | 192.0 | 750933 | 1.83 |
| ab-m0-a | 42.6 | 75.9 | 101.3 | 152.3 | — | 750933 | 1.83 |
| ab2-m0-a | 44.7 | 77.7 | 101.1 | 156.8 | — | 750933 | 1.83 |
| d-self-scale2 | 40.6 | 69.3 | 97.7 | 142.9 | — | 750933 | 1.83 |
| b-k10-curve-k4-medium | 41.0 | 73.3 | 96.0 | 138.3 | — | 750933 | 1.83 |
| b-medium-template | 37.0 | 67.8 | 95.2 | 139.0 | — | 750933 | 1.83 |
| d-dynamic-ceiling-9 | 38.6 | 67.2 | 91.9 | 147.8 | — | 750933 | 1.83 |
| finalist-curve-k5 | 39.6 | 70.6 | 91.3 | 146.4 | — | 750933 | 1.83 |
| b-kvfp8-spec7 | 37.6 | 64.5 | 90.8 | 140.2 | — | 1458424 | 3.56 |
| m0-reference-clean-full | 17.1 | 56.6 | 89.8 | 131.8 | 202.6 | 750933 | 1.83 |
| d-self-head0 | 34.0 | 62.0 | 85.7 | 127.1 | — | 750933 | 1.83 |
| eugr-b12x-base | 33.2 | 64.7 | 78.2 | 99.9 | — | 586074 | 2.24 |
| d-draftk10-spec6 | 45.5 | 83.2 | 68.3 | 162.9 | — | 750933 | 1.83 |
| eugr-b12x-spec7 | 27.2 | 43.6 | 64.5 | 81.5 | — | 524442 | 2.00 |
| eugr-b12x-spec6 | 28.8 | 47.5 | 64.2 | 89.9 | — | 547533 | 2.09 |

### Speculative decoding efficiency

| run | MTP | draft depth | acceptance | Δaccept pp | tok/step | step ms | steps/s | tok/chunk | decode tok/s |
|---|---|---|---|---|---|---|---|---|---|
| b-k10-curve-k4 | 4 | 4.00 | 0.520 | 35.0 | 3.09 | 60.6 | 16.5 | 3.24 | 51.1 |
| eugr-b12x-base | 4 | 4.00 | 0.510 | 34.0 | 3.04 | 64.7 | 15.5 | 3.46 | 47.0 |
| finalist-winner-full | — | 4.00 | 0.510 | 34.0 | 3.03 | 54.3 | 18.4 | 3.30 | 55.8 |
| b-k10-curve-k4-medium | 4 | 4.00 | 0.500 | 33.0 | 2.99 | 58.9 | 17.0 | 3.25 | 50.8 |
| finalist-winner-temp06 | — | 4.00 | 0.500 | 33.0 | 3.00 | 56.1 | 17.8 | 3.15 | 53.5 |
| ab2-k4 | 4 | 4.00 | 0.490 | 32.0 | 2.94 | 52.0 | 19.2 | 3.28 | 56.6 |
| d-curve-k4 | 4 | 4.00 | 0.490 | 32.0 | 2.94 | 48.3 | 20.7 | 3.18 | 60.9 |
| ab-k4 | 4 | 4.00 | 0.480 | 31.0 | 2.91 | 54.5 | 18.4 | 3.12 | 53.4 |
| finalist-curve-k5 | 5 | 5.00 | 0.390 | 22.0 | 2.97 | 58.5 | 17.1 | 3.18 | 50.7 |
| eugr-b12x-spec6 | 6 | 6.00 | 0.380 | 21.0 | 3.27 | 77.1 | 13.0 | 3.69 | 42.4 |
| eugr-b12x-spec7 | 7 | 7.00 | 0.350 | 18.0 | 3.48 | 84.7 | 11.8 | 3.94 | 41.0 |
| d-draftk10-spec6 | 6 | 6.00 | 0.340 | 17.0 | 3.04 | 53.9 | 18.6 | 3.21 | 56.4 |
| finalist-curve-k6 | 6 | 6.00 | 0.330 | 16.0 | 2.98 | 53.5 | 18.7 | 3.23 | 55.7 |
| b-kvfp8-spec7 | 7 | 7.00 | 0.270 | 10.0 | 2.88 | 59.0 | 17.0 | 3.12 | 48.8 |
| d-dynamic-ceiling-9 | 9 | 9.00 | 0.220 | 5.0 | 2.95 | 58.1 | 17.2 | 3.17 | 50.7 |
| ab-m0-b | 12 | 12.00 | 0.170 | 0.0 | 3.04 | 55.3 | 18.1 | 3.20 | 54.9 |
| ab2-m0-b | 12 | 12.00 | 0.170 | 0.0 | 3.00 | 55.0 | 18.2 | 3.20 | 54.6 |
| d-self-scale2 | 12 | 12.00 | 0.170 | 0.0 | 3.00 | 58.0 | 17.2 | 3.21 | 51.8 |
| m0-reference | 12 | 12.00 | 0.170 | 0.0 | 3.04 | 57.0 | 17.6 | 3.37 | 53.3 |
| m0-reference-clean-full | 12 | 12.00 | 0.170 | 0.0 | 3.07 | 65.5 | 15.3 | 3.61 | 46.9 |
| m0-reference-full | 12 | 12.00 | 0.170 | 0.0 | 3.06 | 59.3 | 16.9 | — | 51.6 |
| ab-m0-a | 12 | 12.00 | 0.160 | -1.0 | 2.97 | 53.5 | 18.7 | 3.23 | 55.4 |
| ab2-m0-a | 12 | 12.00 | 0.160 | -1.0 | 2.93 | 55.5 | 18.0 | 3.10 | 52.8 |
| b-medium-template | 12 | 12.00 | 0.160 | -1.0 | 2.97 | 60.2 | 16.6 | 3.17 | 49.2 |
| d-self-head0 | 12 | 12.00 | 0.160 | -1.0 | 2.94 | 67.9 | 14.7 | 3.18 | 43.2 |

### Prefill / long-context

| run | 1k tok/s | 8k tok/s | 32k tok/s | 32k TTFT ms | 128k TTFT ms | 128k needle |
|---|---|---|---|---|---|---|
| finalist-winner-full | 1267 | 2545 | 2506 | 9567 | 39373 | yes |
| m0-reference-clean-full | 1108 | 2397 | 2390 | 10031 | 41002 | yes |
| m0-reference-full | 1108 | 2235 | 2278 | 10526 | 43477 | yes |
| m0-reference | 1266 | 2341 | 2260 | 10609 | 42489 | yes |
| ab-k4 | 1268 | 2582 | — | — | — | — |
| ab-m0-a | 1382 | 2568 | — | — | — | — |
| ab-m0-b | 1304 | 2572 | — | — | — | — |
| ab2-k4 | 1351 | 2605 | — | — | — | — |
| ab2-m0-a | 1282 | 2579 | — | — | — | — |
| ab2-m0-b | 1291 | 2576 | — | — | — | — |
| b-k10-curve-k4 | 1143 | 2323 | — | — | — | — |
| b-k10-curve-k4-medium | 1234 | 2378 | — | — | — | — |
| b-kvfp8-spec7 | 1275 | 2387 | — | — | — | — |
| b-medium-template | 1250 | 2327 | — | — | — | — |
| d-curve-k4 | 1417 | 2919 | — | — | — | — |
| d-draftk10-spec6 | 1429 | 2785 | — | — | — | — |
| d-dynamic-ceiling-9 | 1187 | 2190 | — | — | — | — |
| d-self-head0 | 1135 | 2314 | — | — | — | — |
| d-self-scale2 | 1306 | 2326 | — | — | — | — |
| eugr-b12x-base | 1298 | 1854 | — | — | — | — |
| eugr-b12x-spec6 | 1302 | 1844 | — | — | — | — |
| eugr-b12x-spec7 | 1255 | 1776 | — | — | — | — |
| finalist-curve-k5 | 1300 | 2422 | — | — | — | — |
| finalist-curve-k6 | 1370 | 2613 | — | — | — | — |
| finalist-winner-temp06 | 1348 | 2601 | — | — | — | — |

### Anomalies / warnings (do not rank on the flagged level)

- **d-draftk10-spec6**: scaling: C4 (68.3) < 0.9*C2 (83.2); cold-start stall: C4 TTFT p50 3108 ms (>1500 ms)
- **m0-reference-clean-full**: token-ratio decode=0.70 outside 0.95-1.05; token-ratio concurrency=0.83 outside 0.95-1.05

### Boot & run health

| run | boot s | profile | suite s | errs | warns | image |
|---|---|---|---|---|---|---|
| d-curve-k4 | 330 | quick | 50 | 0 | 0 | sha256:bc42593a5bbf |
| ab2-k4 | 338 | quick | 55 | 0 | 0 | sha256:bc42593a5bbf |
| b-k10-curve-k4 | 340 | quick | 139 | 0 | 0 | sha256:bc42593a5bbf |
| b-k10-curve-k4-medium | 342 | quick | 141 | 0 | 0 | sha256:bc42593a5bbf |
| finalist-curve-k5 | 342 | quick | 60 | 0 | 0 | sha256:bc42593a5bbf |
| ab-k4 | 346 | quick | 56 | 0 | 0 | sha256:bc42593a5bbf |
| d-self-scale2 | 348 | quick | 59 | 0 | 0 | sha256:bc42593a5bbf |
| finalist-curve-k6 | 348 | quick | 55 | 0 | 0 | sha256:bc42593a5bbf |
| ab-m0-b | 348 | quick | 55 | 0 | 0 | sha256:bc42593a5bbf |
| d-draftk10-spec6 | 350 | quick | 124 | 0 | 0 | sha256:bc42593a5bbf |
| ab2-m0-b | 352 | quick | 56 | 0 | 0 | sha256:bc42593a5bbf |
| ab-m0-a | 352 | quick | 56 | 0 | 0 | sha256:bc42593a5bbf |
| b-medium-template | 354 | quick | 144 | 0 | 0 | sha256:bc42593a5bbf |
| m0-reference-clean-full | 356 | full | 342 | 0 | 1 | sha256:bc42593a5bbf |
| ab2-m0-a | 356 | quick | 56 | 0 | 0 | sha256:bc42593a5bbf |
| m0-reference | 356 | full | 331 | 0 | 0 | sha256:bc42593a5bbf |
| d-dynamic-ceiling-9 | 358 | quick | 61 | 0 | 0 | sha256:bc42593a5bbf |
| d-self-head0 | 362 | quick | 70 | 0 | 0 | sha256:bc42593a5bbf |
| b-kvfp8-spec7 | 362 | quick | 149 | 0 | 0 | sha256:bc42593a5bbf |
| eugr-b12x-base | — | quick | 68 | 0 | 0 | sha256:19ab778dd57f |
| eugr-b12x-spec6 | — | quick | 77 | 0 | 0 | sha256:19ab778dd57f |
| eugr-b12x-spec7 | — | quick | 80 | 0 | 0 | sha256:19ab778dd57f |
| finalist-winner-full | — | full | 478 | 0 | 0 | — |
| finalist-winner-temp06 | — | quick | 55 | 0 | 0 | — |
| m0-reference-full | — | full | 271 | 0 | 2 | sha256:bc42593a5bbf |

### Raw rows

See `summary.csv` for the full machine-readable table and each run's `summary.json` / `requests.jsonl` for per-request detail.

