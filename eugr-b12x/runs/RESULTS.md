# eugr b12x — GPU-window results (2026-11-12)

One GPU window, one DGX Spark GB10, model `azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ`
mounted read-only from the HF cache, image `eugr/spark-vllm-b12x:latest`
(vLLM `0.1.dev21830+g8244d1c44.d20261102`), container `qwen38-flash-next-b12x`,
port 8000. Measured with `bench/suite.py --profile quick --concurrency 1,2,4,8`
(bench-harness), one server at a time, sequential boots.

Common flags: `--load-format b12x --quantization modelopt_mixed --kv-cache-dtype fp8
--block-size 16 --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 4096
--gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x
--gpu-memory-utilization 0.75` + `VLLM_PLE_TABLE_MEMORY=disk B12X_DISK_BACKEND=io_uring`
+ **`AUTOTUNE=0`** (`--kernel-config '{"enable_b12x_autotune": false}'`) for every run.

## Results

| config | boot wall | decode single-stream | code cat. | TTFT p50 | TPOT p50 | c=1 | c=2 | c=4 | c=8 | acceptance | tok/step | KV pool (tokens) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **base MTP 4** | 483 s | **47.0 tok/s** | 48.5 | 319 ms | 21.4 ms | 33.2 | 64.7 | **78.2** | **99.9** | **51.1 %** | 3.04 | **586,074** |
| spec6 MTP 6 | 363 s | 42.4 | 44.9 | 310 ms | 23.7 ms | 28.8 | 47.5 | 64.2 | 89.9 | 37.8 % | 3.27 | 547,533 |
| spec7 MTP 7 | 363 s | 41.0 | 45.2 | 325 ms | 24.7 ms | 27.2 | 43.6 | 64.5 | 81.5 | 35.4 % | 3.48 | 524,442 |
| spec9 MTP 9 | **boot fails** | — | — | — | — | — | — | — | — | — | — | — |

c=1/2/4/8 are total output tok/s per concurrency wave. Prefill (1k/8k prompt
targets) is ~1.3k / ~1.85k tok/s for all booted configs (spec7: 1.26k / 1.78k).

All runs: 0 request errors, 0 suite warnings. First-boot 483 s includes reuse of
879 cached b12x kernel compilations from an aborted tuned boot; later boots ~363 s.

## Findings

1. **Fixed depth 4 is the best eugr b12x config measured.** Increasing depth makes
   throughput *worse* at every concurrency (decode 47.0 → 42.4 → 41.0 tok/s;
   c=8 99.9 → 89.9 → 81.5) even though tokens/step rises 3.04 → 3.27 → 3.48:
   acceptance falls faster than the extra accepted tokens repay (51.1 % → 37.8 % →
   35.4 %). Draft compute, not acceptance, is the bottleneck on this stack.
2. **MTP 9 cannot boot at all** — a hard b12x GDN-decode limit:
   `ValueError: state_index_columns must be at most 8, got 10` from
   `flashinfer/experimental/b12x/sequence/gdn_decode/_impl.py` via
   `qwen_gdn_linear_attn._make_b12x_gdn_caps` (columns = num_spec + 1). This image
   supports **max 7** speculative tokens with this hybrid GDN model; there is no
   flag/env workaround. Evidence: `runs/20261112T140838Z-boot.log`.
3. **Acceptance vs myllmbox** (same checkpoint, same box): eugr fixed-MTP-4 gets
   51.1 % acceptance at 3.04 tokens/step; the myllmbox v5.4 M0 reference gets 17.2 %
   at ~3.06 tokens/step. Same effective tokens per draft step at ~3× less draft
   compute — i.e. myllmbox's shipped K=12 dynamic-depth path is over-drafting for
   this 5-expert checkpoint, even though myllmbox is still faster in absolute terms
   (code c=1 ≈56 vs 48.5 tok/s here; c=8 ≈139 vs 99.9) thanks to its
   load-format/mmap/dynamic-depth stack and larger KV budget.
4. **KV capacity is under-allocated by the eugr recipe on this box.** fp8 KV at
   gpu-mem 0.75 gives 586,074 tokens (max concurrency 2.24× at 262,144 ctx);
   myllmbox uses 23 GB bf16 KV = 750,933 tokens. There is no `--kv-cache-memory`
   in this image; the lever is `--gpu-memory-utilization` (0.80–0.85) with
   `PYTORCH_ALLOC_CONF` kept equal. Depth also costs KV: 586k (MTP4) → 547k (MTP6)
   → 524k (MTP7).
5. **AUTOTUNE=0 is required for a practical window.** The b12x first-boot kernel
   autotune is a 241-unit search (≈16 s/unit, all candidates compiled fresh) —
   estimated 55–65 min. With `--kernel-config '{"enable_b12x_autotune": false}'`
   boot is ~6–8 min (mandatory preparation still runs, compile cache reused).
   A tuned boot may add single-digit %; it was not measured. Any eugr/myllmbox
   comparison must use the same AUTOTUNE setting on the eugr side.
6. PLE disk mode worked exactly as designed: no table map file, model mount stayed
   read-only, no ~59 GB `ple-nvme` map (unlike myllmbox). io_uring + O_DIRECT
   preflight passed (io_uring_setup OK, dd iflag=direct OK).

## Artifacts

* Runs (bench-harness): `bench/results/eugr-b12x-{base,spec6,spec7}/` (summary.json,
  requests.jsonl, metrics/, meta.json, server.log). spec9 failure: no run dir.
* Boot evidence: `eugr-b12x/runs/20261112T135039Z-*` (base), `…T140035Z-*` (spec6),
  `…T140924Z-*` (spec7), `…T140838Z-*` (spec9 failure boot log),
  `runs/aborted-autotune-boot.log` (tuned first attempt, aborted to save the window).
* Per-config rollups: `runs/{eugr-base,spec6,spec7}-summary.txt` (boot seconds,
  run ids, suite dirs, AUTOTUNE state).
* b12x compile cache persisted across boots in `eugr-b12x/cache/b12x/`
  (started at 30 MB / 879 files after the aborted tuned boot; 37 MB / 1170 files
  after the matrix). This is what makes later boots ~6 min.

## Exact commands used

```bash
cd /home/pgx/Documenti/vllm-gb10/eugr-b12x
AUTOTUNE=0 CFG_ID=eugr-base MTP=4 KV_DTYPE=fp8 GPU_UTIL=0.75 ./run_config.sh
AUTOTUNE=0 CFG_ID=spec6    MTP=6 KV_DTYPE=fp8 GPU_UTIL=0.75 ./run_config.sh
AUTOTUNE=0 CFG_ID=spec7    MTP=7 KV_DTYPE=fp8 GPU_UTIL=0.75 ./run_config.sh
# spec9 intentionally fails; see finding 2 (max supported depth = 7).
./launch.sh --stop
```

`run_config.sh` = boot (`launch.sh --replace --wait`, records boot wall/log/inspect)
→ `validate.sh` (health/models/one completion/metrics) → `bench/serve.py adopt` →
`bench/suite.py --profile quick --concurrency 1,2,4,8 --quality off` → stop.
