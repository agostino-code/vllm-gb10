# eugr B12X stack — azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ (DGX Spark / GB10)

Owner: teammate `eugr-b12x` (shared task `task-2`). This directory is the eugr-side
counterpart of the myllmbox v5.4 benchmark: it launches the **same checkpoint**
(`azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ`, 5-of-512 experts, 102.4 GB, 21 shards)
on the **eugr b12x image** (`eugr/spark-vllm-b12x:latest`) so the two stacks can be
compared with identical prompts/tasks on port 8000.

Everything here is plain `docker run` / plain YAML: no eugr submodule files were
modified, nothing is written outside this workspace, and the model weights are
mounted **read-only** from the existing HF cache (no re-download).

---

## TL;DR

```bash
cd /home/pgx/Documenti/vllm-gb10/eugr-b12x

./launch.sh --dry-run     # show the exact docker argv, change nothing
./launch.sh --wait        # start + wait for /health, save boot log + inspect
./validate.sh             # (once healthy) health/models/completion/metrics -> runs/
./launch.sh --stop        # remove container qwen38-flash-next-b12x
```

Container: `qwen38-flash-next-b12x`, network host, port `8000`,
served model name `azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ`.
Image: `eugr/spark-vllm-b12x:latest` (vLLM `0.1.dev21830+g8244d1c44.d20261102`,
built 2026-11-02 from `local-inference-lab/vllm` `dev/karmic-kraken`).

## Benchmark matrix (four configs the Lead asked for)

All four are one launcher command each; `recipe-*.yaml` files are the same configs
in eugr `run-recipe.py` form. Only the listed knob changes between them.

| # | config | command | speculative-config |
|---|--------|---------|--------------------|
| a | **base**: spec 4, fp8 KV, block 16, maxlen 262144, seqs 8, gpu-mem 0.75 | `./launch.sh --wait` | `{"method":"mtp","num_speculative_tokens":4}` |
| b | **spec 6**: same, MTP 6 | `MTP=6 ./launch.sh --wait` | `…"num_speculative_tokens":6` |
| c | **spec 9**: same, MTP 9 | `MTP=9 ./launch.sh --wait` | `…"num_speculative_tokens":9` |
| d | **spec 6 + bf16 KV**: `--kv-cache-dtype auto`, gpu-mem 0.80 | `MTP=6 KV_DTYPE=auto GPU_UTIL=0.80 ./launch.sh --wait` | `…"num_speculative_tokens":6` |

Optional one-knob sweeps (cheap, same image):

```bash
MTP=0 ./launch.sh --wait                 # no speculative decoding (baseline)
SEQS=16 ./launch.sh --wait               # concurrency
BLOCK=64 ./launch.sh --wait              # block-size 64
MTP=4 GPU_UTIL=0.80 ./launch.sh --wait   # more KV headroom at fp8 KV
```

Equivalent recipe files (for the eugr `run-recipe.sh` route — see below):

| file | differs from base |
|------|-------------------|
| `recipe-az-b12x.yaml` | base (spec 4, fp8 KV, block 16, seqs 8, gpu-mem 0.75) |
| `recipe-az-b12x-spec0.yaml` | no `--speculative-config` at all |
| `recipe-az-b12x-spec6.yaml` | `num_speculative_tokens: 6` |
| `recipe-az-b12x-spec9.yaml` | `num_speculative_tokens: 9` |
| `recipe-az-b12x-spec6-bf16kv.yaml` | spec 6, `--kv-cache-dtype auto`, gpu-mem 0.80 |

Run one config at a time (one model server per GPU). `launch.sh` refuses to touch
the myllmbox container (`qwen38-flash-next`); it only manages `qwen38-flash-next-b12x`.

## Knob mapping

`launch.sh` environment variables → vLLM flag / docker effect:

| env | default | vLLM flag / effect |
|-----|---------|--------------------|
| `MTP` | 4 | `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'` (0 = omit) |
| `KV_DTYPE` | `fp8` | `--kv-cache-dtype` (`fp8` or `auto` = bf16 KV) |
| `GPU_UTIL` | 0.75 | `--gpu-memory-utilization` |
| `ALLOC_FRAC` | = `GPU_UTIL` | `PYTORCH_ALLOC_CONF=per_process_memory_fraction:N` (keep in sync) |
| `SEQS` | 8 | `--max-num-seqs` |
| `BLOCK` | 16 | `--block-size` |
| `MAXLEN` | 262144 | `--max-model-len` (checkpoint's native window, no YaRN here) |
| `BATCHED` | 4096 | `--max-num-batched-tokens` |
| `AUTOTUNE` | 1 | if `0`: `--kernel-config '{"enable_b12x_autotune": false}'` — skips the first-boot 241-unit b12x kernel search (≈55–65 min on GB10) but keeps mandatory kernel preparation. **All benchmark runs in this study used `AUTOTUNE=0`; keep it identical across configs when comparing.** |
| `PORT` / `NAME` | 8000 / `qwen38-flash-next-b12x` | host port / container name (`--network host`) |
| `MODEL_SNAP` | auto-detected | HF-cache snapshot dir mounted at `/models/Qwen3.8-Flash-Next-FP4-Local-AZ` (ro) |
| `EXTRA_ARGS` | empty | appended verbatim to `vllm serve` |
| `WAIT` | 0 (`--wait` sets 1) | poll `/health`, then save boot log + `docker inspect` under `runs/` |
| `BOOT_TIMEOUT` | 2400 s | health wait limit |

Fixed command content (same as eugr's `qwen3.8-flash-next-nvfp4-solo` recipe):
`--quantization modelopt_mixed`, `--load-format b12x`, `--moe-backend b12x`,
`--linear-backend b12x`, `--gdn-decode-kernel b12x`, `--mamba-cache-mode align`,
`--enable-prefix-caching`, `--enable-chunked-prefill`, `--dtype bfloat16`,
`--no-enable-flashinfer-autotune`, `--mm-encoder-tp-mode data`,
`--reasoning-parser qwen3`, `--tool-call-parser qwen3_xml`, `--enable-auto-tool-choice`,
`--compilation-config '{"pass_config":{"fuse_act_quant":true}}'`,
`--served-model-name azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ`.

Environment (`launch.sh`): `VLLM_PLE_TABLE_MEMORY=disk`, `B12X_DISK_BACKEND=io_uring`,
`PYTORCH_ALLOC_CONF`, `CUTE_DSL_ARCH=sm_121a`, `SAFETENSORS_FAST_GPU=1`,
`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_SSM_CONV_STATE_LAYOUT=DS`,
`VLLM_USE_AOT_COMPILE=1`, `VLLM_USE_MEGA_AOT_ARTIFACT=1`, `VLLM_USE_V2_MODEL_RUNNER=1`,
`VLLM_MXFP8_LM_HEAD=1`, `VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13`,
`VLLM_B12X_MOE_FP4_FORCE_A16=1`, `B12X_W4A16_A4_PREFILL=1`, plus
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. `B12X_POLICY_MODE=auto` is passed
for parity with eugr's recipe but is **not referenced anywhere in this image**
(no-op; keep or drop, it changes nothing).

Variant recipe files contain the same flags, one knob changed; e.g. the bf16-KV
file uses `--kv-cache-dtype auto` and `gpu_memory_utilization: 0.80` with
`PYTORCH_ALLOC_CONF=per_process_memory_fraction:0.80`.

## How the model is mounted (and why nothing is downloaded)

`launch.sh` locates the snapshot automatically under
`~/.cache/huggingface/hub/models--azampatti--Qwen3.8-Flash-Next-FP4-Local-AZ/snapshots/`
(newest snapshot having both `config.json` and the real
`model.safetensors.index.json`), validates that `tensors/` holds the 21 shards,
then runs:

```
-v <snapshot>:/models/Qwen3.8-Flash-Next-FP4-Local-AZ:ro
```

and serves the local path `/models/Qwen3.8-Flash-Next-FP4-Local-AZ`. The weight
index maps tensors to `tensors/model-000NN-of-00021.safetensors`, i.e. a
subdirectory; vLLM's loader (and the b12x loader, which resolves `weight_map`
entries relative to the index's parent directory) handles that fine.

Writable caches are workspace-local and mounted at the same paths eugr's
`launch-cluster.sh` uses, so nothing is written to `~/.cache`:
`eugr-b12x/cache/{vllm,flashinfer,b12x,triton,tilelang,torchinductor}` →
`/root/.cache/...`, plus `cache/flashinfer-workspace` → `/cache/flashinfer-workspace`
(`FLASHINFER_WORKSPACE_BASE`). The `cache/` directory is gitignored.

## PLE n-gram table on b12x (`VLLM_PLE_TABLE_MEMORY=disk`)

Resolved by reading the image source (`vllm/models/qwen4_exp/nvidia/b12x_ple.py`,
`flashinfer/experimental/b12x/sequence/_shared/disk_table.py`):

* `VLLM_PLE_TABLE_MEMORY=disk` → b12x storage mode **`io_uring`**
  (`_resolve_ple_table_memory`); `B12X_DISK_BACKEND` accepts `io_uring` (default)
  or `gds`.
* In this mode the n-gram table is **not materialised as a file**. The loader
  registers each `ngram_embedding.shard_N.*` tensor as a *source* (path + offset)
  via `DiskTable.add_shard()`, and at serving time it performs bounded io_uring
  row reads straight from the checkpoint shards. No table map, no temp file, no
  writes to the model directory → the read-only mount is safe, and there is
  **no ~59 GB `ple-nvme` map** like myllmbox's `MBX_PLE_MMAP` path.
* Consequence: the checkpoint shards must stay readable for the life of the
  server (they are: the snapshot is mounted ro), and the container needs
  io_uring + O_DIRECT. Host `/` is ext4 (O_DIRECT ok); the launcher uses
  `--privileged` like eugr's launcher so seccomp does not block io_uring.
  If boot fails with io_uring/EINVAL errors, fall back to
  `B12X_DISK_BACKEND=gds` (needs GDS-capable driver path) or
  `EXTRA_ARGS='--additional-config {"ple_table_memory":"ram"}'` as documented
  in `eugr-stack/recipes/README.md`.

## eugr `run-recipe.sh` route (optional, for eugr tooling users)

The recipe files use the **local image tag** (`container: eugr/spark-vllm-b12x:latest`)
because eugr's stock recipes say `vllm-node-b12x`, which does not exist on this host.
`run-recipe.py` uses `container:` verbatim as the image tag.

```bash
cd /home/pgx/Documenti/vllm-gb10/eugr-stack
SNAP=/home/pgx/.cache/huggingface/hub/models--azampatti--Qwen3.8-Flash-Next-FP4-Local-AZ/snapshots/5d927eabd4e4766a69bf12f7a1387b01d936a78f

# dry-run (no GPU, no state change)
./run-recipe.sh ../eugr-b12x/recipe-az-b12x.yaml --dry-run --solo \
  -v "$SNAP:/models/Qwen3.8-Flash-Next-FP4-Local-AZ:ro"

# real launch (only after the Lead frees the GPU)
./run-recipe.sh ../eugr-b12x/recipe-az-b12x.yaml --solo \
  -v "$SNAP:/models/Qwen3.8-Flash-Next-FP4-Local-AZ:ro"
```

Caveat: `run-recipe.sh` → `launch-cluster.sh` always adds
`-v ~/.cache/huggingface:/root/.cache/huggingface` (rw) and cache mounts from
`~/.cache`. The model is served from `/models` and `HF_HUB_OFFLINE=1` is set, so
vLLM does not write to the HF cache, but if a strict "no writes under ~/.cache"
guarantee is needed, use `launch.sh` instead. `launch.sh` is the reproducible,
self-contained path.

## Differences vs the myllmbox v5.4 stack (key finding)

Same checkpoint, same model code family (both vLLM forks carry `qwen4_exp`
incl. `b12x_ple`/`ngram_embedding`/`mtp`), but the stacks expose different knobs.

| | myllmbox v5.4 (`myllmbox/qwen38-flash-next-vllm:v5.4`) | eugr b12x (`eugr/spark-vllm-b12x:latest`) |
|---|---|---|
| weight loader | `--load-format mbx` | `--load-format b12x` (O_DIRECT read ring / GDS) |
| MoE backend | `--moe-backend marlin` | `--moe-backend b12x` |
| linear backend | `--kernel-config '{"linear_backend_per_quant":{"mxfp8":"b12x"}}'` | `--linear-backend b12x` |
| PLE table | `MBX_PLE_MMAP=1`, `MBX_PLE_MMAP_MODE=nvme` → 31 GB map per boot under `./cache/ple-nvme` | `VLLM_PLE_TABLE_MEMORY=disk` + `B12X_DISK_BACKEND=io_uring` → reads the shards directly, no map file |
| speculative depth | **dynamic**: `MBX_DEPTH_MODE=dynamic`, `MBX_DEPTH_MIN/WINDOW/PROMOTE/DEMOTE/CEIL`, `MBX_DEPTH_SYNCFREE`, `MBX_DRAFT_HEAD` | **fixed** `num_speculative_tokens` only; no `MBX_*` envs exist in this image |
| KV sizing | `--kv-cache-memory <bytes>` (explicit) | only `--gpu-memory-utilization` (derive KV from the budget) |
| GDN state | `--use-replayssm`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | not available (`--use-replayssm` is myllmbox-only) |
| sampling / parsing | `--override-generation-config`, `qwen3_coder` tool parser | no override; `qwen3_xml` tool parser, `qwen3` reasoning |
| default budget in our runs | gpu-mem 0.88 + 23 GB KV | gpu-mem 0.75 / 0.80, fp8 or bf16 KV |

So the comparison is **fixed-depth MTP vs myllmbox's dynamic draft depth**, and
**io_uring disk PLE vs myllmbox's 31 GB NVMe mmap** — both worth flagging in the
report as inherent stack differences, not tuning choices.

Untested alternative for adaptive depth in this image: a fixed
`--speculative-config` supports a **batch-size schedule** via
`num_speculative_tokens_per_batch_size` (`[(range_start, range_end, tokens), …]`,
see `vllm/config/speculative.py`). It is the closest eugr-side analogue to
`MBX_DEPTH_CEIL`, but it is not exercised here; keep the matrix at fixed depths.

## Validation evidence

Pre-GPU evidence (no state change):

* `dryrun-launch-base.txt` — exact `docker run` argv for the base config plus the
  spec0/spec6/spec9/bf16-KV/seqs16/block64 knob checks.
* `dryrun-run-recipe-base.txt` — `run-recipe.sh --dry-run --solo` for the base
  recipe (generated vLLM command + launch-cluster invocation).
* `dryrun-run-recipe-variants.txt` — same for all five recipe files.

Real boot evidence is in `runs/` (`<stamp>-boot.log`, `<stamp>-docker-inspect.json`,
`<stamp>-cmd.txt`, `<stamp>-meta.txt`), bench runs in `bench/results/eugr-b12x-*`
(adopted manifests + `summary.json`), and the consolidated table in
`runs/RESULTS.md`.

GPU window 2026-11-12 (all runs `AUTOTUNE=0`, `--kv-cache-dtype fp8` unless noted,
`--block-size 16`, `max-num-batched-tokens 4096`, fp8 KV, gpu-mem 0.75,
`--speculative-config {"method":"mtp","num_speculative_tokens":N}`):

| config | boot wall | decode single-stream | c=1 / c=2 / c=4 / c=8 (total tok/s) | acceptance | tokens/step | KV pool |
|---|---|---|---|---|---|---|
| base (MTP 4) | 483 s | 47.0 tok/s (code 48.5) | 33.2 / 64.7 / 78.2 / 99.9 | 51.1 % | 3.04 | 586,074 tok |
| spec6 (MTP 6) | 363 s | 42.4 tok/s (code 44.9) | 28.8 / 47.5 / 64.2 / 89.9 | 37.8 % | 3.27 | 547,533 tok |
| spec7 (MTP 7) | 363 s | 41.0 tok/s (code 45.2) | 27.2 / 43.6 / 64.5 / 81.5 | 35.4 % | 3.48 | 524,442 tok |
| spec9 (MTP 9) | — | **boot fails** | — | — | — | — |

Headline: fixed-depth **MTP 4 beats MTP 6** on this stack (throughput −10 % at
higher depth even though tokens/step rises 3.04→3.27: the extra draft forwards cost
more than the accepted tokens save). `MTP 9` cannot boot at all (see risks).
Against the myllmbox v5.4 M0 reference on the same box (code c=1 ≈56 tok/s,
c=8 ≈139 tok/s, acceptance 17.2 %, tokens/step ≈3.0): eugr b12x is slower on
throughput but gets ~3× the acceptance at the same tokens/step — i.e. myllmbox's
shipped K=12 dynamic depth is over-drafting, while eugr's ceiling is fixed depth 4
(and cannot exceed 7).

## Risks / what to watch in the boot log

1. **b12x loader vs this checkpoint** — the checkpoint is NVIDIA's ModelOpt
   `MIXED_PRECISION` format with a granular `quantization_config.ignore` list and
   `quantized_layers`; this image's `ModelOptMixedPrecisionConfig` reads exactly
   that, and the sibling `nvidia/Qwen3.8-Flash-Next-FP4-Local` runs on the same
   image, so compatibility is expected. The only repack difference relevant to
   loading is shard 15 (shared expert 640→1280) and `config.json` routing width;
   both are config-driven. Watch for `KeyError`/shape errors on
   `shared_expert.*` or `mtp.*` in the boot log.
2. **O_DIRECT / io_uring** — served from `/` (ext4), privileged container. If
   PLE reads fail, the boot log says so before `/health`; fallbacks in the PLE
   section above.
3. **Memory** — 102.4 GB checkpoint on a 128 GB unified box. PLE disk mode keeps
   the n-gram rows off the GPU, which is why `gpu-memory-utilization 0.75`
   (96 GB) leaves room for host-side weight prep. `PYTORCH_ALLOC_CONF` must match
   the util budget; the launcher keeps them in sync (`ALLOC_FRAC`).
4. **Boot time** — first boot compiles b12x kernels into
   `eugr-b12x/cache/b12x/compile` (empty by design; boot wall time is recorded in
   `runs/<stamp>-meta.txt`). Later boots reuse that cache.
5. **Speculative-depth hard cap.** `num_speculative_tokens >= 9` fails at engine
   start with `ValueError: state_index_columns must be at most 8, got 10` raised by
   the flashinfer b12x **GDN decode** caps (`sequence/gdn_decode/_impl.py`) via
   `qwen_gdn_linear_attn._make_b12x_gdn_caps`. The columns count is
   `num_speculative_tokens + 1`, so this image supports at most **7** speculative
   tokens for this hybrid GDN model. Evidence: `runs/20261112T140838Z-boot.log`.
   There is no env/flag workaround in this build; the alternatives are fixed 4
   (fastest measured) or a different stack (myllmbox's MBX depth path).
6. **KV capacity is under-allocated by the eugr recipe for this box.** At
   gpu-mem 0.75 / fp8 KV the server advertises 586,074 KV tokens (max concurrency
   2.24× at max_model_len 262144); myllmbox uses 23 GB bf16 KV = 750,933 tokens.
   If long-context seats matter, raise `GPU_UTIL` (0.80 → ~0.85) and/or
   `MAXLEN`/KV budget; `--kv-cache-memory` does not exist in this image, so the
   util fraction is the only lever. Keep `ALLOC_FRAC` equal to `GPU_UTIL`.
7. **Do not restore `config.json.azfix1404.bak`** (stale modified copy) and do not
   rewrite `model.safetensors.index.json` (must stay the real 375,487-tensor index).
