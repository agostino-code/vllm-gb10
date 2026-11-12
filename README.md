# vLLM on one DGX Spark (GB10): Qwen3.8-Flash-Next FP4-Local-AZ

Deployment + benchmarking stack for one model on one DGX Spark:

**`azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ`** — NVIDIA Qwen3.8-Flash-Next FP4 checkpoint,
5 of 512 experts per token with a widened (1280) smart expert; ~102 GB; 262,144-token native
context, 409,600 with YaRN.

Two serving stacks are included and measured here:

| stack | image | loader | status |
|---|---|---|---|
| **myllmbox v5.4** (default) | `myllmbox/qwen38-flash-next-vllm:v5.4` | `--load-format mbx`, dynamic draft depth, NVFP4 draft head, PLE table on NVMe | recommended — best measured throughput |
| eugr b12x | `eugr/spark-vllm-b12x:latest` | `--load-format b12x`, fixed-depth MTP, PLE via io_uring | works, quality-equal, ~30 % slower at C=8, much lower host-RAM use |

Everything in this repository was measured on the same box:
**[BENCHMARK-REPORT.md](BENCHMARK-REPORT.md)** (30+ configurations, quality gate, controlled A/B tests).

---

## TL;DR — best preset

**The shipped myllmbox v5.4 recipe is already the best configuration measured.** No tested depth,
drafter, KV or template variant beat it beyond run-to-run noise.

```bash
presets/run_qwen38_best.sh
# equivalent:  MTP=0 ./run.sh recipes/qwen38-flash-solo-optimized.yaml
```

| setting | value |
|---|---|
| image | `myllmbox/qwen38-flash-next-vllm:v5.4` |
| speculative decoding | MTP, K=12 dynamic depth ladder, block verification, probabilistic draft |
| draft head | `MBX_DRAFT_HEAD=1` (NVFP4 copy; disabling it costs 19 % decode) |
| KV cache | bf16, 23 GiB → 750,933 tokens (1.83× for a 400k request) |
| seats / block / context | 16 / 3456 / 409,600 (YaRN) |
| quality | quality suite 0.9667; 128k needle OK; 262k strict needle 3/3 |

Measured on this box (thinking off, temp 0; see report for methodology and caveats):

| metric | shipped preset |
|---|---|
| decode, single stream | ~53 tok/s cross-category median · **65–70 tok/s on code-only full answers** · ~45 prose/mixed |
| throughput C=8 / C=16 | ~157 / ~200 tok/s (suite) · **198.5 tok/s sustained C=8** |
| prefill | ~1100 / ~2300 / ~2300 tok/s at 1k / 8k / 32k |
| boot | ~6 min |

Alternatives and when to use them:

| preset | command | when |
|---|---|---|
| shipped K=12 | `presets/run_qwen38_best.sh` | default |
| K=4 depth | `presets/run_qwen38_k4.sh` | same throughput, **half the draft compute** (not faster) |
| fp8 KV, 1.46M-token pool | `MTP=0 ./run.sh matrix/recipes/az-kvfp8-spec7.yaml` | need >750k tokens of KV (−8 % decode) |
| eugr b12x | `presets/run_qwen38_eugr_b12x.sh` | low host-RAM box / eugr image preferred |

Notes: `MYLLMBOX`'s published 80+ tok/s numbers are code-only full answers on a quiet box; this
repo's benchmark medians are cross-category 256-token completions, so they read lower. The report
reconciles the two (same preset, 70.1 tok/s reproduced here on pure code, remainder is host load).

## Quick start

Requirements: DGX Spark (GB10), Docker with GPU access, ~120 GB free disk, the model in the HF cache.

```bash
git clone --recursive https://github.com/agostino-code/vllm-gb10.git
cd vllm-gb10
./setup.sh                    # pulls/builds the serving image; see config.env for MODEL_REPO etc.

presets/run_qwen38_best.sh    # boot the best preset (waits until the API is healthy)
./check.sh                    # health + active config + test prompt
./view.sh                     # follow logs/stats;  ./stop.sh stops
```

One-command alternatives:

```bash
MTP=0 ./run.sh recipes/qwen38-flash-solo-optimized.yaml   # the shipped recipe directly
MTP=0 ./run.sh recipes/qwen38-k4.yaml                     # the K=4 alternative
```

**Do not set `MTP=3` or any depth ≤ 3**: `run.sh` appends a second `--speculative-config`, and on
the v5.4 image speculative depth ≤ 3 makes the server fail at CUDA-graph capture. `config.env`
defaults `MTP=0` for this reason (recipes carry their own speculative config).

## Repository layout

```
run.sh, config.env, recipes/   universal recipe runner + YAML recipes (server/env/vllm sections)
presets/                       ready-to-run launchers (best, K=4, eugr)
bench/                         reproducible benchmark harness (serve/suite/aggregate/matrix runner)
quality/                       deterministic capability & long-context quality suite
matrix/                        the preset-study matrix, generated recipes, A/B scripts, findings log
draft-lab/                     speculative-draft variants + fixed draft-folder builder
eugr-b12x/                     self-contained eugr b12x stack launcher (docker run, no eugr tooling)
steering/                      runtime steering vectors (orthogonal projection, DS4-style)
tools/                         helper scripts (draft folders, MTP checks, simple benchmark)
docs/                          ADRs and design notes
BENCHMARK-REPORT.md            full study: results, quality gate, boot constraints, recommendations
```

## Benchmarking

`bench/` launches each configuration in its own container, waits for `/health`, records image
digest + exact docker argv/env + boot time, runs a standardized suite (decode, concurrency
1/2/4/8/16, prefill, 128k needle, `/metrics` spec-decode counters, client/server token accounting),
then stops the container. Results are machine-readable and aggregated into ranked tables.

```bash
python3 bench/check_recipes.py matrix/matrix.yaml      # static preflight
python3 bench/run_matrix.py --matrix matrix/matrix.yaml --dry-run
python3 bench/run_matrix.py --matrix matrix/matrix.yaml  # long: run in background
python3 bench/aggregate.py --results-dir bench/results   # -> RESULTS.md + summary.csv
```

Controlled comparisons (the runs that settle noise questions) are in
`matrix/ab.sh` (suite A/B/A) and `matrix/ab_load.sh` (sustained-load A/B/A);
`matrix/finalist_phase.sh` does the full-profile + full-quality + temp-0.6 + soak validation.
Published ranked output: [bench/results/RESULTS.md](bench/results/RESULTS.md).

## Quality suite

```bash
python3 quality/run_quality.py --mode quick --output /tmp/q.json   # 55 requests, ~80 s
python3 quality/run_quality.py --mode full  --output /tmp/q.json   # + 32k/128k/262k needle
python3 quality/compare.py baseline.json candidate.json --noise 1  # acceptance gate
```

Batteries: math, JSON schema, code (hidden unit tests, executed), instruction following,
tool calling, repetition/degeneracy, long-context needle. All graders are deterministic and
run offline (133 unit tests).

## eugr b12x stack

```bash
presets/run_qwen38_eugr_b12x.sh                 # base: MTP=4, fp8 KV, util 0.75, AUTOTUNE=0
MTP=6 ./presets/run_qwen38_eugr_b12x.sh         # fixed depth 6
./presets/run_qwen38_eugr_b12x.sh --stop
```

Caveats measured here: fixed MTP=4 is its best point; depth ≥ 8 fails
(`state_index_columns must be at most 8`), so 7 is the maximum; b12x autotune must stay off for
practical boot times (~55–65 min tuned first boot); raise `--gpu-memory-utilization` to 0.80–0.85
for a larger KV pool. Quality is identical to myllmbox (0 item flips). Details:
[eugr-b12x/README.md](eugr-b12x/README.md).

## Known constraints (v5.4 image)

- Speculative depth **must be ≥ 4**; `MBX_DEPTH_MODE=off` does not boot either. Static depth is
  emulated on the dynamic path (`MIN=CEIL=K`, `PROMOTE=101`, `DEMOTE=0`).
- `run.sh` + `config.env MTP != 0` appends a duplicate speculative config — keep `MTP=0`.
- The model snapshot must carry the **real 375,487-tensor `model.safetensors.index.json`**
  (sha256 `60e0e2d8…`). A draft-folder builder once clobbered it with a 4,638-tensor draft index,
  which breaks every boot with `NVFP4 weight_scale ... never loaded`; `tools/make_draft_dir.py`
  now refuses to write into the HF cache. See `bench/refs/README.md` for the repair procedure.
- The NVFP4 draft head must stay on (`MBX_DRAFT_HEAD=1`).
- Absolute throughput on a shared desktop/agent host can be 10–25 % below a quiet box; the study
  therefore relies on same-window A/B runs and sustained-load averages.

## Steering (runtime, optional)

Generate vectors with `steering/generate_vector.py` and apply them via environment variables:

```bash
export STEERING_VECTOR_PATH="/workspace/steering/your_vector.pt"
export STEERING_STRENGTH="1.0"
MTP=0 ./run.sh recipes/qwen38-flash-solo-optimized.yaml
```

The projection is orthogonal (DS4-style) so the residual stream norm is preserved; see
`docs/ADR/001-steering-projection.md`.

## Host tuning

`./tune-host.sh` applies DGX Spark host settings (`vm.compaction_proactiveness=0`) that the
benchmark numbers rely on. `check.sh` diagnoses a running server; `bench.sh` is the original
simple multi-stream benchmark.

## Credits

Model: [azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ)
(built on NVIDIA's Qwen3.8-Flash-Next FP4-Local). Serving image and recipe design:
[myllmbox](https://github.com/myllmbox/qwen38-flash-next-recipe). Alternative stack:
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker).
