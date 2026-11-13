# vLLM on one DGX Spark (GB10): Qwen3.8-Flash-Next FP4-Local-AZ

A complete, **measurement-driven** serving stack for one model on one NVIDIA DGX Spark (GB10):
the best preset we could find, a reproducible benchmark harness, a capability/quality gate, an
alternative eugr b12x stack, and a runtime-steering hook.

The model is **[`azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ)** —
NVIDIA's Qwen3.8-Flash-Next FP4 checkpoint re-routed at **5 of 512 experts per token**, with a
widened/re-trained shared expert (1280), ~102 GB on disk. Native 262,144-token context, 409,600
with YaRN. It runs on one 128 GB unified-memory Spark box.

Everything below was measured on the same machine, with the same harness, and the results are
reproducible from this repository. The full study (30+ configurations, controlled A/B tests,
quality gate and limitations) is in **[BENCHMARK-REPORT.md](BENCHMARK-REPORT.md)**.

---

## Table of contents

- [TL;DR — best preset](#tldr--best-preset)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [The model](#the-model)
- [Repository layout](#repository-layout)
- [Configuration reference](#configuration-reference)
- [Presets](#presets)
- [Measured results](#measured-results)
  - [Benchmark methodology](#benchmark-methodology)
  - [Best preset, in detail](#best-preset-in-detail)
  - [Content-dependent throughput and the published numbers](#content-dependent-throughput-and-the-published-numbers)
  - [Alternatives and negative results](#alternatives-and-negative-results)
- [Benchmark harness (`bench/`)](#benchmark-harness-bench)
- [Quality suite (`quality/`)](#quality-suite-quality)
- [Runtime steering (`steering/`)](#runtime-steering-steering)
- [eugr b12x stack (`eugr-b12x/`)](#eugr-b12x-stack-eugr-b12x)
- [Known constraints and troubleshooting](#known-constraints-and-troubleshooting)
- [Operations](#operations)
- [FAQ](#faq)
- [Credits and licensing](#credits-and-licensing)

---

## TL;DR — best preset

**The shipped myllmbox v5.4 recipe is the best configuration we measured.** None of the tested
depth, drafter, KV-cache or chat-template variants beat it beyond run-to-run noise.

```bash
presets/run_qwen38_best.sh
# equivalent:  MTP=0 ./run.sh recipes/qwen38-flash-solo-optimized.yaml
```

| setting | value |
|---|---|
| image | `myllmbox/qwen38-flash-next-vllm:v5.4` (vLLM 0.32 + RecoverSSM + dynamic draft depth + NVFP4 draft head) |
| loader | `--load-format mbx` |
| speculative decoding | MTP, **K=12 dynamic depth ladder**, block verification, probabilistic draft |
| draft head | `MBX_DRAFT_HEAD=1` (NVFP4 copy of the head: **do not disable**, costs ~19 % decode) |
| KV cache | bf16, 23 GiB → **750,933 tokens** (1.83× a 400k request at max concurrency) |
| seats / block / context | 16 / 3456 / 409,600 tokens (YaRN 1.5625) |
| chat template | model stock template |
| quality | quick gate **0.9667**; long-context needle 3/3 strict at 32k/128k/262k |

Expected performance on this class of box (thinking off, temperature 0; see
[measured results](#measured-results) for methodology):

| metric | shipped preset |
|---|---|
| single-stream decode | ~53 tok/s cross-category median · **65–70 tok/s on code-only full answers** · ~45 tok/s prose/mixed |
| throughput at C=8 / C=16 | ~157 / ~200 tok/s (suite) · **198.5 tok/s sustained at C=8** |
| prefill | ~1.1k / ~2.3k / ~2.3k tok/s at 1k / 8k / 32k tokens |
| first token, 1k prompt | ~170–190 ms |
| boot to healthy | ~6–9 min (first boot with a cold PLE map: ~15 min once) |
| KV pool | 750,933 tokens, 242 blocks, 1.83× at 400k |

**Do not** set `MTP=3` (or any speculative depth ≤ 3) when using `run.sh`: it appends a second
`--speculative-config`, and depth ≤ 3 makes the v5.4 image fail at CUDA-graph capture. `config.env`
defaults `MTP=0` for exactly this reason (recipes carry their own speculative config).

---

## Requirements

| item | requirement |
|---|---|
| GPU / host | NVIDIA DGX Spark (GB10), 128 GB unified memory, aarch64, DGX OS (driver 580.x, CUDA 13) |
| Docker | Docker Engine with GPU support (`docker run --gpus all` works) |
| disk | ~250 GB free: serving image ~35 GB + model ~102 GB (+ ~31 GB PLE table map and compile caches) |
| model | `azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ` in the HF cache (or let `run.sh` download it) |
| Python (host) | Python 3.10+ for tools; no host CUDA/PyTorch needed for serving (everything runs in the container) |

The serving container needs the whole unified memory pool: ~102 GB weights + 23 GB KV + runtime
buffers. Close other large memory consumers before benchmarking.

## Quick start

```bash
# 1. clone (eugr-stack is a submodule)
git clone --recursive https://github.com/agostino-code/vllm-gb10.git
cd vllm-gb10

# 2. get the serving image (~35 GB, once)
docker pull myllmbox/qwen38-flash-next-vllm:v5.4

# 3a. model already in the HF cache? just launch:
presets/run_qwen38_best.sh

# 3b. otherwise run.sh downloads it (~102 GB) into the HF cache on first launch:
MTP=0 ./run.sh recipes/qwen38-flash-solo-optimized.yaml

# 4. verify / monitor / stop
./check.sh        # health, served model, context length, experts-per-token, test prompt
./view.sh         # follow engine logs and live stats
./stop.sh         # stop and remove the server container
```

Host tuning (recommended, needs sudo once):

```bash
./tune-host.sh    # sets vm.compaction_proactiveness=0 on DGX Spark (removes periodic freezes)
```

Optional Python tooling environment (for host-side helper scripts that import torch, e.g.
`steering/generate_vector.py`): `./setup.sh` creates `.venv` from `requirements.txt`;
`./setup_benchmark_env.sh` creates the legacy `venv/` with the `openai` client.

The OpenAI-compatible API is served on `http://127.0.0.1:8000/v1` (`/health`, `/v1/models`,
`/v1/chat/completions`, `/metrics`).

## The model

| property | value |
|---|---|
| base | NVIDIA `Qwen3.8-Flash-Next-FP4-Local` (quantization-aware-distilled FP4) |
| architecture | Qwen4Exp hybrid: 48 layers, gated-delta-net linear-attention + full attention, hyper-connections (HC=4), 512 routed experts, top-**5**, shared expert **1280** (MXFP8), NVFP4 routed experts |
| hidden size | 2560 |
| context | 262,144 native; **409,600** via YaRN (`hf_overrides` in the recipe) |
| MTP layer | NVIDIA's 10-expert-trained MTP head, served at the config's 5 experts unless a draft folder is supplied; its shared expert is zero-padded to 1280 |
| size | ~102 GB, 21 shards under `tensors/`, 375,487 tensors in the index |
| extra | PLE n-gram table read from NVMe, multi-modal (vision) tower included |

Model-card facts and the "draft at 10 experts" option are documented in the HF model card; this
repo contains a rebuilt 10-expert draft folder workflow (`draft-lab/`), but measurements showed it
is **not** a serving win (see [negative results](#alternatives-and-negative-results)).

## Repository layout

```
run.sh, config.env            universal single-node runner + settings (recipes carry most knobs)
recipes/                      YAML recipes: qwen38-flash-solo-optimized.yaml (best),
                              qwen38-k4.yaml (alternative), qwen38-best-steering.yaml,
                              plus the upstream recipe collection for other models
presets/                      ready-to-run launchers
  run_qwen38_best.sh          best preset (shipped recipe, safe MTP=0)
  run_qwen38_k4.sh            K=4 alternative (equal throughput, half the draft compute)
  run_qwen38_eugr_b12x.sh     eugr b12x stack
  run_qwen38_best_steering.sh best preset + runtime steering (experimental)
bench/                        reproducible benchmark harness
  serve.py                    recipe -> docker launch manifest, health wait, logs, boot time
  suite.py                    standardized measurement suite (decode/concurrency/prefill/needle/spec)
  aggregate.py                results/*/summary.json -> RESULTS.md + summary.csv + ranking
  run_matrix.py               sequential, resumable matrix runner (boot -> suite -> stop)
  check_recipes.py            static preflight of recipes/matrix (no GPU)
  common.py, prompts.py, recompute_metrics.py, watch_matrix.py, tests/mock_vllm.py
quality/                      deterministic capability & long-context quality gate
  run_quality.py, compare.py, qlib/, data/, tests/ (133 offline tests)
matrix/                       the preset study: matrix definitions, generated recipe variants,
                              A/B scripts (ab.sh, ab_load.sh), finalist_phase.sh, soak.py,
                              codeprobe.py, PLAN.md, FINDINGS.md, templates/
draft-lab/                    draft/MTP experiments, rebuilt draft folders (k10/k5),
                              fixed builder tools/make_draft_dir.py, MEASURED.md
eugr-b12x/                    self-contained eugr b12x launcher (plain docker run)
steering/                     runtime steering: vector builder, model patch, eval, VALIDATION.md
docs/ADR/                     design records (orthogonal-projection steering)
tools/                        helper scripts (draft dirs, MTP drafter check, simple benchmark)
examples/                     prompt sets: benchmarks, long_context, red_teaming, steering_pairs
BENCHMARK-REPORT.md           full study: methodology, results, quality, constraints, recommendations
```

Build/verification scripts: `check.sh`, `view.sh`, `stop.sh`, `tune-host.sh`, `bench.sh`
(original multi-stream benchmark), `validate_steering.sh` (steering compliance pipeline).

## Configuration reference

`run.sh` reads a YAML **recipe** (`recipes/*.yaml`, three sections: `server:`, `env:`, `vllm:`)
and `config.env` for a few top-level overrides. Environment variables win over defaults.

### `config.env` — most useful knobs

| variable | default | meaning |
|---|---|---|
| `MODEL_REPO` | `azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ` | HF repo to serve |
| `SERVED_NAME` | same as repo | `model` name clients send |
| `IMAGE` | `myllmbox/qwen38-flash-next-vllm:v5.4` | serving image |
| `PORT` | `8000` | host API port |
| `CTX` | `409600` | max tokens per request (prompt+output) |
| `SEQS` | `4` | informational; the recipe sets `--max-num-seqs` (16) |
| `KV_BYTES` | `16g` | informational; the recipe pins `--kv-cache-memory` (23 GiB) |
| `MTP` | **`0`** | if non-zero, `run.sh` appends a **second** `--speculative-config`; keep `0` |
| `CHAT_TEMPLATE` | `medium` | `model` (stock), `medium`/`xhigh` if a template file exists, or a path |
| `HF_HOME` | `~/.cache/huggingface` | HF cache root (mounted read-only) |
| `MODELS_DIR` | `./models` | local model folder (mounted at `/models`) |
| `DOCKER_EXTRA_ARGS` | empty | extra `docker run` flags (mounts, `-e`, …) |
| `DOWNLOAD_MODE` | `cache` | `cache` = HF snapshot, `local` = plain folder |

### Recipe format (excerpt from `recipes/qwen38-flash-solo-optimized.yaml`)

```yaml
server:
  image: myllmbox/qwen38-flash-next-vllm:v5.4
  docker_args: --security-opt seccomp=unconfined
  cpuset: "5-9,15-19"          # the 10 Cortex-X925 cores
  model: azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ
  port: 8000
  models_dir: ./models
  cache_dir: ./cache

env:
  MBX_PLE_MMAP: "1"           # n-gram table off the GPU, read from NVMe
  MBX_DRAFT_HEAD: "1"         # NVFP4 draft head (do not disable)
  MBX_DEPTH_MODE: dynamic     # dynamic draft depth
  ...

vllm:                         # name: value -> --name value | name: true -> --name
  served-model-name: azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ
  load-format: mbx
  kv-cache-memory: "23000000000"
  speculative-config: '{"method":"mtp","num_speculative_tokens":12,...}'
  ...
```

Notes:
- `recipes/` also contains the upstream recipe collection for other models — this repo's focus is
  the AZ checkpoint (`qwen38-*.yaml`).
- `run.sh` mounts `./models` at `/models`, `./cache` at `/cache`, and the HF hub read-only at its
  host path; it waits for `/health` (up to 40 min) and streams engine logs.
- `MTP=0` is the safe setting: every recipe already carries exactly one `--speculative-config`.

## Presets

| preset | command | notes |
|---|---|---|
| **best (default)** | `presets/run_qwen38_best.sh` | shipped myllmbox v5.4 recipe; best measured configuration |
| K=4 depth | `presets/run_qwen38_k4.sh` | throughput-equal alternative, halved draft compute; not faster |
| long-context KV | `MTP=0 ./run.sh matrix/recipes/az-kvfp8-spec7.yaml` | fp8 KV, 1,458,424-token pool (~1.94×), −8 % decode |
| eugr b12x | `presets/run_qwen38_eugr_b12x.sh` | alternative image/loader, quality-equal, ~30 % slower at C=8, low host-RAM use |
| best + steering | `presets/run_qwen38_best_steering.sh` | experimental runtime steering (see below) |

`STEERING_STRENGTH=<float>` tunes the steering preset; other presets honour `BACKEND`,
`MODEL_REPO`, `CTX`, `KV_BYTES`, `CHAT_TEMPLATE`.

## Measured results

### Benchmark methodology

- **Box**: one DGX Spark GB10, 128 GB unified memory; `vm.compaction_proactiveness=0`; desktop
  session and agent processes present (documented in the report; ±10–15 % run-to-run variance).
- **Each configuration boots its own container**, and the harness records: image digest, exact
  docker argv/env/mounts, boot seconds, full server log, per-request JSONL, Prometheus spec-decode
  counters and a **client-vs-server token cross-check** (ratio 1.000 = no external traffic).
- **Profiles**: `quick` (~2–2.5 min: decode 4 categories × 2 repeats at 192 tokens, C=1/2/4/8,
  prefill, spec counters) and `full` (~4–5.5 min: 3 repeats at 256 tokens, C=16, 128k needle).
- **Decode** numbers are medians at temperature 0, thinking off; TTFT/TPOT are stream percentiles.
- **Spec decode** metrics come from `/metrics` deltas: acceptance rate, tokens/step, draft depth,
  accepted tokens per position.
- **Noise control**: the study's decisive comparisons were back-to-back A/B/A runs
  (`matrix/ab.sh`, `matrix/ab_load.sh`) and sustained-load soaks; single-run rankings were shown to
  swing ±10–15 % on this shared host (the report documents one such reversal).

### Best preset, in detail

**Shipped recipe (K=12 dynamic depth)** — clean run, full profile:

| metric | value |
|---|---|
| decode (median) | **53.3 tok/s** (code 54.6 · JSON 59.9 · math 51.6 · QA 41.4) |
| TTFT / TPOT p50 | 181 ms / 18.8 ms |
| C=1 / C=2 / C=4 / C=8 / C=16 | 49.0 / 67.3 / 103.6 / 142.0 / 198.2 tok/s |
| sustained C=8 (60 s soak) | **198.5 tok/s** (repeat: 190.9) |
| sustained C=16 (60 s soak) | 292.4 tok/s (repeat: 287.1) |
| prefill 1k / 8k / 32k | 1108 / 2341 / 2260 tok/s |
| 96k-token needle | TTFT 42.5 s, prefill 2266 tok/s, recall OK |
| speculative | acceptance 17.0 %, 3.04 tokens/step, depth 12 |
| KV pool | 750,933 tokens, 1.83× |
| quality (quick) | 0.9667 (math 20/20, JSON 10/10, code 8/8, tool 6/6, instruction 8/10 — two genuine model misses, repetition non-degenerate) |
| boot | 356 s |

Full quality profile on the K=4 variant (same stack, equal throughput): **0.9714**, including
**3/3 strict needle recall at 32k / 128k / 262k tokens** (validated prompt token counts; TTFT
12.4 / 51.7 / 105.0 s).

### Content-dependent throughput and the published numbers

Single-stream throughput depends strongly on **draft acceptance**, which depends on content:
code/JSON accept 5–6 tokens/step, prose accepts ~2.8–3.2. That is why myllmbox's published table
lists code at 86.9 tok/s and mixed at 60.5. Measured here with the same shipped preset:

| content (single stream, natural stop, temp 0) | decode tok/s | acceptance | tokens/step |
|---|---|---|---|
| pure Python code, 2k-token answer | **70.1** | 38 % | 5.6 |
| TypeScript code, ~1–1.5k tokens | 65.0 | 30 % | 4.6–5.2 |
| code + prose explanation | 45.0 | 16 % | 2.9 |
| thinking-on (reasoning phase) | 51.6 | 21 % | 3.5 |

The remaining 10–25 % vs the published figures is host environment (quiet box vs shared desktop/
agent load), not configuration; the report reconciles this in §2b.

### Alternatives and negative results

| config | measured vs shipped | verdict |
|---|---|---|
| **K=4 depth** (`recipes/qwen38-k4.yaml`) | decode 53.4–56.7 · sustained C=8 identical 198.5 · C=16 300.7 · same quality | throughput-equal; only if you want lower draft compute |
| depth 5 / 6 / 9 | decode 50.7 / 55.7 / 50.7 (noisy) | no gain |
| **10-expert draft folder** (`draft-lab/`) | K=6: +6 % decode, C=8 +15 %; K=4: −4 % decode | not worth it; extra drafter cost cancels the acceptance gain |
| `MBX_DRAFT_HEAD=0` | **−19 % decode** at unchanged acceptance | keep the NVFP4 draft head |
| `MBX_MTP_DRAFT_SCALE=2` | no change (sampling-time knob; no-op at temp 0) | not useful for greedy |
| medium chat template | −8 % decode, no acceptance gain, quality unchanged | rejected |
| fp8 KV, K=7 | −8 % decode, C=8 ≈ parity, KV pool **1,458,424 tokens** | use only for context headroom |
| **eugr b12x** (MTP=4, AUTOTUNE=0) | 47.0 tok/s decode, C=8 99.9, acceptance 51 %, quality 0.9667 | quality-equal but slower; best when host RAM is tight |

Depth story in one line: **the accepted-token curve dies after position 2 on every drafter/stack
tested**, so more than ~4 drafts per step only adds compute; depth 4 and the shipped depth 12 are
throughput-equal because the fused multi-step drafter makes the extra drafts nearly free.

## Benchmark harness (`bench/`)

The harness is self-contained (Python stdlib only; no host `nvidia-smi` dependency) and is what
produced every number above.

```bash
# static preflight of recipes/matrix (parses everything, no GPU)
python3 bench/check_recipes.py matrix/matrix.yaml

# print the exact commands without launching anything
python3 bench/run_matrix.py --matrix matrix/matrix.yaml --dry-run

# run a matrix (sequential, resumable; boot -> suite -> stop per config)
nohup python3 bench/run_matrix.py --matrix matrix/matrix.yaml > bench/results/matrix.out 2>&1 &

# rank everything collected so far
python3 bench/aggregate.py --results-dir bench/results     # RESULTS.md + summary.csv + summary_all.json
```

Single-config usage:

```bash
python3 bench/serve.py up --recipe recipes/qwen38-flash-solo-optimized.yaml --run-id my-run
python3 bench/suite.py --base-url http://127.0.0.1:8000 \
    --run-dir bench/results/my-run --config-id my-run --profile quick
python3 bench/serve.py down --name bench-qwen38-flash-solo-optimized
```

Outputs per run: `meta.json` (image digest, docker argv/env/mounts, boot time, effective spec
config), `requests.jsonl` (per request timings/tokens/errors), `metrics/` snapshots, `server.log`,
`summary.json`, `status.json`. `bench/results/RESULTS.md` and `summary.csv` from the study are
tracked in git; raw per-run logs are not.

Matrix schema (YAML): each config has `name`, `recipe` (any repo path, native or eugr `command:`
format), optional `image`/`port`/`chat_template`, `set:` (recipe overrides, e.g.
`vllm.kv-cache-dtype: fp8`), `env:`, `mounts:`, `suite_profile: quick|full`, `suite:` overrides
(including `quality: quick|full`), `priority`, `enabled`, `notes`. The study matrices are
`matrix/matrix.yaml` (phase A) and `matrix/matrix-phaseB.yaml` (targeted phase B).

Controlled-comparison scripts used to settle noise questions:

```bash
bash matrix/ab.sh          # interleaved suite A/B/A (same conditions)
bash matrix/ab_load.sh     # interleaved sustained-load A/B/A (C=8 and C=16, 60 s each)
bash matrix/finalist_phase.sh   # full suite + full quality + temp-0.6 + soak on the winner
```

## Quality suite (`quality/`)

A deterministic, rule-based capability gate (pure stdlib; 133 offline tests).

```bash
python3 quality/run_quality.py --mode quick --output /tmp/q.json   # 55 requests, ~80 s
python3 quality/run_quality.py --mode full  --output /tmp/q.json   # + 32k/128k/262k needle (~4 min)
python3 quality/compare.py baseline.json candidate.json --noise 1  # acceptance gate (exit 1 on regression)
```

Batteries:

| battery | items | what is measured |
|---|---|---|
| math | 20 | exact numeric answer after normalization |
| JSON | 10 | valid JSON + mini-schema conformance (types, required keys, counts, enums) |
| code | 8 | model-written Python executed against 37 hidden assertions (subprocess, timeout) |
| instruction | 10 | rule-based format constraints (word counts, bullets, no commas, ordering) |
| tool calling | 6 | server-parsed `qwen3_xml` tool calls: name + argument schema |
| repetition | 1 | 2k-token generation: unique ratio, distinct-2, max 20-gram repeat, identical lines |
| long context (full) | 3 | strict needle recall at 32k / 128k / 262k with real `prompt_tokens` validation |

Scoring: per-category score, overall score, JSON + per-item raw outputs. `compare.py` supports
item-level noise tolerance (`--noise N`), prints pass→fail flips with attribution, and exits
non-zero on regression — use `--noise 0` for finalist acceptance.

## Runtime steering (`steering/`)

The stock v5.4 image has **no steering support**. This repo adds a CUDA-graph-safe hook without
rebuilding the image:

```bash
presets/run_qwen38_best_steering.sh                          # best preset + steering, strength 1.0
STEERING_STRENGTH=2.0 presets/run_qwen38_best_steering.sh    # stronger projection
STEERING_STRENGTH=0   presets/run_qwen38_best_steering.sh    # patched stack, steering off
```

What happens under the hood:

1. `steering/make_steering_vector.py` builds `steering/succinct.pt` (2560-dim, unit norm) from the
   model's own output head: mean `lm_head` row of verbose-style markers minus mean row of
   concise-style markers (`steering/succinct.json` lists them).
2. `steering/install_patch.sh` extracts the image's `Qwen4ExpModel` source and injects the
   orthogonal-projection rule from `docs/ADR/001` on the final residual stream:
   `y = y − strength · (y · v) · v`. The preset bind-mounts the patched file over the image copy.
   The op is captured inside the CUDA graphs; the MTP drafter is untouched.
3. The server log then shows `[steering] active on final residual stream: …`.

**Honest validation status:** the hook is real and reproducible, but this head-space style vector
does **not** give clean verbosity control at the strengths tested — at 1.5 the output-length change
is inside noise; at ±8 outputs change but become longer/more padded. Details and limitations:
[`steering/VALIDATION.md`](steering/VALIDATION.md). Treat steering as an experimental hook.
`steering/generate_vector.py` (random/zero/safetensors conversion) and `steering/eval_steering.py`
(measure response length/speed) are also available.

## eugr b12x stack (`eugr-b12x/`)

An alternative serving stack using [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)
b12x kernels, launched with plain `docker run` (no eugr tooling, model mounted read-only).

```bash
presets/run_qwen38_eugr_b12x.sh                    # base: MTP=4, fp8 KV, block 16, 8 seats, util 0.75
MTP=6 ./presets/run_qwen38_eugr_b12x.sh            # fixed depth 6
KV_DTYPE=auto GPU_UTIL=0.80 ./presets/run_qwen38_eugr_b12x.sh
./presets/run_qwen38_eugr_b12x.sh --stop
```

Measured (all with `AUTOTUNE=0`, quality 0.9667 = identical to myllmbox):

| config | boot | decode | C=8 | acceptance | tokens/step | KV tokens |
|---|---|---|---|---|---|---|
| MTP=4 (base) | 483 s | 47.0 tok/s | 99.9 | 51.1 % | 3.04 | 586,074 |
| MTP=6 | 365 s | 42.4 | 89.9 | 37.8 % | 3.27 | 547,533 |
| MTP=7 | 363 s | 41.0 | 81.5 | 35.4 % | 3.48 | 524,442 |
| MTP=9 | boot fails | b12x GDN cap: max 7 speculative tokens | | | | |

Caveats: fixed depth 4 is the best point; depth ≥ 8 fails with
`state_index_columns must be at most 8`; b12x autotune must stay off for practical boot times
(tuned first boot ~55–65 min), so numbers are ~single-digit-% below a fully tuned build; the
recipe under-allocates KV — raise `--gpu-memory-utilization` to 0.80–0.85 for ~700k+ tokens.
See [eugr-b12x/README.md](eugr-b12x/README.md).

## Known constraints and troubleshooting

| symptom | cause | fix |
|---|---|---|
| `NVFP4 weight_scale for layer 'column_parallel_linear' was never loaded (still NaN)` | the HF snapshot's `model.safetensors.index.json` was overwritten by a draft-only index | restore the real index (sha256 `60e0e2d8…`, 375,487 tensors): procedure in [`bench/refs/README.md`](bench/refs/README.md) |
| `num_speculative_tokens_per_batch_size is required for dynamic speculative decoding` at CUDA-graph capture | speculative depth ≤ 3, or `MBX_DEPTH_MODE=off` | use depth ≥ 4; static depth must be emulated as `MIN=CEIL=K`, `PROMOTE=101`, `DEMOTE=0` |
| duplicate `--speculative-config` / same error with depth 3 | `run.sh` appends a second spec config when `MTP != 0` | keep `MTP=0` (default since the study) |
| eugr: `state_index_columns must be at most 8, got 10` | b12x GDN kernel cap | MTP ≤ 7 on the b12x image |
| server boots but very slow first token | PLE table map being built (~31 GB, once) or kernel compile caches cold | wait; the next boots are ~6–9 min |
| periodic multi-second stalls / lower throughput | DGX Spark host compaction (unified memory) | run `./tune-host.sh` (needs sudo) once |
| throughput far below published numbers | shared desktop/agent host load (±10–15 %), and content mix | compare like-for-like (code vs prose); see report §2b |
| `✗ container exited` in `run.sh` | engine failure | `docker logs qwen38-flash-next`; the specific error maps to one of the rows above |
| memory pressure (`free -g` shows <8 GB free while serving) | 102 GB weights + 23 GB KV in 128 GB unified memory | that is expected; close other large consumers, or lower `KV_BYTES`/`kv-cache-memory` for a smaller pool |

Also: the container runs unprivileged but needs `--security-opt seccomp=unconfined` (recipe
`docker_args`) for the image's PLE table reader; `docker run` must pass `--gpus all` (run.sh does).

## Operations

```bash
./check.sh    # is it up, which model/config, quick test completion
./view.sh     # docker logs -f + live stats
./stop.sh     # stop and remove the container (weights/caches stay)
tail -f bench/results/matrix.log          # follow a matrix run
cat bench/results/<config>/status.json    # per-config state (done/failed/timeout/...)
```

Measured-state guidance: one model server at a time (the harness enforces it via port/container
preflight); benchmark on an otherwise-idle box for absolute numbers and use same-window A/B runs
for decisions.

## FAQ

**Why does myllmbox advertise 80+ tok/s while the README shows ~53?**
Those numbers are code-only full answers on a quiet box. This box reproduces ~70 tok/s on pure
code with the same preset; the cross-category benchmark median (53) is dominated by shorter prose
completions where acceptance is ~3 tokens/step instead of ~5.6. See report §2b.

**Is the K=4 variant faster?**
No. Controlled sustained-load A/B/A: both are 198.5 tok/s at C=8; C=16 300.7 vs 287–292. It is
an equal-throughput option with half the draft compute.

**Should I use eugr b12x?**
Only if its image fits your workflow or you need its lower host-memory footprint (no 31 GB PLE
map). With this recipe it is ~30 % slower at C=8; quality is identical.

**Can I run a different model?**
`recipes/` contains the upstream recipe collection and `run.sh` is model-agnostic (set
`MODEL_REPO`); the measured presets, constraints and report in this repo are specific to the AZ
checkpoint and the v5.4 image.

**Can I use the 10-expert MTP drafter from the model card?**
`draft-lab/` has a working rebuild (`tools/make_draft_dir.py`) and recipes, but measurements show
it is not a throughput win on this box; keep the self-draft.

## Credits and licensing

- Model: [azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ),
  built from [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local);
  the base Qwen model is governed by the Qwen Community License, the NVIDIA checkpoint by the
  NVIDIA Open Model Agreement.
- Serving image and recipe design: [myllmbox/qwen38-flash-next-recipe](https://github.com/myllmbox/qwen38-flash-next-recipe).
- Alternative stack: [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (checked
  out as the `eugr-stack/` submodule; its own license applies).
- Steering approach: orthogonal projection per `docs/ADR/001`, following the DS4 research stack.

This repository ships code, recipes and measured documentation; no model weights are included and
no license file is provided for the repo itself — third-party components keep their upstream
licenses.
