# Qwen3.8-Flash-Next on one DGX Spark (GB10)

A measurement-driven serving stack for two checkpoints of Qwen3.8-Flash-Next on a single
NVIDIA DGX Spark (GB10, 128 GB unified memory): the shipped **myllmbox v5.4** configuration,
and a tuned **524k-context single-stream** configuration.

Four commands. Two checkpoints, two configurations, one recipe each.

```bash
./presets/run_az_standard.sh          # AZ checkpoint, shipped best
./presets/run_az_524k.sh              # AZ checkpoint, 524k context, one stream
./presets/run_uncensored_standard.sh  # abliterated checkpoint, shipped best   ⚠ no guardrails
./presets/run_uncensored_524k.sh      # abliterated checkpoint, 524k single    ⚠ no guardrails
```

Everything here was measured on this box with the harness in [`bench/`](bench/README.md) and the
gate in [`quality/`](quality/). The full study — 30+ configurations, controlled A/B/A runs, quality
validation, and the negative results — is in **[BENCHMARK-REPORT.md](BENCHMARK-REPORT.md)**.

---

## Table of contents

- [Why these four and nothing else](#why-these-four-and-nothing-else)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [The two configurations](#the-two-configurations)
- [The two checkpoints](#the-two-checkpoints)
- [How the weights are loaded](#how-the-weights-are-loaded)
- [Repository layout](#repository-layout)
- [Recipes and overrides](#recipes-and-overrides)
- [Measured results](#measured-results)
- [Benchmark harness](#benchmark-harness-bench)
- [Quality gate](#quality-gate-quality)
- [The alternative eugr b12x stack](#the-alternative-eugr-b12x-stack)
- [Troubleshooting](#troubleshooting)
- [Credits and licensing](#credits-and-licensing)

---

## Why these four and nothing else

This repo used to carry ~40 recipes. The study said:

- **The shipped myllmbox v5.4 recipe is already the best configuration measured.** Twenty-plus
  depth / drafter / KV / chat-template variants were tested and none beat it beyond the
  ±10–15 % run-to-run noise of a shared host.
- **K=4 is not faster than K=12.** Controlled sustained-load A/B/A: both 198.5 tok/s at C=8.
  It only halves the draft compute, so it was dropped.
- **The 10-expert draft folder is not a win.** The extra drafter cost cancels the acceptance gain.
  Dropped, and `draft-lab/` with it.
- **Runtime steering does not work on this checkpoint family.** Refusal 82 % → 82 % at λ = ±1 on
  all 96 residual writers. The fix is weights, not hooks: use the abliterated checkpoint.
  `steering/` is gone.
- **fp8 KV buys context, not speed.** −8 % decode. It is one `set:` override in the matrix, not a
  shipped preset.

What survived is the two things you actually use, on the two checkpoints you actually have.
Everything else is a `set:` override away in [`bench/matrix.yaml`](bench/matrix.yaml).

## Requirements

| item | requirement |
|---|---|
| host | NVIDIA DGX Spark (GB10), 128 GB unified memory, aarch64, DGX OS (driver 580.x, CUDA 13) |
| docker | Docker Engine with GPU support (`docker run --gpus all` works) |
| disk | ~250 GB free: image ~35 GB + one checkpoint ~100 GB + compile caches |
| python | 3.10+ on the host, only for `huggingface_hub` (everything else is stdlib) |
| weights | in the Hugging Face cache; `./setup.sh` fetches and verifies them |

The container needs most of the unified pool: ~102 GB of weights + 23 GB KV + runtime buffers.
Close other large memory consumers before measuring.

## Quick start

```bash
git clone --recursive <this repo> && cd vllm-gb10

./setup.sh                 # check the host, pull the image, verify the weights
./tune-host.sh             # sudo once: removes the periodic GB10 memory freezes

./presets/run_az_standard.sh
# first boot ~15 min (the PLE table map is built once); after that ~6-9 min

./check.sh                 # what is actually being served, from the live server
./view.sh                  # follow the engine log
./stop.sh                  # remove the container; weights and caches stay
```

`./setup.sh` is idempotent: it re-checks and only does the missing work. Run it again after a
`git pull` or when something behaves oddly — most first-boot failures are a half-finished download
or a corrupted index, and it checks for both.

## The two configurations

Same image, same kernels, same drafter. They differ only where the goal differs.

| | **standard** | **524k-single** |
|---|---|---|
| purpose | many streams, the shipped best | one agent, the whole box |
| `max-model-len` | 409,600 (YaRN 1.5625) | **524,288** (YaRN 2.0) |
| `max-num-seqs` | 16 | **1** |
| `max-num-batched-tokens` | 8,192 | **32,768** |
| `gpu-memory-utilization` | 0.88 | **0.90** |
| `block-size` | 3,456 | **860** |
| `prefix-match-unit` | 64 | **32** |
| `cudagraph_capture_sizes` | to 96 | to 96 **+17, 112, 128** |
| `pass_config` fusions | default | **norm/act/attn quant + rope_kvcache_cat_mla** |
| speculative depth | K=12, dynamic ladder | **K=16**, ceiling pinned `[1 -> 16]` |
| KV pool | 750,933 tokens (1.83× a 400k request) | 750,933 tokens (**1.43× a 524k request**) |

The 23 GiB bf16 KV pool is what makes the 524k configuration fit at all. Do not lower
`kv-cache-memory` there.

Both configurations exist for both checkpoints, which is the point: `standard` is the reference
measured preset, `524k-single` is the tuned one.

## The two checkpoints

| | |
|---|---|
| **`azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ`** | NVIDIA's FP4-Local re-routed at **5 of 512** experts per token, with a widened re-trained shared expert (1280). ~102 GB, native 262,144 context. The default. |
| **`myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored`** | OrcaRouter's abliterated body — the refusal direction removed in the weights (`W' = W − r(rᵀW)`, λ = 1, every residual writer) with myllmbox's NVFP4 output head. **Gated**: accept the licence on its HF page, then `hf auth login`. |

Both are the same Qwen4Exp hybrid: 48 layers, gated-delta-net linear attention + full attention,
hyper-connections, 512 routed experts, NVFP4 experts with MXFP8 dense layers, one MTP layer, and a
PLE n-gram table read from NVMe.

Measured side by side (full detail in the report):

| | AZ | uncensored |
|---|---|---|
| refusal rate, 50 held-out harmful prompts | 78–82 % | **0 %** |
| quality gate (quick) | 0.9667 | 0.9667 |
| decode, single stream | 53.3 tok/s | 52.8 tok/s |
| C=8 | 142.0 tok/s | 134.5 tok/s |

The uncensored preset needs one extra env knob, `MBX_QSA_UNFUSE_INDEXER=1`, which is set in its
recipe. This checkpoint stores `q/k/v` under mixed schemes (int8 on 12 of 48 layers, bf16 on the
rest); the image unfuses the indexer projection natively with that flag and keeps `qkv_proj`
fused. **Do not** replace it with the old `UNFUSE_QKV` model patch — dropping the whole fusion was
the repetition-loop cause, and that patch is deleted.

## How the weights are loaded

One decision worth stating, because it is the difference between "works" and "loads then dies at
CUDA-graph capture":

> The model is served from the **snapshot's absolute path inside the HF hub**, with the hub
> bind-mounted read-only at its own host path. There is no `models/` folder and no symlink farm.

A Hugging Face snapshot is a directory of **relative** symlinks into `../../blobs/<sha256>`.
Bind-mounting the snapshot at `/models/<name>` makes those links resolve to `/blobs/<sha256>`
inside the container, which does not exist. The AZ checkpoint happens to survive this because its
weights are plain files under `tensors/`; the uncensored one does not, because its weights *are*
blob symlinks. Serving from the absolute hub path makes both work, and it was verified by reading
one shard of each from inside a container.

Consequences:
- nothing is ever written into the HF cache by this repo;
- `DOWNLOAD_MODE=local` still exists (it mounts `$MODELS_DIR/<name>`), but the default is the cache;
- if a snapshot looks corrupted, `./setup.sh` re-verifies it shard by shard against the index.

## Repository layout

```
run.sh                       the launcher. Reads one recipe, launches, waits for /health.
config.env                   only what run.sh actually reads (image, port, container, HF paths)
setup.sh                     idempotent bootstrap + verification: host, image, weights, licence gate
check.sh                     what is actually being served, read from the live server and container
view.sh · stop.sh · tune-host.sh

recipes/
  az-standard.yaml                 the shipped myllmbox v5.4 recipe — the measured best
  az-524k-single.yaml              524k context, one stream
  uncensored-standard.yaml         abliterated checkpoint, shipped settings
  uncensored-524k-single.yaml      abliterated checkpoint, 524k single stream
presets/
  run_az_standard.sh  run_az_524k.sh
  run_uncensored_standard.sh  run_uncensored_524k.sh
  run_az_b12x.sh                 the alternative stack

bench/                       the reproducible benchmark harness (stdlib only)
  serve.py                 recipe -> docker manifest, health wait, boot time, meta.json
  suite.py                 the measurement suite: decode / concurrency / prefill / needle / spec
  run_matrix.py            sequential, resumable matrix runner
  aggregate.py             results -> RESULTS.md + summary.csv + ranking
  check_recipes.py         static preflight, no GPU
  matrix.yaml              the four shipped configs, ready to measure
quality/                     deterministic capability + long-context gate (133 offline tests)
tools/                       mtp_drafter_check.sh, verify_cache.py
eugr-b12x/                   the alternative eugr b12x launcher (plain docker run)
BENCHMARK-REPORT.md          the full study: methodology, results, constraints, recommendations
```

## Recipes and overrides

A recipe has three sections, and it is the only place the model is described:

```yaml
server:      # image, docker_args, cpuset, model, port, cache_dir
env:         # container environment (MBX_*, VLLM_*, allocator)
vllm:        # name: value -> --name value | name: true -> --name
```

`run.sh` adds nothing to the engine command line beyond the mounts, the port and the model path.
There is deliberately no `MTP=`, `CTX=` or `KV_BYTES=` knob: every recipe carries exactly **one**
`--speculative-config`, and the old habit of appending a second one from the environment was the
single biggest footgun here (a duplicate at depth ≤ 3 makes the image refuse to boot).

Overrides come from the environment, and they win over the recipe:

```bash
PORT=8010 ./presets/run_az_standard.sh
SERVED_NAME=my-model ./presets/run_az_524k.sh
IMAGE=myllmbox/qwen38-flash-next-vllm:v5.4 ./presets/run_az_standard.sh
CHAT_TEMPLATE=/path/to/template.jinja ./presets/run_az_standard.sh   # empty = the stock template
./run.sh recipes/az-standard.yaml --dry-run                            # print the argv, change nothing
```

## Measured results

Methodology, in one paragraph: each configuration boots its own container; the harness records the
image digest, the exact argv/env/mounts, boot seconds, the full server log, per-request JSONL, the
Prometheus spec-decode counters, and a **client-vs-server token cross-check** (ratio 1.000 = no
external traffic). Decode numbers are medians at temperature 0, thinking off. The decisive
comparisons were interleaved A/B/A runs, because single-run rankings swing ±10–15 % on this box.

**`az-standard`, full profile:**

| metric | value |
|---|---|
| decode (median) | **53.3 tok/s** — code 54.6 · JSON 59.9 · math 51.6 · QA 41.4 |
| TTFT / TPOT p50 | 181 ms / 18.8 ms |
| C=1 / 2 / 4 / 8 / 16 | 49.0 / 67.3 / 103.6 / 142.0 / 198.2 tok/s |
| sustained C=8 / C=16 (60 s) | **198.5** / 292.4 tok/s |
| prefill 1k / 8k / 32k | 1108 / 2341 / 2260 tok/s |
| 96k needle | TTFT 42.5 s, prefill 2266 tok/s, recall OK |
| spec decode | 17.0 % acceptance, 3.04 tokens/step at depth 12 |
| quality gate | 0.9667 quick · 3/3 strict needle at 32k / 128k / 262k |
| boot | 356 s warm |

**Content matters more than any knob.** Single-stream throughput tracks draft acceptance, which
tracks content: code and JSON accept 5–6 tokens/step, prose ~2.8–3.2. Same preset, same box:

| content | decode | acceptance | tokens/step |
|---|---|---|---|
| pure Python code, 2k-token answer | **70.1** | 38 % | 5.6 |
| TypeScript, 1–1.5k tokens | 65.0 | 30 % | 4.6–5.2 |
| code + prose explanation | 45.0 | 16 % | 2.9 |
| thinking-on, reasoning phase | 51.6 | 21 % | 3.5 |

That is the whole reconciliation with the 80+ tok/s published elsewhere: those are code-only
answers on a quiet box.

**Negative results, kept on purpose** — each of these costs you throughput:

| change | effect |
|---|---|
| `MBX_DRAFT_HEAD=0` | **−19 % decode** at unchanged acceptance |
| non-stock chat template | −8 % decode, no acceptance gain |
| fp8 KV | −8 % decode, buys ~1.94× the KV pool |
| 10-expert draft folder | +6 % decode at K=6, −4 % at K=4 — the drafter cost cancels it |
| `MBX_MTP_DRAFT_SCALE=2` | no-op at temperature 0 |
| depth 5 / 6 / 9 | 50.7 / 55.7 / 50.7 — inside noise |

The depth story in one line: **the accepted-token curve dies after position 2 on every drafter
tested**, so more than ~4 drafts per step only adds compute — and depth 4 and depth 12 are
throughput-equal because the fused multi-step drafter makes the extra drafts nearly free.

## Benchmark harness (`bench/`)

```bash
python3 bench/check_recipes.py bench/matrix.yaml          # static preflight, no GPU
python3 bench/run_matrix.py --matrix bench/matrix.yaml --dry-run
nohup python3 bench/run_matrix.py --matrix bench/matrix.yaml > bench/results/matrix.out 2>&1 &
python3 bench/aggregate.py --results-dir bench/results    # RESULTS.md + summary.csv + ranking
```

Single configuration:

```bash
python3 bench/serve.py up --recipe recipes/az-524k-single.yaml --run-id my-run
python3 bench/suite.py --base-url http://127.0.0.1:8000 --run-dir bench/results/my-run \
                       --config-id my-run --profile quick
python3 bench/serve.py down --name bench-az-524k-single
```

A variant is an override, not a copied file:

```yaml
- name: az-standard-fp8kv
  recipe: recipes/az-standard.yaml
  set:
    vllm.kv-cache-dtype: fp8
    vllm.speculative-config: '{"method":"mtp","num_speculative_tokens":7,"rejection_sample_method":"block","draft_sample_method":"probabilistic"}'
```

## Quality gate (`quality/`)

```bash
python3 quality/run_quality.py --mode quick --output /tmp/q.json    # 55 requests, ~80 s
python3 quality/run_quality.py --mode full  --output /tmp/q.json    # + 32k/128k/262k needle
python3 quality/compare.py baseline.json candidate.json --noise 1   # exit 1 on regression
```

| battery | items | measured |
|---|---|---|
| math | 20 | exact numeric answer after normalization |
| JSON | 10 | valid JSON + mini-schema conformance |
| code | 8 | model-written Python executed against 37 hidden assertions |
| instruction | 10 | rule-based format constraints |
| tool calling | 6 | server-parsed `qwen3_xml` calls: name + argument schema |
| repetition | 1 | 2k-token generation: unique ratio, distinct-2, max 20-gram repeat |
| long context | 3 | strict needle recall at 32k / 128k / 262k with real token validation |

Use `--noise 0` for a finalist acceptance gate; `--noise 1` tolerates one known flaky item.

## The alternative eugr b12x stack

```bash
./presets/run_az_b12x.sh                     # MTP=4, fp8 KV, util 0.80, autotune off
MTP=6 ./presets/run_az_b12x.sh
./presets/run_az_b12x.sh --dry-run
./presets/run_az_b12x.sh --stop
```

Quality is identical (0.9667) and host-RAM pressure is lower (no ~31 GB PLE table map), but it is
**~30 % slower at C=8** (99.9 vs 142.0 tok/s) and the b12x GDN kernel caps speculative depth at 7.
Use it when its image fits your workflow, not for speed. Details in
[`eugr-b12x/README.md`](eugr-b12x/README.md).

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `NVFP4 weight_scale ... was never loaded (still NaN)` | the snapshot's index was overwritten (a draft-only index is the usual suspect) | `./setup.sh --weights <repo>` re-verifies every shard against the index |
| `num_speculative_tokens_per_batch_size is required for dynamic speculative decoding` | depth ≤ 3, or `MBX_DEPTH_MODE=off` | depth ≥ 4; static depth must be emulated as `MIN=CEIL=K`, `PROMOTE=101`, `DEMOTE=0` |
| weight load fails with `No such file or directory` on a `.safetensors` | the snapshot was bind-mounted at `/models/<name>` and its blob symlinks dangle | serve from the absolute hub path — which is what `run.sh` now does |
| `state_index_columns must be at most 8, got 10` | b12x GDN kernel cap | MTP ≤ 7 on the b12x image |
| very slow first token ever | the PLE table map is being built (~31 GB, once) or compile caches are cold | wait; later boots are 6–9 min |
| periodic multi-second stalls | GB10 host page compaction | `./tune-host.sh` |
| throughput well below the published numbers | shared desktop/agent load (±10–15 %) and content mix | compare like for like; use same-window A/B |
| `✗ container exited` | engine failure | `docker logs qwen38-flash-next`; the error maps to one of the rows above |
| `free -g` shows < 8 GB while serving | 102 GB weights + 23 GB KV in 128 GB unified | expected; close other consumers |

Two open questions worth settling before you trust the tuning:

- **`presence_penalty` in `--override-generation-config`.** Stock vLLM ignores it without a patch.
  `./check.sh --penalty` shows whether the flag reached the engine config; whether it is *applied*
  needs an A/B at temperature 0.6.
- **`MBX_DEPTH_CEIL` has no key for 1–3 requests** in the standard recipe
  (`[4 -> 8, 6 -> 7, 12 -> 6, 14 -> 5]`), while the comment claims 1–3 → 12. Either the default
  below the first key is the configured K, or the ladder is not doing what the comment says.
  `./check.sh --spec` and the spec counters in `bench/results/*/meta.json` will tell you.

## Credits and licensing

- Model: [azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ),
  built from [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local).
  The base Qwen model is under the Qwen Community License, the NVIDIA checkpoint under the NVIDIA
  Open Model Agreement.
- Uncensored checkpoint: [myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored)
  (gated).
- Serving image and recipe design: [myllmbox/qwen38-flash-next-recipe](https://github.com/myllmbox/qwen38-flash-next-recipe).
- Alternative stack: [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker).
- The INT4-AutoRound lineage, the `setup.sh`/`check.sh` discipline and the drafter-health check:
  [azampatti/Qwen3.8-Flash-Next-Int4-FAST](https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST).

This repository ships code, recipes and measured documentation. No model weights are included.
