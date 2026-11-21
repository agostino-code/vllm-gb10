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
- [Getting the baseline](#getting-the-baseline)
- [The two configurations](#the-two-configurations)
- [The two checkpoints](#the-two-checkpoints)
- [How the weights are loaded](#how-the-weights-are-loaded)
- [Repository layout](#repository-layout)
- [Recipes and overrides](#recipes-and-overrides)
- [Measured results](#measured-results)
- [Open questions and next tests](#open-questions-and-next-tests)
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

## Getting the baseline

One command measures all four configurations and prints a block you can paste back here:

```bash
./bench/baseline.py
```

It boots each config in its own container, runs the suite and the quality gate, tears it down,
moves to the next, and writes `BASELINE.md` at the repo root. Full profile is roughly 40 minutes
for the four; `--profile quick` is about a tenth of that and is enough to spot a regression.

Before you run it:

- **nothing else may be on the GPU.** It refuses to start if a server is running, and tells you
  what to stop. If you want to measure a server you started yourself, without the harness
  touching it: `./bench/baseline.py --attach http://127.0.0.1:8000`.
- **run `./tune-host.sh` once.** The GB10 page compaction is the single biggest source of noise
  on this box.
- **run it twice for anything you intend to decide on.** Single runs swing ±10–15 %; only
  same-window repeats are comparable.

The report is designed to be pasted: summary table, decode by content, the concurrency ladder, the
accepted-tokens-per-position depth curve, prefill, the quality battery, and the exact
configuration used — plus a pointer to the raw `summary.json` per config.

## The two configurations

Same image, same kernels, same drafter. They differ only where the goal differs.

| | **standard** | **524k-single** |
|---|---|---|
| purpose | many streams, the shipped best | one agent, the whole box |
| `max-model-len` | 409,600 (YaRN 1.5625) | **524,288** (YaRN 2.0) |
| `max-num-seqs` | 16 | **1** |
| `max-num-batched-tokens` | 8,192 | **32,768** |
| `gpu-memory-utilization` | 0.88 | **0.90** |
| `block-size` | 3,456 | **3,520** |
| `prefix-match-unit` | 64 | 64 |
| `cudagraph_capture_sizes` | to 96 | to 96 **+17, 112, 128** |
| `pass_config` fusions | default | **norm/act/attn quant + rope_kvcache_cat_mla** |
| speculative depth | K=12, dynamic ladder | **K=16**, ceiling pinned `[1 -> 16]` |
| KV pool | 750,933 tokens (1.83× a 400k request) | **767,013** tokens (**1.46× a 524k request**, measured) |

`block-size` differs by 64 tokens for a hard reason, not a tuning preference: the QSA drafter
ring has capacity **K+4** and the engine refuses to boot unless it divides the block size
(`QSA ring capacity 20 must divide the attention block size 3456`, measured). At K=12 the ring
is 16 and 3,456 is legal; at K=16 the ring is 20, so 3,520 = 20×176 = 64×55. The first cut of
the 524k recipe used 860/32 inherited from the container it replaced; the 2026-11-21 baseline
measured the cost — the engine reserved the same 23 GB and returned 604,947 KV tokens, because
each of the 703 small blocks pays a per-block state charge (visible as group pages `13760` vs
`3520` in the boot log). With 3,520 the same 23 GB yield **767,013 tokens = 1.46×** a full
524,288-token request, confirmed in the boot log of both checkpoints.

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

Measured side by side, standard configuration, 2026-11-21 baseline (refusal study and full
detail in the report):

| | AZ | uncensored |
|---|---|---|
| refusal rate, 50 held-out harmful prompts | 78–82 % | **0 %** |
| quality gate (full) | 0.9714 | 0.9643 (524k variant: **0.9786**, best of all four) |
| decode, single stream | 64.1 tok/s | 63.9 tok/s |
| C=8 | 172.4 tok/s | 140.3 tok/s |

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
  baseline.py              THE ENTRY POINT: boot all four configs, measure, print one pasteable report
  serve.py                 recipe -> docker manifest, health wait, boot time, meta.json
  suite.py                 the measurement suite: decode / concurrency / prefill / needle / spec
  run_matrix.py            sequential, resumable matrix runner (what baseline.py drives)
  aggregate.py             results -> RESULTS.md + summary.csv + ranking
  check_recipes.py         static preflight, no GPU
  matrix.yaml              the four shipped configs, ready to measure
quality/                     deterministic capability + long-context gate (133 offline tests)
tools/                       mtp_drafter_check.sh, verify_cache.py
eugr-b12x/                   the alternative eugr b12x launcher (plain docker run)
eugr-stack/                  git submodule -> github.com/eugr/spark-vllm-docker (keep at upstream tip)
BENCHMARK-REPORT.md          the previous study: methodology, results, constraints, negative results
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

**The current numbers are in [`BASELINE.md`](BASELINE.md)** — one pasteable view of all four
configurations, rebuilt from the raw results with `./bench/baseline.py --report <dirs>`. As
measured 2026-11-21 on this box (image `v5.4`, bc42593a5bbf, full profile + full quality):

| config | decode | C=1 | C=8 | TTFT | KV pool | quality | 524k needle |
|---|---|---|---|---|---|---|---|
| az-standard | 64.1 | 56.6 | 172.4 | 130 ms | 750,933 | 0.9714 | 96k OK |
| az-524k-single | 63.4 | 55.6 | — | 266 ms | 767,013 | 0.9714 | **518k OK** |
| uncensored-standard | 63.9 | 55.9 | 140.3 | 154 ms | 750,933 | 0.9643 | 96k OK |
| uncensored-524k-single | 63.3 | 54.6 | — | 287 ms | 767,013 | **0.9786** | **518k OK** |

The two 524k configs cost ~1 % decode versus standard and give 524,288 real context with a
verified needle at 518k tokens (2,216 tok/s prefill, 234 s to first token). The abliteration
costs nothing measurable — the uncensored 524k config posted the best quality of the four.

Methodology, in one paragraph: each configuration boots its own container; the harness records the
image digest, the exact argv/env/mounts, boot seconds, the full server log, per-request JSONL, the
Prometheus spec-decode counters, and a **client-vs-server token cross-check** (ratio 1.000 = no
external traffic). Decode numbers are medians at temperature 0, thinking off. The decisive
comparisons were interleaved A/B/A runs, because single-run rankings swing ±10–15 % on this box.

**Content matters more than any knob.** Single-stream throughput tracks draft acceptance, which
tracks content: code and JSON accept 5–6 tokens/step, prose ~2.8–3.2. Same preset, same box
(from the earlier study, still the case):

| content | decode | acceptance | tokens/step |
|---|---|---|---|
| pure Python code, 2k-token answer | **70.1** | 38 % | 5.6 |
| TypeScript, 1–1.5k tokens | 65.0 | 30 % | 4.6–5.2 |
| code + prose explanation | 45.0 | 16 % | 2.9 |
| thinking-on, reasoning phase | 51.6 | 21 % | 3.5 |

That is the whole reconciliation with the 80+ tok/s published elsewhere: those are code-only
answers on a quiet box.

**Negative results, kept on purpose** — each of these costs you throughput (full study with
methodology and A/B/A tables: [`BENCHMARK-REPORT.md`](BENCHMARK-REPORT.md)):

| change | effect |
|---|---|
| `MBX_DRAFT_HEAD=0` | **−19 % decode** at unchanged acceptance |
| non-stock chat template | −8 % decode, no acceptance gain |
| fp8 KV | −8 % decode, buys ~1.94× the KV pool |
| 10-expert draft folder | +6 % decode at K=6, −4 % at K=4 — the drafter cost cancels it |
| `MBX_MTP_DRAFT_SCALE=2` | no-op at temperature 0 |
| depth 5 / 6 / 9 | 50.7 / 55.7 / 50.7 — inside noise |

The depth story in one line: **the accepted-token curve dies after position 2 on every drafter
tested** (confirmed again on all four configs in the 2026-11-21 baseline: p0–p2 carry 93–94 % of
accepted tokens, p5 and beyond are zero), so more than ~4 drafts per step only adds compute —
and depth 4 and depth 12 are throughput-equal because the fused multi-step drafter makes the
extra drafts nearly free.

## Open questions and next tests

Ranked by expected value, all expressible as `bench/matrix.yaml` overrides — no new code:

1. **The uncensored C=8 gap (−19 %, 140.3 vs 172.4) is the biggest unexplained number in the
   baseline** and exceeds the ±10–15 % band. Decode at C=1 is identical (63.9 vs 64.1) and
   acceptance is *higher* (18.3 vs 17.4), so it smells like batch-scaling, not drafting:
   `MBX_QSA_UNFUSE_INDEXER=1` unfuses the indexer projection, i.e. more kernel launches per
   step, which only bites at high batch. The 2026-11-12 study saw only 5 % — but it ran
   uncensored second of two, this run had it third of four. Test: order-swapped A/B/A
   (`az-standard`, `uncensored-standard`, `az-standard`) at C=8 sustained, 60 s each.
2. **The 524k TTFT penalty (+136 ms at C=1, 266 vs 130) is unattributed.** Three candidate
   causes changed together: `prefix-match-unit` 32→64, `block-size` 860→3520, run order.
   Test: `az-524k-single` with `set: vllm.prefix-match-unit: 32` (legal at 3520), measured
   twice back-to-back against the shipped one.
3. **The 1k prefill sample is a cold-start artifact** (475/263/344/1108/1267 across runs of
   configs that should be identical; a single sample, TTFT up to 3 s). Fix the harness:
   warm-up request before the prefill phase, then median of 3. Until then, read the 1k column
   as noise.
4. **~~`instruction` 0.80 — investigated and closed, it is honest.~~** Two of ten tasks fail on
   every config and every run because the model cannot count: `instr-01` ("exactly 12 words")
   answers 13, `instr-09` ("repeat 'ab' exactly 20 times") outputs `ab`×10 — both with
   `finish_reason=stop` and no truncation. Already documented in `quality/README.md` as
   baseline behaviour identical in M0. Keep the 0.80: it is a real capability signal, not a
   harness bug.
5. **`presence_penalty` via `--override-generation-config` may be a no-op** on stock vLLM
   (open since the study). Test: repetition probe with and without, compare `repetition` score.
6. **A second seat on 524k**: the 767k pool is 1.46× a full request, so two ~350k sessions fit.
   If dual long-context agents are a real use case, measure `set: vllm.max-num-seqs: 2` at
   C=2 — expect a decode cost; the fp8-KV variant instead buys 2.9× the pool for −8 % decode.
7. **520k prefill at 65,536 batched tokens**: halves the chunk count (16→8); the needle run
   says 2,216 tok/s, worth one boot to see if the ceiling is chunking or bandwidth.
8. **eugr b12x against the new baseline.** The old comparison used the pre-refactor numbers
   (157 at C=8); the standard config now measures 172.4. Re-run
   `./bench/baseline.py --attach http://127.0.0.1:8000 --only az-standard` against b12x before
   repeating the "~30 % slower" claim.

## Benchmark harness (`bench/`)

The entry point is **`bench/baseline.py`**: it boots every shipped configuration, runs the
measurement suite and the quality gate on each, tears it down, and prints one consolidated block
you can paste straight into a conversation.

```bash
./bench/baseline.py                      # all four configs, full profile + full quality
./bench/baseline.py --profile quick      # fast pass (~10 min total instead of ~40)
./bench/baseline.py --only az-standard   # one config (repeatable)
./bench/baseline.py --dry-run            # show the plan, change nothing
./bench/baseline.py --report             # rebuild BASELINE.md from the newest results
./bench/baseline.py --report DIR [DIR…]  # …or merge specific runs (later dirs override
                                         # the same config — how this file combines
                                         # 033710Z's standards with 161012Z's 524k)
```

It refuses to start while another model server is running — this box serves one model at a time
and the numbers are meaningless with two. If you already have a server up and want to measure it
without touching it:

```bash
./bench/baseline.py --attach http://127.0.0.1:8000 --only az-standard
```

Each run writes `bench/baseline/<run-id>/<config>/` (`summary.json`, `quality.json`, `meta.json`,
`server.log`) and a single `BASELINE.md` at the repo root. That file is the pasteable artifact:
summary table, decode-by-content, the concurrency ladder, the accepted-tokens-per-position depth
curve, prefill, the quality battery, and the configuration actually used.

The pieces it composes, if you want to drive them by hand:

```bash
python3 bench/check_recipes.py bench/matrix.yaml          # static preflight, no GPU
python3 bench/run_matrix.py --matrix bench/matrix.yaml --dry-run
python3 bench/aggregate.py --results-dir bench/results    # RESULTS.md + summary.csv + ranking

python3 bench/serve.py up --recipe recipes/az-524k-single.yaml --run-id my-run
python3 bench/suite.py --base-url http://127.0.0.1:8000 --run-dir bench/results/my-run \
                       --config-id my-run --profile quick --quality full
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

`eugr-stack/` is a **real git submodule** of
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker), not a vendored copy. Keep it
at the upstream tip:

```bash
./setup.sh --eugr        # sync eugr-stack to upstream latest and print the SHA
git add eugr-stack && git commit -m "eugr-stack @ <sha>"
```

The workflow we intend for it, once the myllmbox baseline is validated: take the winning
configuration from `BASELINE.md` and run the same measurement on b12x, so the comparison is
like-for-like on one box.

```bash
./setup.sh --eugr                                    # pin the upstream SHA first
./presets/run_az_b12x.sh --wait                      # boot b12x
./bench/baseline.py --attach http://127.0.0.1:8000   # measure it without touching the container
./presets/run_az_b12x.sh --stop
```

Two things to know before you trust a b12x number:

- `launch.sh` pulls `eugr/spark-vllm-b12x:latest`, which is **mutable**. For a reproducible run,
  pin the digest: `IMAGE=eugr/spark-vllm-b12x@sha256:<digest> ./presets/run_az_b12x.sh`.
- The b12x GDN kernel caps speculative depth at **7** (`state_index_columns must be at most 8`),
  so the 524k single-stream configuration at K=16 cannot be reproduced there — that one is
  myllmbox-only. The standard configuration at K=4/K=7 can.

Measured on this box with `AUTOTUNE=0` (autotune costs ~55–65 min on first boot), during the
study: quality identical, host-RAM pressure lower (no ~31 GB PLE table map), but **~30 % slower
at C=8** (99.9 vs 142.0 tok/s). The standard baseline has since measured 172.4 at C=8, so that
comparison needs a re-run before the number is quoted again — open test #7. Details in
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

One open question left, plus one the baseline settled:

- **`presence_penalty` in `--override-generation-config` may be a no-op.** Stock vLLM ignores it
  without a patch. `./check.sh --penalty` shows whether the flag reached the engine config;
  whether it is *applied* needs an A/B at temperature 0.6 (repetition probe, open test #4).
- **Settled: the depth ladder below its first key.** The standard ladder is `[4 -> 8, 6 -> 7,
  12 -> 6, 14 -> 5]` with no key for 1–3, and the 2026-11-21 baseline measured **average draft
  depth 12 at C=1** (= the configured K) from the engine's own counters. Below the first key the
  ceiling defaults to K, which is what the comment claims. The 524k recipe's explicit
  `[1 -> 16]` measured depth 16 at C=1, consistent.

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
