# Qwen3.8-Flash-Next on one DGX Spark (GB10)

A measurement-driven serving stack for two checkpoints of Qwen3.8-Flash-Next on a single
NVIDIA DGX Spark (GB10, 128 GB unified memory). Seven configurations, each measured on this
box, one recipe each — pick by what you are doing:

| I want | config | context each | how |
|---|---|---|---|
| the safe default / max throughput | **az-standard** | 47k × 16 seats | `./presets/run_az_standard.sh` |
| **a sub-agent fleet, low latency** | **az-standard-seats8** | 94k × 8 seats | `./presets/run_az_seats8.sh` |
| **a sub-agent fleet, 256k each** | **az-256k-fleet** | 256k × 4–5 seats | `./run.sh recipes/az-256k-fleet.yaml` |
| **one agent, maximum context** | **az-786k-single** | 786k × 1 stream | `./presets/run_az_786k.sh` |
| one long agent, two mid streams | az-524k-single | 524k × 1 (1.46× pool) | `./presets/run_az_524k.sh` |
| any of the above, no refusals | `uncensored-*` | same | `./presets/run_uncensored_*.sh` ⚠ |

The three fleet/long-context configs were promoted from a 12-candidate measurement run
(2026-11-21/22); the numbers behind every row are in **[BASELINE.md](BASELINE.md)**. Two caveats
are documented honestly: `az-256k-fleet` has a ~2.6 s concurrent-admission barrier (fine at
≤2 agents), and the abliterated checkpoint costs ~25 % at C=8 — use it single-stream, not as a
fleet. Details in [Which config for what](#which-config-for-what).

Everything here was measured on this box with the harness in [`bench/`](bench/README.md) and the
gate in [`quality/`](quality/). The full study — 30+ configurations, controlled A/B/A runs, quality
validation, and the negative results — is in **[BENCHMARK-REPORT.md](BENCHMARK-REPORT.md)**.

**Built on the work of others.** The checkpoints are
[azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ)
and [myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored);
the serving image, the recipe format and the tuned defaults come from
[myllmbox/qwen38-flash-next-recipe](https://github.com/myllmbox/qwen38-flash-next-recipe); the
setup/check discipline and the alternative b12x lineage from
[azampatti/Qwen3.8-Flash-Next-Int4-FAST](https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST);
the b12x stack itself from [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker).
This repository adds the measurements, the harness, the 524k/786k/fleet configurations and the
reasoning — full attribution in [Credits](#credits-and-licensing).

---

## Table of contents

- [Why these seven and nothing else](#why-these-seven-and-nothing-else)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Getting the baseline](#getting-the-baseline)
- [Which config for what](#which-config-for-what)
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

## Why these seven and nothing else

This repo used to carry ~40 recipes. The study and the 2026-11-21/22 candidate run kept seven,
each for a measured reason:

- **az-standard** — the shipped myllmbox v5.4 recipe is the best single config measured.
  Twenty-plus depth / drafter / KV / chat-template variants were tested; none beat it beyond the
  ±10–15 % noise of a shared host.
- **az-standard-seats8** — for a sub-agent fleet, 8 seats × 94k beats 16 × 47k: same C=8
  throughput (174.9 vs 172), 335 ms TTFT, but no seat preempts another. One knob from standard.
- **az-256k-fleet** — the only way to give every agent a full native 256k window (4–5 seats,
  recall verified). fp8 KV is what makes the pool fit. Carries a documented 2.6 s barrier.
- **az-786k-single** — YaRN 3.0 works: 786k context, 778k needle recalled. The max-context config.
- **az-524k-single** — the middle long-context point; its 1.46× pool is what fits two ~380k
  streams, which 786k (1.14×) cannot.
- **uncensored-standard / uncensored-524k-single** — the abliterated checkpoint, for when you
  need 0 % refusals. Costs ~25 % at C=8, so single-stream, not a fleet.

Dropped by measurement, each still a `set:` override away: **K=4** (not faster than K=12 —
198.5 tok/s both at C=8, just less draft compute); the **10-expert draft folder** (cost cancels
the acceptance gain); **runtime steering** (refusal 82 %→82 % at λ=±1 — the fix is weights, not
hooks); **fp8 KV as a preset** (buys context, −8 % decode — it is a knob, except where it is the
enabler, i.e. az-256k-fleet).

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

One command measures all seven configurations and prints a block you can paste back here:

```bash
./bench/baseline.py
```

It boots each config in its own container, runs the suite and the quality gate, tears it down,
moves to the next, and writes `BASELINE.md` at the repo root. Full profile is roughly 40 minutes
for the seven; `--profile quick` is a fraction of that and is enough to spot a regression.

Before you run it:

- **nothing else may be on the GPU.** It refuses to start if a server is running, and tells you
  what to stop. If you want to measure a server you started yourself, without the harness
  touching it: `./bench/baseline.py --attach http://127.0.0.1:8000`.
- **enough free RAM.** On the GB10 there is no separate VRAM: weights, KV and activations share
  the 128 GB. The preflight refuses any boot that cannot fit (`~74 GB weights + the recipe's
  KV reservation + 9 GB slack`) rather than dying 4 minutes in, so a too-large `kv-cache-memory`
  fails fast with the numbers, not as an OOM. Close the desktop's heavy apps first; `--force`
  overrides the check.
- **run `./tune-host.sh` once.** The GB10 page compaction is the single biggest source of noise
  on this box.
- **run it twice for anything you intend to decide on.** Single runs swing ±10–15 %; only
  same-window repeats are comparable.

The report is designed to be pasted: summary table, decode by content, the concurrency ladder, the
accepted-tokens-per-position depth curve, prefill, the quality battery, and the exact
configuration used — plus a pointer to the raw `summary.json` per config.

## Which config for what

All seven share the image, kernels and drafter. They differ only in the KV/context/seat trade-off,
so pick by the one number you care about: **context per concurrent agent**.

The KV pool is the budget. Measured on this box: **29,986 bytes/token bf16**, half that in fp8.
Every seat needs `context × 29,986` bytes of pool reserved. That single fact decides everything:

| config | seats | context each | KV pool | C=N TTFT | when |
|---|---|---|---|---|---|
| az-standard | 16 | 47k | 750,933 (1.83× a 400k) | 335 ms | many short-lived agents, max throughput |
| az-standard-seats8 | 8 | 94k | 750,933 | 335 ms | **fleet default** — every realistic agent fits, no eviction |
| az-256k-fleet | 4–5 | **262,144 (native)** | 1,384,355 fp8 (5.28×) | **2.6 s ⚠** | agents that genuinely need 256k each |
| az-524k-single | 1 | 524,288 | 767,013 (1.46×) | 267 ms | one long agent; 1.46× also fits two ~380k |
| az-786k-single | 1 | **786,432** | 892,616 (1.14×) | 352 ms | **max context**; 6 min to first token on a full prompt |

**The fleet decision.** For sub-agents, more seats with less context each is faster and has no
latency cliff: 8 seats × 94k (seats8) gives 175 tok/s at 335 ms TTFT, while 4 seats × 256k
(256k-fleet) gives 71 tok/s and a 2.6 s barrier when all seats fire together. Only pay for 256k
per seat if an agent actually holds a 256k window — otherwise seats8 is strictly better. The
256k-fleet barrier is reproduced and unexplained (see [open tests](#open-questions-and-next-
tests)); it does not appear at ≤2 concurrent agents, so a few long-running agents are fine.

**The max-context decision.** 786k is YaRN 3.0 and the 778k needle is verified — extrapolation
to 3× works on this checkpoint. But its 1.14× pool is one stream only. For two long streams,
524k's 1.46× fits two ~380k sessions, or 786k-on-fp8 would give two full 786k (not yet measured).

### The knobs that are not free

`block-size` differs between standard (3,456) and the 524k/786k configs (3,520) for a hard
reason: the QSA drafter ring has capacity **K+4** and the engine refuses to boot unless it
divides the block size (`QSA ring capacity 20 must divide the attention block size 3456`,
measured). At K=12 the ring is 16 and 3,456 is legal; at K=16 the ring is 20, so 3,520 =
20×176 = 64×55. The first cut of the 524k recipe used 860/32 inherited from the container it
replaced, which cost 19 % of the KV pool (604,947 tokens — each of 703 small blocks pays a
per-block state charge, visible as group pages `13760` vs `3520`); 3,520 yields the full
767,013. The 23 GiB KV pool is what makes 524k fit — do not lower `kv-cache-memory` there.

The 524k/786k configs also raise `max-num-batched-tokens` (32,768 / 16,384), pin the depth
ceiling to the full budget for one stream, and add fusion passes — all documented inline in each
recipe. `az-standard-seats8` and `az-256k-fleet` change only seats (and KV dtype/size for the
latter); they inherit the standard's kernels untouched.

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
| quality gate (full) | 0.9714 | 0.9643 (524k variant: **0.9714**, matching AZ) |
| decode, single stream | 63.3 tok/s | 63.3 tok/s |
| C=8 | 163 tok/s | 136 tok/s (−17 %, the abba block confirms it is real, not drift) |

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
  az-standard-seats8.yaml          FLEET: 8 seats x 94k, low latency
  az-256k-fleet.yaml               FLEET: 4-5 seats x 256k native, fp8 KV (2.6s barrier)
  az-524k-single.yaml              524k context, one stream
  az-786k-single.yaml              MAX CONTEXT: 786k, YaRN 3.0, needle verified
  uncensored-standard.yaml         abliterated checkpoint, shipped settings
  uncensored-524k-single.yaml      abliterated checkpoint, 524k single stream
  tests/                           still-open candidates, one knob each — see below
    524k-pmu32.yaml                  answered #2: 524k TTFT was the prefix unit
    524k-dual.yaml                   answered #6: two long seats (shape bug at K=16 fixed)
    524k-fp8kv.yaml                  answered #6b: fp8 KV = 2.85x pool
    524k-batched48k-fp8.yaml         answered #7: 64k OOM'd, 48k is the ceiling
    256k-tri.yaml                    3 seats x 256k bf16 — barrier investigation
    256k-quad.yaml                   4 seats x 256k bf16 — barrier investigation
presets/
  run_az_standard.sh  run_az_seats8.sh  run_az_524k.sh  run_az_786k.sh
  run_uncensored_standard.sh  run_uncensored_524k.sh
  run_az_b12x.sh                 the alternative stack

bench/                       the reproducible benchmark harness (stdlib only)
  baseline.py              THE ENTRY POINT: boot all shipped configs, measure, print one pasteable report
  serve.py                 recipe -> docker manifest, health wait, boot time, meta.json
  suite.py                 the measurement suite: decode / concurrency / prefill / needle / spec
  run_matrix.py            sequential, resumable matrix runner (what baseline.py drives)
  aggregate.py             results -> RESULTS.md + summary.csv + ranking
  check_recipes.py         static preflight, no GPU
  matrix.yaml              the seven shipped configs, ready to measure
  matrix-tests.yaml        the still-open candidates + the A/B/A that settled the uncensored gap
quality/                     deterministic capability + long-context gate (133 offline tests)
tools/                       mtp_drafter_check.sh, verify_cache.py
eugr-b12x/                   the alternative eugr b12x launcher (plain docker run)
eugr-stack/                  git submodule -> github.com/eugr/spark-vllm-docker (keep at upstream tip)
BASELINE.md                  the current numbers, all seven configs, pasteable
BENCHMARK-REPORT.md          the study behind the shipped configs: methodology, constraints, negative results
LICENSE                      MIT for this code; model weights and images keep their own terms
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

**The current numbers are in [`BASELINE.md`](BASELINE.md)** — one pasteable view of all seven
configurations, rebuilt from the raw results with `./bench/baseline.py --report <dirs>`. As
measured 2026-11-21/22 on this box (image `v5.4`, bc42593a5bbf; the four core at full profile +
full quality, the three promoted at the candidate run's quick profile — a fresh
`./bench/baseline.py` now re-measures all seven at full quality):

| config | decode | C=1 | C=N | TTFT C=1 | KV pool | quality | needle |
|---|---|---|---|---|---|---|---|
| az-standard | 63.3 | 55.5 | 163 @C8 | 141 ms | 750,933 | 0.9714 | 96k OK |
| az-standard-seats8 | 60.9 | 47.2 | 174.9 @C8 | 104 ms | 750,933 | — | — |
| az-256k-fleet | 58.0 | 45.8 | 71.3 @C4 | 106 ms | 1,384,355 fp8 | — | **259k OK** |
| az-524k-single | 63.7 | 53.8 | — | 267 ms | 767,013 | 0.9643 | **518k OK** |
| az-786k-single | 60.7 | 42.4 | — | 352 ms | 892,616 | — | **778k OK** |
| uncensored-standard | 63.3 | 57.2 | 136 @C8 | 157 ms | 750,933 | 0.9643 | 96k OK |
| uncensored-524k-single | 63.7 | 54.5 | — | 287 ms | 767,013 | **0.9714** | **518k OK** |

Three headline findings from the full candidate run: **YaRN 3.0 works** (786k, 778k needle
recalled); **fp8 KV gives 2.85–5.28× the pool** for ~9 % decode (what makes the 256k fleet fit);
and the **abliterated checkpoint costs ~25 % at C=8** (the abba block, zero drift) — so it is a
single-stream choice, not a fleet one. The 256k-fleet's 2.6 s concurrent barrier is documented in
its recipe and tracked below.

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
tested** (confirmed again on every config in the 2026-11-21/22 baseline: p0–p2 carry 93–94 % of
accepted tokens, p5 and beyond are zero), so more than ~4 drafts per step only adds compute —
and depth 4 and depth 12 are throughput-equal because the fused multi-step drafter makes the
extra drafts nearly free.

## Open questions and next tests

The 2026-11-21/22 candidate run **settled six questions and promoted three configs**. What is
answered, and what is still open:

### Settled by measurement (do not re-litigate without new evidence)

- **The uncensored C=8 gap is real (−25 %).** The abba block booted az-standard first and last:
  177.1 and 177.4 (0.2 % apart, zero drift), uncensored between them 132.0. It is the unfused
  indexer (`MBX_QSA_UNFUSE_INDEXER=1`) biting at batch, free at C=1. → abliterated is a
  single-stream choice, not a fleet one.
- **The 524k TTFT penalty was the 64-unit prefix match** (pmu32 → 130 ms, exactly standard's),
  but 32 costs TPOT/decode in return — neither dominates, so 524k keeps 64.
- **fp8 KV = 2.85–5.28× the pool for ~9 % decode** (measured at 524k and 256k). It is the enabler
  for the 256k fleet.
- **Prefill chunking is memory-bound.** A 64k chunk × 524k context needs ~32 GB of activations
  and OOM-froze the box; 48k on fp8 is the ceiling (524k-batched48k-fp8).
- **YaRN 3.0 works.** 786k boots (engine accepts factor 3.0) and the 778k needle is recalled.
  Promoted to az-786k-single.
- **`instruction` 0.80 is honest** — the model cannot count (12→13 words, "ab"×20→×10), identical
  everywhere, a real capability signal not a harness bug.

### Still open

1. **The 256k-fleet 2.6 s concurrent-admission barrier — characterised, not a bug to fix.** At
   C=N (all seats firing together) every 256k request shows an identical ~2.6–2.7 s TTFT (stdev
   < 1 ms), vs 106 ms at C=1. Across 9 controlled configs (run `barrier-fix`) the barrier appears
   **if and only if: `kv-cache-dtype` is fp8, OR `max-model-len == 262144` (native) — either one,
   independently.** Every barrier-free config on this box is bf16 AND at an extended length
   (409600/524288/786432). Falsified one by one against a same-window control: the rope path
   (yarn factor 1.0 = identical 2.7 s), max-model-len alone (409600 still stalls, but via fp8),
   and the seat count (8 seats still stalls, via 262144+fp8). **Consequence:** a barrier-free
   fleet caps per-agent context at ~94k (that is az-standard-seats8); going bigger per-agent
   needs fp8 (to fit 4-5 seats) or native 262144 (for trained recall), and both stall. So
   "big context per agent" and "no barrier" are in genuine tension on this image — not a config
   error. az-256k-fleet is correct for a few long agents (≤2 concurrent: 106 ms, no barrier);
   az-standard-seats8 is correct for a wide fleet. One clean prediction remains untested:
   az-524k-fp8kv (fp8) would also stall at C=N.
2. **`524k-dual` per-seat cost.** The K=16 2-seat shape bug is fixed (depth 8), but the honest
   C=2 number needs a re-run of the fixed recipe (in matrix-tests).
3. **`presence_penalty` via `--override-generation-config` may be a no-op** on stock vLLM. Test:
   repetition probe with/without at temperature 0.6.
4. **The 1k prefill sample is a cold-start artifact** (single sample, TTFT up to 3 s). Harness
   fix: warm-up + median of 3 before the prefill phase.
5. **eugr b12x against the new baseline.** The old "~30 % slower" used pre-refactor numbers
   (157 at C=8); standard now measures 163–175. Re-run
   `./bench/baseline.py --attach http://127.0.0.1:8000 --only az-standard` against b12x before
   repeating it.

To run the still-open candidates (the promoted configs moved to the baseline matrix):

```bash
python3 bench/run_matrix.py --matrix bench/matrix-tests.yaml            # candidates + A/B/A
python3 bench/run_matrix.py --matrix bench/matrix-tests.yaml --only 524k-dual   # or one
```

## Benchmark harness (`bench/`)

The entry point is **`bench/baseline.py`**: it boots every shipped configuration, runs the
measurement suite and the quality gate on each, tears it down, and prints one consolidated block
you can paste straight into a conversation.

```bash
./bench/baseline.py                      # all seven configs, full profile + full quality
./bench/baseline.py --profile quick      # fast pass (~10 min total instead of ~40)
./bench/baseline.py --only az-standard   # one config (repeatable)
./bench/baseline.py --dry-run            # show the plan, change nothing
./bench/baseline.py --report             # rebuild BASELINE.md from the newest results
./bench/baseline.py --report DIR [DIR…]  # …or merge specific runs (later dirs override
                                         # the same config — how this file combines
                                         # 161012Z's 524k into the standards from 033710Z)
```

**The full picture — baseline + all twelve open-question candidates, merged into one table:**

```bash
./bench/baseline.py --run-id b-today \
 && ./bench/baseline.py --run-id t-today --matrix bench/matrix-tests.yaml \
 && ./bench/baseline.py --report bench/baseline/t-today bench/baseline/b-today
```

~2.5–3 h: 4 full-quality boots + 12 quick ones. Where a config appears in both matrices the
full-quality row wins; the abba block is compared inside its own run
(`./bench/baseline.py --report bench/baseline/t-today`).

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
