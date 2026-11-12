# draft-lab — speculative-decoding / MTP draft variants for Qwen3.8-Flash-Next-FP4-Local-AZ

Owner: **draft-lab** (shared task `task-4`). Scope: the MTP drafter, the draft folders it loads from, and
the recipes that vary draft depth / acceptance behaviour. **No GPU work happens here** — the Lead runs
every server and the benchmark matrix; this directory is build + offline verification + recipes.

Model: `azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ` (NVFP4 MoE, routes **5 of 512** experts, trunk
shared expert 1280) served by `myllmbox/qwen38-flash-next-vllm:v5.4` (vLLM 0.32 + myllmbox patches).
The MTP layer keeps NVIDIA's drafter **trained at 10 experts**, with its shared expert zero-padded
640 → 1280 so one config describes both trunk and head. That mismatch is the whole hypothesis of this
lab: the model routes 5, but its drafter was trained at 10.

```
draft-lab/
├── README.md                     this file
├── matrix.yaml                   machine-readable manifest: tiers, priority scan, exact specs/env
├── gen_recipes.py                one table -> recipes/*.yaml + matrix.yaml + bench-matrix.yaml
├── bench-matrix.yaml             drop-in configs for bench/run_matrix.py (same recipes, run order)
├── recipes/                      28 recipes: 24 bootable, 4 doc-only depth<4 points (not bootable)
├── recipes-runsh/                6 run.sh deployment copies of the draft-k10 recipes (model=/models/draft-k10)
├── drafts/
│   ├── qwen38-az-draft-k10/      draft folder, top-k 10 (the model-card optional MTP-at-10 drafter)
│   └── qwen38-az-draft-k5/       control copy, top-k 5 (= self-draft routing)
└── verification/                 build/verify logs, guard tests, checksums, exact file list
```

## TL;DR — what to run first

The Lead's tier-1 scan, in order (all launch with `bench/serve.py`, no edits needed):

| # | recipe | K | drafter | what it tests |
|---|---|---|---|---|
| 1 | `draft-lab/recipes/dynamic-ceiling-9.yaml` | 9 | self | dynamic depth with max depth 9 |
| 2 | `draft-lab/recipes/fixed-5.yaml` | 5 | self | static depth 5 (`MBX_DEPTH_MODE=off`) |
| 3 | `draft-lab/recipes/draftk10-spec6.yaml` | 6 | **draft-k10** | the model-card 10-expert drafter |
| 4 | `draft-lab/recipes/self-head0.yaml` | 12 | self | bf16 drafter head instead of NVFP4 |
| 5 | `draft-lab/recipes/self-scale2.yaml` | 12 | self | `MBX_MTP_DRAFT_SCALE=2` |
| 6 | `draft-lab/recipes/fixed-9.yaml` | 9 | self | static depth 9 |

`dynamic-ceiling-12-ref.yaml` is the measured myllmbox reference (**M0**) — control, do **not** re-run it
in the scan; use its numbers as the acceptance baseline. The full tiered list is in `matrix.yaml` and the
table at the bottom of this file.

## The draft folders

The model card (HF snapshot README) specifies the optional MTP-at-10-experts mode: a **draft folder**
whose `config.json` is the model config with `text_config.num_experts_per_tok = 10`, whose
`model.safetensors.index.json` lists only the `mtp.*`, `lm_head.*` and `embed_tokens` tensors, and whose
shard files are links to the shards holding those tensors (this checkpoint: **shards 1, 14, 15** =
`tensors/model-00001/14/15-of-00021.safetensors`). It is passed as `"model": "<draft folder>"` in
`--speculative-config`. Without a `"model"` key the drafter is built from the target itself ("self-draft").

What `drafts/qwen38-az-draft-k10/` contains (15 entries; 3 regular files, 11 **absolute** symlinks, 2 dirs):

| entry | kind | content |
|---|---|---|
| `config.json` | regular, 75,218 B | copy of the model config; **only** `text_config.num_experts_per_tok: 5 → 10` |
| `model.safetensors.index.json` | regular, 467,418 B | `metadata: {}`, `weight_map` = exactly **4,639** tensors: **4,637 `mtp.*` + `lm_head.weight` + `model.language_model.embed_tokens.weight`** |
| `tensors/model-00001-of-00021.safetensors` | symlink (abs) | holds 4,608 of the draft tensors |
| `tensors/model-00014-of-00021.safetensors` | symlink (abs) | holds `lm_head.weight` |
| `tensors/model-00015-of-00021.safetensors` | symlink (abs) | holds 30 draft tensors incl. the embedding and `mtp.pre_fc_norm_hidden` |
| `chat_template.jinja`, `generation_config.json`, `merges.txt`, `preprocessor_config.json`, `tokenizer.json`, `tokenizer_config.json`, `video_preprocessor_config.json`, `vocab.json` | symlinks (abs) | tokenizer/config files for the drafter |
| `.make_draft_dir.json` | regular | build marker (source snapshot, top-k, counts, source checksums); also the marker that allows the builder to replace the folder |

**Why 4,639 and not 4,638?** The old broken folder in `models/` had 4,638: its filter matched the
prefix `embed_tokens.`, but this checkpoint's key is `model.language_model.embed_tokens.weight`, so the
embedding was silently dropped. The Lead's successful self-draft boot log
(`/tmp/boot-fixed-index.log`) shows the loader reading **4,639** draft tensors from 3 files
(`4,637 mtp + lm_head + embed_tokens`), so 4,639 is the correct/expected count (and `run.sh`'s own
builder uses the `endswith("embed_tokens.weight")` rule).

**MTP layer structure (verified from the index + safetensors headers).** The MTP head is one layer whose MoE
keeps **all 512 experts** (9 tensors per expert = 4,608 tensors in shard 1) with its own BF16 router gate
`mtp.layers.0.mlp.gate.weight` of shape `[512, 2560]` — bit-distinct from every one of the 48 trunk-layer
routers. So the draft folder's `num_experts_per_tok` changes only **how many of those 512 experts the MTP
router selects per token** (10 vs 5); no MTP weight changes. The router was trained at 10, which is why the
hypothesis is that top-k 10 recovers the drafter's native behaviour.

`drafts/qwen38-az-draft-k5/` is identical except `num_experts_per_tok` is 5 (a copy of the model routing).
It is a **control**: it should behave like the self-draft at the same K. If it does not, the draft-folder
plumbing (mount, index filter) is wrong — that is its only purpose.

### Build / re-verify (offline, no GPU)

```bash
SNAP=/home/pgx/.cache/huggingface/hub/models--azampatti--Qwen3.8-Flash-Next-FP4-Local-AZ/snapshots/5d927eabd4e4766a69bf12f7a1387b01d936a78f

# build (safe: the snapshot is read-only; the destination must be outside the HF cache)
python3 tools/make_draft_dir.py "$SNAP" draft-lab/drafts/qwen38-az-draft-k10 --num-experts-per-tok 10
python3 tools/make_draft_dir.py "$SNAP" draft-lab/drafts/qwen38-az-draft-k5  --num-experts-per-tok 5

# re-verify an existing folder against the snapshot (no writes)
python3 tools/make_draft_dir.py "$SNAP" draft-lab/drafts/qwen38-az-draft-k10 --verify-only
```

`tools/make_draft_dir.py` v2.0 verifies, at build time: the config differs from the model config **only**
in `text_config.num_experts_per_tok`; the index is exactly the filtered model weight_map (and is
non-empty, has `mtp.*`, `lm_head.*` and an embed_tokens tensor); every indexed tensor name exists in the
safetensors header of the shard it maps to; every symlink resolves to the intended snapshot file; the
folder contains nothing unexpected. It also fingerprints the snapshot before/after (sha256 of index +
config, plus the full top-level listing) and fails if anything changed.

## Mount contract (confirmed with the Lead)

The container sees the host filesystem at the **same absolute paths**, read-only. `bench-harness` adds
two mounts to every launch; `run.sh` already mounts the HF hub that way in cache mode:

```
-v /home/pgx/Documenti/vllm-gb10:/home/pgx/Documenti/vllm-gb10:ro          # recipe + draft folders
-v /home/pgx/.cache/huggingface/hub:/home/pgx/.cache/huggingface/hub:ro   # snapshot the links point at
```

Therefore the draft-folder path in `--speculative-config` is its host absolute path:

```json
{"method":"mtp","num_speculative_tokens":6,
 "model":"/home/pgx/Documenti/vllm-gb10/draft-lab/drafts/qwen38-az-draft-k10",
 "rejection_sample_method":"block","draft_sample_method":"probabilistic"}
```

Self-draft recipes simply omit `"model"` — the myllmbox stack then builds the drafter from the target.

**Duplicate-flag rule.** `run.sh` appends its own `--speculative-config` when `MTP != 0`, and it takes
`MTP` from `config.env`/the environment, not from a recipe's `env:` section. Every recipe here contains
exactly one `speculative-config:` line, so:
* `bench/serve.py` (parses the recipe itself) — launch as-is;
* `./run.sh` fallback — prefix `MTP=0`, e.g. `MTP=0 ./run.sh draft-lab/recipes/draftk10-spec6.yaml`.

Sanity check on a live server: the boot log must show
`mbx loader: drafter reads 3 of 21 files` / `4639 tensors, 3 files` and, for a draft-folder run,
`draft model … /home/pgx/Documenti/vllm-gb10/draft-lab/drafts/qwen38-az-draft-k10`. If it reads 21 files
or fails to open the folder, the mount is wrong.

## Deployment outside bench/: `/models/draft-k10` (run.sh)

`bench/serve.py` mounts the workspace root at its absolute path, so the canonical draft folders under
`draft-lab/drafts/` can be referenced directly. **`run.sh` does not mount the workspace root**: it mounts
`./models` → `/models`, `./cache` → `/cache`, and the HF hub at its own absolute path. For run.sh the k10
drafter is therefore pre-packaged in the served models directory:

```
models/draft-k10/                  # built by tools/make_draft_dir.py (same verification as draft-lab/drafts)
├── config.json                    # REAL FILE (not a symlink: the workspace root is not mounted there)
├── model.safetensors.index.json   # REAL FILE, 4,639 draft tensors
├── tensors/model-0000{1,14,15}-of-00021.safetensors   -> ABSOLUTE symlinks into the HF hub
├── tokenizer/config files         -> ABSOLUTE symlinks into the HF hub
└── .make_draft_dir.json
```

so `--speculative-config` gets `"model": "/models/draft-k10"`. Launch-ready copies of every draft-k10
variant are generated in `draft-lab/recipes-runsh/*-models.yaml` (identical except the model path):

```bash
MTP=0 ./run.sh draft-lab/recipes-runsh/draftk10-spec6-models.yaml   # tier-1 k10 hypothesis
```

(`MTP=0` matters: otherwise run.sh appends its own `--speculative-config`.) The absolute shard symlinks
stay valid because run.sh mounts the HF hub at its same absolute host path in cache mode. Verified inside a
container with exactly that mount set *and no workspace-root mount* —
`verification/mount-check-runsh-models.log`: `/models/draft-k10` visible, config top-k 10, all three shards
readable through the hub mount, tokenizer resolves. Rebuild with:

```bash
python3 tools/make_draft_dir.py "$SNAP" models/draft-k10 --num-experts-per-tok 10
```

## Recipes and matrix

`draft-lab/recipes/*.yaml` use the repository format (`server:` / `env:` / `vllm:` with `name: value`
flags) and are self-describing: the **full** env and vllm sections from the M0 reference, with only the
axis under test changed. `draft-lab/matrix.yaml` is the machine-readable manifest (tiers, priority scan,
exact `speculative_config` string, env deltas, drafter, draft-model path, notes). Regenerate/validate:

```bash
python3 draft-lab/gen_recipes.py          # writes recipes + matrix.yaml + bench-matrix.yaml
python3 draft-lab/gen_recipes.py --check  # validate only
```

**Run it without editing anything** — the generated `draft-lab/bench-matrix.yaml` is a drop-in for the
Lead's runner (verified with `bench/run_matrix.py --dry-run`: it resolves every recipe and prints the
serve/suite/down commands; `bench/serve.py` already mounts the workspace root and the HF hub at their
absolute host paths, so the draft folders resolve with no extra mounts):

```bash
python3 bench/run_matrix.py --matrix draft-lab/bench-matrix.yaml --dry-run
python3 bench/run_matrix.py --matrix draft-lab/bench-matrix.yaml --max-configs 6   # tier 1 only
python3 bench/run_matrix.py --matrix draft-lab/bench-matrix.yaml                   # tier 1 -> 2 -> 3
```

Run order is the `name`/`priority` order in that file; tier-3 entries use `suite_profile: quick`, and the
M0 reference is `enabled: false` so it is skipped (already measured).

### Env reference (v5.4 image; verified against the image and the boot log)

| env | values | meaning |
|---|---|---|
| `MBX_DEPTH_MODE` | **`dynamic` only** | dynamic draft depth (concurrency/timing adaptive). `off` is **not bootable** in v5.4 — see the static-depth note below. |
| `MBX_DEPTH_MIN` | e.g. `3` | lowest depth dynamic mode may pick |
| `MBX_DEPTH_WINDOW` | e.g. `48` | measurement window (steps) for the depth controller |
| `MBX_DEPTH_PROMOTE` / `MBX_DEPTH_DEMOTE` | e.g. `60,45` / `25,15` | acceptance thresholds to raise/lower depth |
| `MBX_DEPTH_CEIL` | `[N -> M, ...]` | concurrency-threshold → max depth. Reference `[4 -> 8, 6 -> 7, 12 -> 6, 14 -> 5]`: ≤3 reqs may use K, 4-5 → 8, 6-11 → 7, 12-13 → 6, 14+ → 5 |
| `MBX_DEPTH_SYNCFREE` | `0` \| `1` | keep depth bookkeeping on the GPU |
| `MBX_DRAFT_HEAD` | `0` \| `1` | drafter scores with an NVFP4 Marlin copy of `lm_head` (1: 0.36 GB/rank vs 1.27 GB bf16; 0: full target head) |
| `MBX_MTP_DRAFT_SCALE` | float, `1.0` = off | multiplies drafter logits before the accept test (exact: the verifier still uses the target distribution). `mtp.py` builds `LogitsProcessor(config.vocab_size, scale=…)`; 2 is the repo's measured +2.5pp acceptance |
| `MBX_DRAFT_FP8` | `0` \| `1` | fp8 drafter linears (faster drafts; acceptance may drop) |
| `MBX_B12X_WARM_DRAFTER` | `0` \| `1` | warm the drafter at load (low priority; value semantics undocumented in the image — check the boot log) |
| `MBX_LOADER_PREPARE`, `MBX_LOADER_EMPTY_CACHE`, `MBX_PLE_*`, `MBX_MARLIN_CSF`, … | | loader/PLE/kernel knobs carried over unchanged from the M0 reference so every recipe is a controlled A/B |

**Static depth is emulated on the dynamic path (v5.4 known issue).** `MBX_DEPTH_MODE=off` alone does
**not** boot on the v5.4 image: the patched speculator falls through to vLLM's own dynamic-SD schedule,
which at CUDA-graph capture fails with
`ValueError: num_speculative_tokens_per_batch_size is required for dynamic speculative decoding`.
Observed on Lead matrix runs **d-fixed-5** and **d-fixed-3** (both failed before `/health`, container
exit 1; evidence: `verification/boot-failure-static-depth.log`). Every `fixed-*` recipe therefore keeps
`MBX_DEPTH_MODE=dynamic` and pins it so it can never move:

```yaml
MBX_DEPTH_MODE: "dynamic"
MBX_DEPTH_MIN: "<K>"
MBX_DEPTH_CEIL: "[1 -> <K>]"      # every concurrency level may use at most K
MBX_DEPTH_PROMOTE: "101,101"      # unreachable threshold -> never promote
MBX_DEPTH_DEMOTE: "0,0"           # never demote
MBX_DEPTH_WINDOW: "48"
MBX_DEPTH_SYNCFREE: "1"
```

Same depth-law as the Lead's `matrix/recipes/self-fixed{3,4,5}.yaml` / `draftk10-fixed*.yaml`; the
generated `matrix.yaml` carries it under `known_issues` for `fixed-3/fixed-5/fixed-9`.

**The minimum bootable speculative depth is 4 (v5.4 known issue).** Depth 1 and 3 do **not** boot on
v5.4 — not with static `MBX_DEPTH_MODE=off`, not with the pinned ladder above, and not with plain
dynamic depth. At CUDA-graph capture the image's dynamic-SD validation still requires
`num_speculative_tokens_per_batch_size`, which v5.4 only supplies for depth >= 4:

```
ValueError: num_speculative_tokens_per_batch_size is required for dynamic speculative decoding.
```

Observed on the Lead's phase-B runs `b-curve-k3` (self K=3) and `b-k10-curve-k3` (draft-k10 K=3), both
failed before `/health` — evidence in `verification/boot-failure-min-depth.log` and
`bench/results/b-curve-k3/server.log`, `bench/results/b-k10-curve-k3/server.log`. Consequences:
* depth 0 (no `speculative-config` at all) still boots — `no-spec` is valid;
* the usable token curve starts at **4**;
* `curve-k1`, `curve-k3`, `fixed-3` and `draftk10-spec3` are kept only to document the unreachable curve
  points — they are marked `bootable: false` (`boot_issue: min-bootable-spec-depth-4`) in `matrix.yaml`
  and `enabled: false` in `bench-matrix.yaml`. Do not run them.

**Not available / not used**
* `MBX_DRAFT_HEAD2` (two-stage vocabulary head) — the image ships basis files only for
  `Qwen3.8-Flash-Next-FP4-Local` and `hibrid48` under `/opt/mbx/draft/`, not for this AZ checkpoint (Lead
  verified). Excluded.
* `tools/draft_scale.py` — patches an *older* module path; v5.4 reads `MBX_MTP_DRAFT_SCALE` natively, so
  the recipe env is sufficient and the patcher is unnecessary here.

### Full catalog (`matrix.yaml` is authoritative)

Baseline = self-draft, K=12, `MBX_DEPTH_MODE=dynamic`, reference ceiling ladder, `MBX_DRAFT_HEAD=1`,
`MBX_MTP_DRAFT_SCALE=1`, `MBX_DRAFT_FP8=0`, `block` + `probabilistic`. Only the delta is shown.

| tier | recipe | K | drafter | delta vs reference |
|---|---|---|---|---|
| 1 | `dynamic-ceiling-9` | 9 | self | (ceiling = K = 9, ladder unchanged) |
| 1 | `fixed-5` | 5 | self | pinned static depth: MIN=CEIL=5, CEIL=`[1 -> 5]`, PROMOTE=`101,101`, DEMOTE=`0,0` |
| 1 | `draftk10-spec6` | 6 | **draft-k10** | `"model": …/qwen38-az-draft-k10` |
| 1 | `self-head0` | 12 | self | `MBX_DRAFT_HEAD=0` |
| 1 | `self-scale2` | 12 | self | `MBX_MTP_DRAFT_SCALE=2` |
| 1 | `fixed-9` | 9 | self | pinned static depth: MIN=CEIL=9, CEIL=`[1 -> 9]`, PROMOTE=`101,101`, DEMOTE=`0,0` |
| 2 | `self-draftfp8` | 12 | self | `MBX_DRAFT_FP8=1` |
| 2 | `draftk10-spec9` | 9 | draft-k10 | draft folder |
| 2 | `draftk5-spec6` | 6 | draft-k5 | control folder (top-k 5) |
| 2 | `dynamic-ceiling-12-ref` | 12 | self | **M0 reference — control, do not re-run** |
| 2 | `no-spec` | off | — | no `speculative-config` at all (MTP off floor) |
| 2 | `draftk10-dynamic` | 12 | draft-k10 | draft folder + dynamic depth |
| 3 | `self-greedy` | 12 | self | `draft_sample_method=greedy` |
| 3 | `self-rejection-standard` | 12 | self | `rejection_sample_method=standard` |
| 3 | `fixed-3` | 3 | self | pinned static depth 3 — **NOT BOOTABLE** (min spec depth 4) |
| 3 | `ceil-aggressive` | 12 | self | `MBX_DEPTH_CEIL=[8 -> 8, 16 -> 6]` |
| 3 | `ceil-flat9` | 12 | self | `MBX_DEPTH_CEIL=[1 -> 9]` |
| 3 | `curve-k1` | 1 | self | token-curve point — **NOT BOOTABLE** (min spec depth 4) |
| 3 | `curve-k3` | 3 | self | token-curve point — **NOT BOOTABLE** (min spec depth 4) |
| 3 | `curve-k4` | 4 | self | token-curve point |
| 3 | `curve-k5` | 5 | self | token-curve point (dynamic; compare `fixed-5`) |
| 3 | `curve-k6` | 6 | self | token-curve point (anchor for `draftk10-spec6`) |
| 3 | `draftk10-greedy` | 12 | draft-k10 | greedy draft sampling |
| 3 | `draftk10-head0` | 12 | draft-k10 | `MBX_DRAFT_HEAD=0` |
| 3 | `draftk10-spec3` | 3 | draft-k10 | low-depth draft-k10 point — **NOT BOOTABLE** (min spec depth 4) |
| 3 | `draftk5-spec9` | 9 | draft-k5 | control at the ceiling-9 point |
| 3 | `self-scale3` | 12 | self | `MBX_MTP_DRAFT_SCALE=3` |
| 3 | `warmdrafter` | 12 | self | `MBX_B12X_WARM_DRAFTER=1` |

## Over-drafting hypothesis (why low fixed depth / draft-k10 can win)

Measured on this host (Lead's numbers, same checkpoint):

| stack | draft depth | acceptance | tokens / step |
|---|---|---|---|
| eugr b12x, **fixed MTP=4** | 4 | **51.1 %** | 3.04 |
| myllmbox v5.4 shipped preset, **K=12 dynamic** | up to 12 | **17.2 %** | 3.06 |

The effective output is the same (~3.0 tokens per engine step) but the K=12 dynamic preset burns roughly
**3x the draft compute** to get there: accepted-per-position collapses after position 2, so positions
3..12 are mostly speculation the verifier rejects. That is **over-drafting** — extra drafted positions
cost a drafter forward each and buy almost no extra emitted tokens.

Prediction this matrix tests (not assumed):
1. **`fixed-5`** and **`dynamic-ceiling-9`** will beat the shipped K=12 dynamic on tokens/s at comparable
   effectiveness, because they stop paying for draft positions that rarely survive.
2. At a low K, the **draft-k10** drafter (routing the 10 experts its MTP was trained with) raises
   acceptance per drafted token vs self-draft-at-5 — rows `draftk10-spec6` / `draftk10-spec9` /
   `draftk10-dynamic`.
3. `no-spec` is the floor and `fixed-9` / `curve-k6` bracket where extra depth stops paying.

Interpret acceptance_rate (accepted / proposed draft tokens) and tokens_per_step (emitted / draft step)
*together*: a config that lowers raw acceptance but keeps tokens/step while raising tok/s is a win, not a
regression. These are exactly the numbers the Lead's `matrix/matrix.yaml` records; this lab only
pre-packages and pre-verifies the variants.

### Measured results (2026-11-12): over-drafting confirmed, shallow self-draft wins

Lead's matrix runs (full detail [`MEASURED.md`](MEASURED.md), raw in `bench/results/`). Decode tok/s is
the decode-phase p50; C8 is the concurrency-phase total at 8 streams.

| metric | M0 self K=12 | draft-k10 K=6 | self K=4 | draft-k10 K=4 |
|---|---|---|---|---|
| acceptance_rate | 16.97 % | 33.96 % | 48.54 % | 52.36 % |
| avg draft depth | 12 | 6 | 4 | 4 |
| tokens / step | 3.036 | 3.038 | 2.942 | 3.094 |
| decode out tok/s p50 | 53.3 | 56.4 | **60.9** | 51.1 |
| C8 total tok/s | 142.0 | 162.9 | **178.4** | 148.4 |
| boot s | 356.3 | 350.2 | 329.9 | 340.1 |
| quality (quick) | 0.9667 | 0.9667 | n/a | 0.9583 |

1. **Over-drafting confirmed; shallow self-draft is the winner.** `d-curve-k4` (self K=4 dynamic)
   reaches **60.9 tok/s** decode (+14 % vs shipped K=12) and **178.4 tok/s at C8** (+26 %), at 48.5 %
   acceptance and 2.94 tokens/step. The shipped K=12 pays drafter forwards for positions that almost
   never survive (`accepted_per_position` collapses after position 2 at every K), so depth 4 is the
   measured sweet spot. (Depth 3 cannot be tested: not bootable, see above.)
2. **The 10-expert draft folder: clean negative at shallow depth.** At K=4 the k10 drafter does raise
   acceptance (52.4 % vs 48.5 %) and tokens/step (3.09 vs 2.94), but decode drops to **51.1 tok/s
   (−16 % vs self K=4)** and C8 to 148.4: the 10-expert drafter's heavier per-forward compute costs more
   than its acceptance gain at depth 4. At K=6 it is 56.4 tok/s (+6 % vs M0) — positive against the
   shipped preset but still below self K=4. So the MTP-at-10 option is not the winner for serving; it is
   an acceptance curiosity whose cost dominates when depth is shallow. The folder/plumbing itself is
   proven correct (the k5-top-k control and the 4,639-tensor load).
3. The k10 draft folder booted exactly as built: `drafter reads 3 of 3 files`, 4,639 tensors, 4.6 s,
   boot 350 s — the rebuild and the mount contract are solid.

## Measurement guidance for the Lead's matrix

* **Acceptance** (the point of this lab): scrape `vllm:spec_decode_num_drafts_total`,
  `vllm:spec_decode_num_accepted_tokens_total`, `vllm:spec_decode_num_emitted_tokens_total` from
  `/metrics` before/after each run; acceptance = accepted/drafts, effective tokens per step =
  (accepted + emitted)/steps. Report per C ∈ {1,2,4,8,16} — dynamic depth deliberately trades depth (and
  acceptance at fixed K) against tail latency as concurrency rises, so single-stream numbers alone hide
  the effect.
* **`draft_sample_method` / `rejection_sample_method` only matter at temperature > 0** (repo
  `config.env`): run the A/B at the served default (`temperature 0.6`, see
  `override-generation-config`) and keep the temperature-0 throughput runs separate.
* **Expectations**: `draftk10-*` should raise next-token acceptance vs self-draft at the same K if the
  10-expert-training hypothesis holds (strongest at low K, where the drafter is used near its training
  regime); `draftk5-*` should match self-draft at the same K (test of the plumbing); `no-spec` is the
  floor; `fixed-*` isolates depth from adaptivity; `self-head0`/`self-draftfp8`/`self-scale2..3` are
  drafter-quality knobs whose acceptance changes are exact-output-safe except `DRAFT_FP8`/`HEAD=0`
  (quantization can shift the draft, never the verified output distribution).
* **Boot cost**: the draft folder loads only 3 shards (~4.1 GB, ~5 s in the equivalent self-draft boot
  log) — a draft-folder run should not add meaningfully to the ~205 s model load time.

## Safety: the 05:38 clobber and the v2.0 guarantees

At ~05:38 a modified `tools/make_draft_dir.py` dumped its filtered draft index **into the model
snapshot** (`src/model.safetensors.index.json`) instead of the draft folder, replacing the real
375,487-tensor index with a 4,638-tensor one; every server boot then failed
(`Engine core initialization failed`, see `/tmp/boot-test-mtp3.log`). The Lead restored the file; the
restored index is byte-identical to `bench/refs/model.safetensors.index.original.json`
(**sha256 `60e0e2d8…d71a`**, 375,487 tensors, 21 shards, 47,280,973 B). The working-tree version had
already changed the write target to `dst`, but still had no guards: its default destination was *inside*
`~/.cache/huggingface/hub`, it never checked `dst` against `src`, and it would `rmtree()` any destination.

`tools/make_draft_dir.py` **v2.0** closes that class of bug explicitly:

1. refuses a destination that is the source snapshot, inside it, **or an ancestor of it**;
2. refuses any destination inside `$HF_HOME` / `$HF_HUB_CACHE` / `~/.cache/huggingface`;
3. only deletes an existing destination if it carries this script's `.make_draft_dir.json` marker
   (otherwise `--force` is required, and the same guards still apply);
4. every write/symlink goes through a helper that asserts the resolved parent is inside the destination;
5. shard paths from the weight_map must be relative and cannot escape the snapshot (`..` refused);
6. the only files written are `<draft>/config.json`, `<draft>/model.safetensors.index.json`,
   `<draft>/.make_draft_dir.json` and absolute symlinks — no weight bytes are copied anywhere;
7. source shas + full directory listing are compared before/after; any change aborts the build;
8. the build fails unless every indexed tensor is present in the header of the shard it maps to, all
   links resolve, and the folder has no unexpected entries.

### Verification evidence (offline, all in `verification/`)

| file | what it proves |
|---|---|
| `build-draft-k10.log`, `build-draft-k5.log` | index 4,639 tensors (4,637 mtp + 1 lm_head + 1 embed_tokens); shards 1/14/15 with 4,608/1/30 header matches; config diff only `num_experts_per_tok`; `PASS`; source index sha `60e0e2d8…` unchanged |
| `verify-draft-k10.log`, `verify-draft-k5.log` | independent `--verify-only` re-check: `PASS` |
| `guard-tests.log` | five hostile destinations (inside snapshot, snapshot itself, inside HF cache, ancestor of snapshot, unmarked dir) all refused with `exit=1` |
| `checksums.txt` | snapshot index sha == `bench/refs` backup sha; config sha; no snapshot file newer than the build window; per-draft config/index shas; marker JSONs; scoped `git status` |
| `FILES-WRITTEN.txt` | every entry the builder created (15 per folder) — nothing outside the two draft folders |
| `boot-failure-min-depth.log` | why speculative depth < 4 can never be used on v5.4: Lead phase-B runs b-curve-k3 and b-k10-curve-k3 both failed at CUDA-graph capture with `ValueError: num_speculative_tokens_per_batch_size is required for dynamic speculative decoding`; min bootable depth is 4 |
| `boot-failure-static-depth.log` | why `MBX_DEPTH_MODE=off` can never be used: Lead runs d-fixed-5/d-fixed-3 both failed before `/health` with `ValueError: num_speculative_tokens_per_batch_size is required for dynamic speculative decoding`; records the pinned-ladder fix |
| `measured-d-draftk10-spec6.md` | first live evidence: the k10 draft folder is loaded by the served container (`drafter reads 3 of 3 files`, 4,639 tensors, 4.6 s), boot 350 s, acceptance/tokens-per-step/quality vs M0, plus the measurement caveats |
| `boot-draftk10-spec6-keylines.log` | the exact server.log lines proving the draft folder was used |
| `build-models-draft-k10.log` | the run.sh-deployment copy `models/draft-k10` built and verified by the same builder: 4,639 tensors, 3 shard symlinks, config top-k 10, source snapshot unchanged |
| `mount-check-runsh-models.log` | container with run.sh's mount set only (models→/models, cache→/cache, HF hub at its absolute path, **no workspace root**): `/models/draft-k10` visible, config/index are real files, all 3 shard links readable, tokenizer resolves |
| `image-config-load-check.log` | the v5.4 image's own libraries load the draft folder: `vllm.transformers_utils.config.get_config` returns `Qwen4ExpConfig` with `text_config.num_experts_per_tok=10`, `num_experts=512`, `shared_expert_intermediate_size=1280`; `transformers.AutoTokenizer` returns `Qwen3_5Tokenizer` (vocab 248,044) |
| `container-mount-check.log` | inside a disposable container with exactly the bench mount contract (workspace root + HF hub, both `:ro` at their absolute paths): the draft folder is visible, `config.json` says top-k 10, the index is present, all 3 absolute shard links resolve and are readable, the tokenizer link resolves |
| `recipes-parser-check.log` | all 28 recipes parse through a faithful simulation of run.sh's `rkey`/`rsection`: exactly one `--speculative-config`, no duplicate flags, no `MTP` env export |
| `gen-recipes-check.log` | `gen_recipes.py --check` passes: recipes + matrix.yaml + bench-matrix.yaml consistent |

## Limitations

* Draft-lab performed **no GPU boots**; the draft folders and recipes are offline-verified only. The
  first real proof is the Lead's `draftk10-spec6` boot: expect `qwen38-az-draft-k10` in the draft-model
  log line and quickly rising acceptance.
* DRAFT_FP8 / MBX_DRAFT_HEAD=0 acceptance effects are hardware/runtime facts that only the matrix can
  measure; the README states the expected direction, not a measured number.
* `MBX_B12X_WARM_DRAFTER` value semantics are not documented in the v5.4 image; the recipe is
  exploratory and should be checked against the boot log.
* Absolute symlinks rely on the mount contract above; if the workspace or HF hub is mounted at a
  different path, the draft folder's links will dangle (the builder deliberately uses absolute targets
  because the container is standardised on same-path mounts).

## Ownership / write scopes

draft-lab owns `draft-lab/` and `tools/make_draft_dir.py`. It does not touch `bench/`, `recipes/`,
`run.sh`, `config.env`, `quality/`, `eugr-b12x/`, or the HF cache. Recipes here are *variants for the
matrix*; the promoted winner should later be folded into `recipes/` by whoever owns that directory.
