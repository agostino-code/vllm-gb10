# Preset study: `azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ` on one DGX Spark (GB10)

**Date:** 2026-11-12. **Hardware:** one NVIDIA DGX Spark / GB10, 128 GB unified LPDDR5X,
aarch64, DGX OS, driver 580.178.04, CUDA 13.0.
**Model:** `azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ` (NVIDIA Qwen3.8-Flash-Next FP4 checkpoint,
5 of 512 experts/token, shared expert widened to 1280; ~102 GB; 262,144-token native context,
409,600 with YaRN).
**Question:** which preset — stack, speculative-draft configuration, KV cache, serving knobs —
is best for this model on this box?

---

## TL;DR — the shipped preset is already the best one measured

**No tested configuration beat the shipped myllmbox v5.4 AZ recipe** (`recipes/qwen38-flash-solo-optimized.yaml`:
K=12 dynamic speculative depth, bf16 KV 23 GiB, 16 seats, block 3456, YaRN 409,600, block/probabilistic
draft verification, NVFP4 draft head). Across ~30 runs covering depth 4–12, self-draft vs the
10-expert draft folder, draft-head on/off, draft-logit scaling, the medium chat template, fp8 KV,
static vs dynamic depth, and the eugr b12x stack:

- **All myllmbox depth variants (K=4, 5, 6, 9, 12) are throughput-equivalent within run-to-run
  noise.** Controlled back-to-back A/B/A under sustained load: **198.5 tok/s at C=8 for both the
  shipped K=12 and the K=4 variant** (soak, 60 s, 0 errors); C=16: shipped 287–292 tok/s vs K=4
  301 tok/s (+2.5 %, inside noise); quick-suite decode: shipped 52.8–55.4, K=4 53.4–56.7 tok/s.
- The shipped preset measured: **~53–55 tok/s single-stream decode, ~157 tok/s at C=8,
  ~290 tok/s at C=16, 199 tok/s sustained at C=8** (thinking off, temp 0), KV pool 750,933
  tokens; quality quick 0.9667.
- **Real, reproducible findings:**
  1. `MBX_DRAFT_HEAD=0` (bf16 draft head instead of the NVFP4 copy) costs **−19 % decode** — keep
     the default (`=1`).
  2. **The eugr b12x stack is slower** with its current recipe: 47.0 tok/s decode, 99.9 tok/s at
     C=8 (vs ~53 and ~157) at `AUTOTUNE=0`; it caps at 7 speculative tokens for this model and
     under-allocates KV (586k vs 751k tokens). Its acceptance is ~3x higher but throughput lower.
  3. The **10-expert draft folder** (`draft-k10`, the model card's MTP-at-10 option) raises
     acceptance (34 % at K=6, 52 % at K=4) but is **not a throughput win**: at K=4 it costs 12 ms
     more per step; at K=6 it is within noise of the shipped preset. Do not use for serving.
  4. **fp8 KV** doubles the KV pool (750,933 → 1,458,424 tokens) for ~8 % less decode — the right
     choice only when you need the extra context headroom.
  5. The **medium chat template** (from the related INT4 checkpoint) is neutral for this
     checkpoint: no acceptance/throughput gain, no quality change.
  6. **`MBX_MTP_DRAFT_SCALE=2` is a no-op at temperature 0** (it only affects sampled decoding).
- **Two image-level boot constraints** (both crash at CUDA-graph capture):
  - `MBX_DEPTH_MODE=off` (static depth) does not boot in v5.4;
  - **speculative depth ≤ 3 does not boot; 4 is the minimum** (and the default depth 12 is fine).
- **One real configuration bug fixed:** `run.sh` appends a *second* `--speculative-config` from
  `config.env` when `MTP != 0`; with the old default `MTP=3` this made the server unbootable on
  this image. `config.env` now defaults to `MTP=0` (recipes carry their own speculative config),
  and `presets/run_qwen38_best.sh` launches the shipped recipe safely.
- **Measurement caveat (important):** this box is shared with a desktop session and agent
  processes. Single-run quick measurements vary by **±10–15 %** — enough to produce a spurious
  "+14 % from depth 4" in the first scan that did **not** reproduce in controlled A/B/A runs
  (it became −3 % in the next pair). Only same-window comparisons or sustained-load averages are
  trustworthy; that is how the conclusions above were reached.

**Recommendation:** keep the shipped preset. Use `presets/run_qwen38_best.sh`.
Choose `recipes/qwen38-k4.yaml` only if you want lower draft compute at equal throughput
(it is not faster). For >750k-token KV capacity use `matrix/recipes/az-kvfp8-spec7.yaml`.
Do not use eugr b12x unless you need its lower host-memory footprint.

---

## 1. What was broken before the study (and how it was fixed)

The workspace could not boot the model at all when the study started:

1. **The model snapshot's `model.safetensors.index.json` had been overwritten** (~05:38) by an
   early draft-folder builder, which wrote its 4,638-tensor draft index into the model snapshot.
   Every boot died in weight loading with
   `NVFP4 weight_scale for layer 'column_parallel_linear' was never loaded (still NaN)`.
   Restored the published 375,487-tensor index (backup:
   `bench/refs/model.safetensors.index.original.json`, sha256 `60e0e2d8…`).
2. `config.json` had been edited with an `exclude_modules` list; this vLLM build reads `ignore`
   for that format, so the edit was ineffective and was reverted to the published config.
3. The draft builder was fixed permanently (`tools/make_draft_dir.py` v2.0: refuses destinations
   inside/equal to the HF snapshot or HF cache, marker-protected rmtree, before/after checksums,
   verifies every tensor against its shard header).

After the repair the shipped recipe boots in ~6 min and serves correctly (quality-verified).

## 2. Test setup

| | myllmbox stack | eugr b12x stack |
|---|---|---|
| image | `myllmbox/qwen38-flash-next-vllm:v5.4` (sha256 `bc42593a…`, vLLM 0.32.0) | `eugr/spark-vllm-b12x:latest` (vLLM 0.1.dev21830) |
| loader | `--load-format mbx` | `--load-format b12x` (O_DIRECT) |
| speculative | MTP, dynamic draft depth, NVFP4 draft head, fused multi-step draft | fixed-depth MTP only |
| PLE table | 31 GB NVMe map in `./cache/ple-nvme` | read from checkpoint via io_uring (no map) |
| other | RecoverSSM, 16 seats | b12x autotune **off** (tuned first boot ≈55–65 min); depth capped at 7 |

**Harness** (`bench/`, written for this study): launches a recipe as a docker container, waits for
`/health`, records image digest, exact argv/env/mounts, boot time and the server log, runs a
standardized suite (decode with 4 prompt categories at temp 0/thinking off, concurrency 1/2/4/8
(+16), prefill 1k/8k/32k, 128k needle, `/metrics` spec-decode deltas), then stops the container.
A client-vs-server token counter cross-check flags external load (all accepted runs = 1.000).
Results: `bench/results/<config>/{summary.json,requests.jsonl,server.log,meta.json}`;
`bench/aggregate.py` → `RESULTS.md` + `summary.csv`.

**Quality** (`quality/`): math (20), JSON (10), code with hidden unit tests (8), instruction
following (10), tool calling (6), repetition, and full-profile long-context needle (32k/128k/262k).
Deterministic, rule-based; M0 quick = **0.9667**; 133 offline tests.

Host caveat: desktop session + agent processes share the box. Absolute numbers sit ~35 % below
myllmbox's published figures for the same preset (published: 86.9 tok/s code c=1 vs our 53–55
median across categories); we could not attribute that gap to any configuration knob (all variants
are affected equally). CPU contention from the shared host (vLLM is pinned to 10 cores, but host
processes can still run there) and page-cache pressure (96 GB weights + 23 GB KV leaves <5 GB for
the 26.8 GB PLE table) are the likely contributors. **Relative comparisons in the same window are
valid; absolute values are box-specific.**

## 2b. Why myllmbox's published numbers look higher than our benchmark median

myllmbox publishes per-prompt averages of 3 **full answers** (thinking off): code (TypeScript)
c=1 **86.9 tok/s**, mixed c=1 **60.5**, thinking-on **56.5** (per their README). Our screening
number "~53 tok/s" is a **median across four prompt categories at 256-token completions** — a
cross-category, short-window metric, dominated by prose/Q&A. Throughput on this model is strongly
content-dependent because draft acceptance is:

| measurement here (same shipped recipe, this box) | decode tok/s | acceptance | tokens/step |
|---|---|---|---|
| long pure-code answer (Python RB-tree module, 2048 tok, 3 reps) | **70.1 median** (69.7–71.5) | 38 % | **5.6–5.7** |
| long TypeScript debounce answer (1024 tok, 3 reps) | 65.0 median (61.6–72.2) | 30 % | 4.6–5.2 |
| TypeScript code + prose explanation (1024 tok) | 45.0 median | 16 % | 2.9 |
| thinking-on (reasoning phase only, 3072-token budget) | 51.6 | 21 % | 3.5 |

The K=4 variant was re-measured on the same code content: py-code 67.6 tok/s (acceptance 84 %,
4.4 tok/step), ts-code 65.0 (acceptance ~80 %, 4.2 tok/step). So on code-only content the shipped
K=12 is equal-to-slightly-better (70.1 vs 67.6 on py-code; identical 65.0 on ts-code): K=12 buys
more tokens/step (5.6 vs 4.4) and pays for it with more drafts per step. The choice remains a tie,
with the shipped preset never worse.

So the published 80+ tok/s class is **reproducible on this box for code-only content** (70 tok/s,
79 % of their 86.9); the remaining 10–25 % is host environment: their figures are from a quiet box
(`vm.compaction_proactiveness=0`, no desktop/agent load), ours shares the box with a desktop
session and agent processes (measured GPU clocks 2522/3003 MHz, power 35–38 W, temp 47–62 °C
under full load). Their own table shows the same prompt dependence (mixed 60.5 vs code 86.9), which
confirms the mechanism is content/tokens-per-step, not a different configuration.
Our config is their shipped recipe (K=12 dynamic, bf16 KV, 16 seats) — unchanged.

## 3. Reference measurement (shipped recipe, K=12 dynamic)

| metric | value |
|---|---|
| decode median (temp 0, thinking off) | **52.8–55.4 tok/s** across clean runs (code ~54, JSON ~60, math ~52, QA ~41) |
| TTFT / TPOT p50 | ~170–190 ms / ~18.2–19.5 ms |
| concurrency (suite) | C1 ~49 · C2 ~67–76 · C4 ~101–114 · C8 ~152–160 · C16 ~198–203 tok/s |
| sustained load (60 s, 8 streams) | **198.5 tok/s** (also 190.9 in the A/B repeat) |
| sustained load (60 s, 16 streams) | 287–292 tok/s |
| prefill | 1k ~1108 · 8k ~2235–2397 · 32k ~2278–2390 tok/s |
| 96k needle | TTFT ~41–43 s, prefill ~2215–2349 tok/s, recall OK |
| speculative | acceptance 16–17 %, 2.9–3.1 tokens/step, depth 12 |
| KV pool | 750,933 tokens, 1.83x max concurrency at 400k |
| boot / quality | ~330–360 s / 0.9667 |

## 4. Experiments

### 4.1 Depth and drafter scan (single-run quick protocol — screening only)

| config | spec K | decode tok/s | acceptance | tok/step | C4 | C8 |
|---|---|---|---|---|---|---|
| shipped (K=12 dynamic) | 12 | 53.3 | 17 % | 3.04 | 103.6 | 142.0 |
| K=4 self (curve-k4) | 4 | 60.9 | 49 % | 2.94 | 116.1 | 178.4 |
| K=5 self | 5 | 50.7 | 39 % | 2.97 | 91.4 | 146.4 |
| K=6 self | 6 | 55.7 | 33 % | 2.98 | 102.8 | 155.4 |
| K=9 (dynamic ceiling) | 9 | 50.7 | 22 % | 2.95 | 91.9 | 147.8 |
| K=6 + draft-k10 (10-expert) | 6 | 56.4 | 34 % | 3.04 | 68.3* | 162.9 |
| K=4 + draft-k10 | 4 | 51.1 | 52 % | 3.09 | 107.4 | 148.4 |
| K=12, `MBX_DRAFT_HEAD=0` | 12 | 43.2 | 16 % | 2.94 | 85.7 | 127.1 |
| K=12, `MBX_MTP_DRAFT_SCALE=2` | 12 | 51.8 | 17 % | 3.00 | 97.7 | 142.9 |

\* cold-start stall in the single-wave quick profile.

**This table was the trap.** It suggests K=4 is +14 % and deeper is worse. Controlled repeats
(§4.2) show the ordering is not stable: adjacent runs of the *same* configuration differ by
±10–15 % under host load. The **only** robust statements from this scan are the *mechanism*
values — acceptance, tokens/step, draft depth — which are measured from exact counters, and the
draft-head effect, which is far larger than the noise.

### 4.2 Controlled A/B/A (same conditions, interleaved)

| run | decode tok/s | C4 | C8 (suite) | sustained C8 (60 s) | sustained C16 (60 s) |
|---|---|---|---|---|---|
| shipped — A | 52.8 | 101.1 | 156.8 | 198.5 | 292.4 |
| K=4 | 56.7 | 103.1 | 158.1 | 198.5 | 300.7 |
| shipped — B | 54.6 | 104.9 | 155.5 | 190.9 | 287.1 |

Shipped mean: decode 53.7, C4 103.0, C8 156.2, sustained C8 194.7, C16 289.8.
K=4: decode 56.7 (+5.5 %, inside the shipped spread), C4 103.1, C8 158.1, sustained C8 **198.5**,
C16 300.7 (+3.8 % vs shipped mean, inside noise).
An earlier A/B/A pair under the same protocol gave the opposite decode ordering
(shipped 55.4/54.9 vs K=4 53.4). **Conclusion: depth 4 and depth 12 are throughput-equivalent.**

Why: at ~3 tokens/step both configs produce the same useful output; the K=12 config proposes more
drafts but accepts fewer (17 % vs ~49 %), and the fused multi-step drafter makes the extra drafts
nearly free on this stack — step time is ~equal (≈54 ms shipped vs ≈55 ms K=4), so throughput is
equal. Depth tuning shifts *draft compute*, not *delivered throughput*.

### 4.3 Serving knobs

| config | change | decode | C8 | quality | verdict |
|---|---|---|---|---|---|
| fp8 KV (`az-kvfp8-spec7`) | kv-cache-dtype fp8, K=7 | 48.8 (−8 %) | 140.2 | 0.9583 (1 math item flip) | **KV pool 1,458,424 tokens** (1.94x) — use only for context headroom |
| medium chat template | `matrix/templates/medium_chat_template.jinja` | 49.2 (−8 %) | 139.0 | 0.9667 | no gain; rejected |
| draft scale 2 | `MBX_MTP_DRAFT_SCALE=2` | 51.8 (−3 %) | 142.9 | — | no-op at temp 0 |
| k10 + K4 + medium | — | 50.8 | 138.3 | 0.9667 | no gain |

### 4.4 eugr b12x stack

All runs `AUTOTUNE=0`, model mounted read-only, PLE via io_uring.

| config | boot s | decode tok/s | C4 | C8 | acceptance | tok/step | KV tokens |
|---|---|---|---|---|---|---|---|
| base MTP=4, fp8 KV, block 16, 8 seats, util 0.75 | 483 | 47.0 | 78.2 | 99.9 | 51.1 % | 3.04 | 586,074 |
| MTP=6 | 365 | 42.4 | 64.2 | 89.9 | 37.8 % | 3.27 | 547,533 |
| MTP=7 | 363 | 41.0 | 64.5 | 81.5 | 35.4 % | 3.48 | 524,442 |
| MTP=9 | boot fails | b12x GDN cap: at most 8 state-index columns → max depth 7 | | | | | |

The b12x stack serves the same checkpoint with **identical quality** (quick battery 0.9667, 0 item
flips vs the myllmbox M0) and needs far less host memory (no 59 GB PLE map), but it is clearly
slower with this recipe: C=8 about 100 tok/s vs ~157 for myllmbox. Fixed depth 4 is its best point
(deeper loses), matching the myllmbox finding that more drafts do not pay off.
A tuned eugr preset should raise `--gpu-memory-utilization` to 0.80–0.85 (KV ~700k+ tokens);
full b12x autotune was not measured (55–65 min first boot, expected single-digit %).

### 4.5 Quality validation (acceptance gate)

| config | profile | score | vs shipped (shared categories, `--noise 0`) |
|---|---|---|---|
| shipped preset | quick | 0.9667 | baseline |
| K=4 variant | full | **0.9714** | PASS — 0 item flips (needle battery accounts for the higher overall) |
| eugr b12x base | quick | 0.9667 | PASS — 0 item flips |
| K=6 draft-k10 / k10-K4-medium / medium-template | quick | 0.9667 | PASS |
| K=4 draft-k10 / fp8 KV | quick | 0.9583 | OK at ±1 item (same single math item, ambiguity) |

K=4 long-context (full profile): **3/3 strict needle recall at 32k/128k/262k** (TTFT 12.4 / 51.7 /
105.0 s, actual prompt tokens 31.1k / 124.1k / 248.1k, validated against `max_model_len=409600`).
No hard-category regression in any measured configuration.

## 5. What actually moves throughput on this box

- **Not the draft depth**, between 4 and 12: acceptance and draft count trade off almost exactly;
  sustained throughput is equal (198.5 vs 198.5 tok/s at C=8). The dynamic-depth machinery is fine
  as shipped.
- **The NVFP4 draft head is essential**: disabling it costs 19 % decode. It is the one draft knob
  that clearly matters.
- **The eugr loader/kernels cost ~30 %** at C=8 relative to the myllmbox image with this model.
- **KV dtype** is a capacity/throughput trade: fp8 doubles the pool, costs ~8 % decode.
- **Quality is insensitive** to all serving/draft knobs tested (speculative decoding preserves the
  model distribution; fp8 KV and the medium template did not shift scores beyond item noise).

## 6. Boot constraints discovered (image-level)

1. `MBX_DEPTH_MODE=off` with the dynamic env block present → server dies at CUDA-graph capture:
   `num_speculative_tokens_per_batch_size is required for dynamic speculative decoding`.
   Static depth must be emulated on the dynamic path (MIN=CEIL=K, PROMOTE=101, DEMOTE=0).
2. **`num_speculative_tokens ≤ 3` fails the same way** — depth 4 is the minimum bootable, on both
   the static and dynamic paths (depth 0 / no speculation is fine).
3. eugr b12x: depth ≥ 8 fails (`state_index_columns must be at most 8`) → max 7.
4. `run.sh` + `config.env MTP=3` would append a duplicate K=3 flag and hit constraint 2; fixed by
   defaulting `MTP=0` in `config.env` (recipes already carry one speculative-config).

## 7. Recommendation

### 7.1 Default: keep the shipped preset
```bash
presets/run_qwen38_best.sh          # safely launches the shipped recipe (MTP=0, stock template)
# equivalent, explicit:
MTP=0 ./run.sh recipes/qwen38-flash-solo-optimized.yaml
```
It is the best measured configuration on this box: ~53–55 tok/s decode, ~199 tok/s sustained at
C=8, ~290 tok/s at C=16, 750,933-token KV pool, quality 0.9667, 128k needle OK.

### 7.2 Alternative with equal throughput, lower draft compute
`recipes/qwen38-k4.yaml` (K=4). Measured throughput-equal to the shipped preset within noise
(one A/B favored it by 5 %, the other disfavored it by 3 %; sustained C=8 identical). It halves
the drafter forwards per step (~49 % vs 17 % acceptance at ~3 tokens/step), so it burns less GPU
work for the same output — but it is **not faster**. Use it if you value the lower draft overhead
or run in a power/thermal-constrained setting; otherwise stay with the shipped preset.

### 7.3 Long-context variant
If you need more than ~750k tokens of KV: `matrix/recipes/az-kvfp8-spec7.yaml` (fp8 KV,
1,458,424 tokens = 1.94x, decode −8 %, C8 near parity). Keep bf16 KV otherwise.

### 7.4 eugr stack
Only if its image or lower host-memory footprint matters: use `MTP=4`, raise
`--gpu-memory-utilization` to 0.80–0.85, and expect ~30–35 % less throughput at C=8 than
myllmbox. Never exceed 7 speculative tokens.

### 7.5 Do not
- Disable `MBX_DRAFT_HEAD` (−19 %).
- Use `MBX_DEPTH_MODE=off` or speculative depth ≤ 3 (does not boot).
- Use the 10-expert draft folder for serving (equal or worse throughput at higher cost).
- Expect the medium template to add throughput for this checkpoint.

## 8. Reproduction

```bash
# preflight / dry run
python3 bench/check_recipes.py matrix/matrix.yaml
python3 bench/run_matrix.py --matrix matrix/matrix.yaml --dry-run

# screening matrices (all data behind §4.1-4.3)
python3 bench/run_matrix.py --matrix matrix/matrix.yaml        # phase A
python3 bench/run_matrix.py --matrix matrix/matrix-phaseB.yaml # phase B
# controlled comparisons (the decisive runs behind §4.2)
bash matrix/ab.sh && bash matrix/ab_load.sh
# finalist validation (full suite + full quality + temp 0.6 + soak)
bash matrix/finalist_phase.sh
# aggregate
python3 bench/aggregate.py --results-dir bench/results   # RESULTS.md + summary.csv

# eugr stack
cd eugr-b12x && AUTOTUNE=0 ./launch.sh --wait
python3 ../bench/suite.py --profile quick --concurrency 1,2,4,8 ...
```

Raw results: `bench/results/` — 30+ run directories with `summary.json`, `requests.jsonl`,
`server.log`, `meta.json`; `RESULTS.md` (auto-ranked), `summary.csv`, `summary_all.json`.
Quality artifacts: `quality/runs/` and `bench/results/*/quality.json`.
Supporting evidence: `matrix/FINDINGS.md` (chronological log incl. the noise reversal),
`draft-lab/MEASURED.md`, `eugr-b12x/runs/RESULTS.md`, `bench/README.md`.

## 9. Limitations

- **Host noise.** Shared desktop + four agent processes: single-run quick measurements vary by
  ±10–15 %. The initial scan's +14 % depth claim did not reproduce under controlled A/B/A; all
  conclusions above are from same-window or sustained-load comparisons. Deltas below ~5 % are not
  meaningful here.
- **Absolute numbers are depressed** (~35 % below myllmbox's published figures for the same
  preset) by the shared host environment; relative comparisons are valid.
- Single box / single GPU; one measurement per configuration per suite profile unless stated.
- Ranking used temperature 0 (greedy); the K=4 alternative was re-checked at 0.6 (acceptance 50 %,
  decode −4 % vs greedy). Sampling-time knobs (`MBX_MTP_DRAFT_SCALE`, draft-sample method) can only
  be judged at temperature > 0; they made no difference at 0.
- eugr b12x measured with `AUTOTUNE=0` (tuned first boot 55–65 min; single-digit % expected).
- `MBX_DRAFT_HEAD2` (two-stage draft head) is unavailable for this checkpoint (image ships basis
  files only for the NVIDIA and hibrid48 checkpoints).
- Long-context needle prompts are synthetic; TTFT includes client-side/queueing latency.
- Untested (in the matrix for follow-up): native 262k context, `MBX_MXFP8_A16=1`, seats=8,
  batched-tokens=4096, PLE prewarm, cpuset variants.
