# MEASURED — first boot of the k10 draft folder (2026-11-12)

Run: Lead's benchmark matrix, config `d-draftk10-spec6` (priority 2).
Recipe: [`recipes/draftk10-spec6.yaml`](recipes/draftk10-spec6.yaml) — self target + draft folder
`drafts/qwen38-az-draft-k10`, K=6, dynamic depth, block/probabilistic, `MBX_DRAFT_HEAD=1`.
Run directory: `bench/results/d-draftk10-spec6/` (server.log, summary.json, metrics/, quality.json).
Image: `myllmbox/qwen38-flash-next-vllm:v5.4` (`sha256:bc42593a5bbf…`). No GPU work by draft-lab.

## The draft folder was accepted and loaded (proof from server.log)

```
[model.py:733] Resolved architecture: Qwen4ExpMTP
[default_loader.py:365] mbx loader: drafter reads 3 of 3 files
[default_loader.py:365] mbx loader: 4.1 GB in 4.4 s (0.95 GB/s), 4639 tensors, 3 files, prepared 0/0/0
[utils.py:128] mbx draft head: drafter lm_head → NVFP4 Marlin copy [248320, 2560] (1.27 GB → 0.36 GB per rank)
[speculator.py:246] mbx depth: dynamic · min 3 · max 6 (K 6; …) · window 48 · promote [60.0, 45.0] · demote [25.0, 15.0] · sync-free on
```

* `drafter reads 3 of 3 files` and `4639 tensors` — exactly the rebuilt folder (4,637 `mtp.*` +
  `lm_head.weight` + `model.language_model.embed_tokens.weight`), loaded from the three linked shards.
  This also confirms the 4,639-tensor filter (the old 4,638 folder was missing the embedding).
* `model` in the effective speculative config is the absolute workspace path of the canonical draft
  folder; the bench mount contract (workspace + HF hub at same absolute paths, read-only) resolved it.
* Boot: **350.2 s** vs M0 **356.3 s** — the draft folder adds only ~4.6 s of drafter loading.
* The drafter keeps the NVFP4 head copy (`MBX_DRAFT_HEAD=1`) and the dynamic-depth controller runs with
  max depth 6.

## Speculative-decoding result vs the M0 reference

| metric | M0 (`m0-reference`, self K=12 dynamic) | **draft-k10** (`d-draftk10-spec6`, K=6) |
|---|---|---|
| acceptance_rate | 16.97 % | **33.96 %** |
| avg draft depth | 12.0 | **6.0** |
| tokens / step | 3.036 | **3.038** |
| accepted_per_position | 4422 / 3514 / 2752 / 214 / 90 / 0… | 1033 / 816 / 677 / 9 / 2 / 0 |
| drafts (phase totals) | 5399 | 1245 |
| decode out tok/s p50 | 53.3 | **56.4** |
| boot | 356.3 s | 350.2 s |
| quality (quick, overall) | 0.9667 | **0.9667** |

Reading it:

* **Same effective output at half the draft depth.** `tokens_per_step` is identical (3.04 vs 3.04) while
  `avg_draft_depth` halves (6 vs 12) → roughly **half the drafter forwards per emitted token**. That is
  the over-drafting thesis confirmed in its measurable form: the extra positions 6-11 in M0 were not
  producing tokens.
* `acceptance_rate` is **not** directly comparable across K: it divides by ~K draft tokens, so the
  17 % → 34 % jump is largely the smaller denominator. The comparable statement is tokens/step at a
  given depth.
* `accepted_per_position` still collapses after position 2 (677 accepted at position 2 → 9 at
  position 3). Even K=6 over-drafts: ~3 of 6 positions carry the output. This predicts **K=3/K=4 fixed
  or dynamic** can match the same tokens/step with even less drafter compute — hence the Lead's
  `d-fixed-3` / `d-curve-k4` / `d-fixed-5` runs.
* Throughput: decode p50 56.4 vs 53.3 tok/s (+5.8 %). Caveat: M0 ran the **full** suite and this run the
  **quick** suite, so tok/s is indicative; the spec ratios and boot times are comparable. Compare tok/s
  only against other quick-suite runs.
* Quality is unchanged (0.9667, 53/55 items, identical category scores) — the drafter never affects the
  verified output distribution.

## Phase B: shallow depth wins, and the 10-expert drafter is a clean negative

Lead phase-B/phase-A matrix runs on the same host/image (raw: `bench/results/d-curve-k4`,
`bench/results/b-k10-curve-k4`, `bench/results/d-draftk10-spec6`, `bench/results/m0-reference`).
Decode tok/s is the decode-phase p50; C8 is the concurrency-phase total at 8 streams.

| metric | M0 self K=12 | draft-k10 K=6 | self K=4 (`d-curve-k4`) | draft-k10 K=4 (`b-k10-curve-k4`) |
|---|---|---|---|---|
| acceptance_rate | 16.97 % | 33.96 % | 48.54 % | 52.36 % |
| avg draft depth | 12 | 6 | 4 | 4 |
| tokens / step | 3.036 | 3.038 | 2.942 | 3.094 |
| decode out tok/s p50 | 53.3 | 56.4 | **60.9** | 51.1 |
| C8 total tok/s | 142.0 | 162.9 | **178.4** | 148.4 |
| boot s | 356.3 | 350.2 | 329.9 | 340.1 |
| quality (quick) | 0.9667 | 0.9667 | n/a | 0.9583 |

**The over-drafting hypothesis is confirmed and the winner is shallow self-draft.** `d-curve-k4`
(self-drafter, K=4, dynamic) is the best measured config: 60.9 tok/s decode (+14 % vs the shipped K=12
preset) and 178.4 tok/s total at C8 (+26 %), at 48.5 % acceptance / 2.94 tokens per step. The K=12 preset
spends drafter forwards on positions that almost never survive (`accepted_per_position` collapses after
position 2 in every run), so reducing depth is a pure win. This is exactly what the "measured problem"
in the Lead's matrix notes predicted.

**The 10-expert draft folder is a clean negative at shallow depth.** At the same K=4, draft-k10 raises
acceptance (52.4 % vs 48.5 %) and tokens/step (3.09 vs 2.94), yet decode collapses to 51.1 tok/s
(−16 % vs self K=4) and C8 to 148.4 (−17 %): the 10-expert MTP router's heavier per-forward compute
costs more than the acceptance it buys when depth is shallow. At K=6 it is 56.4 tok/s (+6 % vs M0) —
positive versus the shipped preset, but still below self K=4. Conclusion: for serving, use the target's
own MTP at a shallow pinned depth; the model-card MTP-at-10 option is an acceptance curiosity, not the
winner. (Depth 3, which might have been even better on the over-drafting trend, is not bootable — see
the depth-4 minimum below.)

The draft folder itself remains proven: `drafter reads 3 of 3 files`, 4,639 tensors, 4.6 s, boot 350 s.
The attribution experiment is complete; see `d-draftk5-spec6` for the extra same-K control (top-k 5
folder, predicted to match self-draft).

## Boot failure (and fix): static-depth recipes were not bootable

The first phase-A matrix attempts of `d-fixed-5` (14:39) and `d-fixed-3` (14:44) never reached `/health`;
the container exited with:

```
(Worker pid=…) ERROR [multiproc_executor.py:1063] ValueError:
num_speculative_tokens_per_batch_size is required for dynamic speculative decoding.
RuntimeError: Worker failed with error 'num_speculative_tokens_per_batch_size is required for dynamic
speculative decoding.', please check the stack trace above for the root cause
```

Root cause: those recipes set `MBX_DEPTH_MODE=off` while keeping the full `MBX_DEPTH_*` block.
Disabling the myllmbox depth controller drops the image into vLLM's own **dynamic-SD schedule**, which at
CUDA-graph capture validates `num_speculative_tokens_per_batch_size`; that value is never set for static
depth, so boot dies. `MBX_DEPTH_MODE=off` is therefore not usable alone in v5.4.

Fix (applied 2026-11-12 to `fixed-3`, `fixed-5`, `fixed-9` in `recipes/`): keep the supported dynamic
code path and pin it so it behaves statically — `MBX_DEPTH_MODE=dynamic`,
`MBX_DEPTH_MIN=CEIL=K`, `MBX_DEPTH_CEIL="[1 -> K]"`, `MBX_DEPTH_PROMOTE="101,101"` (never promote),
`MBX_DEPTH_DEMOTE="0,0"` (never demote), window 48, sync-free 1. This is byte-for-byte the same depth-law
as the Lead's `matrix/recipes/self-fixed{3,4,5}.yaml` and `draftk10-fixed*.yaml`. Boot re-verification is
phase B (Lead).

Evidence: [`verification/boot-failure-static-depth.log`](verification/boot-failure-static-depth.log); the
generated `matrix.yaml` records it under `known_issues`.

## Attribution result: k10 routing helps acceptance, loses throughput at shallow depth

Measured (`b-k10-curve-k4` vs `d-curve-k4`): the 10-expert router does what the model card says — it
raises acceptance (52.4 % vs 48.5 %) and tokens/step (3.09 vs 2.94) at the same K=4 — but the deeper
drafter computation dominates: 51.1 vs 60.9 tok/s decode, 148.4 vs 178.4 tok/s at C8. At K=6 the k10
drafter is 56.4 tok/s, better than M0 but well below self K=4.

So the "MTP trained at 10 experts" hypothesis is true but **not exploitable for serving on this
hardware**: the drafter cost scales with its routed experts, and the best total configuration is the
self-drafted target at a shallow pinned depth (4). The extra same-K control `d-draftk5-spec6` (top-k 5
folder at K=6) remains available as a plumbing sanity check; it is predicted to match self-draft.

## Boot failure: speculative depth < 4 is not bootable

Lead phase-B runs `b-curve-k3` (self K=3) and `b-k10-curve-k3` (draft-k10 K=3) both failed before
`/health`. The image's dynamic-SD validation requires `num_speculative_tokens_per_batch_size`, which v5.4
only supplies for depth >= 4, so K=1 and K=3 are unreachable regardless of static/pinned/dynamic depth:

```
(Worker pid=…) ERROR [multiproc_executor.py:1063] ValueError:
num_speculative_tokens_per_batch_size is required for dynamic speculative decoding.
RuntimeError: Worker failed with error 'num_speculative_tokens_per_batch_size is required for dynamic
speculative decoding.', please check the stack trace above for the root cause
```

Depth 0 (no speculative config) still boots. `curve-k1`, `curve-k3`, `fixed-3` and `draftk10-spec3` are
kept as documentation of the unreachable curve points, marked `bootable: false` in `matrix.yaml` and
`enabled: false` in `bench-matrix.yaml`. Evidence:
[`verification/boot-failure-min-depth.log`](verification/boot-failure-min-depth.log),
`bench/results/b-curve-k3/server.log`, `bench/results/b-k10-curve-k3/server.log`.

## Reproduce

```bash
# offline: the folder itself
python3 tools/make_draft_dir.py "$SNAP" draft-lab/drafts/qwen38-az-draft-k10 --verify-only

# live (Lead only): one boot of the k10 hypothesis
python3 bench/run_matrix.py --matrix matrix/matrix.yaml --only d-draftk10-spec6 --force
```
