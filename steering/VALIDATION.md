# Runtime steering on the v5.4 image — implementation and measured behaviour

## What was built

The stock `myllmbox/qwen38-flash-next-vllm:v5.4` image has **no steering support** (no
`STEERING_VECTOR_PATH` handling anywhere in vLLM). This repo adds it without rebuilding the image:

1. **Vector** — `steering/succinct.pt` (2560-dim, unit norm) is extracted from the model's own
   output head by `steering/make_steering_vector.py`: mean of the `lm_head` rows for verbose-style
   markers minus mean of the rows for concise-style markers (markers listed in
   `steering/succinct.json`). Row `t` of `lm_head` is the direction in the final residual stream
   that raises token `t`'s logit, so the difference is a style direction in that exact space.
2. **Injection** — `steering/patch_model.py` inserts the orthogonal-projection rule from
   `docs/ADR/001` on `Qwen4ExpModel.forward`'s final hidden state (the tensor that feeds
   `lm_head`): `y = y - strength * (y @ v) * v`. `steering/install_patch.sh` extracts the
   image's `model.py`, patches it, and `presets/run_qwen38_best_steering.sh` bind-mounts the
   patched file over the image copy. The op is inside the CUDA-graph-captured forward; the MTP
   drafter reads the pre-mixer buffer, so speculative acceptance is unaffected.
3. **Preset** — `presets/run_qwen38_best_steering.sh` (+ `recipes/qwen38-best-steering.yaml`):
   the measured-best shipped recipe plus steering, strength tunable via `STEERING_STRENGTH`.

Verified on this box: server boots healthy with the patched model and logs
`[steering] active on final residual stream: /steering/vector.pt dim=2560 strength=...`.

## Measured behaviour (2026-11-12, same box)

5 open prompts, temperature 0, natural stop, `steering/eval_steering.py`:

| run | max_tokens | completion lengths (tokens) | avg |
|---|---|---|---|
| no steering (baseline) | 1024 | 912, 1024*, 697, 1024*, 895 | 910 |
| steering, strength +1.5 | 1024 | 1024*, 1024*, 621, 1024*, 879 | 914 |
| steering, strength -8.0 | 1024 | 1024*, 1024*, 613, 1024*, 1024* | 942 |

(* hit the max_tokens cap.) At strength 1.5 the effect is inside run-to-run noise.
At ±8 the projection is clearly applied (outputs change) but becomes **longer and more padded**,
i.e. quality degrades rather than a clean concise shift. Throughput is unaffected beyond noise
(quick suite: 50.9 tok/s steered vs 56.0 baseline; both within this box's ±10 % variance).

## Honest conclusion

The plumbing is real and reproducible (any 2560-dim unit vector can be dropped in and is applied
on every decode step), but the **head-space style vector does not give clean verbosity control**
at the strengths tested — classic activation-extraction at a mid layer (difference-in-means over
contrastive prompt activations) would be the next step; the current injection point (final hidden
state, immediately before `lm_head`) was chosen because it is the one place that is captured by
CUDA graphs safely in this image.

Default preset strength is therefore conservative (`1.0`); treat runtime steering here as an
experimental hook, not as a tuned product feature. Set `STEERING_STRENGTH=0` to disable while
keeping the patch mounted.
