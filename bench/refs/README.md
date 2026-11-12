# Model-snapshot repair (local reference)

The study found the HF cache snapshot of `azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ` with its
`model.safetensors.index.json` **overwritten by a draft-only index** (4,638 tensors: `mtp.*`,
`lm_head`, no embedding) instead of the published 375,487-tensor model index. Every server boot
then failed while loading weights with:

```
RuntimeError: NVFP4 weight_scale for layer 'column_parallel_linear' was never loaded (still NaN).
The checkpoint likely stores this layer as BF16 (not FP4).
```

Cause: an early version of `tools/make_draft_dir.py` wrote its filtered index into the *source*
snapshot path instead of the draft destination. It is fixed in v2.0 (it now refuses destinations
inside/equal to the HF cache or snapshot, checksums the source before/after, and verifies output).

## Repair procedure (if it happens again)

```bash
SNAP=~/.cache/huggingface/hub/models--azampatti--Qwen3.8-Flash-Next-FP4-Local-AZ/snapshots/<sha>
curl -L -o /tmp/index.json \
  "https://huggingface.co/azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ/resolve/main/model.safetensors.index.json"
python3 -c "import json; d=json.load(open('/tmp/index.json')); print(len(d['weight_map']))"  # 375487
# write it into the snapshot (writes outside the workspace need docker or normal permissions)
```

Expected sha256 of the published index:

```
60e0e2d847ca469105f697ae7721227b07b3bda788cf23c67853cad50b10d71a
```

The local copy of the index (`model.safetensors.index.original.json`, 47 MB) and the reference
boot log/container manifest are **gitignored** — this directory only tracks this README.
