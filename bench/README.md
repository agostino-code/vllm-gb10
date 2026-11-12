# bench/ — reproducible vLLM preset benchmark harness

Measurement backbone for the **azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ** preset
study on a single DGX Spark / GB10. It launches a vLLM container from a
recipe-like YAML, runs a standardized measurement suite against the
OpenAI-compatible API, records Prometheus spec-decode accounting, and produces
machine-readable + ranked results.

Everything is stdlib-only Python 3 (PyYAML is used when present; a small
fallback parser handles the flat recipe/matrix YAML). No venv, no pip install.
**No code path depends on host `nvidia-smi`** (it is broken in this environment):
container state is read via `docker inspect`, health via HTTP.

```
bench/
  serve.py            launch/adopt/stop a vLLM container from a recipe, wait /health
  suite.py            standardized measurement suite against a live server
  aggregate.py        merge results/*/summary.json -> summary.csv + RESULTS.md
  run_matrix.py       sequential, resumable matrix runner (boot -> suite -> stop)
  matrix.yaml         matrix template + example configs (Lead owns the full matrix)
  check_recipes.py    static preflight of every recipe / matrix entry (no GPU)
  recompute_metrics.py re-derive spec-decode deltas from saved raw /metrics text
  common.py           shared HTTP/docker/metrics/log/stats helpers
  prompts.py          deterministic prompt sets + long-context/needle generators
  results/            <run-id>/ outputs (gitignored content, safe to delete manually)
```

Read-only inputs: `run.sh`, `config.env`, `recipes/`. The harness never modifies
them. Write scope: `bench/` + `tools/benchmark.py` only.

---

## Quickstart

### Measure a server that is already running (no GPU lifecycle action)

```bash
cd /home/pgx/Documenti/vllm-gb10
# optional: record the launch manifest (image digest, env, argv, boot log) without touching it
python3 bench/serve.py adopt --name qwen38-flash-next --port 8000 \
    --run-id m0-reference-full --config-id m0-reference

# quick profile (~1.5-2.5 min) or full profile (~4-5.5 min)
python3 bench/suite.py --base-url http://127.0.0.1:8000 \
    --run-dir bench/results/m0-reference-full --config-id m0-reference --profile full
```

### Launch a container yourself, measure, stop

```bash
python3 bench/serve.py up --recipe matrix/recipes/az-ref.yaml \
    --run-id az-ref --config-id az-ref --boot-timeout 1500
python3 bench/suite.py --base-url http://127.0.0.1:8000 \
    --run-dir bench/results/az-ref --config-id az-ref --profile full
python3 bench/serve.py down --run-dir bench/results/az-ref
```

`serve.py up` writes the manifest to `results/<run-id>/meta.json`, waits for
`/health` (default 1200 s, configurable), copies the full container log to
`server.log`, and parses KV-pool/max-concurrency from it. `down` removes the
container and marks the manifest stopped.

### Unattended matrix run

```bash
# validate everything first: recipes parse, no duplicate spec flags, mounts OK
python3 bench/check_recipes.py matrix/matrix.yaml

# print the exact commands, launch nothing
python3 bench/run_matrix.py --matrix matrix/matrix.yaml --dry-run

# real run — launch it in a background job; it takes boot+suite per config sequentially
nohup python3 bench/run_matrix.py --matrix matrix/matrix.yaml > bench/results/matrix.out 2>&1 &

# resume / slice
python3 bench/run_matrix.py --matrix matrix/matrix.yaml --only az-ref
python3 bench/run_matrix.py --matrix matrix/matrix.yaml --start-at az-seqs8
python3 bench/run_matrix.py --matrix matrix/matrix.yaml --force        # redo done configs
```

Runner behaviour:

* strict sequential order (`priority`, then name); refuses to boot while any
  other model server is running and while the port is serving traffic
  (`--force` overrides the "other container" check only).
* for each config: `serve.py up` → `suite.py` → `serve.py down`, always stopping
  the container (even after a suite failure/timeout).
* **resumable**: `results/<name>/status.json` holds
  `pending|booting|running_suite|done|suite_failed|failed|timeout|interrupted`.
  A `done` config is skipped on the next invocation. A stale `bench-<name>`
  container from a crashed run is removed and the config retried.
* **timeouts**: `defaults.per_config_timeout_s` (default 2400 s) is a hard cap
  covering boot + suite; on expiry the container is stopped and the config is
  marked `timeout`; the runner continues with the next config (use
  `--fail-fast` to stop at the first failure). `defaults.boot_timeout_s` is
  passed to `serve.py` (default 1500 s).
* at the end it runs `aggregate.py` automatically (`--no-aggregate` to skip),
  producing `bench/results/summary.csv`, `summary_all.json`, `RESULTS.md`, and
  `matrix.log` with the full output of every serve/suite subprocess.

---

## `serve.py` — launch manifest from a recipe

```
python3 bench/serve.py up --recipe <yaml> [options]
python3 bench/serve.py down --name <container> | --run-dir <results/run-id>
python3 bench/serve.py status --name <container> | --port 8000
python3 bench/serve.py adopt --name <running-container> --run-id <id>
```

Recipe formats supported:

* **native** (this repo's `run.sh` format): `server:` (`image`, `port`,
  `model`, `models_dir`, `cache_dir`, `cpuset`, `docker_args`, `patches`),
  `env:` (mapping), `vllm:` (mapping -> CLI flags; `true` = bare flag,
  `false`/empty = omitted).
* **eugr command format**: `defaults:` + `env:` + a `command: |` template that
  is rendered and executed (`{placeholder}` substituted, `{{`/`}}` literal
  braces). Mods from the eugr repo are **not** applied — use
  `eugr-b12x/launch.sh`/`run-recipe.sh` when a recipe needs mods; the harness
  recipe support is for recipes whose command is self-contained.

Options (most useful):

| flag | meaning |
|---|---|
| `--name` | container name (default `bench-<recipe stem>`); never touches `qwen38-flash-next` unless you name it |
| `--image` | image override (digest + repo digests recorded in meta) |
| `--port` | host port (container always serves 8000 internally) |
| `--run-id` / `--run-dir` / `--config-id` | results directory / identity |
| `--set vllm.<key>=<value>` | recipe override, e.g. `--set vllm.max-num-seqs=8`; `server.<key>` / `env.<KEY>` too |
| `--env KEY=VALUE` | extra/overriding container env |
| `--mtp N` | replace the recipe speculative config with `{"method":"mtp","num_speculative_tokens":N}` (0 = disable); extra keys of the recipe config are preserved |
| `--speculative-config JSON` | full replacement config; **always replaces, never appends** (no duplicate-flag bug) |
| `--chat-template medium\|xhigh\|model\|/path.jinja` | mount the template read-only at `/chat_template.jinja` and pass `--chat-template`; recorded in meta |
| `--cpuset`, `--docker-arg`, `--mount host:container[:ro]` | docker-level overrides |
| `--extra-vllm-arg "..."` | raw extra vllm argv (advanced; not deduplicated) |
| `--boot-timeout`, `--poll`, `--settle`, `--keep-failed`, `--force`, `--dry-run` | boot control |

Safety and reproducibility rules built into `serve.py`:

* refuses to start when the port is healthy/open or another model-server-looking
  container is running (never two model servers on one GB10);
* `--dry-run` prints the full docker argv/env/mounts/spec config and launches
  nothing (side-effect free);
* container is started with `--gpus all --ipc=host`, the recipe's
  `docker_args`/`cpuset`, `-p <port>:8000`, and these mounts:
  * `./models -> /models`, `./cache -> /cache` (rw, recipe `models_dir`/`cache_dir`),
  * `<HF_HOME>/hub -> <HF_HOME>/hub` **ro** at the same absolute host path,
  * the resolved model snapshot -> `/models/<basename>` (HF cache snapshot when
    available, else `./models/<basename>`),
  * **the repo root -> the same absolute host path, read-only** (mount contract
    for draft folders / absolute `speculative-config` paths),
  * `./cache/container-cache/b12x -> /root/.cache/b12x` and
    `.../flashinfer -> /root/.cache/flashinfer` when present (boot caches),
  * optional chat-template mount;
* standard env: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-workspace`,
  `VLLM_CACHE_ROOT=/cache/vllm-cache` (recipe `env:` wins on duplicates);
* boot time = `docker run -d` → first 200 on `/health`; recorded with poll count.

`meta.json` records: recipe path + sha256 + format, image tag/id/repo-digests,
exact docker argv (`launch.argv`, plus a copy-paste shell string), env, mounts,
cpuset, vllm argv and parsed knobs (spec config, max-num-seqs, max-model-len,
kv-cache-dtype, chat template, load-format), model repo/host dir/source,
host facts, timing (requested/started/ready/boot seconds), and the parsed
startup log (KV pool tokens, KV memory GiB, max concurrency, block size,
weights/graph/engine timings, PLE table path).

---

## `suite.py` — measurement suite

```
python3 bench/suite.py --base-url http://127.0.0.1:8000 \
  [--run-dir bench/results/<id>] [--config-id <id>] [--profile quick|full] \
  [--concurrency 1,2,4,8] [--include-16] [--prefill-targets 1024,8192,32768] \
  [--long-smoke|--no-long-smoke] [--quality off|quick|full] ...
```

Profiles (all overridable per flag):

| | quick | full |
|---|---|---|
| warmup (discarded) | 2 | 2 |
| single-stream decode | 4 prompts (1/category) × 2 repeats, 192 tok cap | 8 prompts (2/category) × 3 repeats, 256 tok cap |
| concurrency | C=1,4 × 1 wave | C=1,2,4,8 × 2 waves |
| prefill TTFT | ~1k, ~8k tokens | + ~32k tokens |
| 128k needle smoke | off | on (one request, `--long-target-tokens 131072`) |
| duration | ~1.5-2.5 min | ~4-5.5 min |

Phases and what is reported:

1. **warmup** — discarded, absorbs first-request transients (PLE page cache,
   cudagraph warmup). Never in the aggregates.
2. **decode** — single-stream chat completions per prompt/repeat. Metrics:
   `out_tok_s` (median and p50/p90/p99/min/max/stdev) where
   `out_tok_s = usage.completion_tokens / (last content chunk − first content
   chunk)`; `ttft_ms` (client request start → first content chunk);
   `tpot_ms = (last−first)/(completion_tokens−1)`; `tokens_per_chunk`
   (multi-token MTP SSE chunks — explains chunk-vs-token accounting gaps);
   per-category breakdown (qa/code/json/math).
3. **concurrency** — C requests in parallel at a barrier, 2 waves per level,
   fixed prompt pool. Per level: `total_out_tok_s = Σ completion_tokens / Σ
   wave wall` and per-stream decode tok/s, TTFT stats, errors, plus background
   `/metrics` sampling of peak `num_requests_running`, `num_requests_waiting`,
   `kv_cache_usage_perc`.
4. **prefill** — deterministic generated prompts of ~1k/8k/32k target tokens,
   `max_tokens=1` (TTFT ≈ prefill); reports actual `prompt_tokens`, TTFT,
   `prefill_tok_s = prompt_tokens/TTFT`, cached tokens (should be 0: a unique
   nonce makes the prefix cold even with prefix caching enabled).
5. **long_context** (full profile) — ~128k-token prompt with a needle at ~45 %
   depth; checks the answer contains the code, reports TTFT/prefill tok/s.
6. **metrics** — a raw Prometheus snapshot before/after every phase, saved to
   `metrics/<phase>_{before,after}.txt`. Summary stores per-phase and overall
   deltas: `acceptance_rate = accepted/draft_tokens`,
   `tokens_per_step = emitted/drafts` (or, when an image does not export
   `spec_decode_num_emitted_tokens_total`, derived as
   `(accepted + drafts)/drafts` — accepted + one bonus token per step),
   `avg_draft_depth = draft_tokens/drafts`, and accepted-per-position.
   Missing metric names degrade to `null`; the suite never fails because of
   /metrics.
7. **server-side token cross-check** — per phase, the client token sums are
   compared with the delta of `vllm:generation_tokens_total` /
   `vllm:prompt_tokens_total`. A ratio outside 0.75-1.35 adds a warning: it
   means either accounting drift or (most likely) **another client hitting the
   same server**, i.e. a contaminated run.

Robustness: every request has a timeout (default 300 s, 600/900 s for
prefill/long) and one retry; a failure is recorded in `requests.jsonl` and
`summary.json.errors` and the suite continues. Phase failures are recorded as
`partial`/`failed` instead of crashing; exit code 0 = all core phases ok,
2 = a core phase failed, 3 = server not reachable / no model.

`--quality quick|full` invokes `./quality/run_quality.py` if it exists (after the
performance phases), stores return code/stdout/scores in `summary.json.quality`
and `quality.json`; failure is non-fatal unless `--strict-quality`.

### Output layout

```
bench/results/<run-id>/
  meta.json          launch manifest (serve.py up/adopt)
  suite.json         suite version/command/profile/timestamps
  summary.json       aggregated results (source of truth for aggregation)
  requests.jsonl     one JSON line per request: phase, timings, tokens,
                     decode tok/s, TPOT, per-chunk ITL, errors/attempts
  metrics/*.txt      raw Prometheus snapshots (before/after each phase)
  server.log         full container log copied by serve.py (if any)
  quality.json       optional quality-hook output
  status.json        matrix runner state (matrix runs only)
```

Key `summary.json` fields (stable schema, `schema_version: 1`):

```
phases.decode.out_tok_s_median / .out_tok_s.{p50,p90,p99} / ttft_ms_p50 /
             tpot_ms_p50 / tokens_per_chunk / by_category.* / n_ok / n_errors /
             server_generation_tok_s / client_to_server_token_ratio
phases.concurrency.levels[] {c,total_out_tok_s,per_stream_tok_s,ttft_ms,
             waves[],n_ok,n_errors}; peaks.{peak_num_requests_running,
             peak_kv_cache_usage_perc}
phases.prefill.by_target["1024"|"8192"|"32768"].{prompt_tokens,ttft_ms,
             prefill_tok_s,cached_tokens}
phases.long_context.{prompt_tokens,ttft_ms,prefill_tok_s,needle_ok}
metrics.overall.{acceptance_rate,tokens_per_step,tokens_per_step_source,
             avg_draft_depth,drafts,draft_tokens,accepted_tokens,
             accepted_per_position}
metrics.per_phase.<phase>.{...same spec fields...,server_generation_tok_s,
             client_to_server_ratio}
server.{image,image_id,stack,recipe,vllm.{speculative_config,
             num_speculative_tokens,max_num_seqs,max_model_len,kv_cache_dtype},
             kv_cache.{tokens,num_gpu_blocks,block_size,max_concurrency,
             max_num_seqs},boot_seconds}
errors[] / warnings[] / timing.duration_s / harness.suite_version
```

---

## `aggregate.py` — ranked comparison

```bash
python3 bench/aggregate.py                                  # bench/results -> csv+md
python3 bench/aggregate.py --results-dir matrix/results --md matrix/RESULTS.md
python3 bench/aggregate.py --run-dir bench/results/<id> --no-md
```

Reads only `summary.json` (+`status.json`), never contacts the server. Writes
`summary.csv` (flat, fixed columns), `summary_all.json`, and `RESULTS.md` with
ranked tables: single-stream decode, concurrency scaling, speculative-decode
efficiency (acceptance, tok/step, tok/chunk), prefill/long-context, boot/health,
and failed runs.

---

## `matrix.yaml` — matrix schema

`bench/matrix.yaml` is the template; the Lead authors the full matrix from it.
Entries are executed by `run_matrix.py`.

```yaml
defaults:
  results_dir: bench/results        # <results_dir>/<name>/ per config
  port: 8000
  boot_timeout_s: 1500              # passed to serve.py (checkpoint boots ~6-9 min)
  per_config_timeout_s: 2400        # hard cap for boot+suite per config
  stop_server: true                 # always stop container after the config
  suite_profile: full               # quick|full default for all configs
  chat_template: ""                 # default template for all configs
  image: ""                         # default image override
  env: {}                           # container env for all configs
  mounts: []                        # extra "host:container[:ro]" for all configs
  docker_args: []                   # extra docker run flags for all configs
  suite: {}                         # suite.py overrides for ALL configs (profile <- this <- config.suite)
  keep_failed: false                # leave a container behind when boot fails

configs:
  - name: az-ref                    # required, unique, [A-Za-z0-9_.-]+
    recipe: matrix/recipes/az-ref.yaml   # required, repo-relative
    priority: 10                    # lower first (default: file order)
    image: myllmbox/qwen38-flash-next-vllm:v5.4   # optional override
    port: 8000                      # optional override
    chat_template: matrix/templates/medium_chat_template.jinja
    set:                            # recipe overrides: <section>.<key>: value
      vllm.max-num-seqs: "8"
      vllm.kv-cache-dtype: "fp8"
      server.cpuset: "0-9"
    env:                            # container env override/add
      MBX_DEPTH_MODE: static
    mounts: ["/home/pgx/Documenti/vllm-gb10/draft-lab/drafts/k3:/drafts/k3:ro"]
    docker_args: ["--shm-size=16g"]
    extra_args: ""                  # raw extra vllm argv (advanced, not deduplicated)
    suite_profile: quick            # per-config profile
    suite:                          # per-config suite.py overrides
      repeats: 2
      concurrency: [1, 4]
      include_16: true              # opt-in C=16 (can hang low-headroom configs)
      prefill_targets: [1024, 8192]
      long_smoke: true              # full profile default; false for quick scans
      quality: off                  # off|quick|full (runs ./quality/run_quality.py)
    boot_timeout_s: 1500
    tag: az-ref                     # label copied into summary.json
    enabled: true
    notes: "free text, ignored"
```

Allowed `suite:` keys: `repeats`, `decode_max_tokens`,
`decode_prompts_per_category`, `concurrency` (list), `include_16`, `conc_waves`,
`conc_max_tokens`, `prefill_targets` (list), `long_smoke`, `long_target_tokens`,
`request_timeout`, `retries`, `quality`, `extra` (list of raw CLI flags).
Unknown keys fail fast in `check_recipes.py`.

---

## Measurement methodology notes

* temperature **0**, `top_p 1`, `presence_penalty 0`, fixed **seed 42**,
  `chat_template_kwargs.enable_thinking=false` unless `--thinking` is passed;
  server-side defaults (`override-generation-config`) are overridden per request.
* prompts are fixed; each request carries a unique run nonce at the very start
  of the message, so every measured prefix is cold relative to vLLM's prefix
  cache (prefix hits would otherwise inflate TTFT on repeats).
* token counts are always `usage.completion_tokens` / `usage.prompt_tokens`
  from the final SSE chunk (`stream_options.include_usage`), never raw chunk
  counts — with speculative decoding one SSE chunk can carry several tokens
  (`tokens_per_chunk` reports the observed grouping).
* TTFB/TTFT is client-side from request start (includes queue + prefill), TPOT
  over the decode window; per-chunk inter-token latencies are in
  `requests.jsonl.chunk_itl_ms_median`.
* spec-decode numbers come from cumulative Prometheus counters sampled
  before/after each phase (deltas), so they are per-phase and not polluted by
  previous runs on the same server. Raw snapshots are kept so
  `recompute_metrics.py` can re-derive them later.
* prefill uses `max_tokens=1` so only prefill + first token is measured.
* the suite is deliberately workload-like rather than pure-synthetic: natural
  EOS is allowed in decode/concurrency (reported completion tokens vary by
  prompt), while concurrency throughput is measured over complete waves.

## Self-test without a GPU

`bench/tests/mock_vllm.py` is a tiny OpenAI-compatible mock (SSE chat chunks
with multi-token speculative-style batches, usage accounting, Prometheus
counters including the vLLM spec-decode names). It is how the harness is
regression-tested off the GPU:

```bash
python3 bench/tests/mock_vllm.py --port 18000 &
python3 bench/suite.py --base-url http://127.0.0.1:18000 --run-dir /tmp/mockrun \
    --profile quick --concurrency 1,2,4 --prefill-targets 512,1024 \
    --long-smoke --long-target-tokens 3000 --quality off

# graceful degradation: /metrics 404
MOCK_NO_METRICS=1 python3 bench/tests/mock_vllm.py --port 18001 &
python3 bench/suite.py --base-url http://127.0.0.1:18001 --run-dir /tmp/mockrun2 --quality off

# graceful degradation: /metrics without spec_decode_* counters
MOCK_NO_SPEC_COUNTERS=1 python3 bench/tests/mock_vllm.py --port 18002 &
python3 bench/suite.py --base-url http://127.0.0.1:18002 --run-dir /tmp/mockrun3 --quality off
```

Expected: all phases `ok`, `client_to_server_token_ratio ≈ 1.0`,
`tokens_per_step` derived, no crashes and no duplicate `requests.jsonl` lines
with missing metrics. Static recipe preflight: `python3 bench/check_recipes.py`
(all recipes in the repo) and `python3 bench/check_recipes.py matrix/matrix.yaml`
(validates every matrix entry, including the exactly-one-speculative-config
invariant). Matrix resume logic can be tested without a GPU by pre-seeding a
`results/<name>/status.json` with `{"status": "done"}`.

## Known limitations / things to watch

* **One server at a time.** The runner/serve enforce it, but an external client
  can still hit the measured server. The `client_to_server_ratio` warning is the
  contamination detector — always check `summary.json.warnings` before
  comparing runs (the first M0 baseline has a known ~28 s overlap with a manual
  probe; see its warnings).
* Boot times include image pull/caches; first boot of an image may be much
  slower (kernel compile). `boot_s` is recorded per config.
* `serve.py` does not apply eugr `mods:` or run `run-recipe.sh`; recipes that
  rely on eugr tooling/mods must be launched by eugr's own scripts. Command
  recipes are rendered and served as-is.
* Host `nvidia-smi` is unavailable; GPU/clock diagnostics must go through
  `docker exec <container> nvidia-smi` (read-only) when needed.
* Single-node only (no multi-node/ray cluster jobs).
* `quality/run_quality.py` CLI is a convention (`--base-url`, `--model`,
  `--mode`, `--output`); adapt `run_quality_hook()` in `suite.py` if it differs.
* `--include-16` is opt-in: C=16 can hang some low-headroom configs; request
  timeouts are 300 s and the suite records timeouts as errors.

## Known image-level boot failures (v5.4 myllmbox build)

These are engine constraints, not harness bugs — recipes must avoid them (the
runner records the failure in `results/<name>/status.json` + `server.log` and
continues with the next config).

* **id: min-bootable-spec-depth-4** — `num_speculative_tokens=3` boots fine
  through weight load and KV init but dies at CUDA-graph capture with
  `ValueError: num_speculative_tokens_per_batch_size is required for dynamic
  speculative decoding.` (in
  `vllm/v1/spec_decode/dynamic/utils.py: validate_and_normalize_dynamic_sd_schedule`,
  via `speculator.capture()`), followed by
  `RuntimeError: Engine core initialization failed`. **Depth 4 is the minimum
  bootable depth** on this image for both the self-drafter and the draft-k10
  drafter; depth 3 recipes (`b-curve-k3`, `b-k10-curve-k3`, `fixed-3`) always
  fail here. Use `curve-k4`/depth >= 4 instead.
* **id: static-depth-mode-off** — `MBX_DEPTH_MODE=off` with the full
  `MBX_DEPTH_*` env block (CEIL/PROMOTE/DEMOTE/WINDOW/MIN) left in place
  (recipes `fixed-3`, `fixed-5`) fails at graph capture with the SAME
  `num_speculative_tokens_per_batch_size` ValueError: the image's dynamic-SD
  schedule is not populated for the static path. If a static-depth run is ever
  needed, drop the `MBX_DEPTH_CEIL/PROMOTE/DEMOTE/WINDOW` env vars so vLLM takes
  the static decoding path (untested on this image — verify with a one-off boot).
* Both failures occur ~5.5-6 min after container start (after weights + KV
  cache), so a bad recipe costs ~6 min of the matrix budget. `check_recipes.py`
  cannot detect these (they are runtime/image constraints); the runner records
  them cleanly as `status=failed`, `exit_codes.serve=2`.

## Troubleshooting

| symptom | fix |
|---|---|
| `up` refuses: "another model-server-looking container is running" | stop it (`docker rm -f <name>`) or use `--force` only for deliberate adoption |
| boot timeout | the checkpoint can take 6-9 min; check `results/<id>/server.log`, raise `--boot-timeout` |
| `suite.py` exits 3 | `/health` or `/v1/models` not ready; wait for boot |
| `summary.json.metrics.overall` null | image doesn't expose spec-decode counters; check raw `metrics/*.txt` |
| tokens/step null | image lacks `spec_decode_num_emitted_tokens_total`; newer harness derives `(accepted+drafts)/drafts` |
| matrix skips a config unexpectedly | `results/<name>/status.json` says done; use `--force` |
| contaminated numbers | check `warnings` / `client_to_server_ratio`; re-run with no other clients |
