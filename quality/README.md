# quality/ — capability & quality gates for Qwen3.8-Flash-Next FP4 presets

This directory measures whether a serving preset **degrades model capability**:
math reasoning, JSON adherence, code generation, instruction following, tool
calling, long-context recall and repetition/degeneracy.  It complements `bench/`
(throughput): a preset is not a winner if it is fast but answers worse.

Everything is **Python 3.10+ stdlib only** — no pip installs, no network — and
the graders are deterministic.  The suite talks to any OpenAI-compatible server
(vLLM here) at a configurable base URL.

```
quality/
├── run_quality.py            CLI: runs batteries, writes scores.json + raw/
├── compare.py                acceptance gate: diff two scores.json runs
├── qlib/
│   ├── client.py             stdlib HTTP client (health poll, retries, SSE streaming)
│   ├── graders.py            deterministic graders (math/json/code/instr/tool/repetition)
│   ├── needle.py             deterministic long-context prompt builder + recall grader
│   └── batteries.py          per-category runners, aggregation, offline replay
├── data/                     shipped prompts + hidden tests (hand-written, license-safe)
│   ├── math.json             20 arithmetic/word problems with exact answers
│   ├── json_tasks.json       10 constrained JSON tasks + mini validation schemas
│   ├── code_tasks.json       8 function prompts + hidden assertion suites
│   ├── instruction.json      10 format/constraint prompts + rule checks
│   ├── tools.json            6 tool-calling prompts + OpenAI tool schemas + expected args
│   ├── repetition.json       1 long-generation degeneration probe
│   └── needle.json           needle fact/question for long-context recall
├── tests/
│   ├── test_graders.py       62 offline grader unit tests
│   ├── test_client.py        client/parsing tests against a local fake server
│   ├── test_replay.py        end-to-end CLI tests via canned outputs
│   ├── test_compare.py       acceptance-gate tests (noise/tolerance semantics)
│   ├── test_data.py          data/CLI validation
│   ├── build_canned.py       regenerates tests/canned/model_outputs.json
│   └── canned/model_outputs.json   fake model outputs with known pass/fail
├── runs/                     default output root (git-ignored, created on demand)
│                             (kept: m0-reference quick baseline, eugr-b12x-base)
└── README.md
```

## Quick start

```bash
# 1) full default suite against a live server (~8-12 min)
python3 quality/run_quality.py --out quality/runs/preset-A

# 2) fast capability gate without the long-context battery (~2-3 min)
python3 quality/run_quality.py --categories quick --label gating

# 3) long-context only, at 32k + 128k tokens (~1-2 min)
python3 quality/run_quality.py --categories needle --needle-sizes 32k,128k

# 4) offline self-test: grades canned fake outputs, no server required (~3 s)
python3 quality/run_quality.py --self-test

# 5) bench-harness integration form (--mode quick is an alias for categories=quick,
#    --output PATH.json writes scores exactly there; artifacts go to PATH.artifacts/)
./quality/run_quality.py --base-url http://127.0.0.1:8000/v1 --model <id> \
    --mode quick --output bench/results/<cfg>/quality.json

# 6) acceptance gate: compare a candidate preset against the M0 baseline
python3 quality/compare.py quality/runs/m0-reference/quality.json \
                            quality/runs/candidate/quality.json --noise 1

# 7) plan without touching the server
python3 quality/run_quality.py --dry-run

# run the offline unit tests
python3 -m unittest discover -s quality/tests -v   # 133 offline tests, ~8 s
```

Outputs (default `quality/runs/<timestamp>[_label]/`):

| file | content |
|---|---|
| `scores.json` | overall + per-category + per-item verdicts, config, runtimes, limitations |
| `raw/<category>.jsonl` | one line per request: response text, tool calls, usage, latency, grade |
| `summary.md` | human-readable table of the same numbers |
| `run.log` | progress log |

`scores.json` is stable enough for scripts:

```json
{
  "overall": {"score": 0.94, "categories_run": 7, "items_total": 58,
              "items_passed": 55, "category_scores": {"math": 1.0, "...": 0.0},
              "aborted": false},
  "categories": {
    "math": {"score": 1.0, "passed": 20, "total": 20, "runtime_s": 18.2, "items": [...]},
    "needle": {"score": 1.0, "recall_rate": 1.0, "ttft_s": 5.1, "items": [...]},
    "...": {}
  },
  "config": {"model_resolved": "...", "thinking": false, "temperature": 0.0, "...": "..."},
  "limitations": ["..."]
}
```

Exit codes: `0` ok · `2` bad args/`--min-score` gate failed · `3` server not
reachable · `4` run aborted because the server died mid-run (partial scores are
still written).

### Matrix hook (bench-harness integration)

The throughput matrix calls the suite as a subprocess, so the CLI has a stable
short form:

```bash
./quality/run_quality.py --base-url http://127.0.0.1:8000/v1 --model <id> \
    --mode quick --output bench/results/<config>/quality.json
```

- `--mode quick` = every battery except the long-context needle; `--mode full`
  = all batteries (needle 32k/128k/262k + repetition).
- `--output PATH.json` writes the scores exactly at `PATH`; artifacts land in
  `PATH-without-suffix.artifacts/` (`raw/`, `summary.md`, `run.log`).
- Measured on this box (DGX Spark GB10, M0 reference server, 2026-11-12):
  **quick profile = 78.7 s wall**, 55 requests, 0 errors (`math` 27.8 s,
  `json` 6.9 s, `code` 5.1 s, `instruction` 2.5 s, `tool` 4.3 s,
  `repetition` 32.1 s).  Budget ~90 s per matrix config; full profile is
  ~4-5 min because of the 262k-token prefill.
- The tool is safe to call unattended: exit `3` when no server is up, exit `4`
  with partial scores if the server disappears mid-run, and the JSON schema is
  stable (`overall.score`, `overall.category_scores`, `categories.<name>.score`).

**Absolute-speed caveat:** scores in this directory are capability-only.  On
this box the served model is roughly 2x slower than published Qwen3.8-Flash
numbers because of draft/speculative acceptance behaviour of the current
recipe, which is a *throughput/latency* fact measured by `bench/` — it does not
imply degraded answers, and it does not change any quality score here.

## Batteries and scoring

All batteries use `temperature=0`, `top_p=1`, a fixed `seed` (requested; vLLM
batching can still vary tokens slightly).  `thinking` is controlled with
`chat_template_kwargs: {"enable_thinking": <bool>}`; default is **off**
(`--thinking` enables CoT, which needs larger `max_tokens`).

| category | items | what the model sees | scoring |
|---|---|---|---|
| `math` | 20 | short problem; must end with `ANSWER: <value>` | exact match after normalisation: numeric equality when both sides parse as numbers (`5,050` = `5050`, `.70` = `0.7`, `150 km` = `150`), otherwise canonical string equality; `\boxed{}` and last-line fallbacks supported |
| `json` | 10 | constrained JSON generation | `json.loads` on the extracted payload (fences and surrounding prose are tolerated) **and** a mini-schema check (types, required/exact keys, array counts, uniqueness, enums, numeric bounds, regex patterns, nested objects); item score = valid ∧ conformant |
| `code` | 8 | write one function with a given name | code block extracted, concatenated with **hidden assertions**, executed in an isolated subprocess (`-I -S`, CPU rlimit, 10 s timeout); item score = 1 only if all assertions pass; category also reports assertion-level pass rate |
| `instruction` | 10 | format/constraint prompts (word counts, bullet-only, no commas, ordering, JSON keys…) | rule engine; item score = 1 only if every rule passes; category reports per-check pass rate |
| `tool` | 6 | prompt + OpenAI tool schemas | requires parsed `message.tool_calls` (server-side `qwen3_xml`); verifies tool name and expected argument values (typed comparison, case-insensitive strings, numeric strings coerced, `contains`/alias tolerance per argument); falls back to parsing `<tool_call>` XML out of `content` and records which path was used |
| `needle` | 3 (32k/128k/262k tokens) | deterministic synthetic document (~3.5 chars/token) with a unique passphrase at 50% depth | streamed request; recalls the exact passphrase. Score = lenient recall (all passphrase components in order, case-insensitive: tolerates `quasar 4419 zebra`); `strict` recall (exact hyphenated form) is reported too. Also records real `usage.prompt_tokens`, `chars_per_token_actual` and client-side TTFT |
| `repetition` | 1 | ≥150-line code generation, 2048 tokens | degeneracy detector: max repeated 20-gram, unique-token ratio, distinct-2 bigram diversity, longest identical-line run, output length. Score = 0 if degenerate |

Metric details:

- **category score** = mean of per-item 0/1 scores; **overall score** = mean of
  the category scores that actually ran (skipped categories excluded).
- Diagnostics that matter for triage are reported next to the score:
  `valid_json_rate`, `schema_conformance_rate`, `tests_passed/tests_total`,
  `assertion_pass_rate`, `check_pass_rate`, `recall_rate`, `strict_recall_rate`,
  `ttft_s`, `prompt_tokens`, `reason_counts` (tool failures), `metrics`
  (repetition), plus `errors[]` and per-item `failed_tests`/`failed_checks`.
- The degeneracy thresholds were calibrated on real repo prose/code (unique
  token ratio ≈0.20-0.30, distinct-2 ≈0.55-0.65, max 20-gram repeat 1-2) versus
  genuine loops (unique ratio <0.10, distinct-2 <0.20, same 20-gram many times).

### Category selection

`--mode quick|full` is the short form bench-harness uses: `quick` = all batteries
except needle, `full` = every battery.  It is an alias for `--categories`
(`--mode` wins if both are given).  Longer forms:

```
--categories quick            # everything except needle (fast gate)
--categories math,json,code   # explicit list
--categories all,-needle      # everything except needle
--categories needle           # long-context only
--categories long-context     # aliases: tools, instruction-following, degeneracy, ...
```

`--output PATH` is an alias for `--out`.  If `PATH` ends in `.json` the scores
JSON is written exactly there (artifact directory `PATH-stem.artifacts/` holds
`raw/`, `summary.md`, `run.log` and a copy of `scores.json`), which lets the
throughput matrix consume `.../quality.json` directly.  Otherwise `PATH` is a
directory like `--out`.

`--needle-sizes 32k,128k` accepts `32k`/`32768` forms.  Default sizes are
`32768`, `131072`, `262144` tokens (the last one is the "262k" variant; each
request is guarded against the server context limit).

**Sizing heuristic and runtime validation.**  The tokenizer lives in the model
snapshot, so the builder does not tokenize locally: it targets
`tokens * 3.5` characters of deterministic filler (calibrated on the real
Qwen tokenizer: measured 3.53-3.59 chars/token on this filler).  Every live
needle item therefore records the server-reported `usage.prompt_tokens` and
validates it at run time: the request is accepted only if
`prompt_tokens >= 90% of target` **and** `prompt_tokens + max_tokens <=
max_model_len`.  Violations are surfaced in `categories.needle.errors` and as
`prompt_tokens_ok: false` on the item; `chars_per_token_actual` is recorded so
future sizing can be recalibrated.  With the guard above, the 262k request
costs ~916k chars ≈ 259k real tokens, comfortably under `max_model_len=409600`.
Items are skipped (with a reason) if the estimate cannot fit the context.

### Runtime budget

Measured request counts: 58 total (20+10+8+10+6+3+1).  With the defaults
(`--workers 4` for text categories, needle serial) the full suite targets
~8-12 min, dominated by the 262k-token prefill and the 2k-token repetition
generation; `quick` is ~2-3 min.  Use `--needle-sizes 32k,128k` to drop the
biggest single request.

## Acceptance gate: comparing runs

`quality/compare.py` diffs a candidate run against the baseline and is the
disqualification gate for a winning preset:

```bash
python3 quality/compare.py quality/runs/m0-reference/quality.json \
                            quality/runs/winner/quality.json --noise 1
# exit 0 = OK, 1 = regression, 2 = usage/input error
```

- Tolerance is **item-based**: `--noise 1` allows one item of drift per
  category (e.g. 0.05 on math's 20 items, 0.10 on instruction's 10 items).
  `--noise 0` is strict; `--max-drop 0.03` sets an absolute score tolerance
  instead.  Default noise is 1 item.
- The table reports per-category baseline/candidate/delta and **which items
  flipped pass -> fail** (and fail -> pass), so a regression is attributable
  rather than just a number.
- `--ignore instruction` excludes a noisy category from the verdict;
  `--require-categories needle` makes a missing/skipped category a regression
  (useful when comparing a full baseline against a quick-profile candidate).
- `--json-diff diff.json` writes the machine-readable comparison.

Recommended gate for the preset search: candidate must be within noise of M0 on
`math`, `json`, `code`, `tool` (spec-decoding changes are expected to be
output-exact); `needle` and `repetition` are only meaningful when the candidate
was run with the full profile.  `kv-fp8` and template variants are allowed to
move, but a drop beyond noise in the hard categories disqualifies.

Two settings are worth distinguishing:

- **screening (default `--noise 1`)**: one item of drift per category is
  tolerated, which is right for noisy categories such as `instruction`.
- **finalist acceptance (`--noise 0` or `--max-drop 0.05`)**: any category drop
  is a regression.  This matters for short batteries — one `tool` item is
  1/6 ≈ 0.167 of the category and would otherwise slip through `--noise 1`.
  Because `temperature=0` + fixed seed is only a request, an isolated
  single-item flip can still be sampling noise: re-run it once before
  disqualifying a preset (the `pass->fail` column tells you exactly which item
  to re-check).

## Live-run notes (vLLM specifics)

- Boot takes minutes: use `--wait-server 600` to poll `/health` instead of
  failing.  The Lead restarts the server per preset, so runs may be aborted
  mid-flight — partial `scores.json` is still written (exit code 4).
- The suite never starts/stops the server or touches GPU state.  It is safe to
  point at a server that is also being benchmarked, but parallel benchmarking
  will inflate latencies/TTFT and can perturb throughput measurements — run
  quality when the bench is idle.
- `--model` is auto-detected from `GET /v1/models`; pass it explicitly if the
  server exposes several models.
- If content comes back empty, thinking is almost certainly enabled without
  enough `max_tokens`: use `--thinking` (which raises all token budgets) or keep
  thinking off.  The runner surfaces empty-content items as a warning.
- Raw prompts are **not** stored by default (the 262k prompt is ~0.9 MB);
  `--save-prompts` opts in, and every raw record always carries a prompt
  `sha256` + char count.

## Adding tests (and keeping the graders honest)

1. Add prompts/expected values to `quality/data/*.json` (keep `id`, `prompt`
   and the category-specific fields; see the existing items).
2. If a new *check type* or grader behaviour is needed, add it to
   `qlib/graders.py` and a matching offline unit test in `tests/test_graders.py`.
3. Add at least one canned model output for the new item to
   `tests/build_canned.py`, regenerate
   (`python3 quality/tests/build_canned.py`) and assert the verdict in
   `tests/test_replay.py` (or `test_data.py` for structural invariants).
4. Run `python3 -m unittest discover -s quality/tests -v` — it must stay fully
   green without a server.  Nothing in the graders may touch the network or
   depend on the clock/randomness.

The canned dataset is intentionally imperfect (2 wrong math answers, one broken
fibonacci, one malformed JSON, one schema violation, one violated word count,
one wrong tool argument, one missed needle) so the tests prove the suite both
**passes good output and catches bad output**.  Expected canned outcome:
`math 18/20`, `json 8/10`, `code 7/8`, `instruction 9/10`, `tool 5/6`,
`needle 2/3`, `repetition 1/1`, overall `0.8536`.

## Limitations (read before quoting numbers)

- Rule-based scores are capability *proxies*, not a substitute for human review.
- `temperature=0` + seed is a request, not a guarantee: vLLM continuous
  batching can change sampled tokens between runs.  Treat small deltas (<2
  items) as noise.
- Code execution in a subprocess with CPU/temp limits and a timeout is a
  robustness guard, **not** a security sandbox; do not grade untrusted models
  on a machine you care about.
- The long-context battery uses synthetic English filler; real documents are
  harder.  Recall at a given size is an optimistic-ish bound, and TTFT is a
  client-side, moment-in-time measurement — use `bench/` for latency truth.
- Tool grading checks name + expected argument values; extra/optional arguments
  are tolerated and reported, so a pass does not prove the call is semantically
  ideal.
- Repetition is probed with a single 2k-token generation; a preset could still
  degenerate on other prompts.

## Quality validation — finalist acceptance (2026-11-12)

Baseline = shipped M0 recipe (myllmbox AZ, stock chat template, K=12 dynamic
depth, bf16 KV, thinking off), **quick** profile.  Candidate = winning preset
(K=4 self-draft), **full** profile.

```bash
python3 quality/compare.py quality/runs/m0-reference/quality.json \
    bench/results/finalist-winner-full/quality.json --noise 0   # exit 0
python3 quality/compare.py quality/runs/m0-reference/quality.json \
    bench/results/finalist-winner-full/quality.json --noise 1   # exit 0
```

| config | profile | overall | shared-cat mean | needle | gate (`--noise 0`) |
|---|---|---|---|---|---|
| M0 reference | quick | 0.9667 (53/55) | 0.9667 | not run | baseline |
| **winner K=4 self-draft** | full | **0.9714** (56/58) | **0.9667** | **3/3 strict** | **OK, 0 item flips** |
| eugr b12x base | quick | 0.9667 (53/55) | 0.9667 | not run | OK, 0 item flips |
| d-draftk10-spec6 | quick | 0.9667 | 0.9667 | not run | OK |
| b-k10-curve-k4-medium | quick | 0.9667 | 0.9667 | not run | OK |
| b-medium-template | quick | 0.9667 | 0.9667 | not run | OK |
| b-k10-curve-k4 | quick | 0.9583 | 0.9583 | not run | OK at noise 1 (1 math flip) |
| b-kvfp8-spec7 | quick | 0.9583 | 0.9583 | not run | OK at noise 1 (1 math flip) |

Reading the result:

- **Capability-equivalent, not byte-identical.**  On the six shared batteries
  the winner matches M0 item-for-item: `math` 20/20, `json` 10/10, `code` 8/8,
  `tool` 6/6, `repetition` non-degenerate, `instruction` 8/10 — zero
  pass->fail and zero fail->pass flips even at `--noise 0`.  Raw completion
  text still differs on some items (wordings/formatting; `temperature=0` is not
  strictly deterministic under continuous batching), so the claim is
  capability equivalence, not output equality.
- **The 2 instruction misses are baseline behaviour, not a regression:**
  `instr-01` answers 13 words instead of exactly 12 and `instr-09` repeats
  `ab` 10 times instead of 20.  Both fail identically in M0 and in every
  measured config.
- **Profile asymmetry:** M0 ran quick (6 categories), the winner ran full (7).
  `compare.py` compares the 6 shared categories (`overall(common)` 0.9667 on
  both sides) and lists needle separately.  The winner's higher 0.9714 is
  purely the needle battery being present in the full profile.
- **Long-context (winner only):** 3/3 **strict** recall at 32k / 128k / 262k
  tokens.  Server-reported prompt tokens 31,121 / 124,141 / 248,102 =
  94.6-95.0 % of target, so the 3.5 chars/token heuristic is validated by the
  real tokenizer and every request stayed inside `max_model_len=409600`.
  TTFT 12.4 s / 51.7 s / 105.0 s (→ ~2.2-2.4k prefill tok/s), needle battery
  169.8 s, whole full run 244 s.
- **eugr b12x base** uses a 72 GB "selected" view of the 102 GB checkpoint;
  quick quality is 0.9667 with identical category scores and zero item flips
  vs M0, so the reduced loadout does not change these capability outcomes.
- One isolated math flip (`math-08`, GCD of 84 and 126) appears in two
  non-finalist configs (`b-k10-curve-k4`, `b-kvfp8-spec7`) and in neither the
  winner nor M0: one item = within the one-item noise budget, same item in
  both, so treated as borderline arithmetic sampling rather than a systematic
  effect.  If such a flip appeared on a finalist, re-run that single item to
  confirm before disqualifying.

**Verdict:** the winning K=4 self-draft preset is quality-equal to the shipped
preset within item noise (zero item flips on all six shared batteries, plus
3/3 long-context recall at strict match) and passes the acceptance gate.  No
hard-category (math/json/code/tool) regression was observed in any measured
finalist config.
