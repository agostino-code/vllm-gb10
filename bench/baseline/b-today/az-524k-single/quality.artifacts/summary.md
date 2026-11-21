# Quality run summary

- generated: 2026-11-21T20:15:19+01:00
- mode: live
- model: azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ-524k
- base_url: http://127.0.0.1:8000
- thinking: False | temperature: 0.0 | seed: 12345
- git: 7b47594 | host: thinkstationpgx-9213

| category | score | passed | total | runtime_s | notes |
|---|---|---|---|---|---|
| math | 0.95 | 19 | 20 | 59.88 |  |
| json | 1.0 | 10 | 10 | 10.24 | valid_json_rate=1.0 |
| code | 1.0 | 8 | 8 | 9.01 | assertion_pass_rate=1.0 |
| instruction | 0.8 | 8 | 10 | 4.7 | check_pass_rate=0.9048 |
| tool | 1.0 | 6 | 6 | 8.59 |  |
| needle | 1.0 | 3 | 3 | 160.07 | recall_rate=1.0; strict_recall_rate=1.0; ttft_s=52.9592 |
| repetition | 1.0 | 1 | 1 | 26.62 | degenerate=False |

**overall score:** 0.9643 (55/58 items passed) in 279.13s

## Limitations

- Scores are deterministic, rule-based proxies for capability; they are not a substitute for human review.
- temperature=0 and a fixed seed are requested, but server-side batching can still make sampled tokens differ run-to-run.
- Code batteries execute model-written Python in a subprocess with CPU/temp limits and a timeout; this is a robustness guard, not a security sandbox.
- Long-context prompts are synthetic English filler; real-world retrieval difficulty varies, so recall here is an upper-ish bound for a given size.
- TTFT is measured client-side and includes queueing/batching latency of the moment, so long-context TTFT is indicative, not a benchmark.
- Tool-calling checks verify tool name + expected argument values; extra optional arguments are tolerated and reported.
- Thinking mode (--thinking) needs generous max_tokens; an empty-content item usually means the CoT consumed the budget.
