# Quality run summary

- generated: 2026-11-21T03:53:58+01:00
- mode: live
- model: azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ
- base_url: http://127.0.0.1:8000
- thinking: False | temperature: 0.0 | seed: 12345
- git: 51f81e8 | host: thinkstationpgx-9213

| category | score | passed | total | runtime_s | notes |
|---|---|---|---|---|---|
| math | 1.0 | 20 | 20 | 25.99 |  |
| json | 1.0 | 10 | 10 | 5.48 | valid_json_rate=1.0 |
| code | 1.0 | 8 | 8 | 4.5 | assertion_pass_rate=1.0 |
| instruction | 0.8 | 8 | 10 | 2.03 | check_pass_rate=0.9048 |
| tool | 1.0 | 6 | 6 | 3.51 |  |
| needle | 1.0 | 3 | 3 | 150.87 | recall_rate=1.0; strict_recall_rate=1.0; ttft_s=50.0588 |
| repetition | 1.0 | 1 | 1 | 27.48 | degenerate=False |

**overall score:** 0.9714 (56/58 items passed) in 219.88s

## Limitations

- Scores are deterministic, rule-based proxies for capability; they are not a substitute for human review.
- temperature=0 and a fixed seed are requested, but server-side batching can still make sampled tokens differ run-to-run.
- Code batteries execute model-written Python in a subprocess with CPU/temp limits and a timeout; this is a robustness guard, not a security sandbox.
- Long-context prompts are synthetic English filler; real-world retrieval difficulty varies, so recall here is an upper-ish bound for a given size.
- TTFT is measured client-side and includes queueing/batching latency of the moment, so long-context TTFT is indicative, not a benchmark.
- Tool-calling checks verify tool name + expected argument values; extra optional arguments are tolerated and reported.
- Thinking mode (--thinking) needs generous max_tokens; an empty-content item usually means the CoT consumed the budget.
