# Quality run summary

- generated: 2026-11-21T16:43:47+01:00
- mode: live
- model: myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored-524k
- base_url: http://127.0.0.1:8000
- thinking: False | temperature: 0.0 | seed: 12345
- git: d08c278 | host: thinkstationpgx-9213

| category | score | passed | total | runtime_s | notes |
|---|---|---|---|---|---|
| math | 0.95 | 19 | 20 | 64.99 |  |
| json | 1.0 | 10 | 10 | 14.71 | valid_json_rate=1.0 |
| code | 1.0 | 8 | 8 | 9.13 | assertion_pass_rate=1.0 |
| instruction | 0.9 | 9 | 10 | 4.28 | check_pass_rate=0.9524 |
| tool | 1.0 | 6 | 6 | 7.51 |  |
| needle | 1.0 | 3 | 3 | 154.77 | recall_rate=1.0; strict_recall_rate=1.0; ttft_s=51.1508 |
| repetition | 1.0 | 1 | 1 | 29.01 | degenerate=False |

**overall score:** 0.9786 (56/58 items passed) in 284.42s

## Limitations

- Scores are deterministic, rule-based proxies for capability; they are not a substitute for human review.
- temperature=0 and a fixed seed are requested, but server-side batching can still make sampled tokens differ run-to-run.
- Code batteries execute model-written Python in a subprocess with CPU/temp limits and a timeout; this is a robustness guard, not a security sandbox.
- Long-context prompts are synthetic English filler; real-world retrieval difficulty varies, so recall here is an upper-ish bound for a given size.
- TTFT is measured client-side and includes queueing/batching latency of the moment, so long-context TTFT is indicative, not a benchmark.
- Tool-calling checks verify tool name + expected argument values; extra optional arguments are tolerated and reported.
- Thinking mode (--thinking) needs generous max_tokens; an empty-content item usually means the CoT consumed the budget.
