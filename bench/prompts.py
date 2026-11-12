"""Deterministic prompt sets for the benchmark suite.

Everything here is fixed text (no randomness) so runs are comparable across
images/configs.  Each request gets a run-specific nonce at the very start of
the prompt: with vLLM's prefix caching (block-size 3456 here) that guarantees
cold prefixes, so TTFT numbers are not flattered by cache hits from a previous
repeat or a previous run.  The nonce is tiny (~10 tokens) and prompt_tokens is
always read back from the server's ``usage`` field.
"""
from __future__ import annotations

import hashlib

# ──────────────────────────────────────────────────────────────────────────────
# single-stream decode set: two prompts per category (QA / code / JSON / math)
# ──────────────────────────────────────────────────────────────────────────────

DECODE_PROMPTS: list[dict] = [
    {
        "id": "qa-1",
        "category": "qa",
        "prompt": (
            "Explain step by step how a modern inference server keeps GPU utilization high "
            "when serving many users. Cover continuous batching, KV-cache paging, prefill vs "
            "decode, and speculative decoding. Be concrete and mention trade-offs."
        ),
    },
    {
        "id": "qa-2",
        "category": "qa",
        "prompt": (
            "A colleague wants to serve a large mixture-of-experts language model on a single "
            "GPU with unified memory. Describe the main bottlenecks they should expect and how "
            "to measure each one before changing any configuration. Include a paragraph on how "
            "KV-cache size and maximum context length interact."
        ),
    },
    {
        "id": "code-1",
        "category": "code",
        "prompt": (
            "Write a complete Python implementation of an LRU cache with O(1) get and put, a "
            "capacity limit, and a thread-safe wrapper. Then show how you would unit test it, "
            "including edge cases. Keep the explanation short and the code complete."
        ),
    },
    {
        "id": "code-2",
        "category": "code",
        "prompt": (
            "Implement a function in Python that parses a nested JSON document and flattens it "
            "into path->value pairs, handling lists and deeply nested objects, plus a companion "
            "function that rebuilds the original nested structure. Include type hints, docstrings "
            "and a few doctest examples."
        ),
    },
    {
        "id": "json-1",
        "category": "json",
        "prompt": (
            "Produce a single valid JSON object describing a benchmark matrix for three model "
            "serving presets. Each preset must contain name, stack, speculative_method, "
            "num_speculative_tokens, kv_cache_dtype, max_num_seqs, and an array of measured "
            "concurrency levels with throughput values. Output only JSON, no prose."
        ),
    },
    {
        "id": "json-2",
        "category": "json",
        "prompt": (
            "Create a JSON Schema (draft 2020-12) for an API response of an OpenAI-compatible "
            "server: it must validate id, model, choices (with message role/content and "
            "finish_reason enum), and usage with token counts. Output only the schema."
        ),
    },
    {
        "id": "math-1",
        "category": "math",
        "prompt": (
            "Solve this problem carefully and show every step: a server processes requests at a "
            "Poisson rate of 8 per second; each request needs 0.05 s of decoding on average; "
            "there are 4 workers. Using an M/M/c model, compute the utilization, the probability "
            "that an arriving request must wait, and the mean waiting time. Then discuss what "
            "changes when speculative decoding doubles the per-request service rate."
        ),
    },
    {
        "id": "math-2",
        "category": "math",
        "prompt": (
            "A KV cache for a transformer has 60 layers, 8 KV heads, head dimension 128, and is "
            "stored in bf16. Compute the number of bytes per token and how many tokens fit in 23 "
            "GiB. Then work out the maximum number of concurrent requests if every request uses "
            "32768 tokens of context, and explain where the estimate is optimistic."
        ),
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# short pool for concurrency waves (kept small so prefill does not dominate)
# ──────────────────────────────────────────────────────────────────────────────

CONCURRENCY_PROMPTS: list[dict] = [
    {
        "id": "conc-qa",
        "category": "qa",
        "prompt": "Explain the difference between throughput and latency in one short paragraph.",
    },
    {
        "id": "conc-code",
        "category": "code",
        "prompt": "Write a Python function that returns the nth Fibonacci number using matrix exponentiation.",
    },
    {
        "id": "conc-json",
        "category": "json",
        "prompt": 'Output only JSON: {"status":"ok","presets":3,"notes":"all good"} extended with an array of four preset names.',
    },
    {
        "id": "conc-math",
        "category": "math",
        "prompt": "Compute the sum of the first 50 prime numbers and show your reasoning briefly.",
    },
    {
        "id": "conc-lists",
        "category": "code",
        "prompt": "Describe three different ways to detect a cycle in a linked list and compare their complexity.",
    },
    {
        "id": "conc-kv",
        "category": "qa",
        "prompt": "What happens to request latency when the KV cache is almost full on a serving GPU?",
    },
    {
        "id": "conc-yarn",
        "category": "qa",
        "prompt": "Summarize the trade-offs of using YaRN rope scaling to extend context length in an LLM.",
    },
    {
        "id": "conc-review",
        "category": "code",
        "prompt": "Review this idea for a benchmark: measure decode speed with temperature 0 and a fixed seed. What could go wrong?",
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# long-context prompt generation (deterministic)
# ──────────────────────────────────────────────────────────────────────────────

_CORPUS_PARAGRAPHS = [
    "The serving stack schedules incoming requests onto a fixed set of GPU seats. Each request "
    "first runs its prompt through the prefill phase, which fills the key and value caches for "
    "every transformer layer, and then enters the decode phase where one or more tokens are "
    "produced per engine step. Continuous batching lets new prompts join the running batch at "
    "step boundaries, so the engine can trade a little per-request latency for much better "
    "aggregate throughput.",
    "Memory is the limiting resource on a single accelerator. Weights, activation buffers, graph "
    "capture pools and the KV cache all compete for the same unified memory, and the cache size "
    "determines how many tokens can be resident at once. Operators either fix the cache to a "
    "chosen byte budget or give the engine a utilization target and let it profile. The second "
    "option adapts to the checkpoint but makes runs harder to compare unless the resulting cache "
    "size is recorded.",
    "Speculative decoding adds a small draft model that proposes several tokens ahead. The target "
    "model verifies all proposals in one forward pass and accepts the longest correct prefix. "
    "Acceptance depends on the draft quality, the sampling temperature, and the request content; "
    "long structured outputs such as code tend to accept many tokens, while open-ended prose "
    "accepts fewer. Because verification is batched, speculative decoding usually increases "
    "single-stream speed and can still help aggregate throughput.",
    "Quantization reduces weight traffic at the cost of numerical fidelity. Four-bit formats pack "
    "two values per byte and require dequantization kernels, while eight-bit formats have more "
    "native hardware support on some accelerators. Mixed schemes keep sensitive layers, such as "
    "the language-model head or normalization layers, at higher precision. The observed speedup "
    "depends on whether the workload is memory bound or compute bound, and on how much of the "
    "dequantization can be fused into the matrix multiplication.",
    "Prefix caching stores the KV blocks of repeated prompt prefixes. When many requests share a "
    "long system prompt, a large fraction of the prefill work can be skipped. The benefit is "
    "largest for agent-style traffic with long shared instructions; it disappears when every "
    "prompt is unique. Cache hit rates should be logged together with time to first token, "
    "otherwise a fast run may simply have reused somebody else's work.",
    "Collecting Prometheus metrics from the engine is the cheapest way to attribute performance. "
    "Counters for prompt and generation tokens validate the client-side accounting, the time to "
    "first token histogram captures queueing plus prefill, and speculative decoding counters show "
    "how many draft tokens were proposed and accepted. Recording a snapshot before and after each "
    "phase turns those counters into per-phase measurements instead of a single cumulative number.",
    "Long context requests stress both the cache and the attention kernel. A request that occupies "
    "a large fraction of the cache prevents other requests from running concurrently, and the "
    "prefill cost grows with the prompt length. Needle-in-a-haystack tests check that information "
    "placed in the middle of a long prompt survives the attention window, while the time to first "
    "token measures how quickly the engine can ingest the prompt.",
]

_NEEDLE_TEMPLATE = (
    "Important: the secret access code for this exercise is {code}. Remember it for the "
    "question at the end of the document."
)


def _corpus_block() -> str:
    return "\n\n".join(_CORPUS_PARAGRAPHS)


def make_long_prompt(target_tokens: int, nonce: str, needle_code: str | None = None) -> str:
    """Build a deterministic prompt of roughly ``target_tokens`` tokens.

    English prose averages ~4 characters per token for these tokenizers, so we
    aim at ``target_tokens * 4`` characters; the exact prompt length is read
    back from the server's usage field.
    """
    target_chars = max(256, int(target_tokens * 4))
    nonce_line = f"[benchmark nonce {nonce}]\n"
    if needle_code:
        return _build_needle_prompt(target_chars, nonce_line, needle_code)
    block = _corpus_block()
    parts = [nonce_line]
    size = len(nonce_line)
    index = 0
    while size < target_chars:
        section = f"Section {index + 1}. " + block
        parts.append(section)
        size += len(section) + 2
        index += 1
    text = "\n\n".join(parts)[:target_chars]
    return text + "\n\nSummarize the main ideas of the document above in one short paragraph."


def _build_needle_prompt(target_chars: int, nonce_line: str, needle_code: str) -> str:
    block = _corpus_block()
    needle = _NEEDLE_TEMPLATE.format(code=needle_code)
    parts: list[str] = [nonce_line]
    size = len(nonce_line)
    index = 0
    inserted = False
    while size < target_chars:
        section = f"Section {index + 1}. " + block
        parts.append(section)
        size += len(section) + 2
        index += 1
        if not inserted and size >= target_chars * 0.45:
            parts.append(needle)
            size += len(needle) + 2
            inserted = True
    if not inserted:
        parts.insert(len(parts) // 2, needle)
    question = (
        "\n\nQuestion: what is the secret access code mentioned in this document? "
        "Answer with the code only."
    )
    return "\n\n".join(parts) + question


def needle_code(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:6].upper()
    return f"MAGENTA-{digest}"


def nonce(run_id: str, phase: str, index: int) -> str:
    digest = hashlib.sha256(f"{run_id}|{phase}|{index}".encode()).hexdigest()[:10]
    return f"{phase}-{index}-{digest}"
