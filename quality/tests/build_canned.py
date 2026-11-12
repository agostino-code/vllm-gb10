#!/usr/bin/env python3
"""Regenerate quality/tests/canned/model_outputs.json.

The file contains hand-authored fake model outputs with known pass/fail
behaviour.  It lets the graders and the whole CLI be tested offline, with no
server and no network:

    python3 quality/tests/build_canned.py
    python3 quality/run_quality.py --self-test

Expected outcome of the canned run (asserted by quality/tests/test_replay.py):
  math 18/20, json 8/10 (valid 9/10), code 7/8, instruction 9/10,
  tool 5/6 (one content-XML fallback), needle 2/3, repetition 1/1.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "canned" / "model_outputs.json"

PY = "```python\n%s\n```"
TXT = "```\n%s\n```"

# ---------------------------------------------------------------------------
# math: 18 correct, 2 deliberately wrong
# ---------------------------------------------------------------------------
MATH = {
    "math-01": ("17 * 23 = 391.", "391"),
    "math-02": ("144 / 12 = 12, and 12 + 37 = 49.", "49"),
    "math-03": ("5^3 = 125 and 3^4 = 81, so (125 - 81) / 2 = 22.", "22"),
    "math-04": ("60 * 2.5 = 150.", "180"),  # WRONG on purpose
    "math-05": ("96 / 8 = 12, and 12 * 5 = 60.", "60"),
    "math-06": ("12 * 15 = 180 plants; 180 - 24 = 156.", "156"),
    "math-07": ("6! = 720.", "720"),
    "math-08": ("84 = 2^2*3*7 and 126 = 2*3^2*7, so gcd = 42.", "42"),
    "math-09": ("lcm(12, 18) = 36.", "36"),
    "math-10": ("250 * 1.18 = 295.", "295"),
    "math-11": ("0.15 * 260 = 39.", "39"),
    "math-12": ("100 * 101 / 2 = 5050.", "5,050"),  # comma-form must match numerically
    "math-13": ("2^10 = 1024, minus 1 is 1023.", "1024"),  # WRONG on purpose
    "math-14": ("7 * 11 = 77, and 1000 - 77 = 923.", "923"),
    "math-15": ("35 * 6 = 210 pages read; 320 - 210 = 110.", "110"),
    "math-16": ("81 + 144 = 225.", "225"),
    "math-17": ("42 * 42 = 1764, so the square root is 42.", "42"),
    "math-18": ("0.25 * 80 = 20 and 0.5 * 40 = 20, total 40.", "40"),
    "math-19": ("The integers are 27, 29 and 31; the middle one is 29.", "29"),
    "math-20": ("270 / 3 = 90.", "90"),
}

# ---------------------------------------------------------------------------
# json: 9 valid, 1 invalid JSON, 1 schema violation => 8 conformant
# ---------------------------------------------------------------------------
JSON_TASKS = {
    "json-01": json.dumps([
        {"name": "Mercury", "moons": 0, "order": 1},
        {"name": "Venus", "moons": 0, "order": 2},
        {"name": "Earth", "moons": 1, "order": 3},
        {"name": "Mars", "moons": 2, "order": 4},
        {"name": "Jupiter", "moons": 95, "order": 5},
    ]),
    "json-02": '```json\n{"title": "The Silent Harbor", "author": "Iris Vale", "year": 2017, '
               '"genres": ["mystery", "literary fiction"]}\n```',
    "json-03": json.dumps({"user": {"id": 42, "name": "Ada Lovelace", "active": True},
                           "roles": ["admin", "editor", "viewer"]}),
    "json-04": json.dumps([{"id": index, "status": ["ok", "warn", "error"][index % 3]}
                           for index in range(1, 11)]),
    # Valid JSON but only 2 keys: fails the "exactly 3 keys" schema.
    "json-05": json.dumps({"summary": "A short summary of the thing.", "extra": "value"}),
    "json-06": json.dumps([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    "json-07": json.dumps({"items": [
        {"sku": "SKU-0001", "qty": 2, "price": 9.99},
        {"sku": "SKU-0002", "qty": 1, "price": 19.5},
        {"sku": "SKU-0003", "qty": 5, "price": 0.99},
        {"sku": "SKU-0004", "qty": 3, "price": 4.25},
    ], "total": 84.7}),
    "json-08": json.dumps({"config": {"host": "localhost", "port": 8080, "tls": True, "retries": 3},
                           "tags": ["edge", "prod", "eu-west"]}),
    # Deliberately invalid JSON (bare words).
    "json-09": "[alpha, beta, gamma, delta, epsilon, zeta]",
    "json-10": json.dumps({"equation": "2x + 3 = 17", "root": 7,
                           "steps": ["2x = 14", "x = 7"]}),
}

# ---------------------------------------------------------------------------
# code: 7 correct implementations, 1 broken (fibonacci)
# ---------------------------------------------------------------------------
CODE = {
    "code-01": """def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True
""",
    "code-02": """def reverse_words(s):
    if not s:
        return ''
    return ' '.join(reversed(s.split(' ')))
""",
    "code-03": """def count_vowels(s):
    return sum(1 for ch in s.lower() if ch in 'aeiou')
""",
    # Broken: returns the wrong Fibonacci value for most n.
    "code-04": """def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return b
""",
    "code-05": """def flatten(lst):
    out = []
    for item in lst:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
""",
    "code-06": """def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
    "code-07": """def word_frequencies(text):
    freq = {}
    for word in text.lower().split():
        freq[word] = freq.get(word, 0) + 1
    return freq
""",
    "code-08": """def run_length_encode(s):
    if not s:
        return ''
    out = []
    prev = s[0]
    count = 1
    for ch in s[1:]:
        if ch == prev:
            count += 1
        else:
            out.append(prev + str(count))
            prev = ch
            count = 1
    out.append(prev + str(count))
    return ''.join(out)
""",
}

# ---------------------------------------------------------------------------
# instruction: 9 correct, instr-01 deliberately 10 words instead of 12
# ---------------------------------------------------------------------------
INSTRUCTION = {
    "instr-01": "The ocean is deep blue and full of life today.",  # 10 words
    "instr-02": "2\n3\n5\n7\n11",
    "instr-03": "The quokka is a marsupial and the quokka is friendly.",
    "instr-04": "- Alpha\n- Beta\n- Gamma",
    "instr-05": "BANANA",
    "instr-06": "Is the sky blue?\nYes.",
    "instr-07": "red,blue,green,yellow",
    "instr-08": '{"a": 1, "b": 2, "c": 3}',
    "instr-09": "ab" * 20,
    "instr-10": "10 9 8 7 6 5 4 3 2 1",
}

# ---------------------------------------------------------------------------
# tool: 5 correct (one via content-XML fallback), 1 wrong argument
# ---------------------------------------------------------------------------
TOOL = {
    "tool-01": {"content": "", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Rome", "unit": "celsius"}}]},
    "tool-02": {"content": "", "tool_calls": [{"name": "add_event", "arguments": {
        "title": "Sprint Review", "date": "June 10, 2025", "duration_minutes": 45}}]},
    "tool-03": {"content": "", "tool_calls": [{"name": "search_files", "arguments": {
        "path": "/home/user/project", "pattern": "*.py", "max_results": 20}}]},
    # No structured tool_calls: exercises the qwen3_xml content fallback.
    "tool-04": {"content": '<tool_call>\n{"name": "send_email", "arguments": '
                          '{"to": "alice@example.com", "subject": "Meeting tomorrow", '
                          '"body": "Can we meet at 10am?"}}\n</tool_call>',
                "tool_calls": []},
    "tool-05": {"content": "", "tool_calls": [{"name": "get_stock_price", "arguments": {"ticker": "NVDA"}}]},
    # Wrong target currency on purpose.
    "tool-06": {"content": "", "tool_calls": [{"name": "convert_currency", "arguments": {
        "amount": 100, "from": "USD", "to": "GBP"}}]},
}

# ---------------------------------------------------------------------------
# long-context needle: 32k + 128k recalled, 262k missed
# ---------------------------------------------------------------------------
NEEDLE = {
    "needle-32768": {"content": "quasar-4419-zebra", "usage": {"prompt_tokens": 32011},
                   "ttft_s": 1.2, "latency_s": 1.4},
    "needle-131072": {"content": "The passphrase is: quasar-4419-zebra",
                    "usage": {"prompt_tokens": 128014}, "ttft_s": 4.4, "latency_s": 4.7},
    "needle-262144": {"content": "I could not find that information in the document.",
                    "usage": {"prompt_tokens": 260001}, "ttft_s": 9.9, "latency_s": 10.1},
}


_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
          "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
          "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
          "yankee", "zulu", "amber", "basalt", "cobalt", "dune", "ember", "fjord",
          "glacier", "harbor", "iris", "jasper", "kelp", "lumen", "marsh", "nectar"]
_NOUNS = ["entry", "record", "node", "bucket", "segment", "ledger", "channel", "stream"]
_TEMPLATES = [
    "        value_%03d = self._lookup(key_%03d)",
    "        value_%03d = self._cache.get(key_%03d, default_%03d)",
    "        value_%03d = self._tree.search(key_%03d)  # descend the heap",
    "        value_%03d = self._slow_path(key_%03d) if self._dirty else default_%03d",
    "        value_%03d = self._merge(left_%03d, right_%03d)  # combine both branches",
    "        value_%03d = self._rebalance(key_%03d, depth=%d)",
]


def repetition_text() -> str:
    """A long, non-degenerate code-ish output (~4-5k tokens, varied lines).

    Deliberately varied so it exercises the "healthy output" side of the
    degeneracy detector: no 20-gram repeats, high bigram diversity.
    """
    lines = [
        "class PriorityQueue:",
        '    """A binary min-heap backed priority queue with stable ordering."""',
        "",
    ]
    for index in range(1, 81):
        adjective = _WORDS[(index * 7) % len(_WORDS)]
        noun = _WORDS[(index * 13 + 5) % len(_WORDS)]
        template = _TEMPLATES[index % len(_TEMPLATES)]
        lines.append("    def operation_%03d(self, key_%03d):" % (index, index))
        lines.append('        """Handle the %s %s case for %s number %d."""'
                     % (adjective, noun, _NOUNS[index % len(_NOUNS)], index))
        if template.count("%") >= 3:
            lines.append(template % ((index,) * template.count("%")))
        else:
            lines.append(template % (index, index))
        lines.append("        if value_%03d is None:  # guard %d" % (index, index))
        lines.append("            return default_%03d" % index)
        lines.append("        return self._merge(value_%03d, key_%03d)  # %s"
                     % (index, index, adjective))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    canned = {
        "source": "hand-authored fake model outputs for offline grader tests",
        "generated_by": "quality/tests/build_canned.py",
        "categories": {
            "math": [{"id": key, "content": "%s\n\nANSWER: %s" % (rationale, answer)}
                     for key, (rationale, answer) in MATH.items()],
            "json": [{"id": key, "content": content} for key, content in JSON_TASKS.items()],
            "code": [{"id": key, "content": PY % source} for key, source in CODE.items()],
            "instruction": [{"id": key, "content": content} for key, content in INSTRUCTION.items()],
            "tool": [{"id": key, **value} for key, value in TOOL.items()],
            "needle": [{"id": key, **value} for key, value in NEEDLE.items()],
            "repetition": [{"id": "repetition-01", "content": repetition_text()}],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(canned, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote %s (%d bytes)" % (OUT, OUT.stat().st_size))


if __name__ == "__main__":
    main()
