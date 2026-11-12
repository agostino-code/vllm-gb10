#!/usr/bin/env python3
"""Offline unit tests for the deterministic graders.

No server and no network required: every test feeds canned model output to a
grader and asserts the verdict.

    python3 -m unittest discover -s quality/tests -v
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

QUALITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALITY_DIR))

from qlib import graders  # noqa: E402
from qlib.needle import DEFAULT_NEEDLE, build_filler, build_needle_prompt, grade_needle  # noqa: E402


class MathGraderTests(unittest.TestCase):
    def test_answer_marker_exact(self):
        verdict = graders.grade_math({"id": "m", "answer": "391"}, "17 * 23 = 391.\n\nANSWER: 391")
        self.assertEqual(verdict["score"], 1)
        self.assertEqual(verdict["mode"], "numeric")
        self.assertTrue(verdict["has_answer_marker"])

    def test_numeric_thousands_separator(self):
        verdict = graders.grade_math({"id": "m", "answer": "5050"}, "ANSWER: 5,050")
        self.assertEqual(verdict["score"], 1)

    def test_decimal_equivalence(self):
        verdict = graders.grade_math({"id": "m", "answer": "0.7"}, "ANSWER: .70")
        self.assertEqual(verdict["score"], 1)

    def test_units_after_number(self):
        verdict = graders.grade_math({"id": "m", "answer": "150"}, "ANSWER: 150 km")
        self.assertEqual(verdict["score"], 1)

    def test_wrong_answer(self):
        verdict = graders.grade_math({"id": "m", "answer": "150"}, "ANSWER: 180")
        self.assertEqual(verdict["score"], 0)
        self.assertEqual(verdict["mode"], "mismatch")

    def test_boxed_answer(self):
        verdict = graders.grade_math({"id": "m", "answer": "22"}, "So the value is $\\boxed{22}$.")
        self.assertEqual(verdict["score"], 1)

    def test_last_line_fallback(self):
        verdict = graders.grade_math({"id": "m", "answer": "49"}, "After simplifying we get\n49")
        self.assertEqual(verdict["score"], 1)

    def test_accept_list(self):
        item = {"id": "m", "answer": "0.5", "accept": ["1/2"]}
        self.assertEqual(graders.grade_math(item, "ANSWER: 1/2")["score"], 1)

    def test_empty_output(self):
        verdict = graders.grade_math({"id": "m", "answer": "1"}, "")
        self.assertEqual(verdict["score"], 0)


class JsonGraderTests(unittest.TestCase):
    SCHEMA = {
        "type": "object",
        "exactly_keys": ["a", "b"],
        "required": ["a", "b"],
        "properties": {"a": {"type": "integer"}, "b": {"type": "string", "enum": ["x", "y"]}},
    }

    def test_valid_object(self):
        verdict = graders.grade_json({"id": "j", "schema": self.SCHEMA}, '{"a": 1, "b": "x"}')
        self.assertEqual(verdict["score"], 1)
        self.assertTrue(verdict["valid_json"])

    def test_fenced_json(self):
        verdict = graders.grade_json({"id": "j", "schema": self.SCHEMA}, '```json\n{"a": 1, "b": "y"}\n```')
        self.assertEqual(verdict["score"], 1)

    def test_json_embedded_in_prose(self):
        verdict = graders.grade_json(
            {"id": "j", "schema": self.SCHEMA}, 'Here is the JSON:\n{"a": 2, "b": "x"}\nDone.'
        )
        self.assertEqual(verdict["score"], 1)

    def test_invalid_json(self):
        verdict = graders.grade_json({"id": "j", "schema": self.SCHEMA}, "not json at all")
        self.assertEqual(verdict["score"], 0)
        self.assertFalse(verdict["valid_json"])

    def test_missing_required_key(self):
        verdict = graders.grade_json({"id": "j", "schema": self.SCHEMA}, '{"a": 1}')
        self.assertEqual(verdict["score"], 0)
        self.assertTrue(verdict["valid_json"])

    def test_wrong_type(self):
        verdict = graders.grade_json({"id": "j", "schema": self.SCHEMA}, '{"a": "1", "b": "x"}')
        self.assertEqual(verdict["score"], 0)

    def test_enum_violation(self):
        verdict = graders.grade_json({"id": "j", "schema": self.SCHEMA}, '{"a": 1, "b": "z"}')
        self.assertEqual(verdict["score"], 0)

    def test_extra_key_rejected(self):
        verdict = graders.grade_json({"id": "j", "schema": self.SCHEMA}, '{"a": 1, "b": "x", "c": 3}')
        self.assertEqual(verdict["score"], 0)

    def test_bool_is_not_integer(self):
        verdict = graders.grade_json({"id": "j", "schema": self.SCHEMA}, '{"a": true, "b": "x"}')
        self.assertEqual(verdict["score"], 0)

    def test_array_constraints(self):
        schema = {"type": "array", "minItems": 3, "maxItems": 3, "uniqueItems": True,
                  "items": {"type": "integer", "minimum": 0, "maximum": 9}}
        self.assertEqual(graders.grade_json({"id": "j", "schema": schema}, "[1, 2, 3]")["score"], 1)
        self.assertEqual(graders.grade_json({"id": "j", "schema": schema}, "[1, 2]")["score"], 0)
        self.assertEqual(graders.grade_json({"id": "j", "schema": schema}, "[1, 1, 2]")["score"], 0)
        self.assertEqual(graders.grade_json({"id": "j", "schema": schema}, "[1, 2, 10]")["score"], 0)

    def test_pattern_and_nested_schema(self):
        schema = {"type": "array", "minItems": 2,
                  "items": {"type": "object", "required": ["sku"],
                            "properties": {"sku": {"type": "string", "pattern": "^SKU-[0-9]{4}$"}}}}
        self.assertEqual(graders.grade_json({"id": "j", "schema": schema},
                                            '[{"sku": "SKU-1234"}, {"sku": "SKU-0000"}]')["score"], 1)
        self.assertEqual(graders.grade_json({"id": "j", "schema": schema},
                                            '[{"sku": "SKU-12"}, {"sku": "SKU-0000"}]')["score"], 0)

    def test_additional_properties_schema(self):
        schema = {"type": "object", "required": ["summary"], "minProperties": 2,
                  "properties": {"summary": {"type": "string"}},
                  "additionalProperties": {"type": "string"}}
        self.assertEqual(graders.grade_json({"id": "j", "schema": schema},
                                            '{"summary": "ok", "extra": "text"}')["score"], 1)
        self.assertEqual(graders.grade_json({"id": "j", "schema": schema},
                                            '{"summary": "ok", "extra": 5}')["score"], 0)


class CodeGraderTests(unittest.TestCase):
    ITEM = {"id": "c", "function": "add", "tests": ["add(1, 2) == 3", "add(0, 0) == 0"]}

    def test_good_code(self):
        verdict = graders.grade_code(self.ITEM, "```python\ndef add(a, b):\n    return a + b\n```")
        self.assertEqual(verdict["score"], 1)
        self.assertEqual(verdict["tests_passed"], 2)
        self.assertEqual(verdict["status"], "ok")

    def test_broken_code(self):
        verdict = graders.grade_code(self.ITEM, "```python\ndef add(a, b):\n    return 42\n```")
        self.assertEqual(verdict["score"], 0)
        self.assertEqual(verdict["tests_passed"], 0)

    def test_partial_code_gets_zero_item_score(self):
        item = {"id": "c", "function": "add", "tests": ["add(1, 2) == 3", "add(5, 5) == 10"]}
        code = "```python\ndef add(a, b):\n    return a + b if b < 5 else a - b\n```"
        verdict = graders.grade_code(item, code)
        self.assertEqual(verdict["score"], 0)
        self.assertEqual(verdict["tests_passed"], 1)
        self.assertAlmostEqual(verdict["tests_ok_fraction"], 0.5)

    def test_no_code(self):
        verdict = graders.grade_code(self.ITEM, "I cannot help with that.")
        self.assertEqual(verdict["score"], 0)
        self.assertEqual(verdict["status"], "no_code")

    def test_unfenced_code_fallback(self):
        verdict = graders.grade_code(self.ITEM, "def add(a, b):\n    return a + b")
        self.assertEqual(verdict["score"], 1)

    def test_main_guard_is_not_executed(self):
        code = (
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    print(add(1, 1))\n"
            "```"
        )
        self.assertEqual(graders.grade_code(self.ITEM, code)["score"], 1)

    def test_infinite_loop_times_out(self):
        item = {"id": "c", "function": "f", "tests": ["f() == 1"], "timeout": 2}
        started = time.perf_counter()
        verdict = graders.grade_code(item, "```python\ndef f():\n    while True:\n        pass\n```")
        elapsed = time.perf_counter() - started
        self.assertEqual(verdict["score"], 0)
        self.assertEqual(verdict["status"], "timeout")
        self.assertLess(elapsed, 15.0)


class InstructionGraderTests(unittest.TestCase):
    def assert_passes(self, text, checks):
        self.assertTrue(graders.run_checks(text, checks)["passed"], text)

    def assert_fails(self, text, checks):
        self.assertFalse(graders.run_checks(text, checks)["passed"], text)

    def test_word_count(self):
        checks = [{"type": "word_count_eq", "value": 3}]
        self.assert_passes("one two three", checks)
        self.assert_fails("one two", checks)
        self.assert_fails("one two three four", checks)

    def test_line_constraints(self):
        self.assert_passes("a\nb\nc", [{"type": "line_count_eq", "value": 3}])
        self.assert_fails("a\nb", [{"type": "line_count_eq", "value": 3}])
        # All checks must pass: the first is true, the second false -> fail.
        self.assert_fails("a\nb\nc", [{"type": "line_count_eq", "value": 3},
                                       {"type": "each_line_startswith", "value": "x"}])

    def test_bullet_prefix(self):
        self.assert_passes("- a\n- b", [{"type": "line_count_eq", "value": 2},
                                        {"type": "each_line_startswith", "value": "- "}])

    def test_primes(self):
        self.assert_passes("2\n3\n5\n7\n11", [{"type": "all_lines_prime"}])
        self.assert_fails("2\n4\n5", [{"type": "all_lines_prime"}])

    def test_no_space_and_split(self):
        self.assert_passes("red,blue,green,yellow", [{"type": "split_count", "sep": ",", "value": 4},
                                                     {"type": "no_space"}])
        self.assert_fails("red, blue,green,yellow", [{"type": "split_count", "sep": ",", "value": 4},
                                                     {"type": "no_space"}])

    def test_canonical_equals(self):
        self.assert_passes("  BANANA \n", [{"type": "canonical_equals", "value": "BANANA"}])
        self.assert_fails("banana", [{"type": "canonical_equals", "value": "BANANA"}])

    def test_token_sequence(self):
        self.assert_passes("3 2 1", [{"type": "token_sequence", "value": ["3", "2", "1"]}])
        self.assert_fails("3, 2, 1", [{"type": "token_sequence", "value": ["3", "2", "1"]}])

    def test_json_keys_exact_and_int(self):
        checks = [{"type": "json_keys_exact", "keys": ["a", "b"], "values_int": True}]
        self.assert_passes('{"a": 1, "b": 2}', checks)
        self.assert_fails('{"a": 1, "b": true}', checks)
        self.assert_fails('{"a": 1}', checks)

    def test_absent_char(self):
        self.assert_passes("no commas here", [{"type": "absent_char", "value": ","}])
        self.assert_fails("a, b", [{"type": "absent_char", "value": ","}])

    def test_contains_count(self):
        self.assert_passes("quokka and quokka", [{"type": "contains_count_min", "value": "quokka", "min": 2,
                                                  "case_sensitive": False}])
        self.assert_fails("quokka", [{"type": "contains_count_min", "value": "quokka", "min": 2,
                                      "case_sensitive": False}])

    def test_unknown_check_type_fails(self):
        result = graders.run_checks("anything", [{"type": "does_not_exist"}])
        self.assertFalse(result["passed"])

    def test_empty_checks_fail(self):
        self.assertFalse(graders.run_checks("anything", [])["passed"])


class ToolGraderTests(unittest.TestCase):
    ITEM = {
        "id": "t",
        "expect": {"name": "get_weather", "args": {"city": "Rome", "unit": "celsius"}},
        "arg_specs": {"city": {"match": "contains"}},
    }

    def test_server_parsed_ok(self):
        verdict = graders.grade_tool(self.ITEM, "", [
            {"name": "get_weather", "arguments": {"city": "Rome", "unit": "celsius"}}
        ])
        self.assertEqual(verdict["score"], 1)
        self.assertEqual(verdict["source"], "server_parsed")

    def test_missing_arg(self):
        verdict = graders.grade_tool(self.ITEM, "", [
            {"name": "get_weather", "arguments": {"city": "Rome"}}
        ])
        self.assertEqual(verdict["score"], 0)
        self.assertEqual(verdict["reason"], "arg_mismatch")

    def test_wrong_value(self):
        verdict = graders.grade_tool(self.ITEM, "", [
            {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
        ])
        self.assertEqual(verdict["score"], 0)

    def test_contains_match(self):
        verdict = graders.grade_tool(self.ITEM, "", [
            {"name": "get_weather", "arguments": {"city": "Rome, Italy", "unit": "celsius"}}
        ])
        self.assertEqual(verdict["score"], 1)

    def test_wrong_tool(self):
        verdict = graders.grade_tool(self.ITEM, "", [
            {"name": "get_forecast", "arguments": {"city": "Rome"}}
        ])
        self.assertEqual(verdict["score"], 0)
        self.assertEqual(verdict["reason"], "wrong_tool_name")

    def test_no_tool_calls(self):
        verdict = graders.grade_tool(self.ITEM, "The weather in Rome is sunny.", [])
        self.assertEqual(verdict["score"], 0)
        self.assertEqual(verdict["reason"], "no_tool_calls")

    def test_xml_fallback(self):
        content = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome", "unit": "celsius"}}\n</tool_call>'
        verdict = graders.grade_tool(self.ITEM, content, [])
        self.assertEqual(verdict["score"], 1)
        self.assertEqual(verdict["source"], "content_xml_fallback")

    def test_numeric_string_argument(self):
        item = {"id": "t", "expect": {"name": "convert_currency",
                                      "args": {"amount": 100, "from": "USD", "to": "EUR"}}}
        verdict = graders.grade_tool(item, "", [
            {"name": "convert_currency", "arguments": {"amount": "100", "from": "usd", "to": "eur"}}
        ])
        self.assertEqual(verdict["score"], 1)

    def test_alias_argument(self):
        item = {"id": "t", "expect": {"name": "add_event",
                                      "args": {"date": "2025-06-10", "duration_minutes": 45}},
                "arg_specs": {"date": {"aliases": {"2025-06-10": ["June 10, 2025"]}}}}
        verdict = graders.grade_tool(item, "", [
            {"name": "add_event", "arguments": {"date": "June 10, 2025", "duration_minutes": 45}}
        ])
        self.assertEqual(verdict["score"], 1)


_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
          "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
          "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray",
          "yankee", "zulu", "amber", "basalt", "cobalt", "dune", "ember", "fjord",
          "glacier", "harbor", "iris", "jasper", "kelp", "lumen", "marsh", "nectar"]
_TEMPLATES = [
    "        value_%03d = self._lookup(key_%03d)",
    "        value_%03d = self._cache.get(key_%03d, default_%03d)",
    "        value_%03d = self._tree.search(key_%03d)  # descend the heap",
    "        value_%03d = self._slow_path(key_%03d) if self._dirty else default_%03d",
    "        value_%03d = self._merge(left_%03d, right_%03d)  # combine both branches",
    "        value_%03d = self._rebalance(key_%03d, depth=%d)",
]


def varied_code_text(n=80):
    """Deterministic, varied code-like text with no repeated 20-grams."""
    lines = ["class PriorityQueue:", '    """A binary min-heap backed priority queue."""', ""]
    for index in range(1, n + 1):
        adjective = _WORDS[(index * 7) % len(_WORDS)]
        template = _TEMPLATES[index % len(_TEMPLATES)]
        lines.append("    def operation_%03d(self, key_%03d):" % (index, index))
        lines.append('        """Handle the %s case for entry number %d."""' % (adjective, index))
        if template.count("%") >= 3:
            lines.append(template % ((index,) * template.count("%")))
        else:
            lines.append(template % (index, index))
        lines.append("        if value_%03d is None:  # guard %d" % (index, index))
        lines.append("            return default_%03d" % index)
        lines.append("        return self._merge(value_%03d, key_%03d)  # %s" % (index, index, adjective))
        lines.append("")
    return "\n".join(lines)


class RepetitionGraderTests(unittest.TestCase):
    def _unique_text(self, n=80):
        return varied_code_text(n)

    def test_non_degenerate(self):
        metrics = graders.repetition_metrics(self._unique_text())
        self.assertFalse(metrics["degenerate"], metrics["reasons"])
        self.assertGreater(metrics["distinct_2"], 0.30)
        self.assertGreater(metrics["unique_token_ratio"], 0.10)

    def test_repeated_ngram_flagged(self):
        block = "the quick brown fox jumps over the lazy dog again and again now "
        text = block * 40
        metrics = graders.repetition_metrics(text)
        self.assertTrue(metrics["degenerate"])
        self.assertGreaterEqual(metrics["max_ngram_repeat"], 5)

    def test_identical_lines_flagged(self):
        text = ("this exact same line is repeated many times" + "\n") * 20
        metrics = graders.repetition_metrics(text)
        self.assertTrue(metrics["degenerate"])

    def test_too_short_flagged(self):
        metrics = graders.repetition_metrics("short output")
        self.assertTrue(metrics["degenerate"])
        self.assertIn("too short", " ".join(metrics["reasons"]))

    def test_grade_scores_degenerate_zero(self):
        item = {"id": "r", "ngram": 20}
        good = graders.grade_repetition(item, self._unique_text())
        self.assertEqual(good["score"], 1)
        bad = graders.grade_repetition(item, "same line\n" * 200)
        self.assertEqual(bad["score"], 0)


class NeedleTests(unittest.TestCase):
    def test_exact_recall(self):
        verdict = grade_needle("The passphrase is quasar-4419-zebra.", DEFAULT_NEEDLE["fact"])
        self.assertTrue(verdict["strict"])
        self.assertTrue(verdict["lenient"])

    def test_spaced_recall_is_lenient_only(self):
        verdict = grade_needle("quasar 4419 zebra", DEFAULT_NEEDLE["fact"])
        self.assertFalse(verdict["strict"])
        self.assertTrue(verdict["lenient"])

    def test_miss(self):
        verdict = grade_needle("I could not find a passphrase.", DEFAULT_NEEDLE["fact"])
        self.assertFalse(verdict["lenient"])

    def test_long_echo_not_counted(self):
        text = " ".join(["filler"] * 80) + " quasar 4419 zebra"
        verdict = grade_needle(text, DEFAULT_NEEDLE["fact"])
        self.assertFalse(verdict["lenient"])
        self.assertTrue(verdict["echoed_question"])

    def test_prompt_builder_deterministic(self):
        first = build_needle_prompt(2000, seed=7)
        second = build_needle_prompt(2000, seed=7)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.prompt, second.prompt)

    def test_prompt_builder_seed_changes_text(self):
        self.assertNotEqual(build_needle_prompt(2000, seed=7).sha256,
                            build_needle_prompt(2000, seed=8).sha256)

    def test_needle_is_at_midpoint(self):
        prompt = build_needle_prompt(4000)
        position = prompt.prompt.find(DEFAULT_NEEDLE["fact"])
        self.assertGreater(position, 0)
        ratio = position / len(prompt.prompt)
        self.assertGreater(ratio, 0.3)
        self.assertLess(ratio, 0.7)

    def test_filler_does_not_contain_fact(self):
        filler = build_filler(5000)
        self.assertNotIn("quasar", filler.lower())
        self.assertNotIn("zebra", filler.lower())
        self.assertNotIn("4419", filler)


if __name__ == "__main__":
    unittest.main(verbosity=2)
