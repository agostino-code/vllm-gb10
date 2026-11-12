#!/usr/bin/env python3
"""Static validation of the shipped prompt/test data and CLI data plumbing.

No server, no GPU, no network.

    python3 -m unittest discover -s quality/tests -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

QUALITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALITY_DIR))

from qlib import batteries, graders  # noqa: E402
from qlib.needle import build_needle_prompt  # noqa: E402

DATA = batteries.load_data()
CANNED_PATH = QUALITY_DIR / "tests" / "canned" / "model_outputs.json"
EXPECTED_COUNTS = {"math": 20, "json": 10, "code": 8, "instruction": 10, "tool": 6, "repetition": 1}


class DataIntegrityTests(unittest.TestCase):
    def test_expected_counts(self):
        for category, want in EXPECTED_COUNTS.items():
            self.assertEqual(len(DATA[category]), want, category)

    def test_unique_ids(self):
        for category, items in DATA.items():
            if not isinstance(items, list):
                continue
            ids = [item.get("id") for item in items]
            self.assertEqual(len(ids), len(set(ids)), category)
            for item_id in ids:
                self.assertTrue(item_id, category)

    def test_math_answers_are_numeric(self):
        for item in DATA["math"]:
            answer = str(item["answer"]).replace(",", "")
            self.assertIsNotNone(graders._first_number(answer), item["id"])
            self.assertTrue(item["prompt"].strip(), item["id"])

    def test_code_tests_are_valid_expressions(self):
        for item in DATA["code"]:
            self.assertTrue(item["function"], item["id"])
            self.assertIn(item["function"], item["prompt"], item["id"])
            self.assertGreaterEqual(len(item["tests"]), 3, item["id"])
            for expr in item["tests"]:
                compile(expr, "<test:%s>" % item["id"], "eval")
            verdict = graders.grade_code(
                item,
                "```python\ndef %s(*args, **kwargs):\n    raise NotImplementedError\n```" % item["function"],
            )
            self.assertEqual(verdict["score"], 0, item["id"])
            self.assertFalse(verdict["status"] == "no_code", item["id"])

    def test_json_schemas_are_usable(self):
        for item in DATA["json"]:
            schema = item["schema"]
            self.assertIsInstance(schema, dict, item["id"])
            self.assertIn("type", schema, item["id"])
            # A validator that receives a deliberately wrong payload must report errors.
            errors = graders.validate_schema({"__wrong__": []}, schema)
            self.assertTrue(errors, item["id"])
            # Validating the wrong type must not crash.
            graders.validate_schema([], schema)
            self.assertTrue(item["prompt"].strip(), item["id"])

    def test_instruction_checks_are_known_types(self):
        for item in DATA["instruction"]:
            self.assertTrue(item["checks"], item["id"])
            for check in item["checks"]:
                self.assertIn("type", check, item["id"])
                result = graders.run_checks("", [check])
                details = " ".join(entry.get("detail", "") for entry in result["checks"])
                self.assertNotIn("UNKNOWN CHECK TYPE", details, item["id"])

    def test_tool_items_have_valid_schemas(self):
        for item in DATA["tool"]:
            self.assertTrue(item["tools"], item["id"])
            for tool in item["tools"]:
                self.assertEqual(tool["type"], "function")
                fn = tool["function"]
                self.assertTrue(fn["name"], item["id"])
                self.assertEqual(fn["parameters"]["type"], "object")
                self.assertIn("required", fn["parameters"])
            self.assertTrue(item["expect"]["name"], item["id"])
            self.assertIsInstance(item["expect"]["args"], dict, item["id"])

    def test_needle_data(self):
        needle = DATA["needle"]
        self.assertTrue(needle["fact"])
        self.assertIn("{fact}", needle["context_template"])
        self.assertIn("passphrase", needle["question"].lower())
        for default_size in batteries.DEFAULT_NEEDLE_SIZES:
            self.assertLess(default_size, 409600)
        self.assertEqual(batteries.DEFAULT_NEEDLE_SIZES[-1], 262144)

    def test_check_data_clean(self):
        self.assertEqual(run_quality_check_data(), [])


def run_quality_check_data():
    sys.path.insert(0, str(QUALITY_DIR))
    import run_quality

    return run_quality.check_data(DATA)


class CannedCoverageTests(unittest.TestCase):
    def setUp(self):
        if not CANNED_PATH.exists():
            self.skipTest("canned outputs missing; run quality/tests/build_canned.py")
        self.canned = json.loads(CANNED_PATH.read_text(encoding="utf-8"))["categories"]

    def test_all_items_covered(self):
        for category in ("math", "json", "code", "instruction", "tool", "repetition"):
            have = {entry["id"] for entry in self.canned[category]}
            want = {item["id"] for item in DATA[category]}
            self.assertEqual(have, want, category)

    def test_needle_sizes_covered(self):
        have = {entry["id"] for entry in self.canned["needle"]}
        for size in batteries.DEFAULT_NEEDLE_SIZES:
            self.assertIn("needle-%d" % size, have)

    def test_canned_math_has_expected_pass_fail_split(self):
        by_id = {entry["id"]: entry["content"] for entry in self.canned["math"]}
        items = {item["id"]: item for item in DATA["math"]}
        passed = sum(graders.grade_math(items[item_id], by_id[item_id])["score"] for item_id in items)
        self.assertEqual(passed, 18)


class CliParsingTests(unittest.TestCase):
    def test_parse_categories_all(self):
        self.assertEqual(batteries.parse_categories("all"), batteries.CATEGORIES)
        self.assertEqual(batteries.parse_categories(None), batteries.CATEGORIES)

    def test_parse_categories_quick(self):
        self.assertNotIn("needle", batteries.parse_categories("quick"))
        self.assertEqual(len(batteries.parse_categories("quick")), 6)

    def test_parse_categories_lists_and_negation(self):
        self.assertEqual(batteries.parse_categories("math,json"), ["math", "json"])
        self.assertEqual(batteries.parse_categories("all,-needle"),
                         [c for c in batteries.CATEGORIES if c != "needle"])
        self.assertEqual(batteries.parse_categories("needle"), ["needle"])

    def test_parse_categories_aliases(self):
        self.assertEqual(batteries.parse_categories("long-context"), ["needle"])
        self.assertEqual(batteries.parse_categories("tools"), ["tool"])
        self.assertEqual(batteries.parse_categories("instruction-following"), ["instruction"])

    def test_parse_categories_unknown(self):
        with self.assertRaises(ValueError):
            batteries.parse_categories("poetry")

    def test_parse_needle_sizes(self):
        self.assertEqual(batteries.parse_needle_sizes("32k,128k"), [32768, 131072])
        self.assertEqual(batteries.parse_needle_sizes("32768"), [32768])
        self.assertEqual(batteries.parse_needle_sizes("262144"), [262144])
        self.assertEqual(batteries.parse_needle_sizes("all"), batteries.DEFAULT_NEEDLE_SIZES)
        with self.assertRaises(ValueError):
            batteries.parse_needle_sizes("nope")


class NeedlePromptTests(unittest.TestCase):
    def test_deterministic_build(self):
        first = build_needle_prompt(1024)
        second = build_needle_prompt(1024)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.meta()["sha256"], second.meta()["sha256"])

    def test_sizes_scale(self):
        small = build_needle_prompt(1024)
        large = build_needle_prompt(4096)
        self.assertGreater(len(large.prompt), len(small.prompt) * 3)

    def test_actual_prompt_token_estimate_below_context(self):
        # char heuristic 3.5 -> a 262k-token request must stay well under 409600.
        built = build_needle_prompt(262144)
        self.assertLess(built.chars / 3.0, 409600)
        self.assertIn("quasar-4419-zebra", built.prompt)

    def test_default_sizes_fit_context(self):
        for size in batteries.DEFAULT_NEEDLE_SIZES:
            estimated = size * 0.9  # generous: chars_per_token 3.5 vs ~3.9 actual
            self.assertLess(estimated + 256, 409600)


if __name__ == "__main__":
    unittest.main(verbosity=2)
