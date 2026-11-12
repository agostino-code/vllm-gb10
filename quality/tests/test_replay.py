#!/usr/bin/env python3
"""End-to-end offline tests of the CLI using canned model outputs.

These run the same code path as a live run, but with ``--self-test`` (replay),
so they need no server, no GPU and no network.

    python3 -m unittest discover -s quality/tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

QUALITY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QUALITY_DIR))

import run_quality  # noqa: E402

CANNED = QUALITY_DIR / "tests" / "canned" / "model_outputs.json"


class SelfTestRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="quality-selftest-")
        cls.out = Path(cls.tmp.name) / "run"
        cls.rc = run_quality.main(["--self-test", "--out", str(cls.out), "--quiet"])
        cls.payload = json.loads((cls.out / "scores.json").read_text(encoding="utf-8"))
        cls.categories = cls.payload["categories"]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_canned_file_exists(self):
        self.assertTrue(CANNED.exists(), "run quality/tests/build_canned.py to regenerate canned outputs")

    def test_exit_code(self):
        self.assertEqual(self.rc, 0)

    def test_overall(self):
        overall = self.payload["overall"]
        self.assertEqual(overall["categories_run"], 7)
        self.assertEqual(overall["items_total"], 58)
        self.assertEqual(overall["items_passed"], 50)
        self.assertAlmostEqual(overall["score"], 0.8536, places=4)
        self.assertFalse(overall["aborted"])

    def test_math(self):
        math = self.categories["math"]
        self.assertEqual((math["passed"], math["total"]), (18, 20))
        self.assertAlmostEqual(math["score"], 0.9)
        by_id = {item["id"]: item for item in math["items"]}
        self.assertEqual(by_id["math-12"]["score"], 1)  # "5,050" numeric match
        self.assertEqual(by_id["math-04"]["score"], 0)  # wrong answer
        self.assertEqual(by_id["math-13"]["score"], 0)  # wrong answer

    def test_json(self):
        block = self.categories["json"]
        self.assertEqual((block["passed"], block["total"]), (8, 10))
        self.assertAlmostEqual(block["score"], 0.8)
        self.assertEqual(block["valid_json_count"], 9)
        self.assertAlmostEqual(block["valid_json_rate"], 0.9)
        by_id = {item["id"]: item for item in block["items"]}
        self.assertFalse(by_id["json-09"]["valid_json"])  # malformed JSON
        self.assertTrue(by_id["json-05"]["valid_json"])  # valid JSON, schema fail
        self.assertEqual(by_id["json-05"]["score"], 0)

    def test_code(self):
        block = self.categories["code"]
        self.assertEqual((block["passed"], block["total"]), (7, 8))
        self.assertAlmostEqual(block["score"], 0.875)
        self.assertEqual(block["tests_total"], 37)
        self.assertEqual(block["tests_passed"], 33)
        self.assertAlmostEqual(block["assertion_pass_rate"], 0.9, places=4)  # mean per-item test fraction
        by_id = {item["id"]: item for item in block["items"]}
        self.assertEqual(by_id["code-04"]["score"], 0)  # broken fibonacci
        self.assertEqual(by_id["code-04"]["tests_passed"], 1)

    def test_instruction(self):
        block = self.categories["instruction"]
        self.assertEqual((block["passed"], block["total"]), (9, 10))
        self.assertEqual(block["checks_total"], 21)
        self.assertEqual(block["checks_passed"], 20)
        by_id = {item["id"]: item for item in block["items"]}
        self.assertEqual(by_id["instr-01"]["score"], 0)  # word count violated

    def test_tool(self):
        block = self.categories["tool"]
        self.assertEqual((block["passed"], block["total"]), (5, 6))
        self.assertAlmostEqual(block["score"], 0.8333, places=4)
        self.assertEqual(block["server_parsed_count"], 5)
        self.assertEqual(block["fallback_parse_count"], 1)
        self.assertEqual(block["reason_counts"].get("arg_mismatch"), 1)
        by_id = {item["id"]: item for item in block["items"]}
        self.assertEqual(by_id["tool-04"]["source"], "content_xml_fallback")
        self.assertEqual(by_id["tool-06"]["score"], 0)  # wrong currency

    def test_needle(self):
        block = self.categories["needle"]
        self.assertEqual((block["passed"], block["total"]), (2, 3))
        self.assertAlmostEqual(block["score"], 0.6667, places=4)
        self.assertAlmostEqual(block["strict_recall_rate"], 0.6667, places=4)
        self.assertEqual(block["prompt_tokens"], [32011, 128014, 260001])
        self.assertEqual(block["prompt_tokens_ok"], [True, True, True])
        # The synthetic filler is ~3.5 chars/token; canned usage proves the sizing.
        for chars_per_token in block["chars_per_token_actual"]:
            self.assertAlmostEqual(chars_per_token, 3.55, delta=0.15)
        self.assertAlmostEqual(block["ttft_s"], 5.1667, places=4)
        by_id = {item["id"]: item for item in block["items"]}
        self.assertEqual(by_id["needle-32768"]["recall"], True)
        self.assertEqual(by_id["needle-131072"]["recall"], True)
        self.assertEqual(by_id["needle-262144"]["recall"], False)

    def test_repetition(self):
        block = self.categories["repetition"]
        self.assertEqual((block["passed"], block["total"]), (1, 1))
        self.assertFalse(block["metrics"]["degenerate"])
        self.assertGreater(block["metrics"]["n_tokens"], 200)

    def test_raw_outputs_and_summary(self):
        for category in ("math", "json", "code", "instruction", "tool", "needle", "repetition"):
            path = self.out / "raw" / ("%s.jsonl" % category)
            self.assertTrue(path.exists(), path)
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), self.categories[category]["total"])
            for record in lines:
                self.assertIn("response", record)
                self.assertIn("grade", record)
                # Prompts are never saved unless --save-prompts: keep raw small.
                self.assertNotIn("messages", record["request"])
        self.assertTrue((self.out / "summary.md").exists())
        self.assertTrue((self.out / "run.log").exists())


class OutputAliasTests(unittest.TestCase):
    """bench-harness calls: --mode quick --output <run>/quality.json"""

    def test_mode_quick_with_output_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "m0" / "quality.json"
            rc = run_quality.main(["--self-test", "--mode", "quick", "--output", str(target), "--quiet"])
            self.assertEqual(rc, 0)
            self.assertTrue(target.exists(), target)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["mode_arg"], "quick")
            self.assertNotIn("needle", payload["categories"])
            self.assertEqual(len(payload["categories"]), 6)
            artifacts = target.parent / (target.stem + ".artifacts")
            self.assertTrue((artifacts / "raw").is_dir())
            self.assertTrue((artifacts / "summary.md").exists())

    def test_mode_full_includes_needle(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "quality.json"
            rc = run_quality.main(["--self-test", "--mode", "full", "--output", str(target), "--quiet"])
            self.assertEqual(rc, 0)
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertIn("needle", payload["categories"])
            self.assertEqual(len(payload["categories"]), 7)

    def test_output_without_json_suffix_is_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dirout"
            rc = run_quality.main(["--self-test", "--categories", "math", "--output", str(out), "--quiet"])
            self.assertEqual(rc, 0)
            self.assertTrue((out / "scores.json").exists())


class SelectionAndGatingTests(unittest.TestCase):
    def test_category_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            rc = run_quality.main(["--self-test", "--categories", "math,json", "--out", str(out), "--quiet"])
            self.assertEqual(rc, 0)
            payload = json.loads((out / "scores.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(payload["categories"].keys()), ["json", "math"])
            self.assertAlmostEqual(payload["overall"]["score"], 0.85, places=4)  # (0.9 + 0.8) / 2

    def test_quick_profile_excludes_needle(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            rc = run_quality.main(["--self-test", "--categories", "quick", "--out", str(out), "--quiet"])
            self.assertEqual(rc, 0)
            payload = json.loads((out / "scores.json").read_text(encoding="utf-8"))
            self.assertNotIn("needle", payload["categories"])
            self.assertEqual(len(payload["categories"]), 6)

    def test_needle_size_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            rc = run_quality.main(["--self-test", "--categories", "needle", "--needle-sizes", "32k,128k",
                                   "--out", str(out), "--quiet"])
            self.assertEqual(rc, 0)
            payload = json.loads((out / "scores.json").read_text(encoding="utf-8"))
            needle = payload["categories"]["needle"]
            self.assertEqual((needle["passed"], needle["total"]), (2, 2))
            self.assertEqual([item["target_tokens"] for item in needle["items"]], [32768, 131072])

    def test_min_score_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            rc = run_quality.main(["--self-test", "--out", str(out), "--quiet", "--min-score", "0.99"])
            self.assertEqual(rc, 2)

    def test_missing_canned_output_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            canned = json.loads(CANNED.read_text(encoding="utf-8"))
            del canned["categories"]["math"][0]
            path = Path(tmp) / "incomplete.json"
            path.write_text(json.dumps(canned), encoding="utf-8")
            out = Path(tmp) / "run"
            rc = run_quality.main(["--replay", str(path), "--categories", "math", "--out", str(out), "--quiet"])
            self.assertEqual(rc, 0)  # graders still run; the missing id surfaces as an error
            payload = json.loads((out / "scores.json").read_text(encoding="utf-8"))
            errors = payload["categories"]["math"]["errors"]
            self.assertTrue(any("missing canned output" in err for err in errors), errors)

    def test_needle_low_prompt_tokens_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            canned = json.loads(CANNED.read_text(encoding="utf-8"))
            for entry in canned["categories"]["needle"]:
                if entry["id"] == "needle-32768":
                    entry["usage"]["prompt_tokens"] = 1000  # far below target
            path = Path(tmp) / "low_tokens.json"
            path.write_text(json.dumps(canned), encoding="utf-8")
            out = Path(tmp) / "run"
            rc = run_quality.main(["--replay", str(path), "--categories", "needle",
                                   "--needle-sizes", "32k", "--out", str(out), "--quiet"])
            self.assertEqual(rc, 0)
            payload = json.loads((out / "scores.json").read_text(encoding="utf-8"))
            needle = payload["categories"]["needle"]
            self.assertFalse(needle["items"][0]["prompt_tokens_ok"])
            self.assertTrue(any("prompt_tokens" in err for err in needle["errors"]), needle["errors"])

    def test_unreachable_server_exits_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            rc = run_quality.main(["--base-url", "http://127.0.0.1:9/v1", "--categories", "math",
                                   "--out", str(out)])
            self.assertEqual(rc, 3)

    def test_no_server_needed_for_dry_run(self):
        rc = run_quality.main(["--dry-run", "--base-url", "http://127.0.0.1:9/v1"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
