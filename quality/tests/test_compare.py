#!/usr/bin/env python3
"""Offline tests for quality/compare.py (the acceptance gate).

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

import compare  # noqa: E402

M0_RUN = QUALITY_DIR / "runs" / "m0-reference" / "quality.json"


def category(items):
    """items: list of (item_id, 0|1) -> category block with a mean score."""
    scores = [1 if passed else 0 for _, passed in items]
    return {
        "score": round(sum(scores) / len(scores), 4),
        "passed": sum(scores),
        "total": len(scores),
        "skipped": False,
        "items": [{"id": item_id, "score": passed} for item_id, passed in items],
    }


def payload(categories: dict, model: str = "test/model") -> dict:
    return {"categories": categories, "config": {"model_resolved": model}, "overall": {}}


def write_run(tmpdir: Path, name: str, data: dict) -> Path:
    path = tmpdir / ("%s.json" % name)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="quality-compare-")
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _paths(self, base: dict, cand: dict):
        return (str(write_run(self.dir, "base", base)),
                str(write_run(self.dir, "cand", cand)))

    def _gate(self, base: dict, cand: dict, *extra):
        base_path, cand_path = self._paths(base, cand)
        diff_path = self.dir / "diff.json"
        rc = compare.main([base_path, cand_path, "--quiet", "--json-diff", str(diff_path), *extra])
        diff = json.loads(diff_path.read_text(encoding="utf-8"))
        return rc, diff

    def test_identical_runs_pass(self):
        data = payload({"math": category([("m%d" % i, 1) for i in range(20)])})
        rc, diff = self._gate(data, data)
        self.assertEqual(rc, 0)
        self.assertEqual(diff["verdict"], "OK")
        self.assertEqual(diff["regressions"], [])

    def test_one_item_drop_within_default_noise(self):
        base = payload({"math": category([("m%d" % i, 1) for i in range(20)])})
        cand = payload({"math": category([("m%d" % i, 1 if i else 0) for i in range(20)])})
        rc, diff = self._gate(base, cand)
        self.assertEqual(rc, 0)
        self.assertAlmostEqual(diff["rows"][0]["delta"], -0.05)
        self.assertAlmostEqual(diff["rows"][0]["tolerance"], 0.05)

    def test_two_item_drop_is_a_regression(self):
        base = payload({"math": category([("m%d" % i, 1) for i in range(20)])})
        cand = payload({"math": category([("m%d" % i, 0 if i < 2 else 1) for i in range(20)])})
        rc, diff = self._gate(base, cand)
        self.assertEqual(rc, 1)
        self.assertEqual(diff["regressions"], ["math"])
        self.assertEqual(diff["rows"][0]["flipped_pass_to_fail"], ["m0", "m1"])

    def test_small_category_noise_scales_with_items(self):
        # 1 of 10 items = 0.10 drop; tolerance for a 10-item category is 0.10.
        base = payload({"instruction": category([("i%d" % i, 1) for i in range(10)])})
        cand = payload({"instruction": category([("i%d" % i, 1 if i else 0) for i in range(10)])})
        rc, diff = self._gate(base, cand)
        self.assertEqual(rc, 0)
        self.assertAlmostEqual(diff["rows"][0]["tolerance"], 0.1)

    def test_noise_zero_is_strict(self):
        base = payload({"math": category([("m%d" % i, 1) for i in range(20)])})
        cand = payload({"math": category([("m%d" % i, 0 if i else 1) for i in range(20)])})
        rc, _ = self._gate(base, cand, "--noise", "0")
        self.assertEqual(rc, 1)

    def test_max_drop_override(self):
        base = payload({"json": category([("j%d" % i, 1) for i in range(10)])})
        cand = payload({"json": category([("j%d" % i, 0 if i < 2 else 1) for i in range(10)])})
        rc, diff = self._gate(base, cand, "--max-drop", "0.25")
        self.assertEqual(rc, 0)
        self.assertAlmostEqual(diff["rows"][0]["delta"], -0.2)
        self.assertAlmostEqual(diff["rows"][0]["tolerance"], 0.25)

    def test_ignore_category(self):
        base = payload({"instruction": category([("i%d" % i, 1) for i in range(10)])})
        cand = payload({"instruction": category([("i%d" % i, 0 if i < 3 else 1) for i in range(10)])})
        rc, _ = self._gate(base, cand, "--ignore", "instruction")
        self.assertEqual(rc, 0)

    def test_unknown_ignore_name_is_usage_error(self):
        data = payload({"math": category([("m0", 1)])})
        base_path, cand_path = self._paths(data, data)
        rc = compare.main([base_path, cand_path, "--quiet", "--ignore", "poetry"])
        self.assertEqual(rc, 2)

    def test_missing_category_is_reported(self):
        base = payload({"math": category([("m0", 1)]), "needle": category([("needle-32768", 1)])})
        cand = payload({"math": category([("m0", 1)])})
        rc, diff = self._gate(base, cand)
        self.assertEqual(rc, 0)
        self.assertEqual(diff["missing_categories"], ["needle"])

    def test_required_missing_category_is_a_regression(self):
        base = payload({"math": category([("m0", 1)]), "needle": category([("needle-32768", 1)])})
        cand = payload({"math": category([("m0", 1)])})
        rc, diff = self._gate(base, cand, "--require-categories", "needle")
        self.assertEqual(rc, 1)
        self.assertIn("needle", diff["regressions"])

    def test_improvement_is_not_a_regression(self):
        base = payload({"math": category([("m%d" % i, 1 if i else 0) for i in range(20)])})
        cand = payload({"math": category([("m%d" % i, 1) for i in range(20)])})
        rc, diff = self._gate(base, cand)
        self.assertEqual(rc, 0)
        self.assertAlmostEqual(diff["delta_overall_common"], 0.05)

    def test_overall_common_uses_shared_categories(self):
        base = payload({"math": category([("m%d" % i, 1) for i in range(20)]),
                        "json": category([("j%d" % i, 1) for i in range(10)])})
        cand = payload({"math": category([("m%d" % i, 0) for i in range(20)]),
                        "json": category([("j%d" % i, 1) for i in range(10)])})
        rc, diff = self._gate(base, cand)
        self.assertEqual(rc, 1)
        self.assertAlmostEqual(diff["baseline_overall_common"], 1.0)
        self.assertAlmostEqual(diff["candidate_overall_common"], 0.5)

    def test_nonexistent_file_is_usage_error(self):
        rc = compare.main([str(self.dir / "nope.json"), str(self.dir / "nope2.json"), "--quiet"])
        self.assertEqual(rc, 2)

    def test_not_a_scores_file_is_usage_error(self):
        path = self.dir / "random.json"
        path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
        rc = compare.main([str(path), str(path), "--quiet"])
        self.assertEqual(rc, 2)

    def test_real_m0_run_matches_itself(self):
        if not M0_RUN.exists():
            self.skipTest("no live M0 run available (expected on a clean checkout)")
        rc = compare.main([str(M0_RUN), str(M0_RUN), "--quiet"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
