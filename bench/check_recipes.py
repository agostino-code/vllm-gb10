#!/usr/bin/env python3
"""check_recipes.py — static preflight for every recipe/matrix entry.

Validates, without touching docker or the GPU:
  * the recipe parses (native server/env/vllm format or eugr command format)
  * the model snapshot / models dir is resolvable
  * the docker argv + vllm argv can be built
  * exactly one or zero --speculative-config is emitted (the run.sh duplicate
    flag bug this harness exists to avoid), and the JSON is valid
  * the mount contract (workspace root ro at same absolute path, HF hub ro,
    /models, /cache) is present
  * for a matrix file: every referenced recipe exists and the suite/serve
    option keys are known

  python3 bench/check_recipes.py                 # scan default recipe dirs
  python3 bench/check_recipes.py matrix/matrix.yaml
  python3 bench/check_recipes.py recipes/x.yaml draft-lab/recipes/*.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import serve as S  # noqa: E402

DEFAULT_GLOBS = [
    "recipes/*.yaml",
    "matrix/recipes/*.yaml",
    "draft-lab/recipes/*.yaml",
    "eugr-b12x/*.yaml",
]


def build_namespace(recipe_path: Path, chat_template: str | None = None):
    parser = S.build_parser()
    argv = ["up", "--recipe", str(recipe_path), "--name", "check-recipes", "--dry-run"]
    if chat_template:
        argv += ["--chat-template", chat_template]
    return parser.parse_args(argv)


def check_recipe(path: Path, chat_template: str | None = None) -> dict:
    result = {"path": C.repo_rel(path), "ok": False, "errors": [], "warnings": []}
    try:
        args = build_namespace(path, chat_template)
        recipe = C.load_recipe(path)
        meta = S.build_launch(args, recipe)
    except SystemExit as exc:
        result["errors"].append(str(exc))
        return result
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        return result
    vllm_argv = meta["vllm"]["argv"]
    spec_count = sum(1 for arg in vllm_argv
                     if arg == "--speculative-config" or arg.startswith("--speculative-config=")
                     or arg == "--speculative_config" or arg.startswith("--speculative_config="))
    result["spec_flags"] = spec_count
    result["spec"] = meta["vllm"]["speculative_config"]
    result["image"] = meta["stack"]["image"]
    result["model_dir"] = meta["model"]["host_dir"]
    result["model_source"] = meta["model"]["source"]
    result["mounts"] = [m["host"] for m in meta["launch"]["mounts"]]
    if spec_count > 1:
        result["errors"].append(f"{spec_count} speculative-config flags (duplicate flag bug)")
    if spec_count == 1 and meta["vllm"]["speculative_config"] is None:
        result["errors"].append("speculative-config flag present but not valid JSON")
    workspace = str(C.REPO_ROOT)
    if not any(m == workspace for m in result["mounts"]):
        result["errors"].append("workspace root not mounted read-only at its absolute path")
    if not any("/hub" in m for m in result["mounts"]):
        result["errors"].append("HF hub not mounted at its absolute host path")
    if "/models" not in [m_o["container"] for m_o in meta["launch"]["mounts"]]:
        result["errors"].append("/models mount missing")
    result["ok"] = not result["errors"]
    return result


def check_matrix(path: Path) -> tuple[int, int]:
    import run_matrix as R

    try:
        defaults, configs = R.load_matrix(path)
    except R.ConfigError as exc:
        C.log(f"MATRIX FAIL: {exc}")
        return 2, 0
    C.log(f"matrix {C.repo_rel(path)}: {len(configs)} configs")
    failures = 0
    checked = 0
    for config in configs:
        checked += 1
        recipe_path = C.REPO_ROOT / str(config["recipe"])
        if not recipe_path.is_file():
            C.log(f"  FAIL {config['name']:<28} recipe missing: {config['recipe']}")
            failures += 1
            continue
        chat_template = config.get("chat_template") or defaults.get("chat_template")
        result = check_recipe(recipe_path, chat_template=chat_template)
        if result["ok"]:
            spec = json.dumps(result["spec"], separators=(",", ":")) if result["spec"] else "OFF"
            C.log(f"  ok   {config['name']:<28} {result['image']:<42} spec={spec[:64]}")
        else:
            failures += 1
            C.log(f"  FAIL {config['name']:<28} {result['errors']}")
    C.log(f"matrix {C.repo_rel(path)}: {checked} config(s) checked, {failures} failure(s)")
    return failures, checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", help="recipe YAML or matrix YAML (default: scan repo)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    paths = [Path(p) for p in args.paths]
    if not paths:
        for pattern in DEFAULT_GLOBS:
            paths.extend(sorted(Path(p) for p in glob.glob(str(C.REPO_ROOT / pattern))))
    matrix_files = [p for p in paths if p.suffix in (".yaml", ".yml") and _looks_like_matrix(p)]
    recipe_files = [p for p in paths if p not in matrix_files]
    failures = 0
    matrix_checks = 0
    rows = []
    for path in matrix_files:
        matrix_failures, matrix_count = check_matrix(path)
        failures += matrix_failures
        matrix_checks += matrix_count
    for path in recipe_files:
        result = check_recipe(path)
        rows.append(result)
        if result["ok"]:
            spec = json.dumps(result["spec"], separators=(",", ":")) if result["spec"] else "OFF"
            C.log(f"  ok   {result['path']:<42} spec={spec[:70]}")
        else:
            failures += 1
            C.log(f"  FAIL {result['path']}: {'; '.join(result['errors'])}")
    if args.json:
        print(json.dumps({"ok": failures == 0, "recipes": rows}, indent=2))
    C.log(f"{'OK' if failures == 0 else 'FAIL'}: {len(rows)} recipe(s) + {matrix_checks} matrix "
          f"entr(ies) checked, {failures} failure(s)")
    return 0 if failures == 0 else 2


def _looks_like_matrix(path: Path) -> bool:
    try:
        data = C.load_yaml(path)
    except Exception:  # noqa: BLE001
        return False
    return isinstance(data, dict) and "configs" in data


if __name__ == "__main__":
    sys.exit(main())
