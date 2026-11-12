#!/usr/bin/env python3
"""run_matrix.py — sequential, resumable matrix runner for vLLM preset
comparisons across docker stacks and serving configurations.

Every config is measured by launching its own container (``serve.py up``),
running the measurement suite (``suite.py``), then stopping the container.  The
runner is resumable: ``results/<name>/status.json`` records progress, finished
configs are skipped unless ``--force`` is given, and a stale container from a
crashed run is cleaned up before retrying.

  python3 bench/run_matrix.py --matrix bench/matrix.yaml --dry-run
  python3 bench/run_matrix.py --matrix matrix/matrix.yaml --only az-ref
  python3 bench/run_matrix.py --matrix matrix/matrix.yaml          # long: use nohup/background

Never run two model servers at once: configs are strictly sequential and the
runner waits for the port/container to be free before each boot.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

MATRIX_VERSION = "1.0"

# suite.py CLI option for each matrix config `suite:` key
SUITE_KEY_TO_FLAG = {
    "repeats": "--repeats",
    "decode_max_tokens": "--decode-max-tokens",
    "decode_prompts_per_category": "--decode-prompts-per-category",
    "conc_waves": "--conc-waves",
    "conc_max_tokens": "--conc-max-tokens",
    "long_target_tokens": "--long-target-tokens",
    "prefill_timeout": "--prefill-timeout",
    "request_timeout": "--request-timeout",
    "temperature": "--temperature",
    "seed": "--seed",
    "retries": "--retries",
    "inter_request_delay": "--inter-request-delay",
    "quality": "--quality",
    "quality_timeout": "--quality-timeout",
    "tag": "--tag",
}
SUITE_LIST_FLAGS = {
    "concurrency": "--concurrency",
    "prefill_targets": "--prefill-targets",
}
SUITE_BOOL_FLAGS = {
    "include_16": "--include-16",
    "long_smoke": "--long-smoke",
    "strict": "--strict",
    "thinking": "--thinking",
}


# ──────────────────────────────────────────────────────────────────────────────
# matrix loading / validation
# ──────────────────────────────────────────────────────────────────────────────


class ConfigError(Exception):
    pass


def load_matrix(path: Path) -> tuple[dict, list[dict]]:
    raw = C.load_yaml(path)
    defaults = dict(raw.get("defaults") or {})
    configs = raw.get("configs") or []
    if not isinstance(configs, list) or not configs:
        raise ConfigError(f"{path}: no configs list found")
    names = set()
    normalized: list[dict] = []
    for index, entry in enumerate(configs):
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: config #{index} is not a mapping")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ConfigError(f"{path}: config #{index} has no name")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ConfigError(f"{path}: config name {name!r} must match [A-Za-z0-9_.-]+")
        if name in names:
            raise ConfigError(f"{path}: duplicate config name {name!r}")
        names.add(name)
        if not entry.get("recipe"):
            raise ConfigError(f"{path}: config {name!r} has no recipe")
        entry = dict(entry)
        entry["name"] = name
        entry.setdefault("priority", index)
        entry.setdefault("enabled", True)
        normalized.append(entry)
    normalized.sort(key=lambda e: (e.get("priority", 0), e["name"]))
    return defaults, normalized


def expand(value, defaults):
    """Apply simple ${defaults.key} substitution inside matrix strings."""
    if not isinstance(value, str):
        return value

    def repl(match):
        key = match.group(1)
        if key in defaults:
            return str(defaults[key])
        return match.group(0)

    return re.sub(r"\$\{([A-Za-z0-9_.-]+)\}", repl, value)


def merge_suite(defaults: dict, config: dict) -> dict:
    base = dict(defaults.get("suite") or {})
    base.update(config.get("suite") or {})
    return base


def merge_env(defaults: dict, config: dict) -> dict:
    base = dict(defaults.get("env") or {})
    base.update(config.get("env") or {})
    return {str(k): "" if v is None else str(v) for k, v in base.items()}


# ──────────────────────────────────────────────────────────────────────────────
# command building
# ──────────────────────────────────────────────────────────────────────────────


def build_serve_command(args, defaults: dict, config: dict, run_dir: Path) -> list[str]:
    name = config["name"]
    port = str(config.get("port") or defaults.get("port") or C.DEFAULT_PORT)
    recipe = expand(str(config["recipe"]), defaults)
    cmd = [
        sys.executable, str(Path(args.serve_py).resolve()),
        "up",
        "--recipe", recipe,
        "--name", f"bench-{name}",
        "--run-id", name,
        "--run-dir", str(run_dir),
        "--config-id", name,
        "--port", port,
        "--boot-timeout", str(config.get("boot_timeout_s") or defaults.get("boot_timeout_s") or 1500),
    ]
    image = config.get("image") or defaults.get("image")
    if image:
        cmd += ["--image", str(image)]
    chat_template = config.get("chat_template") or defaults.get("chat_template")
    if chat_template:
        cmd += ["--chat-template", expand(str(chat_template), defaults)]
    if config.get("strict_template") or defaults.get("strict_template"):
        cmd.append("--strict-template")
    for key, value in (config.get("set") or {}).items():
        cmd += ["--set", f"{key}={value}"]
    for key, value in merge_env(defaults, config).items():
        cmd += ["--env", f"{key}={value}"]
    for mount in (config.get("mounts") or defaults.get("mounts") or []):
        cmd += ["--mount", expand(str(mount), defaults)]
    for extra in (config.get("docker_args") or defaults.get("docker_args") or []):
        cmd += ["--docker-arg", str(extra)]
    extra_vllm_args = config.get("extra_args") or defaults.get("extra_args")
    if extra_vllm_args:
        for chunk in (extra_vllm_args if isinstance(extra_vllm_args, list) else [extra_vllm_args]):
            cmd += ["--extra-vllm-arg", str(chunk)]
    if defaults.get("keep_failed"):
        cmd.append("--keep-failed")
    return cmd


def build_suite_command(args, defaults: dict, config: dict, run_dir: Path) -> list[str]:
    name = config["name"]
    port = str(config.get("port") or defaults.get("port") or C.DEFAULT_PORT)
    profile = str(config.get("suite_profile") or defaults.get("suite_profile") or "full")
    suite = merge_suite(defaults, config)
    cmd = [
        sys.executable, str(Path(args.suite_py).resolve()),
        "--base-url", f"http://127.0.0.1:{port}",
        "--run-dir", str(run_dir),
        "--config-id", name,
        "--profile", profile,
        "--tag", str(config.get("tag") or name),
    ]
    for key, value in suite.items():
        if key in SUITE_KEY_TO_FLAG:
            cmd += [SUITE_KEY_TO_FLAG[key], str(value)]
        elif key in SUITE_LIST_FLAGS:
            values = value if isinstance(value, (list, tuple)) else [value]
            cmd += [SUITE_LIST_FLAGS[key], ",".join(str(v) for v in values)]
        elif key in SUITE_BOOL_FLAGS:
            if value:
                cmd.append(SUITE_BOOL_FLAGS[key])
            elif key == "long_smoke":
                cmd.append("--no-long-smoke")
        elif key == "extra":
            for chunk in value if isinstance(value, (list, tuple)) else [value]:
                cmd += C.shlex_split(str(chunk))
        else:
            raise ConfigError(f"config {name!r}: unknown suite key {key!r}")
    return cmd


# ──────────────────────────────────────────────────────────────────────────────
# status helpers
# ──────────────────────────────────────────────────────────────────────────────


def write_status(run_dir: Path, name: str, status: dict) -> None:
    run_dir = C.ensure_dir(run_dir)
    status = {**status, "name": name, "updated_at": C.now_iso()}
    tmp = run_dir / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(run_dir / "status.json")


def read_status(run_dir: Path) -> dict:
    path = run_dir / "status.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def is_done(run_dir: Path) -> bool:
    status = read_status(run_dir)
    if status.get("status") == "done":
        return True
    # tolerate status lost after a crash but a valid summary exists
    summary = run_dir / "summary.json"
    if summary.is_file() and status.get("status") not in ("failed", "timeout"):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            return any((phase or {}).get("status") in ("ok", "partial")
                       for phase in (data.get("phases") or {}).values())
        except Exception:  # noqa: BLE001
            return False
    return False


def tail(text: str, limit: int = 4000) -> str:
    text = text or ""
    return text[-limit:]


def run_streamed(cmd: list[str], timeout: float, log_path: Path, label: str) -> tuple[int, str]:
    """Run a command, streaming stdout/stderr to the matrix log and capturing it."""
    lines: list[str] = []
    started = time.time()
    with open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n$ ({label}) {' '.join(cmd)}\n")
        log_fh.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(C.REPO_ROOT),
        )
        assert proc.stdout is not None
        try:
            while True:
                line = proc.stdout.readline()
                if line:
                    lines.append(line)
                    log_fh.write(line)
                    log_fh.flush()
                    print(f"    {line.rstrip()}", flush=True)
                    if time.time() - started > timeout:
                        raise subprocess.TimeoutExpired(cmd, timeout)
                elif proc.poll() is not None:
                    break
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            log_fh.write(f"\n!! {label} TIMEOUT after {timeout:.0f}s\n")
            return 124, "".join(lines)
        finally:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        # drain
        try:
            rest = proc.stdout.read()
        except Exception:  # noqa: BLE001
            rest = ""
        if rest:
            lines.append(rest)
            log_fh.write(rest)
    return proc.returncode, "".join(lines)


def stop_container(name: str, run_dir: Path) -> None:
    container = f"bench-{name}"
    # serve.py down is the sanctioned path (updates meta.json too)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "serve.py"), "down",
         "--name", container],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0 or "not present" in (proc.stdout or ""):
        C.docker_rm(container)
    C.log(f"stop {container}: {(proc.stdout or proc.stderr or '').strip()[:200]}")


def stale_container(name: str) -> str | None:
    container = f"bench-{name}"
    info = C.docker_inspect(container)
    if info is None:
        return None
    state = info.get("State") or {}
    return "running" if state.get("Running") else "stopped"


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", default=str(Path(__file__).with_name("matrix.yaml")))
    parser.add_argument("--results-dir", help="override defaults.results_dir")
    parser.add_argument("--only", action="append", metavar="NAME", help="run only these configs (repeatable)")
    parser.add_argument("--start-at", metavar="NAME", help="skip configs that sort before NAME")
    parser.add_argument("--max-configs", type=int, help="stop after N executed configs")
    parser.add_argument("--force", action="store_true", help="re-run configs already marked done")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    parser.add_argument("--fail-fast", action="store_true", help="stop at the first failed config")
    parser.add_argument("--no-aggregate", action="store_true", help="skip the final aggregate.py step")
    parser.add_argument("--serve-py", default=str(Path(__file__).with_name("serve.py")))
    parser.add_argument("--suite-py", default=str(Path(__file__).with_name("suite.py")))
    parser.add_argument("--aggregate-py", default=str(Path(__file__).with_name("aggregate.py")))
    parser.add_argument("--log", help="matrix log path (default <results>/matrix.log)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix_path = Path(args.matrix)
    if not matrix_path.is_file():
        C.log(f"ERROR: matrix file not found: {matrix_path}")
        return 2
    try:
        defaults, configs = load_matrix(matrix_path)
    except ConfigError as exc:
        C.log(f"ERROR: {exc}")
        return 2
    results_dir = Path(args.results_dir) if args.results_dir else (
        C.REPO_ROOT / str(defaults.get("results_dir") or "bench/results"))
    C.ensure_dir(results_dir)
    log_path = Path(args.log) if args.log else results_dir / "matrix.log"
    per_config_timeout = float(defaults.get("per_config_timeout_s") or 2400)
    stop_server = bool(defaults.get("stop_server", True))

    if not C.docker_ok():
        C.log("ERROR: docker not usable")
        return 2

    C.log(f"matrix {matrix_path} → {results_dir} ({len(configs)} configs, "
          f"{'dry-run' if args.dry_run else 'sequential'})")
    matrix_state = {
        "matrix": C.repo_rel(matrix_path),
        "started_at": C.now_iso(),
        "results_dir": str(results_dir),
        "configs": [],
    }
    executed = 0
    interrupted = False
    failures: list[str] = []
    start_reached = args.start_at is None

    for config in configs:
        name = config["name"]
        run_dir = results_dir / name
        if not config.get("enabled", True):
            C.log(f"skip {name}: disabled")
            matrix_state["configs"].append({"name": name, "status": "disabled"})
            continue
        if args.only and name not in args.only:
            continue
        if not start_reached:
            if name == args.start_at:
                start_reached = True
            else:
                C.log(f"skip {name}: before --start-at")
                continue
        if args.max_configs is not None and executed >= args.max_configs:
            C.log(f"stop: --max-configs {args.max_configs} reached")
            break
        if not args.dry_run:
            C.ensure_dir(run_dir)

        serve_cmd = build_serve_command(args, defaults, config, run_dir)
        suite_cmd = build_suite_command(args, defaults, config, run_dir)
        if args.dry_run:
            print(f"\n# {name}  (priority {config.get('priority')})")
            print(" ".join(serve_cmd))
            print(" ".join(suite_cmd))
            if stop_server:
                print(f"{sys.executable} {args.serve_py} down --name bench-{name}")
            executed += 1
            continue

        status = read_status(run_dir)
        if is_done(run_dir) and not args.force:
            C.log(f"skip {name}: already done ({run_dir / 'summary.json'})")
            matrix_state["configs"].append({"name": name, "status": "already_done",
                                            "run_dir": str(run_dir)})
            continue

        stale = stale_container(name)
        if stale is not None:
            C.log(f"cleaning stale container bench-{name} ({stale})")
            stop_container(name, run_dir)

        C.log(f"▶ config {name} ({executed + 1})")
        started = time.time()
        write_status(run_dir, name, {
            "status": "booting", "started_at": C.now_iso(),
            "recipe": str(config.get("recipe")), "serve_command": serve_cmd,
            "suite_command": suite_cmd, "per_config_timeout_s": per_config_timeout,
        })
        entry = {
            "name": name,
            "recipe": str(config.get("recipe")),
            "priority": config.get("priority"),
            "run_dir": str(run_dir),
            "started_at": C.now_iso(),
            "status": "booting",
        }
        rc_serve = 124
        rc_suite = None
        error = None
        try:
            elapsed = time.time() - started
            remaining = max(60.0, per_config_timeout - elapsed)
            rc_serve, out = run_streamed(serve_cmd, timeout=remaining, log_path=log_path,
                                         label=f"serve {name}")
            if rc_serve != 0:
                error = f"serve.py up failed (rc={rc_serve}): {tail(out, 1200)}"
                write_status(run_dir, name, {"status": "failed", "error": error,
                                             "exit_codes": {"serve": rc_serve},
                                             "started_at": entry["started_at"],
                                             "duration_s": round(time.time() - started, 1)})
                entry.update({"status": "failed", "error": error,
                              "duration_s": round(time.time() - started, 1)})
            else:
                write_status(run_dir, name, {
                    "status": "running_suite", "started_at": entry["started_at"],
                    "serve_command": serve_cmd, "suite_command": suite_cmd,
                    "exit_codes": {"serve": rc_serve},
                })
                remaining = max(60.0, per_config_timeout - (time.time() - started))
                rc_suite, out = run_streamed(suite_cmd, timeout=remaining, log_path=log_path,
                                             label=f"suite {name}")
                suite_ok = rc_suite == 0
                final_status = "done" if suite_ok else "suite_failed"
                if not suite_ok:
                    error = f"suite.py rc={rc_suite}: {tail(out, 1200)}"
                    failures.append(name)
                write_status(run_dir, name, {
                    "status": final_status,
                    "started_at": entry["started_at"],
                    "duration_s": round(time.time() - started, 1),
                    "serve_command": serve_cmd, "suite_command": suite_cmd,
                    "exit_codes": {"serve": rc_serve, "suite": rc_suite},
                    "error": error, "summary_path": str(run_dir / "summary.json"),
                })
                entry.update({"status": final_status, "duration_s": round(time.time() - started, 1),
                              "exit_codes": {"serve": rc_serve, "suite": rc_suite}, "error": error})
        except KeyboardInterrupt:
            C.log("interrupted by user")
            interrupted = True
            write_status(run_dir, name, {"status": "interrupted", "started_at": entry["started_at"]})
            entry.update({"status": "interrupted"})
        finally:
            if stop_server and stale_container(name) is not None:
                stop_container(name, run_dir)
        executed += 1
        matrix_state["configs"].append(entry)
        (results_dir / "matrix_status.json").write_text(
            json.dumps({**matrix_state, "updated_at": C.now_iso()}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if interrupted:
            break
        if entry["status"] in ("failed", "suite_failed") and args.fail_fast:
            C.log("stopping: --fail-fast")
            break

    matrix_state["finished_at"] = C.now_iso()
    (results_dir / "matrix_status.json").write_text(
        json.dumps({**matrix_state, "updated_at": C.now_iso()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.dry_run:
        C.log(f"dry run: {executed} config(s) printed")
        return 0
    C.log(f"matrix finished: {executed} executed, {len(failures)} failed")
    if interrupted:
        return 130
    if failures:
        C.log(f"failed configs: {failures}")
        return 2
    if not args.no_aggregate and (Path(args.aggregate_py).is_file()):
        C.log("aggregating results…")
        proc = subprocess.run(
            [sys.executable, str(Path(args.aggregate_py)), "--results-dir", str(results_dir)],
            capture_output=True, text=True, timeout=300,
        )
        sys.stdout.write(proc.stdout or "")
        sys.stderr.write(proc.stderr or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
