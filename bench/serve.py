#!/usr/bin/env python3
"""serve.py — launch, adopt, stop and inspect a vLLM docker container from a
recipe-like YAML (the ``server:`` / ``env:`` / ``vllm:`` format used by this
repository's ``run.sh``; eugr-style recipes with a ``command:`` block are also
supported).

Why not shell out to run.sh?  This harness must (a) never depend on the host
``nvidia-smi`` (broken in this sandbox), (b) guarantee exactly one
``--speculative-config`` argument, (c) name/port every measurement container
explicitly, and (d) record a reproducible launch manifest (image digest, env,
mounts, argv) next to each run.  The flags/mounts/environment are kept
deliberately close to run.sh so numbers stay comparable.

Examples
--------
  python3 bench/serve.py up --recipe recipes/qwen38-flash-solo-optimized.yaml \\
      --run-id ref-solo --boot-timeout 1200
  python3 bench/serve.py up --recipe recipes/qwen38-mtp3.yaml \\
      --name bench-mtp3 --set vllm.max-num-seqs=8 --speculative-config '{"method":"mtp","num_speculative_tokens":4,"rejection_sample_method":"block"}'
  python3 bench/serve.py adopt --name qwen38-flash-next --run-id live-ref
  python3 bench/serve.py status --port 8000
  python3 bench/serve.py down --run-dir bench/results/live-ref
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DEFAULT_IMAGE = "myllmbox/qwen38-flash-next-vllm:v5.4"
DEFAULT_BOOT_TIMEOUT = 1200.0


# ──────────────────────────────────────────────────────────────────────────────
# launch manifest construction
# ──────────────────────────────────────────────────────────────────────────────


def _render_command_recipe(recipe: dict, values: dict) -> list[str]:
    """Render an eugr-style ``command:`` template into ``vllm`` argv.

    eugr recipes use a str.format-like template: ``{placeholder}`` is
    substituted and ``{{`` / ``}}`` are literal braces (JSON payloads are
    written escaped).  A plain replace-then-format is ambiguous for the common
    ``{num_speculative_tokens}}}`` pattern, so scan the string explicitly.
    """
    template = str(recipe["raw"].get("command") or "").strip()
    if not template:
        raise SystemExit(f"recipe {recipe['path']} has no command block")
    out: list[str] = []
    i, n = 0, len(template)
    while i < n:
        ch = template[i]
        if ch == "{":
            if i + 1 < n and template[i + 1] == "{":
                out.append("{")
                i += 2
                continue
            close = template.find("}", i + 1)
            if close == -1:
                raise SystemExit("recipe command has an unterminated '{'")
            key = template[i + 1:close]
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise SystemExit(f"recipe command has a malformed placeholder {{{key}}}")
            if key not in values:
                raise SystemExit(f"recipe command references unknown placeholder {{{key}}}")
            out.append(str(values[key]))
            i = close + 1
            continue
        if ch == "}":
            if i + 1 < n and template[i + 1] == "}":
                out.append("}")
                i += 2
                continue
            out.append("}")
            i += 1
            continue
        out.append(ch)
        i += 1
    rendered = "".join(out)
    argv = C.shlex_split(rendered)
    if not argv:
        raise SystemExit("rendered recipe command is empty")
    if argv[0].endswith("vllm"):
        argv = argv[1:]
    if argv and argv[0] == "serve":
        return argv
    return ["serve"] + argv


def build_launch(args, recipe: dict) -> dict:
    raw = recipe["raw"]
    server = recipe["server"]
    vllm = dict(recipe["vllm"])
    env = dict(recipe["env"] or {})

    # apply --set overrides (recipe-level)
    overrides = C.parse_set_overrides(args.set or [])
    for key, value in overrides.items():
        if key in ("env", "server") and isinstance(value, dict):
            if key == "vllm":
                vllm.update(value)
            elif key == "env":
                env.update(value)
            elif key == "server":
                server.update(value)
        else:
            vllm[key] = value

    image = args.image or server.get("image") or DEFAULT_IMAGE
    if image == "qwen38-flash-dgx:a5b-int4":  # run.sh's legacy fallback
        image = DEFAULT_IMAGE
    port = int(args.port or server.get("port") or C.DEFAULT_PORT)
    name = args.name or f"bench-{C.safe_name(Path(recipe['path']).stem)}"
    cpuset = args.cpuset if args.cpuset is not None else server.get("cpuset")
    docker_args = C.shlex_split(str(server.get("docker_args") or "")) + list(args.docker_arg or [])

    model_repo = args.model_repo or server.get("model") or C.DEFAULT_MODEL_REPO
    local_name = model_repo.split("/")[-1]
    models_dir = Path(args.models_dir) if args.models_dir else C.REPO_ROOT / str(server.get("models_dir") or "./models")
    cache_dir = Path(args.cache_dir) if args.cache_dir else C.REPO_ROOT / str(server.get("cache_dir") or "./cache")
    hf_home = Path(args.hf_home) if args.hf_home else Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))

    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
        model_source = "explicit"
    else:
        model_dir, model_source = C.resolve_model_dir(model_repo, models_dir, hf_home)
    if model_dir is None:
        raise SystemExit(
            f"model not found: {model_repo} is in neither the HF cache ({hf_home}/hub) "
            f"nor {models_dir / local_name}; pass --model-dir"
        )
    model_dir = Path(model_dir)

    # ── vLLM argv ────────────────────────────────────────────────────────────
    if recipe["format"] == "command":
        values = {
            "host": "0.0.0.0",
            "port": 8000,
            "model_path": f"/models/{local_name}",
            "served_name": args.served_model_name or local_name,
        }
        defaults = raw.get("defaults") or {}
        for key, value in defaults.items():
            values.setdefault(str(key), value)
        for pair in args.default or []:
            key, _, value = pair.partition("=")
            values[key] = value
        argv = _render_command_recipe(recipe, values)
        # Drop an explicit --port from the template and re-add at the end.
        argv = C.set_flag(argv, "port", "8000")
    else:
        flags = C.flags_from_vllm_section(vllm)
        # The served model dir is always ours, exactly like run.sh.
        flags = C.set_flag(flags, "port", "8000") if C.get_flag_value(flags, "port") is not None else flags
        if args.served_model_name:
            flags = C.set_flag(flags, "served-model-name", args.served_model_name)
        argv = ["serve", f"/models/{local_name}"] + flags

    # ── speculative decoding: exactly one flag, replace never append ────────
    if args.speculative_config is not None and args.mtp is not None:
        raise SystemExit("use only one of --speculative-config / --mtp")
    if args.speculative_config is not None:
        try:
            args.speculative_config = json.loads(args.speculative_config)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"--speculative-config is not valid JSON: {exc}") from exc
        argv = C.set_flag(argv, "speculative-config", json.dumps(args.speculative_config, separators=(",", ":")))
    elif args.mtp is not None:
        if int(args.mtp) <= 0:
            argv = C.remove_flag(argv, "speculative-config")
        else:
            spec = C.spec_config_obj(argv) or {"method": "mtp"}
            spec["method"] = spec.get("method") or "mtp"
            spec["num_speculative_tokens"] = int(args.mtp)
            if args.rejection_sample:
                spec["rejection_sample_method"] = args.rejection_sample
            if args.draft_sample:
                spec["draft_sample_method"] = args.draft_sample
            argv = C.set_flag(argv, "speculative-config", json.dumps(spec, separators=(",", ":")))

    # replicate run.sh MTP env behaviour only when the recipe asks for it;
    # env-based overrides are explicit here via --set env.X=Y / --env.

    # port must be the container-internal port 8000 (host mapping is separate)
    argv = C.set_flag(argv, "port", "8000")

    # ── chat template (run.sh parity: mount at /chat_template.jinja) ────────
    template_mount = None
    template_name = args.chat_template
    if template_name:
        template_path = resolve_chat_template(template_name, model_dir)
        if template_path is None:
            msg = f"chat template '{template_name}' not found in {model_dir} or as a path"
            if args.strict_template:
                raise SystemExit(msg)
            C.log(f"WARNING: {msg} — continuing with the model's stock template")
        else:
            template_mount = (str(template_path), "/chat_template.jinja", "ro")
            argv = C.set_flag(argv, "chat-template", "/chat_template.jinja")

    for pair in args.extra_vllm_arg or []:
        argv += C.shlex_split(pair)

    # ── env: run.sh order (standard then recipe then --env overrides) ───────
    env_ordered: dict[str, str] = {}
    env_ordered["HF_HUB_OFFLINE"] = "1"
    env_ordered["TRANSFORMERS_OFFLINE"] = "1"
    env_ordered["FLASHINFER_WORKSPACE_BASE"] = "/cache/flashinfer-workspace"
    env_ordered["VLLM_CACHE_ROOT"] = "/cache/vllm-cache"
    for key, value in env.items():
        env_ordered[str(key)] = "" if value is None else str(value)
    for pair in args.env or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--env expects KEY=VALUE, got {pair!r}")
        env_ordered[key] = value
    for key, value in overrides.get("env", {}).items() if False else ():
        pass

    # ── mounts, mirrors run.sh plus the bench mount contract ────────────────
    # Contract (needed by draft-lab): the workspace root is bind-mounted at the
    # SAME absolute host path, read-only, so speculative-config paths such as
    # /home/pgx/Documenti/vllm-gb10/draft-lab/drafts/<name> resolve inside the
    # container.  The HF hub is likewise mounted at its own absolute path.
    mounts: list[tuple[str, str, str | None]] = []
    mounts.append((str(C.REPO_ROOT), str(C.REPO_ROOT), "ro"))
    mounts.append((str(hf_home / "hub"), str(hf_home / "hub"), "ro"))
    mounts.append((str(models_dir.resolve()), "/models", None))
    mounts.append((str(cache_dir.resolve()), "/cache", None))
    mounts.append((str(model_dir), f"/models/{local_name}", None))
    # boot caches extracted from earlier containers (optional, speeds up boots)
    for sub, target in (("b12x", "/root/.cache/b12x"), ("flashinfer", "/root/.cache/flashinfer")):
        candidate = Path(cache_dir) / "container-cache" / sub
        if candidate.is_dir():
            mounts.append((str(candidate.resolve()), target, None))
    if template_mount:
        mounts.append(template_mount)
    for pair in args.mount or []:
        parts = pair.split(":")
        if len(parts) == 2:
            mounts.append((parts[0], parts[1], None))
        elif len(parts) == 3:
            mounts.append((parts[0], parts[1], parts[2]))
        else:
            raise SystemExit(f"--mount expects host:container[:ro], got {pair!r}")

    # ── docker argv ─────────────────────────────────────────────────────────
    docker_argv = ["docker", "run", "-d", "--name", name, "--gpus", "all", "--ipc=host"]
    if cpuset:
        docker_argv += ["--cpuset-cpus", str(cpuset)]
    docker_argv += docker_args
    docker_argv += ["-p", f"{port}:8000"]
    for host, container, mode in mounts:
        spec = f"{host}:{container}" + (f":{mode}" if mode else "")
        docker_argv += ["-v", spec]
    for key, value in env_ordered.items():
        docker_argv += ["-e", f"{key}={value}"]
    docker_argv += ["--entrypoint", "vllm", image]
    docker_argv += argv

    spec_obj = C.spec_config_obj(argv)
    meta = {
        "schema_version": 1,
        "kind": "serve",
        "run_id": args.run_id or name,
        "config_id": args.config_id or args.run_id or name,
        "created_at": C.now_iso(),
        "status": "pending",
        "recipe": {
            "path": C.repo_rel(recipe["path"]),
            "sha256": C.sha256_file(recipe["path"]),
            "format": recipe["format"],
        },
        "stack": {
            "image": image,
            "stack": detect_stack(image, str(recipe["path"])),
        },
        "container": {
            "name": name,
            "host_port": port,
            "container_port": 8000,
            "gpu": "all",
            "runtime": "docker",
        },
        "launch": {
            "argv": docker_argv,
            "shell": shlex.join(docker_argv),
            "env": env_ordered,
            "mounts": [{"host": h, "container": c, "mode": m or "rw"} for h, c, m in mounts],
            "cpuset": cpuset,
            "docker_args": docker_args,
            "entrypoint": "vllm",
        },
        "vllm": {
            "argv": argv,
            "arguments": argv[2:] if len(argv) > 2 else [],
            "speculative_config": spec_obj,
            "num_speculative_tokens": C.spec_num_tokens(argv),
            "chat_template": template_mount[0] if template_mount else None,
            "max_num_seqs": _as_int(C.get_flag_value(argv, "max-num-seqs")),
            "max_model_len": _as_int(C.get_flag_value(argv, "max-model-len")),
            "kv_cache_dtype": C.get_flag_value(argv, "kv-cache-dtype"),
            "gpu_memory_utilization": C.get_flag_value(argv, "gpu-memory-utilization"),
            "load_format": C.get_flag_value(argv, "load-format"),
        },
        "model": {
            "repo": model_repo,
            "local_name": local_name,
            "host_dir": str(model_dir),
            "container_dir": f"/models/{local_name}",
            "source": model_source,
        },
        "host": C.host_facts(),
        "serve_py_command": shlex.join([sys.executable] + sys.argv),
        "timing": {},
        "logs": {},
    }
    return meta


def _as_int(value):
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def detect_stack(image: str, recipe_path: str) -> str:
    text = f"{image} {recipe_path}".lower()
    if "eugr" in text or "b12x" in text:
        return "eugr-b12x"
    if "myllmbox" in text:
        return "myllmbox-v5.4"
    return "custom"


def resolve_chat_template(template: str, model_dir: Path) -> Path | None:
    if template in ("", "model", "stock", "none"):
        return None
    candidate = Path(template).expanduser()
    if not candidate.is_absolute():
        candidate = C.REPO_ROOT / candidate
    if candidate.is_file():
        return candidate.resolve()
    for name in (template, f"{template}_chat_template.jinja", f"{template}.jinja"):
        for base in (model_dir, C.REPO_ROOT, C.REPO_ROOT / "matrix" / "templates"):
            candidate = base / name
            if candidate.is_file():
                return candidate.resolve()
    return None


# ──────────────────────────────────────────────────────────────────────────────
# preflight / lifecycle
# ──────────────────────────────────────────────────────────────────────────────


def preflight(meta: dict, *, force: bool, quiet: bool = False, port_wait: float = 30.0) -> None:
    name = meta["container"]["name"]
    port = meta["container"]["host_port"]
    if not C.docker_ok():
        raise SystemExit("docker is not usable in this shell")
    if C.port_healthy(port):
        raise SystemExit(
            f"port {port} already serves a healthy /health endpoint — refusing to start a second model server "
            "(use `adopt` to measure it, or stop it first)"
        )
    if C.port_open(port):
        # a container stopped a moment ago can leave the docker-proxy bound for a
        # short while; give it a grace period instead of failing an unattended run
        deadline = time.time() + max(0.0, port_wait)
        while time.time() < deadline and C.port_open(port) and not C.port_healthy(port):
            time.sleep(0.5)
        if C.port_healthy(port):
            raise SystemExit(
                f"port {port} started serving a healthy /health endpoint while we waited — "
                "refusing to start a second model server"
            )
        if C.port_open(port):
            raise SystemExit(f"port {port} is still in use by another process after {port_wait:.0f}s")
    others = []
    for row in C.docker_ps():
        row_name = row.get("Names") or ""
        if row_name == name:
            continue
        if C.looks_like_model_server(row):
            others.append(f"{row_name} ({row.get('Image')})")
    if others and not force:
        raise SystemExit(
            "another model-server-looking container is already running: " + ", ".join(others)
            + "; stop it first (never run two on one GPU) or pass --force"
        )
    if not quiet:
        C.log(f"preflight ok: port {port} free, no other model server")


def wait_for_health(meta: dict, timeout: float, poll: float, run_dir: C.RunDir) -> dict:
    name = meta["container"]["name"]
    port = meta["container"]["host_port"]
    started = time.time()
    polls = 0
    last_log_size = 0
    while time.time() - started < timeout:
        polls += 1
        if C.port_healthy(port):
            ready = time.time()
            return {
                "ready_at": C.now_iso(),
                "boot_seconds": round(ready - started, 3),
                "poll_count": polls,
            }
        info = C.docker_inspect(name)
        if info is None:
            raise RuntimeError("container disappeared while waiting for /health")
        state = info.get("State") or {}
        if not state.get("Running"):
            raise RuntimeError(f"container exited (exit_code={state.get('ExitCode')}) before /health")
        # progress feedback from the tail of the log
        try:
            tail = C.docker_logs(name, tail=2, timeout=20)
            if tail and len(tail) > last_log_size:
                last_log_size = len(tail)
                line = tail.strip().splitlines()[-1] if tail.strip() else ""
                if line:
                    C.log(f"booting… {line[:160]}")
        except Exception:  # noqa: BLE001
            pass
        time.sleep(poll)
    raise TimeoutError(f"/health not ready after {timeout:.0f}s")


def cmd_up(args) -> int:
    recipe = C.load_recipe(Path(args.recipe))
    meta = build_launch(args, recipe)
    run_path = Path(args.run_dir) if args.run_dir else C.RESULTS_DIR / meta["run_id"]

    if args.dry_run:
        print(json.dumps({
            "run_dir": str(run_path),
            "docker_argv": meta["launch"]["argv"],
            "docker_shell": meta["launch"]["shell"],
            "speculative_config": meta["vllm"]["speculative_config"],
            "env": meta["launch"]["env"],
            "mounts": meta["launch"]["mounts"],
        }, indent=2))
        C.log("dry run only — nothing launched")
        return 0

    run_dir = C.RunDir(run_path)
    meta["timing"]["requested_at"] = C.now_iso()
    run_dir.write_json("meta.json", meta)
    preflight(meta, force=args.force, port_wait=args.port_wait)
    name = meta["container"]["name"]

    existing = C.docker_inspect(name)
    if existing is not None:
        C.log(f"removing stale container {name}")
        C.docker_rm(name)

    C.log(f"launching {name} ({meta['stack']['image']}) on port {meta['container']['host_port']}")
    proc = C.run_cmd(meta["launch"]["argv"], timeout=args.docker_timeout)
    if proc.returncode != 0:
        meta["status"] = "failed"
        meta["error"] = (proc.stderr or proc.stdout).strip()[:4000]
        run_dir.write_json("meta.json", meta)
        sys.stderr.write(meta["error"] + "\n")
        return 2
    container_id = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout.strip() else None
    meta["container"]["id"] = container_id
    meta["timing"]["container_started_at"] = C.now_iso()
    run_dir.write_json("meta.json", meta)

    info = C.docker_inspect(name) or {}
    started_at = (info.get("State") or {}).get("StartedAt")
    if started_at:
        meta["timing"]["container_created_at"] = started_at
    image_meta = C.docker_image_meta(meta["stack"]["image"])
    meta["stack"].update({"image_id": image_meta.get("id"), "repo_digests": image_meta.get("repo_digests"),
                          "image_created": image_meta.get("created"), "image_size": image_meta.get("size")})

    try:
        health = wait_for_health(meta, args.boot_timeout, args.poll, run_dir)
    except (RuntimeError, TimeoutError) as exc:
        C.log(f"boot FAILED: {exc}")
        meta["status"] = "failed"
        meta["error"] = str(exc)
        meta["timing"]["failed_at"] = C.now_iso()
        _save_logs(run_dir, name, meta)
        run_dir.write_json("meta.json", meta)
        if not args.keep_failed:
            C.log(f"removing failed container {name}")
            C.docker_rm(name)
        return 2

    meta["timing"].update(health)
    meta["status"] = "running"
    time.sleep(args.settle)
    _save_logs(run_dir, name, meta)
    run_dir.write_json("meta.json", meta)
    print_manifest(meta, run_dir)
    return 0


def _save_logs(run_dir: C.RunDir, name: str, meta: dict) -> None:
    try:
        text = C.docker_logs(name, timeout=180)
    except Exception as exc:  # noqa: BLE001
        meta.setdefault("logs", {})["error"] = str(exc)
        return
    run_dir.write_text("server.log", text)
    startup = C.parse_startup_log(text)
    meta.setdefault("logs", {})["server_log"] = "server.log"
    meta["logs"]["startup"] = startup
    # discover the exact model path the server resolved (entrypoint may mount elsewhere)
    try:
        info = C.docker_inspect(name) or {}
        meta["container"]["id"] = info.get("Id") or meta["container"].get("id")
    except Exception:  # noqa: BLE001
        pass


def print_manifest(meta: dict, run_dir: C.RunDir) -> None:
    timing = meta.get("timing", {})
    startup = (meta.get("logs") or {}).get("startup") or {}
    spec = meta.get("vllm", {}).get("speculative_config")
    print()
    print("─" * 78)
    print(f"✓ server ready: {meta['container']['name']} on http://127.0.0.1:{meta['container']['host_port']}/v1")
    print(f"  run dir      : {run_dir.path}")
    print(f"  image        : {meta['stack']['image']}")
    print(f"  image id     : {meta['stack'].get('image_id')}")
    recipe = meta.get("recipe") or {}
    sha = (recipe.get("sha256") or "")[:12] or "n/a"
    boot = timing.get("boot_seconds")
    boot_text = f"{boot}s" if boot is not None else "n/a (adopted)"
    print(f"  recipe       : {recipe.get('path') or 'adopted container'} (sha256 {sha})")
    print(f"  boot         : {boot_text} ({timing.get('poll_count', '-')} polls)")
    print(f"  spec decode  : {json.dumps(spec, separators=(',', ':')) if spec else 'OFF'}")
    if startup.get("kv_cache_tokens"):
        print(f"  KV pool      : {startup['kv_cache_tokens']:,} tokens"
              + (f" / max concurrency {startup['kv_cache_max_concurrency']}x"
                 if startup.get("kv_cache_max_concurrency") else ""))
    print("─" * 78)


def cmd_down(args) -> int:
    name = args.name
    run_dir = None
    if args.run_dir:
        run_dir = Path(args.run_dir)
        meta = C.RunDir(run_dir).read_json("meta.json", default={}) or {}
        name = name or (meta.get("container") or {}).get("name")
    if not name:
        raise SystemExit("down: pass --name or --run-dir with a meta.json")
    removed = C.docker_rm(name)
    C.log(("stopped " if removed else "container not present: ") + name)
    if run_dir is not None:
        meta = C.RunDir(run_dir).read_json("meta.json", default={}) or {}
        meta.setdefault("timing", {})["stopped_at"] = C.now_iso()
        meta["status"] = "stopped"
        C.RunDir(run_dir).write_json("meta.json", meta)
    return 0


def cmd_status(args) -> int:
    port = args.port
    name = args.name
    row = {"port": port, "name": name, "healthy": False}
    if port is None and name is None:
        print("pass --port or --name", file=sys.stderr)
        return 2
    if name:
        info = C.docker_inspect(name)
        if info:
            row["running"] = (info.get("State") or {}).get("Running")
            row["image"] = (info.get("Config") or {}).get("Image")
            row["started_at"] = (info.get("State") or {}).get("StartedAt")
    if port is None:
        serialized = json.dumps((C.docker_inspect(name) or {}).get("Config") or {}, default=str)
        import re

        match = re.search(r'"-p",\s*"(\d+):8000"', serialized)
        port = int(match.group(1)) if match else C.DEFAULT_PORT
        row["port"] = port
    row["healthy"] = C.port_healthy(port)
    print(json.dumps(row, indent=2))
    return 0 if row["healthy"] else 1


def cmd_adopt(args) -> int:
    """Record an already-running container's launch manifest without touching it."""
    name = args.name or C.STANDARD_CONTAINER
    info = C.docker_inspect(name)
    if info is None:
        raise SystemExit(f"container {name} not found")
    config = info.get("Config") or {}
    state = info.get("State") or {}
    host_config = info.get("HostConfig") or {}
    port = args.port or C.DEFAULT_PORT
    if not C.port_healthy(port):
        raise SystemExit(f"container {name} exists but http://127.0.0.1:{port}/health is not ready yet")
    image = config.get("Image")
    entrypoint = config.get("Entrypoint") or []
    cmd = config.get("Cmd") or []
    binds = host_config.get("Binds") or []
    env_list = config.get("Env") or []
    env = {}
    for item in env_list:
        key, _, value = item.partition("=")
        if key in ("PATH", "HOSTNAME", "HOME", "LD_LIBRARY_PATH", "NVIDIA_REQUIRE_CUDA"):
            continue
        env[key] = value
    image_meta = C.docker_image_meta(image)
    full_argv = ["docker", "run", "-d", "--name", name, "--gpus", "all"] + [b for b in binds]
    meta = {
        "schema_version": 1,
        "kind": "adopt",
        "run_id": args.run_id or f"adopt-{C.safe_name(name)}",
        "config_id": args.config_id or args.run_id or name,
        "created_at": C.now_iso(),
        "status": "running",
        "recipe": {"path": None, "sha256": None, "format": "unknown"},
        "stack": {
            "image": image,
            "image_id": image_meta.get("id"),
            "repo_digests": image_meta.get("repo_digests"),
            "image_created": image_meta.get("created"),
            "image_size": image_meta.get("size"),
            "stack": detect_stack(image or "", ""),
        },
        "container": {
            "name": name,
            "id": info.get("Id"),
            "host_port": port,
            "container_port": 8000,
            "started_at": state.get("StartedAt"),
            "auto_named": True,
        },
        "launch": {
            "argv": full_argv,
            "shell": " ".join(shlex.quote(x) for x in full_argv),
            "env": env,
            "mounts": [{"spec": b} for b in binds],
            "note": "adopted: exact create argv not recoverable from inspect; env/cmd are authoritative",
            "command": cmd,
            "entrypoint": entrypoint,
        },
        "vllm": {
            "argv": list(cmd),
            "speculative_config": C.spec_config_obj(list(cmd)),
            "num_speculative_tokens": C.spec_num_tokens(list(cmd)),
            "max_num_seqs": _as_int(C.get_flag_value(list(cmd), "max-num-seqs")),
            "max_model_len": _as_int(C.get_flag_value(list(cmd), "max-model-len")),
            "kv_cache_dtype": C.get_flag_value(list(cmd), "kv-cache-dtype"),
            "gpu_memory_utilization": C.get_flag_value(list(cmd), "gpu-memory-utilization"),
            "load_format": C.get_flag_value(list(cmd), "load-format"),
            "chat_template": C.get_flag_value(list(cmd), "chat-template"),
        },
        "model": {"repo": C.DEFAULT_MODEL_REPO, "host_dir": None, "source": "adopted"},
        "host": C.host_facts(),
        "serve_py_command": shlex.join([sys.executable] + sys.argv),
        "timing": {"adopted_at": C.now_iso(), "container_started_at": state.get("StartedAt")},
        "logs": {},
    }
    run_dir = C.RunDir(Path(args.run_dir) if args.run_dir else C.RESULTS_DIR / meta["run_id"])
    _save_logs(run_dir, name, meta)
    run_dir.write_json("meta.json", meta)
    print_manifest(meta, run_dir)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def add_launch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recipe", required=False, help="recipe YAML (server/env/vllm or eugr command format)")
    parser.add_argument("--name", help="container name (default bench-<recipe>)")
    parser.add_argument("--image", help="override recipe image")
    parser.add_argument("--port", type=int, help="host port (default from recipe or 8000)")
    parser.add_argument("--run-id", help="run id == results/<run-id> directory name")
    parser.add_argument("--run-dir", help="explicit results directory (overrides --run-id)")
    parser.add_argument("--config-id", help="config id recorded in meta.json")
    parser.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                        help="override a recipe value, e.g. --set vllm.max-num-seqs=8")
    parser.add_argument("--env", action="append", metavar="KEY=VALUE", help="extra/overriding container env")
    parser.add_argument("--mtp", type=int, help="set MTP depth (0 = disable speculative decoding)")
    parser.add_argument("--speculative-config", help="full JSON speculative config (replaces, never appends)")
    parser.add_argument("--rejection-sample", help="with --mtp: rejection_sample_method")
    parser.add_argument("--draft-sample", help="with --mtp: draft_sample_method")
    parser.add_argument("--chat-template", help="medium|xhigh|model|/path/to/template.jinja")
    parser.add_argument("--strict-template", action="store_true", help="fail if the template cannot be resolved")
    parser.add_argument("--extra-vllm-arg", action="append", help="raw extra arg(s) appended to vllm serve (advanced)")
    parser.add_argument("--default", action="append", metavar="KEY=VALUE",
                        help="placeholder value for eugr-style command recipes")
    parser.add_argument("--served-model-name", help="override --served-model-name")
    parser.add_argument("--model-repo", help="override the HF model repo id")
    parser.add_argument("--model-dir", help="explicit host model dir to mount")
    parser.add_argument("--models-dir", help="host models dir (default from recipe)")
    parser.add_argument("--cache-dir", help="host cache dir mounted at /cache (default from recipe)")
    parser.add_argument("--hf-home", help="HF_HOME used to find the model snapshot")
    parser.add_argument("--cpuset", help="--cpuset-cpus value (default from recipe)")
    parser.add_argument("--docker-arg", action="append", help="extra docker run flag")
    parser.add_argument("--mount", action="append", help="extra bind mount host:container[:ro]")
    parser.add_argument("--cgroup-note", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="launch a container and wait for /health")
    add_launch_args(up)
    up.add_argument("--boot-timeout", type=float, default=DEFAULT_BOOT_TIMEOUT, help="seconds to wait for /health")
    up.add_argument("--poll", type=float, default=2.0, help="health poll interval")
    up.add_argument("--settle", type=float, default=2.0, help="extra seconds after healthy before returning")
    up.add_argument("--docker-timeout", type=float, default=120.0)
    up.add_argument("--keep-failed", action="store_true", help="keep the container when boot fails")
    up.add_argument("--force", action="store_true", help="start even if other model containers look present")
    up.add_argument("--port-wait", type=float, default=30.0,
                    help="seconds to wait for a just-freed port to become available")
    up.add_argument("--dry-run", action="store_true", help="print the docker argv without launching")
    up.set_defaults(func=cmd_up)

    down = sub.add_parser("down", help="stop and remove the container")
    down.add_argument("--name")
    down.add_argument("--run-dir")
    down.set_defaults(func=cmd_down)

    status = sub.add_parser("status", help="is a server healthy?")
    status.add_argument("--name")
    status.add_argument("--port", type=int)
    status.set_defaults(func=cmd_status)

    adopt = sub.add_parser("adopt", help="record an already-running container without touching it")
    adopt.add_argument("--name", default=C.STANDARD_CONTAINER)
    adopt.add_argument("--port", type=int, default=C.DEFAULT_PORT)
    adopt.add_argument("--run-id")
    adopt.add_argument("--run-dir")
    adopt.add_argument("--config-id")
    adopt.set_defaults(func=cmd_adopt)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "up" and not args.recipe:
        parser.error("up requires --recipe")
    try:
        return args.func(args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        C.log("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001
        C.log(f"ERROR: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
