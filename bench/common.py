"""Shared helpers for the AZ-model vLLM benchmark harness.

Deliberately dependency-light: stdlib only (PyYAML is used if available, with a
small fallback parser for the flat recipe/matrix YAML files in this repo).

Nothing here touches the GPU directly. Host ``nvidia-smi`` is known broken in
this environment, so no code path may depend on it: container state is read via
``docker inspect`` and the server via HTTP.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"

DEFAULT_MODEL_REPO = "azampatti/Qwen3.8-Flash-Next-FP4-Local-AZ"
DEFAULT_PORT = 8000
STANDARD_CONTAINER = "qwen38-flash-next"

# ──────────────────────────────────────────────────────────────────────────────
# logging / small utilities
# ──────────────────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def repo_rel(path: Path) -> str:
    """Path relative to the repo root when possible, else the absolute path."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path).resolve())


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(text: str, max_len: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")
    return s[:max_len] or "run"


def default_run_id(prefix: str = "") -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return safe_name(f"{prefix}-{stamp}" if prefix else stamp)


# ──────────────────────────────────────────────────────────────────────────────
# YAML: PyYAML when available, tiny fallback parser otherwise.
# ──────────────────────────────────────────────────────────────────────────────

try:  # pragma: no cover - availability differs per interpreter
    import yaml as _yaml  # type: ignore
except Exception:  # noqa: BLE001
    _yaml = None


def _strip_inline_comment(value: str) -> str:
    """Strip ` # comment` outside of quotes (recipes use JSON with ':' inside)."""
    out = []
    quote = None
    i = 0
    while i < len(value):
        ch = value[i]
        if quote:
            if ch == "\\" and quote == '"':
                out.append(ch)
                if i + 1 < len(value):
                    out.append(value[i + 1])
                    i += 2
                    continue
            elif ch == quote:
                quote = None
            out.append(ch)
        else:
            if ch in ("'", '"'):
                quote = ch
                out.append(ch)
            elif ch == "#" and (not out or out[-1] in (" ", "\t")):
                break
            else:
                out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        inner = value[1:-1]
        if value[0] == '"':
            try:
                return json.loads(value)
            except Exception:  # noqa: BLE001
                return inner
        return inner
    return value


def _fallback_yaml(text: str):
    """Parse the restricted YAML used by recipes/ and bench/matrix.yaml.

    Supports: nested mappings by indentation, inline lists ``[a, b]``, block
    scalars (``|``), sequence items (``- value`` / ``- key: value``) and
    scalars kept as strings (callers coerce booleans/ints where needed).
    """
    root: dict = {}
    # stack entries: (indent, container, kind)  kind: 'map' | 'list'
    stack: list[tuple[int, object, str]] = [(-1, root, "map")]

    def container_for(indent: int):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            return root
        return stack[-1][1]

    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        parent = container_for(indent)

        if line.startswith("- "):
            item = line[2:].strip()
            if not isinstance(parent, list):
                # should not happen with our files; shallow fallback
                continue
            if ":" in item and not item.startswith(("'", '"')):
                key, _, val = item.partition(":")
                val = _unquote(_strip_inline_comment(val))
                parent.append({key.strip(): _scalar(val)})
                stack.append((indent, parent[-1], "map"))
            else:
                parent.append(_scalar(_unquote(_strip_inline_comment(item))))
            continue

        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        key = _unquote(key.strip())
        rest = rest.strip()

        if rest == "|" or rest == "|-" or rest == ">":
            block: list[str] = []
            while i < len(lines) and (not lines[i].strip() or (len(lines[i]) - len(lines[i].lstrip(" "))) > indent):
                block.append(lines[i])
                i += 1
            block = [b for b in block if b.strip()]
            base_indent = min((len(b) - len(b.lstrip(" "))) for b in block) if block else indent + 2
            value = "\n".join(b[base_indent:] for b in block)
            parent[key] = value + ("\n" if rest == "|" else "")
            continue

        rest = _strip_inline_comment(rest)
        if rest == "":
            # peek: list or mapping?
            nxt = None
            for candidate in lines[i:]:
                if not candidate.strip() or candidate.lstrip().startswith("#"):
                    continue
                nxt = candidate
                break
            child: object = [] if (nxt is not None and nxt.strip().startswith("- ")) else {}
            parent[key] = child
            stack.append((indent, child, "list" if isinstance(child, list) else "map"))
            continue

        rest = _unquote(rest)
        if rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            parent[key] = [_scalar(_unquote(x.strip())) for x in _split_top_level(inner)] if inner else []
        elif rest.startswith("{") and rest.endswith("}"):
            parent[key] = _parse_inline_map(rest)
        else:
            parent[key] = _scalar(rest)
    return root


def _split_top_level(text: str) -> list[str]:
    items, buf, depth, quote = [], [], 0, None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch in "[{(":
            depth += 1
            buf.append(ch)
        elif ch in "]})":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        items.append("".join(buf))
    return items


def _scalar(text: str):
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text in ("null", "Null", "~", ""):
        return None
    return text


def _parse_inline_map(text: str) -> dict:
    out = {}
    for part in _split_top_level(text[1:-1]):
        if ":" in part:
            k, _, v = part.partition(":")
            out[_unquote(k.strip())] = _scalar(_unquote(v.strip()))
    return out


def load_yaml(path: Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if _yaml is not None:
        data = _yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    return _fallback_yaml(text)


# ──────────────────────────────────────────────────────────────────────────────
# recipes
# ──────────────────────────────────────────────────────────────────────────────


def load_recipe(path: Path) -> dict:
    """Load a recipe YAML.

    Returns a normalized dict:
      format: 'native'  -> sections: server/env/vllm (this repo's run.sh format)
              'command' -> eugr-style recipe with defaults + command template
      raw: the parsed YAML
    """
    path = Path(path)
    raw = load_yaml(path)
    if not raw:
        raise ValueError(f"empty recipe: {path}")
    if isinstance(raw.get("server"), dict) or isinstance(raw.get("vllm"), dict):
        fmt = "native"
    elif "command" in raw:
        fmt = "command"
    else:
        fmt = "native"
    return {
        "path": path,
        "format": fmt,
        "raw": raw,
        "server": raw.get("server") or {},
        "env": raw.get("env") or {},
        "vllm": raw.get("vllm") or {},
    }


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def flags_from_vllm_section(vllm: dict) -> list[str]:
    """Convert a recipe ``vllm:`` mapping into argv, run.sh semantics:
    true -> bare flag, false/null/'' -> omitted, anything else -> "flag value"."""
    flags: list[str] = []
    for key, value in vllm.items():
        if isinstance(value, bool):
            if value:
                flags.append(f"--{key}")
            continue
        if value is None or value == "":
            continue
        flags.extend([f"--{key}", str(value)])
    return flags


_FLAG_EQ_RE = re.compile(r"^--([^=]+)=(.*)$")


def canonical_flag(name: str) -> str:
    """Canonical spelling for flag dedupe: accept ``--speculative_config`` and
    ``--speculative-config`` as the same flag."""
    return "--" + name.lstrip("-").replace("_", "-").lower()


def flag_name(arg: str) -> str | None:
    if arg.startswith("--"):
        m = _FLAG_EQ_RE.match(arg)
        return canonical_flag(m.group(1) if m else arg[2:].split()[0])
    return None


def flag_has_value(arg: str) -> bool:
    return "=" in arg


def remove_flag(flags: list[str], name: str) -> list[str]:
    """Remove every occurrence of flag ``name`` (canonical form) and its value.

    Handles ``--flag value`` and ``--flag=value``; safe for the flags we use.
    """
    target = canonical_flag(name)
    out: list[str] = []
    i = 0
    while i < len(flags):
        arg = flags[i]
        fname = flag_name(arg)
        if fname == target:
            if flag_has_value(arg):
                i += 1
            else:
                i += 2
            continue
        out.append(arg)
        i += 1
    return out


def get_flag_value(flags: list[str], name: str):
    target = canonical_flag(name)
    i = 0
    while i < len(flags):
        arg = flags[i]
        if flag_name(arg) == target:
            if flag_has_value(arg):
                return arg.split("=", 1)[1]
            return flags[i + 1] if i + 1 < len(flags) else None
        i += 1
    return None


def set_flag(flags: list[str], name: str, value) -> list[str]:
    """Replace (or append) exactly one instance of a flag. Never duplicates."""
    flags = remove_flag(flags, name)
    if value is None or value is False:
        return flags
    if value is True:
        return flags + [f"--{name}"]
    return flags + [f"--{name}", str(value)]


def parse_set_overrides(pairs: list[str]) -> dict:
    """Parse ``--set vllm.max-num-seqs=8`` / ``env.FOO=bar`` into nested dict."""
    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        cursor = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


def apply_set_overrides(recipe: dict, overrides: dict) -> dict:
    """Apply nested ``--set`` overrides to a loaded recipe in place."""
    for key, value in (overrides or {}).items():
        if key in ("vllm", "env", "server") and isinstance(value, dict):
            recipe.setdefault(key, {}).update(value)
        else:  # bare key -> vllm section (most common case)
            recipe.setdefault("vllm", {})[key] = value
    return recipe


def spec_config_obj(flags: list[str]) -> dict | None:
    raw = get_flag_value(flags, "speculative-config")
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:  # noqa: BLE001
        return None


def spec_num_tokens(flags: list[str]) -> int | None:
    obj = spec_config_obj(flags)
    if not obj:
        return None
    try:
        return int(obj.get("num_speculative_tokens"))
    except Exception:  # noqa: BLE001
        return None


def hf_snapshot_dir(repo: str, hf_home: Path | None = None) -> Path | None:
    hf_home = Path(hf_home or os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    base = hf_home / "hub" / ("models--" + repo.replace("/", "--"))
    try:
        ref = (base / "refs" / "main").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    snap = base / "snapshots" / ref
    return snap if snap.is_dir() else None


def resolve_model_dir(repo: str, models_dir: Path, hf_home: Path | None = None):
    """Mirror run.sh: prefer the HF cache snapshot, fall back to <models_dir>/<name>."""
    models_dir = Path(models_dir)
    snap = hf_snapshot_dir(repo, hf_home)
    if snap and (snap / "config.json").is_file():
        return snap, "hf-snapshot"
    local = models_dir / repo.split("/")[-1]
    if (local / "config.json").is_file():
        return local, "models-dir"
    return None, "missing"


# ──────────────────────────────────────────────────────────────────────────────
# docker (never nvidia-smi)
# ──────────────────────────────────────────────────────────────────────────────


def run_cmd(argv: list[str], timeout: float = 120.0, check: bool = False) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"command not found: {argv[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"timeout after {timeout}s: {' '.join(argv)}") from exc
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(argv)}\n{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc


def docker_ok() -> bool:
    try:
        return run_cmd(["docker", "version", "--format", "{{.Server.Version}}"], timeout=15).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def docker_inspect(name: str) -> dict | None:
    proc = run_cmd(["docker", "inspect", "--format", "{{json .}}", name], timeout=30)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except Exception:  # noqa: BLE001
        return None


def docker_rm(name: str, timeout: float = 120.0) -> bool:
    return run_cmd(["docker", "rm", "-f", name], timeout=timeout).returncode == 0


def docker_logs(name: str, tail: int | None = None, timeout: float = 120.0) -> str:
    argv = ["docker", "logs", "--timestamps"]
    if tail is not None:
        argv += ["--tail", str(tail)]
    argv.append(name)
    proc = run_cmd(argv, timeout=timeout)
    text = proc.stdout or ""
    if proc.stderr:
        text += proc.stderr
    return _strip_ansi(text)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def docker_ps() -> list[dict]:
    proc = run_cmd(["docker", "ps", "--format", "{{json .}}"], timeout=30)
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return rows


VLLM_IMAGE_HINTS = ("vllm", "myllmbox", "eugr", "qwen38")


def looks_like_model_server(row: dict) -> bool:
    image = (row.get("Image") or "").lower()
    command = (row.get("Command") or "").lower()
    names = (row.get("Names") or "").lower()
    if "vllm serve" in command or command.startswith("vllm"):
        return True
    if any(h in image for h in VLLM_IMAGE_HINTS):
        return True
    return "qwen38-flash" in names


def health_ok(base_url: str, timeout: float = 5.0) -> bool:
    """True when <base_url>/health answers 200 (vLLM's readiness endpoint)."""
    status, _ = http_text(url_join(base_url, "/health"), timeout=timeout)
    return status == 200


def port_healthy(port: int, timeout: float = 3.0) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def docker_image_meta(image: str) -> dict:
    meta = {"image": image, "id": None, "repo_digests": [], "created": None, "size": None}
    proc = run_cmd(
        ["docker", "image", "inspect", image, "--format",
         "{{json .Id}}\t{{json .RepoDigests}}\t{{json .Created}}\t{{.Size}}"],
        timeout=60,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        parts = proc.stdout.strip().split("\t")
        try:
            meta["id"] = json.loads(parts[0])
            meta["repo_digests"] = json.loads(parts[1]) if len(parts) > 1 else []
            meta["created"] = json.loads(parts[2]) if len(parts) > 2 else None
            meta["size"] = parts[3].strip() if len(parts) > 3 else None
        except Exception:  # noqa: BLE001
            pass
    return meta


# ──────────────────────────────────────────────────────────────────────────────
# HTTP / OpenAI-compatible streaming
# ──────────────────────────────────────────────────────────────────────────────


def api_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def url_join(base_url: str, path: str) -> str:
    return api_base(base_url) + path


def http_json(url: str, timeout: float = 30.0, method: str = "GET", body: dict | None = None):
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw.decode("utf-8", "replace")) if raw else None
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read().decode("utf-8", "replace")[:2000]
        except Exception:  # noqa: BLE001
            payload = ""
        return exc.code, {"error": payload}
    except Exception as exc:  # noqa: BLE001
        return None, {"error": f"{type(exc).__name__}: {exc}"}


def http_text(url: str, timeout: float = 30.0) -> tuple[int | None, str]:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read().decode("utf-8", "replace")[:4000]
        except Exception:  # noqa: BLE001
            return exc.code, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def list_models(base_url: str, timeout: float = 20.0):
    status, obj = http_json(url_join(base_url, "/v1/models"), timeout=timeout)
    if status == 200 and isinstance(obj, dict):
        return [m.get("id") for m in obj.get("data") or [] if m.get("id")]
    return []


class StreamingResult(dict):
    """dict with attribute access for convenience."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # noqa: BLE001
            raise AttributeError(item) from exc


def stream_chat(base_url: str, body: dict, timeout: float = 600.0) -> StreamingResult:
    """POST a streaming chat completion and time the token stream.

    Returns a dict with request timing, token counts (usage when the server
    sends it), per-chunk arrival times and error info. Never raises for HTTP or
    network errors: failures are reported in ``status``/``error``.
    """
    url = url_join(base_url, "/v1/chat/completions")
    payload = dict(body)
    payload.setdefault("stream", True)
    payload.setdefault("stream_options", {"include_usage": True})
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    out = StreamingResult(
        ok=False,
        status=None,
        error=None,
        http_status=None,
        error_body=None,
        text="",
        reasoning="",
        finish_reason=None,
        usage=None,
        n_content_chunks=0,
        n_tokens=0,
        token_count_source="none",
        ttft_s=None,
        first_token_at=None,
        last_token_at=None,
        end_at=None,
        duration_s=None,
        itl_ms=[],
        chunk_at=[],
        t0=None,
    )
    t0 = time.perf_counter()
    out["t0"] = t0
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    buf = ""

    def process_line(line: str, arrived: float) -> bool:
        nonlocal buf
        line = line.strip()
        if not line or line.startswith(":"):
            return False
        if not line.startswith("data:"):
            return False
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            return True
        try:
            obj = json.loads(data_str)
        except Exception:  # noqa: BLE001
            return False
        if isinstance(obj.get("usage"), dict):
            out["usage"] = obj["usage"]
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if piece:
                if out["first_token_at"] is None:
                    out["first_token_at"] = arrived
                content_parts.append(piece)
                out["chunk_at"].append(arrived - t0)
                out["n_content_chunks"] += 1
            if reasoning:
                if out["first_token_at"] is None:
                    out["first_token_at"] = arrived
                reasoning_parts.append(reasoning)
                if not piece:
                    out["chunk_at"].append(arrived - t0)
            if choice.get("finish_reason"):
                out["finish_reason"] = choice["finish_reason"]
        return False

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out["http_status"] = resp.status
            while True:
                block = resp.read1(4096)  # type: ignore[attr-defined]
                if not block:
                    break
                arrived = time.perf_counter()
                buf += block.decode("utf-8", "replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    process_line(line, arrived)
            if buf:
                process_line(buf, time.perf_counter())
        out["ok"] = out["error"] is None
    except urllib.error.HTTPError as exc:
        out["http_status"] = exc.code
        try:
            out["error_body"] = exc.read().decode("utf-8", "replace")[:2000]
        except Exception:  # noqa: BLE001
            out["error_body"] = ""
        out["error"] = f"HTTP {exc.code}: {out['error_body']}"
        out["status"] = "error"
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["status"] = "error"

    end = time.perf_counter()
    out["end_at"] = end
    out["duration_s"] = end - t0
    out["text"] = "".join(content_parts)
    out["reasoning"] = "".join(reasoning_parts)

    if out["first_token_at"] is not None:
        out["ttft_s"] = out["first_token_at"] - t0
        out["last_token_at"] = (
            out["first_token_at"] + out["chunk_at"][-1] if out["chunk_at"] else out["first_token_at"]
        )
    usage = out["usage"] or {}
    completion = usage.get("completion_tokens")
    if isinstance(completion, int) and completion > 0:
        out["n_tokens"] = completion
        out["token_count_source"] = "usage"
    elif out["n_content_chunks"]:
        out["n_tokens"] = out["n_content_chunks"]
        out["token_count_source"] = "chunks"
    # inter-token (chunk) latencies
    for prev, cur in zip(out["chunk_at"], out["chunk_at"][1:]):
        out["itl_ms"].append(round((cur - prev) * 1000.0, 3))
    if out["ok"] and out["ttft_s"] is not None:
        decode_window = out["last_token_at"] - out["first_token_at"]
        n_tokens = out["n_tokens"]
        if n_tokens > 1 and decode_window > 0:
            out["decode_tok_s"] = n_tokens / decode_window
            out["tpot_ms"] = decode_window * 1000.0 / (n_tokens - 1)
        else:
            out["decode_tok_s"] = None
            out["tpot_ms"] = None
        out["e2e_tok_s"] = n_tokens / out["duration_s"] if out["duration_s"] > 0 else None
    else:
        out.setdefault("decode_tok_s", None)
        out.setdefault("tpot_ms", None)
        out.setdefault("e2e_tok_s", None)
    if out["status"] is None:
        out["status"] = "ok" if out["ok"] else "error"
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Prometheus /metrics parsing
# ──────────────────────────────────────────────────────────────────────────────

_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_metrics_text(text: str) -> dict:
    """Parse the Prometheus text format into {name: [ {labels, value}, ... ]}."""
    metrics: dict[str, list[dict]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line and "}" in line:
            name, rest = line.split("{", 1)
            label_text, _, value_text = rest.rpartition("}")
            value_text = value_text.strip()
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, value_text = parts[0], parts[1]
            label_text = ""
        try:
            value = float(value_text.split()[0])
        except Exception:  # noqa: BLE001
            continue
        labels = {m.group(1): m.group(2) for m in _LABEL_RE.finditer(label_text)}
        metrics.setdefault(name.strip(), []).append({"labels": labels, "value": value})
    return metrics


def metric_total(metrics: dict, name: str) -> float | None:
    samples = metrics.get(name)
    if not samples:
        return None
    return sum(s["value"] for s in samples)


def metric_gauge(metrics: dict, name: str) -> float | None:
    samples = metrics.get(name)
    if not samples:
        return None
    return samples[-1]["value"]


def metric_by_pos(metrics: dict, name: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for sample in metrics.get(name) or []:
        pos = sample["labels"].get("position")
        if pos is not None:
            try:
                out[int(pos)] = out.get(int(pos), 0.0) + sample["value"]
            except ValueError:
                pass
    return out


HISTOGRAM_QUANTILES = (50, 90, 95, 99)


def histogram_quantile(metrics: dict, base: str, q: float) -> float | None:
    """Prometheus-style quantile from a classic histogram (``<base>_bucket``)."""
    buckets = metrics.get(base + "_bucket")
    if not buckets:
        return None
    points: list[tuple[float, float]] = []
    for sample in buckets:
        le = sample["labels"].get("le")
        if le is None:
            continue
        try:
            upper = float("inf") if le in ("+Inf", "Inf") else float(le)
        except ValueError:
            continue
        points.append((upper, sample["value"]))
    if not points:
        return None
    points.sort()
    total = points[-1][1]
    if total <= 0:
        return None
    rank = q / 100.0 * total
    prev_upper, prev_count = 0.0, 0.0
    for upper, count in points:
        if count >= rank:
            if upper == float("inf"):
                return prev_upper
            span = upper - prev_upper
            inner = (count - prev_count) or 1.0
            return prev_upper + span * ((rank - prev_count) / inner)
        prev_upper, prev_count = upper, count
    return points[-1][0]


SPEC_DECODE_METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_emitted_tokens_total",
)


def spec_decode_snapshot(metrics: dict) -> dict:
    row: dict = {}
    for name in SPEC_DECODE_METRICS:
        row[name.split(":")[-1]] = metric_total(metrics, name)
    row["accepted_per_pos"] = metric_by_pos(metrics, "vllm:spec_decode_num_accepted_tokens_per_pos_total")
    return row


def spec_decode_delta(before: dict, after: dict, num_spec_tokens: int | None) -> dict:
    def delta(key):
        b, a = before.get(key), after.get(key)
        if a is None or b is None:
            return None
        return a - b

    drafts = delta("spec_decode_num_drafts_total")
    draft_tokens = delta("spec_decode_num_draft_tokens_total")
    accepted = delta("spec_decode_num_accepted_tokens_total")
    emitted = delta("spec_decode_num_emitted_tokens_total")
    out = {
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "emitted_tokens": emitted,
        # vLLM semantics: num_drafts = number of speculative draft STEPS,
        # num_draft_tokens = total tokens proposed by the drafter, emitted =
        # tokens actually emitted by the target (accepted + bonus).
        "acceptance_rate": None,          # accepted / proposed draft tokens
        "tokens_per_step": None,          # tokens emitted per draft step
        "avg_draft_depth": None,          # draft tokens proposed per step
        "draft_tokens_per_position": None,
        "accepted_per_position": None,
    }
    if draft_tokens and accepted is not None:
        out["acceptance_rate"] = accepted / draft_tokens
    elif drafts and accepted is not None:
        out["acceptance_rate"] = accepted / drafts
    if emitted is not None and drafts:
        out["tokens_per_step"] = emitted / drafts
        out["tokens_per_step_source"] = "emitted_counter"
    elif accepted is not None and drafts:
        # vLLM images that do not export spec_decode_num_emitted_tokens_total:
        # each draft step emits the accepted tokens plus one bonus token.
        out["tokens_per_step"] = (accepted + drafts) / drafts
        out["tokens_per_step_source"] = "derived_accepted_plus_bonus"
    if draft_tokens and drafts:
        out["avg_draft_depth"] = draft_tokens / drafts
    before_pos = before.get("accepted_per_pos") or {}
    after_pos = after.get("accepted_per_pos") or {}
    pos_delta = {p: after_pos.get(p, 0.0) - before_pos.get(p, 0.0) for p in after_pos}
    out["accepted_per_position"] = pos_delta or None
    return out


KV_USAGE_NAMES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")


def kv_cache_usage(metrics: dict) -> float | None:
    for name in KV_USAGE_NAMES:
        value = metric_gauge(metrics, name)
        if value is not None:
            return value
    return None


def snapshot_metrics(base_url: str, timeout: float = 20.0) -> dict:
    status, text = http_text(url_join(base_url, "/metrics"), timeout=timeout)
    if status != 200 or not text:
        return {"ok": False, "status": status, "error": text[:400] if text else None, "raw": ""}
    parsed = parse_metrics_text(text)
    return {
        "ok": True,
        "status": status,
        "raw": text,
        "parsed": parsed,
        "spec_decode": spec_decode_snapshot(parsed),
        "kv_cache_usage_perc": kv_cache_usage(parsed),
        "num_requests_running": metric_gauge(parsed, "vllm:num_requests_running"),
        "num_requests_waiting": metric_gauge(parsed, "vllm:num_requests_waiting"),
        "gpu_cache_usage_perc": metric_gauge(parsed, "vllm:gpu_cache_usage_perc"),
        "prompt_tokens_total": metric_total(parsed, "vllm:prompt_tokens_total"),
        "generation_tokens_total": metric_total(parsed, "vllm:generation_tokens_total"),
        "request_success_total": metric_total(parsed, "vllm:request_success_total"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# vLLM startup-log parsing (KV pool, max concurrency, timings)
# ──────────────────────────────────────────────────────────────────────────────

_LOG_PATTERNS = {
    "kv_cache_tokens": re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
    "kv_cache_memory_gib": re.compile(
        r"(?:GPU|Available) KV cache memory:\s*([\d.]+)\s*GiB|reserved\s*([\d.]+)\s*GiB memory for KV Cache"
    ),
    "max_concurrency": re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x"),
    "max_num_seqs": re.compile(r"max_num_seqs[\"'\s:=]+(\d+)"),
    "max_num_batched_tokens": re.compile(r"max_num_batched_tokens[\"'\s:=]+(\d+)"),
    "max_model_len": re.compile(r"max_model_len[\"'\s:=]+(\d+)"),
    "gpu_blocks": re.compile(r"# GPU blocks:\s*([\d,]+)"),
    "block_size": re.compile(r"block_size[\"'\s:=]+(\d+)"),
    "engine_init_s": re.compile(r"init engine .*? took\s*([\d.]+)\s*seconds"),
    "weights_load_s": re.compile(r"Loading weights took\s*([\d.]+)\s*seconds"),
    "graph_capture_s": re.compile(r"Graph capturing finished in\s*([\d.]+)\s*sec"),
    "compile_s": re.compile(r"torch\.compile took\s*([\d.]+)\s*seconds"),
    "ple_table": re.compile(r"PLE: table mapped from\s*(\S+)\s*\(([^)]*)\)"),
    "vllm_version": re.compile(r"vLLM version\s*([\w.+\-]+)"),
    "served_model": re.compile(r"served_model_name[=:]\s*\[?([^\],]+)"),
}


def parse_startup_log(text: str) -> dict:
    out: dict = {}
    if not text:
        return out
    for key, pattern in _LOG_PATTERNS.items():
        matches = pattern.findall(text)
        if not matches:
            continue
        last = matches[-1]
        if key == "max_concurrency":
            out["kv_cache_max_concurrency_tokens"] = int(last[0].replace(",", ""))
            out["kv_cache_max_concurrency"] = float(last[1])
        elif key == "ple_table":
            out["ple_table_path"] = last[0]
            out["ple_table_detail"] = last[1]
        elif key in (
            "engine_init_s",
            "weights_load_s",
            "graph_capture_s",
            "compile_s",
            "kv_cache_memory_gib",
            "kv_cache_max_concurrency",
        ):
            value = last if not isinstance(last, tuple) else next((v for v in last if v), "")
            try:
                out[key] = float(value)
            except Exception:  # noqa: BLE001
                pass
        elif key in (
            "max_num_seqs",
            "max_num_batched_tokens",
            "max_model_len",
            "block_size",
            "gpu_blocks",
            "kv_cache_tokens",
        ):
            try:
                out[key] = int(str(last).replace(",", ""))
            except Exception:  # noqa: BLE001
                pass
        else:
            out[key] = last if not isinstance(last, tuple) else " ".join(last)
    # cumulative timings are often logged in phases
    for key in ("engine_init_s", "weights_load_s"):
        values = [float(v) for v in _LOG_PATTERNS[key].findall(text)]
        if values:
            out[key + "_total"] = sum(values)
    kv_tokens = out.get("kv_cache_tokens")
    if kv_tokens and "block_size" in out and out["block_size"]:
        out["kv_cache_blocks"] = kv_tokens // out["block_size"]
    return out


def analyze_server_log(log_path: Path) -> dict:
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    info = parse_startup_log(text)
    info["log_path"] = repo_rel(log_path)
    info["log_bytes"] = len(text)
    return info


# ──────────────────────────────────────────────────────────────────────────────
# stats
# ──────────────────────────────────────────────────────────────────────────────


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def median(values: list[float]) -> float | None:
    return percentile(values, 50)


def stdev(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def stat_block(values: list[float], digits: int = 3) -> dict:
    clean = [float(v) for v in values if isinstance(v, (int, float))]

    def _r(value):
        return round(value, digits) if value is not None else None

    return {
        "n": len(clean),
        "mean": _r(mean(clean)),
        "median": _r(median(clean)),
        "p50": _r(percentile(clean, 50)),
        "p90": _r(percentile(clean, 90)),
        "p99": _r(percentile(clean, 99)),
        "min": _r(min(clean) if clean else None),
        "max": _r(max(clean) if clean else None),
        "stdev": _r(stdev(clean)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# run directory
# ──────────────────────────────────────────────────────────────────────────────


class RunDir:
    def __init__(self, path: Path):
        self.path = Path(path)
        ensure_dir(self.path)
        self.meta_path = self.path / "meta.json"
        self.summary_path = self.path / "summary.json"
        self.requests_path = self.path / "requests.jsonl"

    def write_json(self, name: str, obj) -> Path:
        target = self.path / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(target)
        return target

    def read_json(self, name: str, default=None):
        target = self.path / name
        if not target.is_file():
            return default
        try:
            return json.loads(target.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return default

    def append_jsonl(self, line: dict) -> None:
        with open(self.requests_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, sort_keys=True) + "\n")

    def write_text(self, name: str, text: str) -> Path:
        target = self.path / name
        target.write_text(text, encoding="utf-8")
        return target


def read_requests(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "requests.jsonl"
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    return rows


def host_facts() -> dict:
    facts = {"uname": os.uname().release, "machine": os.uname().machine, "python": sys.version.split()[0]}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines()[:3]:
            if line.startswith("MemTotal:"):
                facts["mem_total_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    try:
        facts["docker_version"] = run_cmd(["docker", "version", "--format", "{{.Server.Version}}"], timeout=15).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return facts


def which(cmd: str) -> str | None:
    for part in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(part) / cmd
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def file_mode_readable(path: Path) -> bool:
    try:
        return bool(Path(path).stat().st_mode & stat.S_IRUSR)
    except OSError:
        return False


def sleep_short(seconds: float = 0.05) -> None:
    time.sleep(seconds)


def shlex_split(text: str) -> list[str]:
    return shlex.split(text) if text else []
