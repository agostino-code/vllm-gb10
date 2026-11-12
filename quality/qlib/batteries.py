"""Battery runners: data + client -> per-item graded records.

Each category is independently runnable.  The heavy long-context battery runs
serially (one request at a time) while the text batteries fan out over a small
thread pool to keep the default suite inside the time budget.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import graders
from .client import ChatClient, ChatResult, ServerUnavailable
from .needle import build_needle_prompt, grade_needle

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CATEGORIES = ["math", "json", "code", "instruction", "tool", "needle", "repetition"]

_CATEGORY_ALIASES = {
    "math": "math",
    "arithmetic": "math",
    "json": "json",
    "json-mode": "json",
    "code": "code",
    "coding": "code",
    "instruction": "instruction",
    "instruction-following": "instruction",
    "tool": "tool",
    "tools": "tool",
    "tool-calling": "tool",
    "toolcalling": "tool",
    "needle": "needle",
    "long_context": "needle",
    "long-context": "needle",
    "longcontext": "needle",
    "repetition": "repetition",
    "degeneracy": "repetition",
    "repetition-degeneracy": "repetition",
}

#: categories whose prompt can be replayed from canned outputs for offline tests
REPLAYABLE = ["math", "json", "code", "instruction", "tool", "needle", "repetition"]
HEAVY = {"needle"}

SYSTEM_PROMPTS = {
    "math": "You are a careful mathematician. Work through the problem briefly and give the exact final answer.",
    "json": "You are a precise JSON generator. Reply with valid JSON only and no markdown fences.",
    "code": "You are an expert Python programmer. Reply with one Python code block.",
    "instruction": "Follow the user's formatting instructions exactly and do not add anything else.",
    "tool": None,
    "needle": None,
    "repetition": None,
}

#: max_tokens per category with thinking disabled (answers are short)
MAX_TOKENS = {
    "math": 512,
    "json": 1024,
    "code": 1024,
    "instruction": 256,
    "tool": 512,
    "needle": 64,
    "repetition": 2048,
}

#: max_tokens per category with thinking enabled (reasoning needs headroom)
THINKING_MAX_TOKENS = {
    "math": 4096,
    "json": 4096,
    "code": 4096,
    "instruction": 3072,
    "tool": 4096,
    "needle": 2048,
    "repetition": 4096,
}

DEFAULT_WORKERS = {
    "math": 4,
    "json": 4,
    "code": 4,
    "instruction": 4,
    "tool": 4,
    "needle": 1,
    "repetition": 1,
}

#: default needle sizes in *tokens* (chosen to stay below max_model_len=409600)
DEFAULT_NEEDLE_SIZES = [32768, 131072, 262144]

MATH_SUFFIX = "\n\nEnd your reply with a final line exactly of the form: ANSWER: <value>"


def _load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_data(data_dir: str | Path | None = None) -> dict:
    """Load every shipped prompt/test file."""
    root = Path(data_dir) if data_dir else DATA_DIR
    return {
        "math": _load(root / "math.json"),
        "json": _load(root / "json_tasks.json"),
        "code": _load(root / "code_tasks.json"),
        "instruction": _load(root / "instruction.json"),
        "tool": _load(root / "tools.json"),
        "repetition": _load(root / "repetition.json"),
        "needle": _load(root / "needle.json"),
    }


def resolve_category(name: str) -> str:
    key = name.strip().lower()
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    raise ValueError("unknown category %r (known: %s)" % (name, ", ".join(CATEGORIES)))


def parse_categories(spec: str | None) -> list:
    """Parse --categories.

    Accepted forms:
      all (default)            -> every battery
      quick | text             -> every battery except the long-context needle
      math,json                -> explicit list
      all,-needle              -> everything except needle
      needle                   -> only the long-context battery
    Unknown names raise ValueError.
    """
    if not spec:
        return list(CATEGORIES)
    tokens = [tok.strip() for tok in str(spec).split(",") if tok.strip()]
    if not tokens:
        return list(CATEGORIES)
    selected: list = []
    started_with_negative = tokens[0].startswith("-")
    for token in tokens:
        negative = token.startswith("-")
        name = token.lstrip("+-").strip().lower()
        if negative and not selected and started_with_negative:
            selected = list(CATEGORIES)
        if name in ("all", "default", "full"):
            cats = list(CATEGORIES)
        elif name in ("quick", "fast", "text"):
            cats = [c for c in CATEGORIES if c not in HEAVY]
        else:
            cats = [resolve_category(name)]
        if negative:
            selected = [c for c in selected if c not in cats]
        else:
            for cat in cats:
                if cat not in selected:
                    selected.append(cat)
    if not selected:
        raise ValueError("category selection %r is empty" % (spec,))
    return [c for c in CATEGORIES if c in selected]


def parse_needle_sizes(spec: str | None) -> list:
    """Parse --needle-sizes: '32k,128k,262k', '32768,131072' or 'all'."""
    if not spec:
        return list(DEFAULT_NEEDLE_SIZES)
    if str(spec).strip().lower() in ("all", "default"):
        return list(DEFAULT_NEEDLE_SIZES)
    sizes = []
    for token in str(spec).split(","):
        token = token.strip().lower().replace("_", "")
        if not token:
            continue
        if token.endswith("k"):
            value = int(float(token[:-1]) * 1024)
        else:
            value = int(float(token))
        if value <= 0:
            raise ValueError("needle size must be positive: %r" % (token,))
        sizes.append(value)
    if not sizes:
        raise ValueError("no needle sizes parsed from %r" % (spec,))
    return sizes


@dataclass
class RunConfig:
    """Knobs shared by all batteries for one run."""

    thinking: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 12345
    workers: int = 0  # 0 -> per-category default
    needle_sizes: list = field(default_factory=lambda: list(DEFAULT_NEEDLE_SIZES))
    needle_depth: float = 0.5
    needle_chars_per_token: float = 3.5
    needle_question_max_tokens: int = 0  # 0 -> MAX_TOKENS table
    save_prompts: bool = False
    request_timeout: float = 600.0

    def max_tokens_for(self, category: str) -> int:
        table = THINKING_MAX_TOKENS if self.thinking else MAX_TOKENS
        return int(table.get(category, 1024))

    def workers_for(self, category: str) -> int:
        if self.workers:
            return max(1, int(self.workers))
        return DEFAULT_WORKERS.get(category, 4)


@dataclass
class ItemRecord:
    """One graded request."""

    category: str
    item_id: str
    prompt: str
    messages: list
    grade: dict
    result: ChatResult
    duration_s: float
    extra: dict = field(default_factory=dict)

    def to_raw(self, save_prompts: bool = False) -> dict:
        raw = {
            "category": self.category,
            "id": self.item_id,
            "duration_s": round(self.duration_s, 3),
            "request": {
                "max_tokens": self.extra.get("max_tokens"),
                "temperature": self.extra.get("temperature"),
                "thinking": self.extra.get("thinking"),
                "tools": len(self.extra.get("tools") or []),
            },
            "response": self.result.to_dict(),
            "grade": self.grade,
        }
        raw["request"]["prompt_chars"] = len(self.prompt)
        raw["request"]["prompt_sha256"] = self.extra.get("prompt_sha256") or _sha256(self.prompt)
        if save_prompts:
            raw["request"]["messages"] = self.messages
        return raw


def _sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _build_messages(category: str, item: dict, needle_system: str | None = None) -> list:
    messages = []
    system = SYSTEM_PROMPTS.get(category)
    if category == "needle" and needle_system:
        system = needle_system
    if system:
        messages.append({"role": "system", "content": system})
    prompt = item["prompt"]
    if category == "math":
        prompt = prompt + MATH_SUFFIX
    messages.append({"role": "user", "content": prompt})
    return messages


def _record(category: str, item: dict, messages: list, result: ChatResult, cfg: RunConfig, duration: float,
            max_tokens: int, tools: list | None = None, prompt_override: str | None = None) -> ItemRecord:
    if category == "needle":
        grade = result  # placeholder, caller replaces
    else:
        grade = graders.grade_item(category, item, result)
    prompt_text = prompt_override if prompt_override is not None else messages[-1].get("content", "")
    extra = {
        "max_tokens": max_tokens,
        "temperature": cfg.temperature,
        "thinking": cfg.thinking,
        "tools": tools,
        "prompt_sha256": _sha256(prompt_text),
    }
    return ItemRecord(
        category=category,
        item_id=str(item.get("id")),
        prompt=prompt_text,
        messages=messages,
        grade=grade,
        result=result,
        duration_s=duration,
        extra=extra,
    )


def run_text_category(category: str, items: list, client: ChatClient, cfg: RunConfig) -> tuple:
    """Run one of the short text batteries. Returns (records, errors)."""
    records: list = []
    errors: list = []
    max_tokens = cfg.max_tokens_for(category)
    messages_list = [_build_messages(category, item) for item in items]

    def work(index: int) -> tuple:
        item = items[index]
        messages = messages_list[index]
        started = time.perf_counter()
        try:
            result = client.chat(
                messages,
                max_tokens=max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                seed=cfg.seed,
                tools=item.get("tools") if category == "tool" else None,
                tool_choice="auto" if category == "tool" else None,
            )
        except ServerUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the suite alive
            result = ChatResult(status="error", error="%s: %s" % (type(exc).__name__, exc))
        duration = time.perf_counter() - started
        return index, _record(category, item, messages, result, cfg, duration, max_tokens,
                              tools=item.get("tools") if category == "tool" else None)

    workers = min(cfg.workers_for(category), max(1, len(items)))
    if workers <= 1:
        for index in range(len(items)):
            _, record = work(index)
            records.append(record)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, i) for i in range(len(items))]
            collected = []
            for future in futures:
                try:
                    collected.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    if isinstance(exc, ServerUnavailable):
                        raise
                    errors.append("%s: %s" % (type(exc).__name__, exc))
            collected.sort(key=lambda pair: pair[0])
            records = [record for _, record in collected]

    for record in records:
        if record.result.status != "ok":
            errors.append("%s: %s" % (record.item_id, record.result.error))
        elif not record.result.content and not record.result.tool_calls:
            errors.append("%s: empty content (thinking enabled but max_tokens too small?)" % (record.item_id,))
    return records, errors


def run_needle_category(client: ChatClient, needle_data: dict, cfg: RunConfig,
                        max_model_len: int | None = None, sizes: list | None = None) -> tuple:
    """Run the long-context needle battery (serial, streamed for TTFT)."""
    records: list = []
    errors: list = []
    sizes = sizes or cfg.needle_sizes
    system = needle_data.get("system")
    default_max = cfg.needle_question_max_tokens or cfg.max_tokens_for("needle")
    for size in sizes:
        item = dict(needle_data)
        item["id"] = "needle-%d" % int(size)
        built = build_needle_prompt(
            target_tokens=int(size),
            depth=cfg.needle_depth,
            chars_per_token=cfg.needle_chars_per_token,
            needle=needle_data,
        )
        messages = _build_messages("needle", {"prompt": built.prompt}, needle_system=system)
        # Guard: never send a request that obviously cannot fit the context.
        est_tokens = int(built.chars / max(0.5, cfg.needle_chars_per_token))
        if max_model_len and est_tokens + default_max + 256 > max_model_len:
            errors.append("%s: skipped, estimated %d tokens + %d output > max_model_len %d"
                          % (item["id"], est_tokens, default_max, max_model_len))
            continue
        started = time.perf_counter()
        try:
            result = client.chat_stream(
                messages,
                max_tokens=default_max,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                seed=cfg.seed,
            )
        except ServerUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            result = ChatResult(status="error", error="%s: %s" % (type(exc).__name__, exc))
        duration = time.perf_counter() - started
        answer = result.content or result.reasoning or ""
        verdict = grade_needle(answer, needle_data.get("fact", ""))
        usage_tokens = (result.usage or {}).get("prompt_tokens")
        prompt_tokens = int(usage_tokens or est_tokens)
        # Validate the sizing heuristic against the server's real tokenizer:
        # the request must land within 10% below the target and inside context.
        tokens_ok = None
        if usage_tokens:
            tokens_ok = prompt_tokens >= 0.9 * size and (
                max_model_len is None or prompt_tokens + default_max <= max_model_len)
            if not tokens_ok:
                errors.append(
                    "%s: prompt_tokens %d outside expected range (target %d, max_model_len %s)"
                    % (item["id"], prompt_tokens, size, max_model_len))
        grade = {
            "id": item["id"],
            "score": 1 if verdict["lenient"] else 0,
            "recall": bool(verdict["lenient"]),
            "strict": bool(verdict["strict"]),
            "answer": verdict["answer"],
            "components": verdict["components"],
            "target_tokens": int(size),
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_ok": tokens_ok,
            "chars_per_token_actual": round(built.chars / prompt_tokens, 3) if prompt_tokens else None,
            "prompt_chars": built.chars,
            "prompt_sha256": built.sha256,
            "ttft_s": None if result.ttft_s is None else round(result.ttft_s, 4),
            "latency_s": round(result.latency_s, 4),
            "status": result.status,
            "error": result.error,
        }
        record = _record(
            "needle",
            item,
            messages,
            result,
            cfg,
            duration,
            default_max,
            prompt_override=built.prompt,
        )
        record.grade = grade
        record.extra["prompt_sha256"] = built.sha256
        record.extra["target_tokens"] = size
        records.append(record)
        if result.status != "ok":
            errors.append("%s: %s" % (item["id"], result.error))
    return records, errors


def _mean(values: list) -> float:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize_category(category: str, records: list, runtime_s: float, errors: list) -> dict:
    """Aggregate graded records into the category block of scores.json."""
    grades = [record.grade for record in records]
    scores = [float(g.get("score", 0)) for g in grades]
    total = len(scores)
    passed = int(sum(1 for s in scores if s >= 1.0))
    summary = {
        "score": _mean(scores),
        "passed": passed,
        "total": total,
        "runtime_s": round(runtime_s, 2),
        "errors": errors[:20],
        "items": grades,
    }
    if category == "json":
        valid = sum(1 for g in grades if g.get("valid_json"))
        summary["valid_json_rate"] = _mean([1.0 if g.get("valid_json") else 0.0 for g in grades]) if grades else 0.0
        summary["schema_conformance_rate"] = summary["score"]
        summary["valid_json_count"] = valid
    elif category == "code":
        tests_total = sum(int(g.get("tests_total", 0)) for g in grades)
        tests_passed = sum(int(g.get("tests_passed", 0)) for g in grades)
        summary["tests_passed"] = tests_passed
        summary["tests_total"] = tests_total
        summary["assertion_pass_rate"] = _mean([g.get("tests_ok_fraction", 0.0) for g in grades])
        summary["status_counts"] = dict(Counter(g.get("status") for g in grades))
    elif category == "instruction":
        checks_total = sum(int(g.get("checks_total", 0)) for g in grades)
        checks_passed = sum(int(g.get("checks_passed", 0)) for g in grades)
        summary["checks_passed"] = checks_passed
        summary["checks_total"] = checks_total
        summary["check_pass_rate"] = round(checks_passed / checks_total, 4) if checks_total else 0.0
    elif category == "tool":
        summary["reason_counts"] = dict(Counter(g.get("reason") for g in grades))
        summary["server_parsed_count"] = sum(1 for g in grades if g.get("source") == "server_parsed")
        summary["fallback_parse_count"] = sum(1 for g in grades if g.get("source") == "content_xml_fallback")
    elif category == "needle":
        summary["recall_rate"] = summary["score"]
        summary["strict_recall_rate"] = _mean([1.0 if g.get("strict") else 0.0 for g in grades])
        summary["ttft_s"] = _mean([g.get("ttft_s") for g in grades])
        summary["prompt_tokens"] = [g.get("prompt_tokens") for g in grades]
        summary["prompt_tokens_ok"] = [g.get("prompt_tokens_ok") for g in grades]
        summary["chars_per_token_actual"] = [g.get("chars_per_token_actual") for g in grades]
        summary["target_tokens"] = [g.get("target_tokens") for g in grades]
    elif category == "repetition" and grades:
        first = grades[0]
        summary["metrics"] = {
            key: first.get(key)
            for key in ("n_tokens", "unique_token_ratio", "distinct_2", "max_ngram_repeat",
                        "longest_identical_line_run", "degenerate", "reasons", "finish_reason")
            if key in first
        }
    return summary


def run_batteries(client: ChatClient, data: dict, cfg: RunConfig, categories: list,
                  max_model_len: int | None = None, log=None) -> dict:
    """Run the requested categories. Returns {"categories": {...}, "aborted": bool}."""
    out = {"categories": {}, "aborted": False}
    for category in categories:
        started = time.perf_counter()
        errors: list = []
        if log:
            log("running %s ..." % category)
        try:
            if category == "needle":
                records, errors = run_needle_category(client, data["needle"], cfg, max_model_len=max_model_len)
            else:
                records, errors = run_text_category(category, data[category], client, cfg)
        except ServerUnavailable as exc:
            out["aborted"] = True
            errors.append("server unavailable: %s" % (exc,))
            runtime = time.perf_counter() - started
            out["categories"][category] = {
                "score": 0.0, "passed": 0, "total": 0, "runtime_s": round(runtime, 2),
                "errors": errors, "items": [], "skipped": True,
            }
            if log:
                log("ABORTED %s: %s" % (category, exc))
            break
        runtime = time.perf_counter() - started
        summary = summarize_category(category, records, runtime, errors)
        summary["skipped"] = False
        out["categories"][category] = summary
        out["_records"] = out.get("_records", [])
        out["_records"].extend(records)
        if log:
            log("%-12s score=%s passed=%d/%d runtime=%.1fs" % (
                category, summary["score"], summary["passed"], summary["total"], runtime))
    return out


# --------------------------------------------------------------------------
# replay (offline) path used by the unit tests and --self-test
# --------------------------------------------------------------------------


def load_canned(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("categories", payload)


def _canned_result(entry: dict) -> ChatResult:
    return ChatResult(
        content=entry.get("content", "") or "",
        reasoning=entry.get("reasoning", "") or "",
        tool_calls=entry.get("tool_calls") or [],
        finish_reason=entry.get("finish_reason"),
        usage=entry.get("usage") or {},
        latency_s=float(entry.get("latency_s", 0.0) or 0.0),
        ttft_s=entry.get("ttft_s"),
        status=entry.get("status", "ok"),
        error=entry.get("error"),
    )


def run_replay(canned: dict, data: dict, cfg: RunConfig, categories: list,
               needle_sizes: list | None = None) -> dict:
    """Grade canned model outputs offline, producing the same structure as a live run."""
    out = {"categories": {}, "aborted": False}
    needle_sizes = needle_sizes or cfg.needle_sizes
    for category in categories:
        started = time.perf_counter()
        errors: list = []
        records: list = []
        entries = canned.get(category) or []
        by_id = {str(e.get("id")): e for e in entries if isinstance(e, dict)}
        if category == "needle":
            for size in needle_sizes:
                item_id = "needle-%d" % int(size)
                entry = by_id.get(item_id)
                if entry is None:
                    errors.append("%s: missing canned output" % item_id)
                    continue
                built = build_needle_prompt(
                    target_tokens=int(size), depth=cfg.needle_depth,
                    chars_per_token=cfg.needle_chars_per_token, needle=data["needle"],
                )
                result = _canned_result(entry)
                verdict = grade_needle(entry.get("content", "") or result.reasoning, data["needle"]["fact"])
                usage_tokens = int((entry.get("usage") or {}).get("prompt_tokens") or 0)
                tokens_ok = None if not usage_tokens else usage_tokens >= 0.9 * size
                if tokens_ok is False:
                    errors.append("%s: prompt_tokens %d outside expected range (target %d)"
                                  % (item_id, usage_tokens, size))
                grade = {
                    "id": item_id,
                    "score": 1 if verdict["lenient"] else 0,
                    "recall": bool(verdict["lenient"]),
                    "strict": bool(verdict["strict"]),
                    "answer": verdict["answer"],
                    "components": verdict["components"],
                    "target_tokens": int(size),
                    "prompt_tokens": usage_tokens,
                    "prompt_tokens_ok": tokens_ok,
                    "chars_per_token_actual": round(built.chars / usage_tokens, 3) if usage_tokens else None,
                    "prompt_chars": built.chars,
                    "prompt_sha256": built.sha256,
                    "ttft_s": entry.get("ttft_s"),
                    "latency_s": entry.get("latency_s"),
                    "status": "ok",
                    "error": None,
                }
                record = ItemRecord(
                    category="needle", item_id=item_id,
                    prompt=built.prompt,
                    messages=[{"role": "user", "content": built.prompt}],
                    grade=grade, result=result, duration_s=0.0,
                    extra={"max_tokens": cfg.max_tokens_for("needle"), "temperature": cfg.temperature,
                           "thinking": cfg.thinking, "prompt_sha256": built.sha256,
                           "target_tokens": size, "replayed": True},
                )
                records.append(record)
        else:
            items = {str(item.get("id")): item for item in data.get(category) or []}
            for item_id, item in items.items():
                entry = by_id.get(item_id)
                if entry is None:
                    errors.append("%s: missing canned output" % item_id)
                    continue
                result = _canned_result(entry)
                record = _record(category, item, _build_messages(category, item), result, cfg, 0.0,
                                 cfg.max_tokens_for(category),
                                 tools=item.get("tools") if category == "tool" else None)
                if entry.get("prompt_sha256"):
                    record.extra["prompt_sha256"] = entry["prompt_sha256"]
                record.extra["replayed"] = True
                records.append(record)
        runtime = time.perf_counter() - started
        summary = summarize_category(category, records, runtime, errors)
        summary["skipped"] = False
        summary["replayed"] = True
        out["categories"][category] = summary
        out.setdefault("_records", []).extend(records)
    return out
