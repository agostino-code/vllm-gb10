"""Deterministic graders for every quality battery.

This module is pure Python (stdlib only) and has NO network dependency: every
grader takes a model output string / tool-call list and returns a JSON-able
verdict.  That property is what makes the suite unit-testable offline with
canned model outputs (see quality/tests/).

Scoring conventions (documented in quality/README.md):
  * per-item score is 0 or 1 (code: 1 only if every hidden assertion passes);
  * category score = mean of item scores (partial categories still report
    diagnostic sub-rates such as JSON validity);
  * math compares *normalised exact answers* (numeric equality when both sides
    parse as numbers, else canonical string equality).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------
# generic text helpers
# --------------------------------------------------------------------------


def canonical(text: str, casefold: bool = False, strip_punct: bool = True) -> str:
    """Collapse whitespace and (optionally) casefold/strip sentence punctuation."""
    s = (text or "").strip()
    s = re.sub(r"\s+", " ", s)
    if strip_punct:
        s = s.strip("\"'` \t")
        s = re.sub(r"[.。]+$", "", s).strip()
    if casefold:
        s = s.casefold()
    return s


def fenced_blocks(text: str) -> list:
    """Return contents of ```...``` fenced blocks, in order."""
    return re.findall(r"```[ \t]*(?:[A-Za-z0-9_+-]+)?[ \t]*\r?\n?(.*?)```", text or "", re.S)


def strip_fenced(text: str) -> str:
    """Remove fences but keep content."""
    return re.sub(r"```[ \t]*(?:[A-Za-z0-9_+-]+)?[ \t]*\r?\n?(.*?)```", r"\1", text or "", flags=re.S)


def _balance_scan(text: str, open_ch: str, close_ch: str) -> list:
    """Return balanced substrings starting with open_ch (string-aware)."""
    out = []
    depth = 0
    start = -1
    in_str = False
    quote = ""
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            quote = ch
            continue
        if ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
    return out


def extract_json_payload(text: str) -> tuple:
    """Extract the first plausible JSON object/array from model output.

    Returns (obj, error).  Tries, in order: fenced blocks, balanced-brace
    substrings, then the whole text.  Deterministic and conservative.
    """
    text = text or ""
    if not text.strip():
        return None, "empty output"
    candidates = []
    candidates.extend(fenced_blocks(text))
    candidates.append(text)
    # Try direct parses first.
    for cand in candidates:
        stripped = cand.strip()
        if not stripped:
            continue
        try:
            return json.loads(stripped), None
        except Exception:  # noqa: BLE001
            pass
    # Then scan for balanced objects / arrays inside each candidate.
    for cand in candidates:
        for opener, closer in (("{", "}"), ("[", "]")):
            for blob in _balance_scan(cand, opener, closer):
                try:
                    return json.loads(blob), None
                except Exception:  # noqa: BLE001
                    continue
    return None, "no parseable JSON found"


def _type_name(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def _type_ok(value, expected) -> bool:
    expected = expected if isinstance(expected, list) else [expected]
    for t in expected:
        if t == "object" and isinstance(value, dict):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "string" and isinstance(value, str):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "null" and value is None:
            return True
    return False


def validate_schema(value, schema: dict, path: str = "$") -> list:
    """Validate a small, explicit JSON-Schema subset used by quality/data.

    Supported: type, properties, required, additionalProperties(bool),
    items, minItems, maxItems, uniqueItems, enum, const, minimum, maximum,
    minLength, maxLength, pattern, minProperties, maxProperties, exactly_keys.
    Returns a list of {"path", "msg"} errors; empty list means conformant.
    """
    errors: list = []
    if not isinstance(schema, dict):
        return errors
    if "type" in schema and not _type_ok(value, schema["type"]):
        errors.append({"path": path, "msg": "expected type %r, got %s" % (schema["type"], _type_name(value))})
        return errors
    if "const" in schema and value != schema["const"]:
        errors.append({"path": path, "msg": "expected const %r, got %r" % (schema["const"], value)})
    if "enum" in schema and value not in schema["enum"]:
        errors.append({"path": path, "msg": "value %r not in enum %r" % (value, schema["enum"])})
    if isinstance(value, dict):
        required = schema.get("required") or []
        for key in required:
            if key not in value:
                errors.append({"path": path, "msg": "missing required key %r" % (key,)})
        if "exactly_keys" in schema:
            want = set(schema["exactly_keys"])
            got = set(value.keys())
            if want != got:
                errors.append({"path": path, "msg": "keys %r != expected %r" % (sorted(got), sorted(want))})
        if schema.get("additionalProperties") is False:
            allowed = set((schema.get("properties") or {}).keys()) | set(schema.get("exactly_keys") or [])
            if allowed:
                extra = set(value.keys()) - allowed
                if extra:
                    errors.append({"path": path, "msg": "unexpected keys %r" % (sorted(extra),)})
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            errors.append({"path": path, "msg": "object has %d keys, need >= %d" % (len(value), schema["minProperties"])})
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            errors.append({"path": path, "msg": "object has %d keys, need <= %d" % (len(value), schema["maxProperties"])})
        properties = schema.get("properties") or {}
        for key, subschema in properties.items():
            if key in value:
                errors.extend(validate_schema(value[key], subschema, "%s.%s" % (path, key)))
        if isinstance(schema.get("additionalProperties"), dict):
            for key, subvalue in value.items():
                if key not in properties:
                    errors.extend(validate_schema(subvalue, schema["additionalProperties"], "%s.%s" % (path, key)))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append({"path": path, "msg": "array has %d items, need >= %d" % (len(value), schema["minItems"])})
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append({"path": path, "msg": "array has %d items, need <= %d" % (len(value), schema["maxItems"])})
        if schema.get("uniqueItems"):
            seen = []
            for item in value:
                key = json.dumps(item, sort_keys=True)
                if key in seen:
                    errors.append({"path": path, "msg": "duplicate item %r" % (item,)})
                    break
                seen.append(key)
        if isinstance(schema.get("items"), dict):
            for idx, item in enumerate(value):
                errors.extend(validate_schema(item, schema["items"], "%s[%d]" % (path, idx)))
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append({"path": path, "msg": "string length %d < %d" % (len(value), schema["minLength"])})
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append({"path": path, "msg": "string length %d > %d" % (len(value), schema["maxLength"])})
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append({"path": path, "msg": "string %r does not match %r" % (value, schema["pattern"])})
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append({"path": path, "msg": "value %r < minimum %r" % (value, schema["minimum"])})
        if "maximum" in schema and value > schema["maximum"]:
            errors.append({"path": path, "msg": "value %r > maximum %r" % (value, schema["maximum"])})
    return errors


# --------------------------------------------------------------------------
# math
# --------------------------------------------------------------------------

_NUM_RE = re.compile(r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)")
_ANSWER_RE = re.compile(r"(?i)\b(?:final\s+)?answer\b\s*[:\-=\*]*\s*(.+)")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


def _first_number(text: str) -> float | None:
    if text is None:
        return None
    m = _NUM_RE.search(text)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def extract_final_answer(text: str) -> tuple:
    """Extract (answer_string, source) from a math completion.

    Order: \\boxed{...} > last ANSWER: marker > last number > last non-empty line.
    """
    text = text or ""
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip(), "boxed"
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        candidate = matches[-1].group(1).strip()
        # Cut markdown bold/emphasis and trailing commentary after a second colon.
        candidate = candidate.strip("*_` ")
        return candidate, "answer_marker"
    # Thinking-mode or chatty models may put the answer on the last line.
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1], "last_line"
    return "", "empty"


def _numeric_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= tol * scale


def grade_math(item: dict, output: str) -> dict:
    extracted, source = extract_final_answer(output or "")
    expected = str(item.get("answer", ""))
    accept = [str(x) for x in (item.get("accept") or [])]
    has_marker = bool(_ANSWER_RE.search(output or "")) or bool(_BOXED_RE.search(output or ""))
    got_num = _first_number(extracted)
    ok = False
    mode = "mismatch"
    # Numeric comparison first (handles "0.70" vs "0.7", "5,050" vs "5050").
    for candidate in [expected] + accept:
        exp_num = _first_number(candidate)
        if got_num is not None and exp_num is not None and _numeric_equal(got_num, exp_num):
            ok = True
            mode = "numeric"
            break
    if not ok:
        got_str = canonical(extracted, casefold=True)
        for candidate in [expected] + accept:
            if got_str and got_str == canonical(candidate, casefold=True):
                ok = True
                mode = "string"
                break
    return {
        "id": item.get("id"),
        "score": 1 if ok else 0,
        "extracted": extracted[:200],
        "expected": expected,
        "mode": mode,
        "has_answer_marker": has_marker,
        "source": source,
    }


# --------------------------------------------------------------------------
# json
# --------------------------------------------------------------------------


def grade_json(item: dict, output: str) -> dict:
    obj, err = extract_json_payload(output or "")
    valid = obj is not None
    schema_errors: list = []
    if valid:
        schema_errors = validate_schema(obj, item.get("schema") or {})
    conformant = valid and not schema_errors
    return {
        "id": item.get("id"),
        "score": 1 if conformant else 0,
        "valid_json": bool(valid),
        "schema_errors": schema_errors[:5],
        "parse_error": err,
        "required_keys": (item.get("schema") or {}).get("required") or [],
    }


# --------------------------------------------------------------------------
# code
# --------------------------------------------------------------------------

CODE_TIMEOUT_S = 10

_CODE_HARNESS = r'''
import json, sys
MARK = "__QRESULTS__"
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))
except Exception:
    pass
SRC = __SRC__
TESTS = __TESTS__
ns = {"__name__": "solution"}
status = "ok"
err = None
try:
    exec(compile(SRC, "<solution>", "exec"), ns)
except BaseException as exc:
    status = "error"
    err = type(exc).__name__ + ": " + str(exc)[:400]
results = []
if status == "ok":
    for i, expr in enumerate(TESTS, 1):
        try:
            ok = bool(eval(expr, ns))
            results.append({"test": i, "expr": expr, "ok": ok})
        except BaseException as exc:
            results.append({"test": i, "expr": expr, "ok": False,
                            "error": type(exc).__name__ + ": " + str(exc)[:200]})
sys.stdout.write("\n" + MARK + json.dumps({"status": status, "error": err, "results": results}) + "\n")
'''


def extract_code(text: str, fn_name: str | None = None) -> str:
    """Extract the most relevant Python code block from model output."""
    text = text or ""
    blocks = fenced_blocks(text)
    if blocks:
        if fn_name:
            hits = [b for b in blocks if re.search(r"\bdef\s+" + re.escape(fn_name) + r"\b", b)]
            if hits:
                return max(hits, key=len)
        py_blocks = [b for b in blocks if "def " in b or "class " in b]
        if py_blocks:
            return max(py_blocks, key=len)
        return max(blocks, key=len)
    # No fences: slice from the first import/def/class line to the end.
    lines = text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if re.match(r"^\s*(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+|class\s+\w+)", line):
            start = idx
            break
    if start is not None:
        return "\n".join(lines[start:])
    return ""


def run_code_tests(code: str, tests: list, timeout: int = CODE_TIMEOUT_S) -> dict:
    """Execute model code + hidden assertions in an isolated subprocess."""
    if not code.strip():
        return {"status": "no_code", "error": "no code extracted", "results": [], "passed": 0, "total": len(tests)}
    harness = _CODE_HARNESS.replace("TIMEOUT_S", str(int(timeout)))
    # Replace the test list before inserting model code so that a model-authored
    # literal "__SRC__"/"__TESTS__" cannot corrupt the harness.
    harness = harness.replace("__TESTS__", repr(list(tests)))
    harness = harness.replace("__SRC__", repr(code))
    with tempfile.TemporaryDirectory(prefix="qcode-") as tmpdir:
        path = Path(tmpdir) / "solution_under_test.py"
        path.write_text(harness, encoding="utf-8")
        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": tmpdir,
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", str(path)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=timeout + 5,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "execution timed out", "results": [], "passed": 0, "total": len(tests)}
        stdout = proc.stdout or ""
        stderr = (proc.stderr or "")[-800:]
        if proc.returncode is not None and proc.returncode < 0:
            return {
                "status": "timeout",
                "error": "process killed by signal %d (CPU limit or timeout)" % (-proc.returncode),
                "results": [],
                "passed": 0,
                "total": len(tests),
            }
        marker = "__QRESULTS__"
        pos = stdout.rfind(marker)
        if pos < 0:
            return {
                "status": "crash",
                "error": "no result marker (exit %s): %s" % (proc.returncode, stderr.strip()[-400:]),
                "results": [],
                "passed": 0,
                "total": len(tests),
            }
        line = stdout[pos + len(marker) :].splitlines()[0]
        try:
            payload = json.loads(line)
        except Exception as exc:  # noqa: BLE001
            return {"status": "crash", "error": "bad result JSON: %s" % exc, "results": [], "passed": 0, "total": len(tests)}
        results = payload.get("results") or []
        passed = sum(1 for r in results if r.get("ok"))
        status = payload.get("status") or ("ok" if passed == len(tests) else "partial")
        return {
            "status": status,
            "error": payload.get("error"),
            "results": results,
            "passed": passed,
            "total": len(tests),
            "stdout_tail": stdout[:400] if status != "ok" else "",
        }


def grade_code(item: dict, output: str) -> dict:
    code = extract_code(output or "", item.get("function"))
    run = run_code_tests(code, item.get("tests") or [], timeout=int(item.get("timeout", CODE_TIMEOUT_S)))
    total = max(1, run["total"])
    return {
        "id": item.get("id"),
        "score": 1 if (run["status"] == "ok" and run["passed"] == run["total"] and run["total"] > 0) else 0,
        "tests_passed": run["passed"],
        "tests_total": run["total"],
        "tests_ok_fraction": round(run["passed"] / total, 4),
        "status": run["status"],
        "error": run.get("error"),
        "failed_tests": [r for r in run.get("results") or [] if not r.get("ok")][:5],
        "code_chars": len(code),
    }


# --------------------------------------------------------------------------
# instruction following
# --------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _lines(text: str) -> list:
    return [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def run_checks(text: str, checks: list) -> dict:
    """Apply rule-based checks. Returns {"passed", "checks":[{...}]}."""
    text = text or ""
    stripped = text.strip()
    norm = _WS_RE.sub(" ", stripped)
    lines = _lines(text)
    results = []

    def record(ok: bool, detail: str = "") -> bool:
        results.append({"ok": bool(ok), "detail": detail})
        return bool(ok)

    for check in checks or []:
        ctype = check.get("type")
        value = check.get("value")
        case_sensitive = check.get("case_sensitive", True)
        haystack = stripped if case_sensitive else stripped.casefold()
        needle = str(value) if case_sensitive else str(value).casefold()
        if ctype == "word_count_eq":
            record(len(stripped.split()) == int(value), "words=%d want=%s" % (len(stripped.split()), value))
        elif ctype == "word_count_max":
            record(len(stripped.split()) <= int(value), "words=%d max=%s" % (len(stripped.split()), value))
        elif ctype == "word_count_min":
            record(len(stripped.split()) >= int(value), "words=%d min=%s" % (len(stripped.split()), value))
        elif ctype == "line_count_eq":
            record(len(lines) == int(value), "lines=%d want=%s" % (len(lines), value))
        elif ctype == "line_count_max":
            record(len(lines) <= int(value), "lines=%d max=%s" % (len(lines), value))
        elif ctype == "one_line":
            record(len(lines) == 1, "lines=%d" % len(lines))
        elif ctype == "each_line_startswith":
            ok = bool(lines) and all(ln.startswith(str(value)) for ln in lines)
            record(ok, "prefix=%r" % (value,))
        elif ctype == "line_prefix_count":
            prefix = str(check.get("prefix", value))
            count = sum(1 for ln in lines if ln.startswith(prefix))
            record(count == int(check.get("value", 0)), "count=%d want=%s" % (count, check.get("value")))
        elif ctype == "lines_startswith_count_eq":
            prefix = str(check.get("prefix"))
            count = sum(1 for ln in lines if ln.startswith(prefix))
            record(count == int(check.get("count", 0)), "count=%d want=%s" % (count, check.get("count")))
        elif ctype == "line_endswith":
            idx = int(check.get("line", 1)) - 1
            ok = 0 <= idx < len(lines) and lines[idx].endswith(str(check.get("value", "")))
            record(ok, "line%d=%r" % (idx + 1, lines[idx] if 0 <= idx < len(lines) else None))
        elif ctype == "line_equals":
            idx = int(check.get("line", 1)) - 1
            ok = 0 <= idx < len(lines) and lines[idx] == str(check.get("value", ""))
            record(ok, "line%d=%r" % (idx + 1, lines[idx] if 0 <= idx < len(lines) else None))
        elif ctype == "contains_count_min":
            count = haystack.count(needle)
            record(count >= int(check.get("min", 1)), "count=%d min=%s" % (count, check.get("min")))
        elif ctype == "absent_char":
            record(str(value) not in stripped, "char %r absent" % (value,))
        elif ctype == "absent_substr":
            record(needle not in haystack, "substr %r absent" % (value,))
        elif ctype == "absent_any_char":
            bad = [ch for ch in str(value) if ch in stripped]
            record(not bad, "found %r" % (bad,))
        elif ctype == "no_space":
            record(not any(ch.isspace() for ch in stripped), "no whitespace")
        elif ctype == "regex_search":
            flags = re.I if check.get("ignore_case") else 0
            record(bool(re.search(str(value), stripped, flags)), "pattern=%r" % (value,))
        elif ctype == "regex_fullmatch":
            flags = re.I if check.get("ignore_case") else 0
            record(bool(re.fullmatch(str(value), stripped, flags)), "pattern=%r" % (value,))
        elif ctype == "canonical_equals":
            record(norm == str(value), "got=%r" % (norm[:120],))
        elif ctype == "token_sequence":
            want = [str(x) for x in (value or [])]
            record(norm.split() == want, "got=%r" % (norm[:120],))
        elif ctype == "split_count":
            sep = str(check.get("sep", ","))
            parts = stripped.split(sep) if stripped else []
            record(len(parts) == int(check.get("value", 0)), "parts=%d want=%s" % (len(parts), check.get("value")))
        elif ctype == "all_lines_numeric":
            ok = bool(lines) and all(re.fullmatch(r"\d+", ln) for ln in lines)
            record(ok, "all numeric lines")
        elif ctype == "all_lines_prime":
            ok = bool(lines) and all(re.fullmatch(r"\d+", ln) and _is_prime(int(ln)) for ln in lines)
            record(ok, "all prime lines")
        elif ctype == "json_keys_exact":
            try:
                obj = json.loads(strip_fenced(stripped).strip())
            except Exception:
                obj = None
            want = list(check.get("keys") or [])
            ok = isinstance(obj, dict) and set(obj.keys()) == set(want)
            if ok and check.get("values_int"):
                ok = all(isinstance(v, int) and not isinstance(v, bool) for v in obj.values())
            record(ok, "json keys")
        elif ctype == "lowercase_ascii":
            record(stripped == stripped.lower(), "lowercase")
        elif ctype == "uppercase_ascii":
            record(stripped == stripped.upper(), "uppercase")
        elif ctype == "max_chars":
            record(len(stripped) <= int(value), "chars=%d max=%s" % (len(stripped), value))
        elif ctype == "min_chars":
            record(len(stripped) >= int(value), "chars=%d min=%s" % (len(stripped), value))
        elif ctype == "first_word":
            first = stripped.split()[0] if stripped.split() else ""
            ok = first == str(value) if case_sensitive else first.casefold() == str(value).casefold()
            record(ok, "first_word=%r want=%r" % (first, value))
        else:
            record(False, "UNKNOWN CHECK TYPE %r" % (ctype,))
    return {"passed": bool(results) and all(r["ok"] for r in results), "checks": results}


def grade_instruction(item: dict, output: str) -> dict:
    run = run_checks(output or "", item.get("checks") or [])
    return {
        "id": item.get("id"),
        "score": 1 if run["passed"] else 0,
        "checks_passed": sum(1 for c in run["checks"] if c["ok"]),
        "checks_total": len(run["checks"]),
        "failed_checks": [c for c in run["checks"] if not c["ok"]][:5],
    }


# --------------------------------------------------------------------------
# tool calling
# --------------------------------------------------------------------------


def parse_tool_calls_from_content(text: str) -> list:
    """Best-effort parse of qwen3_xml style tool calls embedded in content."""
    text = text or ""
    calls = []
    for blob in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        try:
            obj = json.loads(blob)
        except Exception:  # noqa: BLE001
            continue
        calls.append(_coerce_tool_obj(obj))
    if calls:
        return [c for c in calls if c]
    # Fallback: any JSON object mentioning name+arguments.
    for blob in _balance_scan(text, "{", "}"):
        try:
            obj = json.loads(blob)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict) and "name" in obj and ("arguments" in obj or "parameters" in obj):
            coerced = _coerce_tool_obj(obj)
            if coerced:
                calls.append(coerced)
    return calls


def _coerce_tool_obj(obj) -> dict | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    fn = obj.get("function")
    if not name and isinstance(fn, dict):
        name = fn.get("name")
    raw_args = obj.get("arguments", obj.get("parameters"))
    if raw_args is None and isinstance(fn, dict):
        raw_args = fn.get("arguments", {})
    if raw_args is None:
        raw_args = {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except Exception:  # noqa: BLE001
            raw_args = {}
    if not isinstance(raw_args, dict):
        raw_args = {}
    if not name:
        return None
    return {"name": str(name), "arguments": raw_args}


def _norm_scalar(value, casefold: bool = True):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip().strip("\"'`.")
        s = _WS_RE.sub(" ", s)
        return s.casefold() if casefold else s
    return value


def _values_equal(expected, actual, arg_spec: dict | None) -> bool:
    if expected is None:
        return actual is not None
    spec = arg_spec or {}
    aliases = spec.get("aliases") or {}
    match = str(spec.get("match") or "exact").lower()
    exp = _norm_scalar(expected)
    act = _norm_scalar(actual)
    if isinstance(exp, float) and isinstance(act, float):
        return _numeric_equal(exp, act, tol=1e-6)
    # Tolerate numbers serialised as strings ("100" vs 100).
    if isinstance(exp, float) and isinstance(act, str):
        try:
            return _numeric_equal(exp, float(act.replace(",", "")), tol=1e-6)
        except ValueError:
            pass
    if isinstance(act, float) and isinstance(exp, str):
        try:
            return _numeric_equal(act, float(exp.replace(",", "")), tol=1e-6)
        except ValueError:
            pass
    candidates = [exp]
    if isinstance(expected, str) and isinstance(aliases, dict):
        for alias in aliases.get(expected, []) or []:
            candidates.append(_norm_scalar(alias))
    for candidate in candidates:
        if isinstance(candidate, str) and match == "contains":
            if isinstance(act, str) and candidate and candidate in act:
                return True
        elif candidate == act:
            return True
    return False


def grade_tool(item: dict, content: str, tool_calls: list) -> dict:
    """Grade a tool-calling item against expected name + argument values."""
    expected = item.get("expect") or {}
    exp_name = str(expected.get("name", ""))
    exp_args = expected.get("args") or {}
    arg_specs = item.get("arg_specs") or {}
    calls = [c for c in (tool_calls or []) if isinstance(c, dict)]
    source = "server_parsed"
    if not calls:
        fallback = parse_tool_calls_from_content(content or "")
        if fallback:
            calls = fallback
            source = "content_xml_fallback"
    if not calls:
        return {
            "id": item.get("id"),
            "score": 0,
            "reason": "no_tool_calls",
            "expected_name": exp_name,
            "source": source,
            "observed_calls": [],
        }
    name_match = None
    for call in calls:
        if _norm_scalar(call.get("name")) == _norm_scalar(exp_name):
            name_match = call
            break
    if name_match is None:
        return {
            "id": item.get("id"),
            "score": 0,
            "reason": "wrong_tool_name",
            "expected_name": exp_name,
            "source": source,
            "observed_calls": [{"name": c.get("name"), "arguments": c.get("arguments")} for c in calls][:3],
        }
    args = name_match.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    mismatches = []
    for key, want in exp_args.items():
        if key not in args:
            mismatches.append({"arg": key, "want": want, "got": None, "why": "missing"})
        elif not _values_equal(want, args[key], arg_specs.get(key)):
            mismatches.append({"arg": key, "want": want, "got": args[key], "why": "value"})
    extra = sorted(set(args.keys()) - set(exp_args.keys()))
    return {
        "id": item.get("id"),
        "score": 1 if not mismatches else 0,
        "reason": "ok" if not mismatches else "arg_mismatch",
        "expected_name": exp_name,
        "expected_args": exp_args,
        "observed_args": args,
        "mismatches": mismatches[:5],
        "extra_args": extra,
        "source": source,
        "n_calls": len(calls),
    }


# --------------------------------------------------------------------------
# repetition / degeneracy
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[^\sa-z0-9_]")


def repetition_metrics(text: str, n: int = 20) -> dict:
    text = text or ""
    tokens = _TOKEN_RE.findall(text.lower())
    n_tokens = len(tokens)
    unique_ratio = (len(set(tokens)) / n_tokens) if n_tokens else 0.0
    ngrams = Counter(tuple(tokens[i : i + n]) for i in range(max(0, n_tokens - n + 1)))
    max_ngram_repeat = max(ngrams.values()) if ngrams else 0
    worst_ngram = ""
    if max_ngram_repeat > 1:
        worst = max(ngrams.items(), key=lambda kv: kv[1])[0]
        worst_ngram = " ".join(worst)[:200]
    # Longest run of identical consecutive non-empty lines.
    longest_line_run = 0
    current = None
    run = 0
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped and stripped == current:
            run += 1
            longest_line_run = max(longest_line_run, run)
        elif stripped:
            current = stripped
            run = 1
        else:
            current = None
            run = 0
    # Distinct-2 (adjacent token bigram diversity).
    bigrams = [tuple(tokens[i : i + 2]) for i in range(max(0, n_tokens - 1))]
    distinct2 = (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    # Thresholds calibrated on real repo prose/code (unique ratio ~0.2-0.3,
    # distinct-2 ~0.55-0.65, max 20-gram repeat 1-2) versus genuine loops
    # (unique ratio < 0.1, distinct-2 < 0.2, same 20-gram repeated many times).
    degenerate = (
        max_ngram_repeat >= 5
        or distinct2 < 0.30
        or unique_ratio < 0.10
        or longest_line_run >= 8
        or n_tokens < 200
    )
    reasons = []
    if max_ngram_repeat >= 5:
        reasons.append("max %d-gram repeated %d times" % (n, max_ngram_repeat))
    if distinct2 < 0.30:
        reasons.append("distinct-2 %.3f < 0.30" % distinct2)
    if unique_ratio < 0.10:
        reasons.append("unique token ratio %.3f < 0.10" % unique_ratio)
    if longest_line_run >= 8:
        reasons.append("identical line repeated %d times" % longest_line_run)
    if n_tokens < 200:
        reasons.append("too short: %d tokens" % n_tokens)
    return {
        "n_tokens": n_tokens,
        "n_chars": len(text),
        "unique_token_ratio": round(unique_ratio, 4),
        "distinct_2": round(distinct2, 4),
        "max_ngram_repeat": max_ngram_repeat,
        "ngram_n": n,
        "worst_ngram": worst_ngram,
        "longest_identical_line_run": longest_line_run,
        "degenerate": bool(degenerate),
        "reasons": reasons,
    }


def grade_repetition(item: dict, output: str, finish_reason: str | None = None) -> dict:
    metrics = repetition_metrics(output or "", n=int(item.get("ngram", 20)))
    if finish_reason:
        metrics["finish_reason"] = finish_reason
    return {
        "id": item.get("id"),
        "score": 0 if metrics["degenerate"] else 1,
        **metrics,
    }


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def grade_item(category: str, item: dict, result) -> dict:
    """Grade one ChatResult for a category. ``result`` is a client.ChatResult."""
    output = getattr(result, "content", "") or ""
    if category == "math":
        return grade_math(item, output)
    if category == "json":
        return grade_json(item, output)
    if category == "code":
        return grade_code(item, output)
    if category == "instruction":
        return grade_instruction(item, output)
    if category == "tool":
        return grade_tool(item, output, getattr(result, "tool_calls", []) or [])
    if category == "repetition":
        return grade_repetition(item, output, getattr(result, "finish_reason", None))
    raise ValueError("unknown category %r" % (category,))
