"""Minimal, dependency-free client for an OpenAI-compatible vLLM server.

Only the Python standard library is used (urllib + json) so the quality suite
runs on a fresh box with no network and no pip installs.  The client is
deliberately tolerant: a server may be booting, restarting, or temporarily
erroring out, and the suite must survive that without losing partial results.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional


class ClientError(RuntimeError):
    """Unrecoverable client/server error (bad request, auth, ...)."""


class ServerUnavailable(ClientError):
    """The server did not become healthy within the allowed wait."""


@dataclass
class ChatResult:
    """Normalised result of one chat completion."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)  # [{"name": str, "arguments": dict}]
    finish_reason: Optional[str] = None
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    ttft_s: Optional[float] = None
    status: str = "ok"
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)

    def to_dict(self, include_raw: bool = False) -> dict:
        d = {
            "status": self.status,
            "error": self.error,
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "latency_s": round(self.latency_s, 4),
            "ttft_s": None if self.ttft_s is None else round(self.ttft_s, 4),
        }
        if include_raw:
            d["raw"] = self.raw
        return d


def _message_text(message: dict) -> tuple[str, str]:
    """Return (content, reasoning) from a response message, robust to variants."""
    content = message.get("content") or ""
    if isinstance(content, list):  # some servers return content parts
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        content = "".join(parts)
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    return str(content), reasoning


def _normalise_tool_calls(message: dict) -> list:
    """Normalise OpenAI tool_calls to [{"name", "arguments": dict, "raw_arguments"}]."""
    out = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        if not isinstance(fn, dict):
            fn = {}
        name = fn.get("name") or call.get("name") or ""
        raw_args = fn.get("arguments")
        if raw_args is None:
            raw_args = call.get("arguments")
        args, parse_error = _parse_arguments(raw_args)
        out.append(
            {
                "name": str(name),
                "arguments": args,
                "raw_arguments": raw_args,
                "parse_error": parse_error,
            }
        )
    return out


def _parse_arguments(raw: Any) -> tuple[dict, Optional[str]]:
    if isinstance(raw, dict):
        return raw, None
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            return {}, "%s: %s" % (type(exc).__name__, exc)
        if isinstance(parsed, dict):
            return parsed, None
        return {"__value__": parsed}, "arguments is not a JSON object"
    return {}, "unsupported arguments type: %s" % type(raw).__name__


class ChatClient:
    """Thin OpenAI-compatible chat client with health polling and retries."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: Optional[str] = None,
        api_key: str = "EMPTY",
        timeout: float = 600.0,
        retries: int = 2,
        retry_backoff: float = 3.0,
        enable_thinking: Optional[bool] = False,
        chat_template_kwargs: Optional[dict] = None,
        extra_body: Optional[dict] = None,
    ) -> None:
        self.base_url = self._normalise_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = max(0, retries)
        self.retry_backoff = retry_backoff
        self.enable_thinking = enable_thinking
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.extra_body = dict(extra_body or {})
        self.models_info: dict = {}
        # Local endpoints must not go through an HTTP(S) proxy.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @staticmethod
    def _normalise_base_url(base_url: str) -> str:
        """Accept both 'http://host:8000/v1' and 'http://host:8000'."""
        url = (base_url or "http://127.0.0.1:8000/v1").rstrip("/")
        parts = urllib.parse.urlsplit(url)
        if not parts.path or parts.path == "/":
            return url + "/v1"
        return url

    def _origin(self) -> str:
        parts = urllib.parse.urlsplit(self.base_url)
        if parts.scheme and parts.netloc:
            return "%s://%s" % (parts.scheme, parts.netloc)
        return self.base_url

    # ------------------------------------------------------------------ http
    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def _open(self, request: urllib.request.Request, timeout: float):
        return self._opener.open(request, timeout=timeout)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> dict:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self._url(path), data=data, headers=headers, method=method)
        last_exc: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                with self._open(req, timeout or self.timeout) as resp:
                    body = resp.read()
                try:
                    return json.loads(body.decode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    raise ClientError("non-JSON response from %s: %r" % (path, body[:300])) from exc
            except urllib.error.HTTPError as exc:
                body = b""
                try:
                    body = exc.read()
                except Exception:  # noqa: BLE001
                    pass
                text = body.decode("utf-8", "replace")[:2000]
                if exc.code >= 500 and attempt < self.retries:
                    last_exc = exc
                    time.sleep(self.retry_backoff * (attempt + 1))
                    continue
                raise ClientError("HTTP %s on %s: %s" % (exc.code, path, text)) from exc
            except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    continue
        raise ServerUnavailable("cannot reach %s: %s" % (self._url(path), last_exc))

    # ---------------------------------------------------------------- health
    def is_healthy(self, timeout: float = 3.0) -> bool:
        for path in ("/health", "/models"):
            try:
                if path == "/health":
                    url = self._origin() + "/health"
                else:
                    url = self._url(path)
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with self._opener.open(req, timeout=timeout) as resp:
                    resp.read()
                return True
            except Exception:  # noqa: BLE001 - any failure means try the next probe
                continue
        return False

    def wait_ready(self, max_wait: float = 0.0, poll_interval: float = 3.0) -> bool:
        """Poll /health until ready. max_wait=0 means single probe (no waiting)."""
        deadline = time.time() + max(0.0, max_wait)
        while True:
            if self.is_healthy():
                return True
            if time.time() >= deadline:
                return False
            time.sleep(poll_interval)

    def list_models(self) -> list:
        payload = self._request_json("GET", "/models", timeout=30)
        data = payload.get("data") or []
        if not isinstance(data, list):
            return []
        return data

    def resolve_model(self, preferred: Optional[str] = None) -> str:
        """Pick the served model id, preferring an exact match then prefix match."""
        try:
            models = self.list_models()
        except ClientError:
            models = []
        ids = [str(m.get("id")) for m in models if isinstance(m, dict) and m.get("id")]
        self.models_info = {str(m.get("id")): m for m in models if isinstance(m, dict)}
        wanted = preferred or self.model
        if wanted and wanted in ids:
            return wanted
        if wanted:
            for mid in ids:
                if mid.split("/")[-1] == wanted.split("/")[-1]:
                    return mid
        if ids:
            return ids[0]
        if wanted:
            return wanted
        raise ClientError("no model specified and /v1/models returned no models")

    def model_max_len(self) -> Optional[int]:
        for info in self.models_info.values():
            for key in ("max_model_len", "max_context_length", "context_length"):
                val = info.get(key)
                if isinstance(val, int) and val > 0:
                    return val
        return None

    # ------------------------------------------------------------------ chat
    def _build_payload(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        tools: Optional[list],
        tool_choice: Optional[str],
        seed: Optional[int],
        stop: Optional[list],
        extra: Optional[dict],
        enable_thinking: Optional[bool],
    ) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": bool(stream),
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if stop:
            payload["stop"] = list(stop)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if stream:
            payload["stream_options"] = {"include_usage": True}
        # vLLM chat-template controls: enable_thinking is the Qwen3 switch.
        kwargs = dict(self.chat_template_kwargs)
        think = self.enable_thinking if enable_thinking is None else enable_thinking
        if think is not None:
            kwargs.setdefault("enable_thinking", bool(think))
        if kwargs:
            payload["chat_template_kwargs"] = kwargs
        body = dict(self.extra_body)
        body.update(extra or {})
        if body:
            payload.update(body)
        return payload

    def chat(
        self,
        messages: list,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        seed: Optional[int] = 12345,
        stop: Optional[list] = None,
        extra: Optional[dict] = None,
        enable_thinking: Optional[bool] = None,
    ) -> ChatResult:
        payload = self._build_payload(
            messages, max_tokens, temperature, top_p, False, tools, tool_choice, seed, stop, extra, enable_thinking
        )
        started = time.perf_counter()
        try:
            data = self._request_json("POST", "/chat/completions", payload)
        except ClientError as exc:
            if isinstance(exc, ServerUnavailable):
                raise
            return ChatResult(status="error", error=str(exc), latency_s=time.perf_counter() - started)
        latency = time.perf_counter() - started
        choices = data.get("choices") or []
        if not choices:
            return ChatResult(status="error", error="no choices in response: %r" % (data,), latency_s=latency, raw=data)
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content, reasoning = _message_text(message)
        return ChatResult(
            content=content,
            reasoning=reasoning,
            tool_calls=_normalise_tool_calls(message),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage") or {},
            latency_s=latency,
            ttft_s=latency,  # no streaming: total latency is the best available TTFT proxy
            raw=data,
        )

    def chat_stream(
        self,
        messages: list,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = 12345,
        extra: Optional[dict] = None,
        enable_thinking: Optional[bool] = None,
    ) -> ChatResult:
        """Streaming chat completion; records TTFT (first content/reasoning delta)."""
        payload = self._build_payload(
            messages, max_tokens, temperature, top_p, True, None, None, seed, None, extra, enable_thinking
        )
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self._url("/chat/completions"), data=data, headers=headers, method="POST")
        last_exc: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            ttft: Optional[float] = None
            content_parts: list = []
            reasoning_parts: list = []
            usage: dict = {}
            finish_reason = None
            tool_calls_raw: list = []
            try:
                with self._open(req, self.timeout) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", "replace").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        chunk_str = line[5:].strip()
                        if chunk_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(chunk_str)
                        except Exception:  # noqa: BLE001 - skip malformed keep-alive lines
                            continue
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        for choice in chunk.get("choices") or []:
                            delta = choice.get("delta") or {}
                            piece = delta.get("content")
                            rpiece = delta.get("reasoning_content") or delta.get("reasoning")
                            if piece:
                                content_parts.append(piece)
                                if ttft is None:
                                    ttft = time.perf_counter() - started
                            if rpiece:
                                reasoning_parts.append(rpiece)
                                if ttft is None:
                                    ttft = time.perf_counter() - started
                            for tc in delta.get("tool_calls") or []:
                                tool_calls_raw.append(tc)
                                if ttft is None:
                                    ttft = time.perf_counter() - started
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
                latency = time.perf_counter() - started
                return ChatResult(
                    content="".join(content_parts),
                    reasoning="".join(reasoning_parts),
                    tool_calls=[],
                    finish_reason=finish_reason,
                    usage=usage or {},
                    latency_s=latency,
                    ttft_s=ttft if ttft is not None else latency,
                    raw={"streamed": True, "n_chunks": len(content_parts) + len(reasoning_parts)},
                )
            except urllib.error.HTTPError as exc:
                body = b""
                try:
                    body = exc.read()
                except Exception:  # noqa: BLE001
                    pass
                text = body.decode("utf-8", "replace")[:2000]
                # 4xx (except 429) is a request-level problem we cannot fix by
                # retrying -- surface it so one battery item fails, not the run.
                if exc.code < 500 and exc.code != 429:
                    raise ClientError("HTTP %s on streaming chat: %s" % (exc.code, text)) from exc
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    continue
            except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    continue
        raise ServerUnavailable("streaming request failed: %s" % (last_exc,))
