import json
import re
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
# Context budget for the 8192-token window every preset uses (llm.load.contextLength=8192): the four local
# models tokenize English prose and Python at roughly 3.6 characters per token, so 24_000 characters of
# history is about 6_700 prompt tokens. That leaves room for the agent system prompt plus the tool schemas
# (reserved at 1_400 tokens) and a reply of 256-2_048 tokens, which reply_token_budget() sizes per request;
# trim_messages() drops the oldest whole turns (and finally trims the oldest surviving message) whenever the
# history passes this cap, and an attached image is charged IMAGE_CHAR_COST because its base64 payload is
# billed as real prompt tokens by the vision models.
MAX_CHAT_CHARS = 24_000
IMAGE_CHAR_COST = 4_096
CONTEXT_TOKENS = 8192
AVG_CHARS_PER_TOKEN = 3.6
PROMPT_OVERHEAD_TOKENS = 1_400
MIN_REPLY_TOKENS = 256
TRUNCATION_MARK = "\n[...earlier part of this message trimmed...]"
NO_CONTENT_MESSAGE = (
    "No content was returned. The model may still be loading - please try again in a moment."
)
EMPTY_AFTER_LENGTH = (
    "The model hit its token limit before writing any answer. Try a shorter prompt or start a new chat."
)

DONE_SENTINEL = "[DONE]"
SSE_PREFIX = "data:"
FENCE_PATTERN = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)
TAG_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
NAME_KEYS = ("name", "tool_name", "tool")
ARGS_KEYS = ("arguments", "parameters", "args", "input")
TOOL_FUNCTION_KEYS = ("function", "tool_call", "tool_calls")


class LLMError(RuntimeError):
    pass


def normalize_base_url(url: Optional[str]) -> str:
    base = (url or "").strip().rstrip("/")
    if not base:
        base = DEFAULT_BASE_URL
    lowered = base.lower()
    if not lowered.endswith("/v1") and "/v1/" not in lowered:
        base = base + "/v1"
    return base


def text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
                elif "text" in part:
                    chunks.append(str(part["text"]))
            else:
                chunks.append(str(part))
        return "".join(chunks)
    return str(content)


def message_chars(message: Dict[str, Any]) -> int:
    content = message.get("content")
    total = len(str(message.get("role", ""))) + len(str(message.get("name", "")))
    if isinstance(content, str):
        total += len(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "image_url" or "image_url" in part:
                    total += IMAGE_CHAR_COST
                elif "text" in part:
                    total += len(str(part.get("text", "")))
                else:
                    total += len(json.dumps(part, ensure_ascii=False, default=str))
            else:
                total += len(str(part))
    elif content is not None:
        total += len(str(content))
    for call in message.get("tool_calls") or []:
        total += len(json.dumps(call, ensure_ascii=False, default=str))
    return total


def conversation_chars(messages: Iterable[Dict[str, Any]]) -> int:
    return sum(message_chars(message) for message in messages)


def _shrink_message(message: Dict[str, Any], overflow: int) -> Dict[str, Any]:
    clone = dict(message)
    content = clone.get("content")
    if isinstance(content, str):
        keep = max(0, len(content) - overflow - len(TRUNCATION_MARK))
        clone["content"] = content[:keep] + TRUNCATION_MARK
    elif isinstance(content, list):
        remaining = overflow
        parts: List[Any] = []
        for part in content:
            if remaining > 0 and isinstance(part, dict) and "text" in part and "image_url" not in part:
                text = str(part.get("text", ""))
                keep = max(0, len(text) - remaining - len(TRUNCATION_MARK))
                trimmed = text[:keep] + TRUNCATION_MARK
                remaining -= max(0, len(text) - len(trimmed))
                new_part = dict(part)
                new_part["text"] = trimmed
                parts.append(new_part)
            else:
                parts.append(part)
        clone["content"] = parts
    return clone


def trim_messages(messages: Iterable[Dict[str, Any]], max_chars: int = MAX_CHAT_CHARS) -> List[Dict[str, Any]]:
    trimmed = [dict(message) for message in messages]
    head: List[Dict[str, Any]] = []
    while trimmed and trimmed[0].get("role") == "system":
        head.append(trimmed.pop(0))
    while trimmed and conversation_chars(head + trimmed) > max_chars:
        trimmed.pop(0)
    while len(trimmed) > 1 and trimmed[0].get("role") != "user":
        trimmed.pop(0)
    if trimmed:
        overflow = conversation_chars(head + trimmed) - max_chars
        if overflow > 0:
            trimmed[0] = _shrink_message(trimmed[0], overflow)
    return head + trimmed


def estimate_tokens(chars: int) -> int:
    return max(1, int(round(max(0, chars) / AVG_CHARS_PER_TOKEN)))


def reply_token_budget(
    messages: Iterable[Dict[str, Any]],
    context_tokens: int = CONTEXT_TOKENS,
    ceiling: int = 2048,
    floor: int = MIN_REPLY_TOKENS,
) -> int:
    prompt_tokens = estimate_tokens(conversation_chars(messages)) + PROMPT_OVERHEAD_TOKENS
    available = context_tokens - prompt_tokens
    return int(max(floor, min(ceiling, available)))


def iter_json_objects(text: str) -> Iterable[str]:
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : index + 1]
                    start = -1


def normalize_tool_call(raw: Any, index: int = 0, valid_tool_names: Optional[Iterable[str]] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    call_id = raw.get("id") if isinstance(raw.get("id"), str) else None
    name = None
    arguments: Any = None
    nested = raw.get("function")
    if isinstance(nested, dict):
        name = nested.get("name")
        if "arguments" in nested:
            arguments = nested.get("arguments")
        elif "parameters" in nested:
            arguments = nested.get("parameters")
    else:
        for key in NAME_KEYS:
            if isinstance(raw.get(key), str):
                name = raw[key]
                break
        if name is None and isinstance(nested, str):
            name = nested
        for key in ARGS_KEYS:
            if key in raw:
                arguments = raw[key]
                break
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    if valid_tool_names is not None and name not in set(valid_tool_names):
        return None
    if isinstance(arguments, str):
        arguments_text = arguments.strip() or "{}"
    elif arguments is None:
        arguments_text = "{}"
    else:
        arguments_text = json.dumps(arguments, ensure_ascii=False, default=str)
    return {
        "id": call_id or "call_text_%d" % index,
        "type": "function",
        "function": {"name": name, "arguments": arguments_text},
    }


def extract_tool_calls(text: str, valid_tool_names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []
    candidates: List[Any] = []
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            candidates.append(json.loads(stripped))
        except json.JSONDecodeError:
            candidates = []
    for block in FENCE_PATTERN.findall(text):
        block = block.strip()
        if block.startswith("{") or block.startswith("["):
            try:
                candidates.append(json.loads(block))
            except json.JSONDecodeError:
                continue
    for block in TAG_PATTERN.findall(text):
        block = block.strip()
        try:
            candidates.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    for block in iter_json_objects(text):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        candidates.append(parsed)
    results: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        items = candidate if isinstance(candidate, list) else [candidate]
        for item in items:
            if not isinstance(item, dict):
                continue
            raw = item
            for key in TOOL_FUNCTION_KEYS:
                nested = item.get(key)
                if isinstance(nested, dict):
                    raw = nested
                    if "id" not in raw and isinstance(item.get("id"), str):
                        raw = dict(raw)
                        raw["id"] = item["id"]
                    break
                if isinstance(nested, list) and nested:
                    raw = nested[0]
                    break
            call = normalize_tool_call(raw, len(results), valid_tool_names)
            if call is None:
                continue
            fingerprint = (call["function"]["name"], call["function"]["arguments"])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            results.append(call)
    return results


def read_error_body(response: requests.Response, limit: int = 300) -> str:
    try:
        raw = response.content
    except Exception:
        return ""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)
    text = " ".join(text.split())
    return text[:limit]


def friendly_status_message(status: int, detail: str = "") -> str:
    if status in (401, 403):
        message = "LM Studio refused the request (HTTP %d). Restart the local server and try again." % status
    elif status == 404:
        message = "The local server could not find that model (HTTP 404). Pick a model that is downloaded in LM Studio."
    elif status == 400:
        message = (
            "The local server rejected the request (HTTP 400). The conversation may be longer than the "
            "model's context window - try starting a new chat."
        )
    elif status >= 500:
        message = (
            "LM Studio reported a server error (HTTP %d). The model may still be loading or may not fit "
            "in RAM - wait a few seconds and retry." % status
        )
    else:
        message = "The local server returned HTTP %d." % status
    detail = (detail or "").strip()
    if detail:
        message = message + " Details: " + detail
    return message


def friendly_connection_message(exc: Exception) -> str:
    detail = "; ".join(str(part) for part in exc.args) if exc.args else ""
    message = "Could not reach the local model server. Make sure LM Studio is running and the local server is started."
    if detail:
        message = message + " (%s)" % detail
    return message


class LLMClient:
    def __init__(
        self,
        model: str,
        url: str = DEFAULT_BASE_URL,
        thinking: bool = False,
        api_key: str = "lm-studio",
        timeout: tuple = (10, 900),
        max_chat_chars: int = MAX_CHAT_CHARS,
    ) -> None:
        self.model = (model or "").strip()
        self.base_url = normalize_base_url(url)
        self.thinking = bool(thinking)
        self.timeout = timeout
        self.max_chat_chars = int(max_chat_chars)
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = "Bearer %s" % api_key
        self.session.headers["Content-Type"] = "application/json"
        self.session.headers["Accept"] = "text/event-stream"
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._response: Optional[requests.Response] = None

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()
        with self._lock:
            response = self._response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    def reset(self) -> None:
        self._cancel.clear()

    def close(self) -> None:
        self.cancel()
        try:
            self.session.close()
        except Exception:
            pass

    def uses_reasoning_model(self) -> bool:
        return "qwen3" in self.model.lower()

    def disable_thinking(self) -> bool:
        return self.uses_reasoning_model() and not self.thinking

    def prepare_messages(self, messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared = trim_messages(messages, self.max_chat_chars)
        if self.disable_thinking():
            first = prepared[0] if prepared else {}
            already = (
                first.get("role") == "system"
                and isinstance(first.get("content"), str)
                and first["content"].strip().startswith("/no_think")
            )
            if not already:
                prepared = [{"role": "system", "content": "/no_think"}] + prepared
        return prepared

    def build_payload(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self.prepare_messages(messages),
            "stream": True,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.disable_thinking():
            payload["reasoning_effort"] = "none"
        return payload

    def _iter_sse(self, response: requests.Response) -> Iterable[Dict[str, Any]]:
        for raw in response.iter_lines(decode_unicode=False):
            if self._cancel.is_set():
                return
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith(SSE_PREFIX):
                if line.startswith("{"):
                    try:
                        plain = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(plain, dict) and "error" in plain:
                        yield plain
                        return
                continue
            data = line[len(SSE_PREFIX) :].strip()
            if not data:
                continue
            if data == DONE_SENTINEL:
                yield {"_done": True}
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict):
                yield chunk

    def stream_chat(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        on_content: Optional[Callable[[str], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        valid_tool_names: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        self.reset()
        payload = self.build_payload(messages, tools=tools, max_tokens=max_tokens, temperature=temperature)
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        pending_calls: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        cancelled = False
        response: Optional[requests.Response] = None
        try:
            response = self.session.post(
                "%s/chat/completions" % self.base_url,
                data=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
                stream=True,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(friendly_connection_message(exc)) from exc
        with self._lock:
            self._response = response
        try:
            if response.status_code >= 400:
                raise LLMError(friendly_status_message(response.status_code, read_error_body(response)))
            for chunk in self._iter_sse(response):
                if self._cancel.is_set():
                    cancelled = True
                    break
                if chunk.get("_done"):
                    break
                if isinstance(chunk.get("error"), (dict, str)):
                    error = chunk["error"]
                    detail = error.get("message", "") if isinstance(error, dict) else str(error)
                    raise LLMError(friendly_status_message(500, detail))
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    content_parts.append(piece)
                    if on_content is not None:
                        on_content(piece)
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(thought, str) and thought:
                    reasoning_parts.append(thought)
                    if on_reasoning is not None:
                        on_reasoning(thought)
                for call_delta in delta.get("tool_calls") or []:
                    if not isinstance(call_delta, dict):
                        continue
                    index = call_delta.get("index")
                    if not isinstance(index, int):
                        index = len(pending_calls)
                    slot = pending_calls.setdefault(
                        index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if isinstance(call_delta.get("id"), str):
                        slot["id"] = call_delta["id"]
                    if isinstance(call_delta.get("type"), str) and call_delta["type"]:
                        slot["type"] = call_delta["type"]
                    fn = call_delta.get("function") or {}
                    if isinstance(fn.get("name"), str) and fn["name"]:
                        current = slot["function"]["name"]
                        slot["function"]["name"] = fn["name"] if not current else current + fn["name"]
                    if isinstance(fn.get("arguments"), str):
                        slot["function"]["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            if self._cancel.is_set():
                cancelled = True
        except requests.RequestException as exc:
            if self._cancel.is_set():
                cancelled = True
            else:
                raise LLMError(friendly_connection_message(exc)) from exc
        except LLMError:
            raise
        except Exception as exc:
            if self._cancel.is_set():
                cancelled = True
            else:
                raise LLMError("The stream from LM Studio ended unexpectedly (%s)." % exc) from exc
        finally:
            with self._lock:
                self._response = None
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts)
        allowed = set(valid_tool_names) if valid_tool_names else None
        tool_calls: List[Dict[str, Any]] = []
        rejected: List[str] = []
        for index in sorted(pending_calls):
            slot = pending_calls[index]
            name = (slot["function"]["name"] or "").strip()
            if not name:
                continue
            if allowed is not None and name not in allowed:
                rejected.append(name)
                continue
            tool_calls.append(
                {
                    "id": slot["id"] or "call_%d" % index,
                    "type": slot["type"] or "function",
                    "function": {
                        "name": name,
                        "arguments": slot["function"]["arguments"].strip() or "{}",
                    },
                }
            )
        if not tool_calls and not cancelled:
            tool_calls = extract_tool_calls(content, allowed)
            if allowed is not None and tool_calls:
                rejected.extend(
                    name for name in [call["function"]["name"] for call in tool_calls] if name not in allowed
                )
        if cancelled:
            finish_reason = finish_reason or "cancelled"
        if not cancelled and not content.strip() and not tool_calls and not reasoning.strip():
            if finish_reason == "length":
                raise LLMError(EMPTY_AFTER_LENGTH)
            raise LLMError(NO_CONTENT_MESSAGE)
        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason or "stop",
            "cancelled": cancelled,
            "model": self.model,
            "rejected_tool_calls": rejected,
        }

    def complete(
        self,
        messages: Iterable[Dict[str, Any]],
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": self.prepare_messages(messages),
            "stream": False,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        try:
            response = self.session.post(
                "%s/chat/completions" % self.base_url,
                data=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(friendly_connection_message(exc)) from exc
        if response.status_code >= 400:
            raise LLMError(friendly_status_message(response.status_code, read_error_body(response)))
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError("The local server returned a response that was not JSON.") from exc
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(NO_CONTENT_MESSAGE)
        return text_of((choices[0].get("message") or {}).get("content"))
