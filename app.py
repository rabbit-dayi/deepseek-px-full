import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

app = FastAPI(title="DeepSeek PX", version="0.3.0")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("deepseek-px")

DEFAULT_BASE_URL = os.getenv("DEFAULT_BASE_URL", "https://api.deepseek.com").rstrip("/")

# PATCH_MODE:
#   replay           正常调用链：缓存 DeepSeek 真实 reasoning_content，下一轮缺失时回填
#   fake             给 assistant + tool_calls 自动补一个占位 reasoning_content
#   disable_thinking 直接在请求体设置 thinking.type=disabled
#   off              不处理 reasoning_content，只做普通转发和兼容性清理
PATCH_MODE = os.getenv("PATCH_MODE", "replay").lower()

# replay 模式找不到缓存时的兜底策略：
#   none  不兜底，保持原样
#   fake  填 FAKE_REASONING_CONTENT
#   error 直接返回 422，便于定位是哪条历史消息缺缓存
REPLAY_MISS_FALLBACK = os.getenv("REPLAY_MISS_FALLBACK", "none").lower()
FAKE_REASONING_CONTENT = os.getenv("FAKE_REASONING_CONTENT", "done")

# 是否把 DeepSeek 不支持的 developer role 转成 system。
NORMALIZE_DEVELOPER_ROLE = os.getenv("NORMALIZE_DEVELOPER_ROLE", "true").lower() in {"1", "true", "yes", "on"}

# 是否把 OpenAI/Claude 风格 content blocks 转成普通字符串。
# 例如 [{"type":"text","text":"hello"}] -> "hello"
NORMALIZE_CONTENT_BLOCKS = os.getenv("NORMALIZE_CONTENT_BLOCKS", "true").lower() in {"1", "true", "yes", "on"}

# 是否清理 DeepSeek 不支持或容易出错的 OpenAI 扩展字段。
STRIP_UNSUPPORTED_PARAMS = os.getenv("STRIP_UNSUPPORTED_PARAMS", "true").lower() in {"1", "true", "yes", "on"}

# SQLite 缓存位置。docker-compose 默认挂载到 /data。
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", "/data/reasoning_cache.sqlite3")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))

# 安全限制：只允许转发到这些 host。为空表示不限制，但不推荐。
# 示例：api.deepseek.com,api.openai.com
ALLOWED_BASE_HOSTS = os.getenv("ALLOWED_BASE_HOSTS", "api.deepseek.com").strip()

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

# 为了能解析上游 JSON/SSE，尽量要求上游不要压缩。
DROP_REQUEST_HEADERS = {"x-px-base-url", "accept-encoding"}

# DeepSeek /chat/completions 不需要或可能不接受的 OpenAI 扩展字段。
UNSUPPORTED_TOP_LEVEL_FIELDS = {
    "store",
    "metadata",
    "parallel_tool_calls",
    "service_tier",
    "prediction",
    "modalities",
    "audio",
    "web_search_options",
    "response_format",  # 如果你需要 JSON mode，可把它从这里删掉
}

# 某些客户端会把非 DeepSeek 的模型名传进来。
# 默认不强行改；如果设置 FORCE_MODEL，就覆盖 payload["model"]。
FORCE_MODEL = os.getenv("FORCE_MODEL", "").strip()


class ReasoningStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reasoning_cache (
                    cache_key TEXT PRIMARY KEY,
                    reasoning_content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reasoning_cache_created_at ON reasoning_cache(created_at)"
            )

    def put_many(self, keys: list[str], reasoning_content: str, source: str) -> int:
        if not keys or not reasoning_content:
            return 0

        now = int(time.time())
        rows = [(key, reasoning_content, source, now) for key in sorted(set(keys))]

        with self._lock:
            with self._connect() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO reasoning_cache
                    (cache_key, reasoning_content, source, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )
        return len(rows)

    def get_first(self, keys: list[str]) -> tuple[str | None, str | None]:
        if not keys:
            return None, None

        expire_before = int(time.time()) - CACHE_TTL_SECONDS
        unique_keys = list(dict.fromkeys(keys))

        with self._lock:
            with self._connect() as conn:
                placeholders = ",".join("?" for _ in unique_keys)
                row = conn.execute(
                    f"""
                    SELECT cache_key, reasoning_content
                    FROM reasoning_cache
                    WHERE cache_key IN ({placeholders}) AND created_at >= ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    [*unique_keys, expire_before],
                ).fetchone()

        if not row:
            return None, None
        return row[0], row[1]

    def cleanup(self) -> int:
        expire_before = int(time.time()) - CACHE_TTL_SECONDS
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM reasoning_cache WHERE created_at < ?", (expire_before,))
                return cur.rowcount

    def stats(self) -> dict[str, Any]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM reasoning_cache").fetchone()
        return {
            "count": row[0] if row else 0,
            "oldest_created_at": row[1] if row else None,
            "newest_created_at": row[2] if row else None,
            "ttl_seconds": CACHE_TTL_SECONDS,
            "db_path": self.db_path,
        }


store = ReasoningStore(CACHE_DB_PATH)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_tool_call(tool_call: dict[str, Any], include_id: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {}

    if include_id and tool_call.get("id"):
        result["id"] = tool_call.get("id")

    if tool_call.get("type"):
        result["type"] = tool_call.get("type")

    function = tool_call.get("function")
    if isinstance(function, dict):
        result["function"] = {
            "name": function.get("name", ""),
            "arguments": function.get("arguments", ""),
        }

    return result


def normalize_tool_calls(tool_calls: Any, include_id: bool = True) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    return [normalize_tool_call(tc, include_id=include_id) for tc in tool_calls if isinstance(tc, dict)]


def build_reasoning_keys_from_message(message: dict[str, Any]) -> list[str]:
    """
    给同一条 assistant tool-call 消息构造多种匹配键。
    优先靠 tool_call_id；如果客户端保留的 id 变化了，再用函数名/参数签名兜底。
    """
    keys: list[str] = []
    tool_calls = message.get("tool_calls")
    normalized_with_id = normalize_tool_calls(tool_calls, include_id=True)
    normalized_without_id = normalize_tool_calls(tool_calls, include_id=False)

    for tc in normalized_with_id:
        tc_id = tc.get("id")
        if isinstance(tc_id, str) and tc_id:
            keys.append(f"tool_call_id:{tc_id}")

        keys.append(f"tool_call_sha256:{sha256_text(stable_json(tc))}")

    for tc in normalized_without_id:
        keys.append(f"tool_call_no_id_sha256:{sha256_text(stable_json(tc))}")

        function = tc.get("function")
        if isinstance(function, dict):
            name = function.get("name", "")
            arguments = function.get("arguments", "")
            if name or arguments:
                keys.append(f"function_sha256:{sha256_text(stable_json({'name': name, 'arguments': arguments}))}")

    if normalized_with_id:
        keys.append(f"tool_calls_sha256:{sha256_text(stable_json(normalized_with_id))}")

    if normalized_without_id:
        keys.append(f"tool_calls_no_id_sha256:{sha256_text(stable_json(normalized_without_id))}")

    message_signature = {
        "role": message.get("role"),
        "content": normalize_content_for_signature(message.get("content", "")),
        "tool_calls": normalized_with_id,
    }
    keys.append(f"message_sha256:{sha256_text(stable_json(message_signature))}")

    message_signature_no_id = {
        "role": message.get("role"),
        "content": normalize_content_for_signature(message.get("content", "")),
        "tool_calls": normalized_without_id,
    }
    keys.append(f"message_no_id_sha256:{sha256_text(stable_json(message_signature_no_id))}")

    return list(dict.fromkeys(keys))


def normalize_content_for_signature(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return stable_json(content)


def is_assistant_tool_call_message(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("tool_calls"), list)
        and len(message.get("tool_calls")) > 0
    )


def store_assistant_message(message: dict[str, Any], source: str) -> int:
    if not is_assistant_tool_call_message(message):
        return 0

    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning:
        return 0

    keys = build_reasoning_keys_from_message(message)
    count = store.put_many(keys, reasoning, source=source)
    logger.info("stored reasoning_content source=%s keys=%s reasoning_chars=%s", source, count, len(reasoning))
    return count


def store_from_response_payload(payload: dict[str, Any], source: str) -> int:
    stored = 0

    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                stored += store_assistant_message(message, source=source)

    return stored


def content_blocks_to_string(content: Any) -> Any:
    """
    兼容 Claude/OpenAI content block：
      [{"type":"text","text":"hello"}] -> "hello"

    非纯文本块会尽量保留可读内容；图片、复杂结构转成 JSON 字符串。
    """
    if not isinstance(content, list):
        return content

    parts: list[str] = []

    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue

        if not isinstance(item, dict):
            parts.append(stable_json(item))
            continue

        item_type = item.get("type")

        if item_type == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif isinstance(item.get("content"), str):
            parts.append(item["content"])
        else:
            parts.append(stable_json(item))

    return "\n".join(part for part in parts if part is not None)


def normalize_deepseek_messages(payload: dict[str, Any]) -> tuple[int, int]:
    """
    修 DeepSeek 兼容性：
    1. developer role -> system role
    2. content block array -> string
    """
    role_changed = 0
    content_changed = 0

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return role_changed, content_changed

    for message in messages:
        if not isinstance(message, dict):
            continue

        if NORMALIZE_DEVELOPER_ROLE and message.get("role") == "developer":
            message["role"] = "system"
            role_changed += 1

        if NORMALIZE_CONTENT_BLOCKS and "content" in message:
            old_content = message.get("content")
            new_content = content_blocks_to_string(old_content)
            if new_content is not old_content and new_content != old_content:
                message["content"] = new_content
                content_changed += 1

    return role_changed, content_changed


def strip_unsupported_payload_fields(payload: dict[str, Any]) -> list[str]:
    removed: list[str] = []

    if not STRIP_UNSUPPORTED_PARAMS:
        return removed

    for key in list(UNSUPPORTED_TOP_LEVEL_FIELDS):
        if key in payload:
            payload.pop(key, None)
            removed.append(key)

    return sorted(removed)


def normalize_tools(payload: dict[str, Any]) -> int:
    """
    DeepSeek 支持 OpenAI 风格 tools。
    这里只做轻量清理，避免某些客户端塞入多余字段导致 400。
    """
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return 0

    changed = 0

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue

        function = tool.get("function")
        if not isinstance(function, dict):
            continue

        # 有些客户端可能带入非标准 display/metadata 字段，DeepSeek 不一定接受。
        allowed_function_keys = {"name", "description", "parameters", "strict"}
        for key in list(function.keys()):
            if key not in allowed_function_keys:
                function.pop(key, None)
                changed += 1

    return changed


def normalize_payload_for_deepseek(payload: dict[str, Any]) -> dict[str, Any]:
    if FORCE_MODEL:
        payload["model"] = FORCE_MODEL

    role_changed, content_changed = normalize_deepseek_messages(payload)
    removed_fields = strip_unsupported_payload_fields(payload)
    tool_changed = normalize_tools(payload)

    if role_changed or content_changed or removed_fields or tool_changed:
        logger.info(
            "normalized payload: developer_to_system=%s content_blocks=%s removed_fields=%s tool_fields=%s",
            role_changed,
            content_changed,
            removed_fields,
            tool_changed,
        )

    return payload


def repair_request_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int, list[int]]:
    patched_count = 0
    missing_indexes: list[int] = []

    # 先做 DeepSeek 兼容性清理，再做 reasoning replay。
    payload = normalize_payload_for_deepseek(payload)

    if PATCH_MODE == "off":
        return payload, patched_count, missing_indexes

    if PATCH_MODE == "disable_thinking":
        payload["thinking"] = {"type": "disabled"}
        return payload, patched_count, missing_indexes

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return payload, patched_count, missing_indexes

    if PATCH_MODE == "fake":
        for index, message in enumerate(messages):
            if not is_assistant_tool_call_message(message):
                continue
            current = message.get("reasoning_content")
            if current is None or current == "":
                message["reasoning_content"] = FAKE_REASONING_CONTENT
                patched_count += 1
        return payload, patched_count, missing_indexes

    if PATCH_MODE != "replay":
        logger.warning("Unknown PATCH_MODE=%s, skip patching.", PATCH_MODE)
        return payload, patched_count, missing_indexes

    for index, message in enumerate(messages):
        if not is_assistant_tool_call_message(message):
            continue

        current = message.get("reasoning_content")
        if isinstance(current, str) and current:
            continue

        keys = build_reasoning_keys_from_message(message)
        matched_key, restored = store.get_first(keys)

        if restored:
            message["reasoning_content"] = restored
            patched_count += 1
            logger.info("restored reasoning_content for messages[%s] by key=%s chars=%s", index, matched_key, len(restored))
        else:
            missing_indexes.append(index)
            logger.warning("missing reasoning_content cache for messages[%s]", index)

            if REPLAY_MISS_FALLBACK == "fake":
                message["reasoning_content"] = FAKE_REASONING_CONTENT
                patched_count += 1
            elif REPLAY_MISS_FALLBACK == "error":
                # 由调用方返回 422
                pass

    return payload, patched_count, missing_indexes


@app.get("/__health")
async def health():
    return {
        "ok": True,
        "version": "0.3.0",
        "default_base_url": DEFAULT_BASE_URL,
        "patch_mode": PATCH_MODE,
        "replay_miss_fallback": REPLAY_MISS_FALLBACK,
        "normalize_developer_role": NORMALIZE_DEVELOPER_ROLE,
        "normalize_content_blocks": NORMALIZE_CONTENT_BLOCKS,
        "strip_unsupported_params": STRIP_UNSUPPORTED_PARAMS,
        "force_model": FORCE_MODEL or None,
        "allowed_base_hosts": ALLOWED_BASE_HOSTS,
        "cache": store.stats(),
    }


@app.post("/__cache/cleanup")
async def cleanup_cache():
    deleted = store.cleanup()
    return {"deleted": deleted, "cache": store.stats()}


def is_json_request(request: Request) -> bool:
    content_type = request.headers.get("content-type", "")
    return "application/json" in content_type.lower()


def check_base_url_allowed(base_url: str) -> bool:
    if not ALLOWED_BASE_HOSTS:
        return True

    allowed_hosts = {item.strip() for item in ALLOWED_BASE_HOSTS.split(",") if item.strip()}
    host = urlparse(base_url).hostname

    return bool(host and host in allowed_hosts)


def build_upstream_url(request: Request, base_url: str) -> str:
    base_url = base_url.rstrip("/")
    path = request.url.path

    # 避免 base_url=https://api.deepseek.com/v1 且 path=/v1/chat/completions
    # 最终拼成 https://api.deepseek.com/v1/v1/chat/completions
    if base_url.endswith("/v1") and path.startswith("/v1/"):
        path = path[len("/v1"):]

    upstream_url = base_url + path

    if request.url.query:
        upstream_url += "?" + request.url.query

    return upstream_url


async def prepare_body(request: Request) -> tuple[bytes, int, list[int], dict[str, Any] | None, JSONResponse | None]:
    raw_body = await request.body()
    patched_count = 0
    missing_indexes: list[int] = []
    payload: dict[str, Any] | None = None

    if not raw_body or not is_json_request(request):
        return raw_body, patched_count, missing_indexes, payload, None

    if not request.url.path.endswith("/chat/completions"):
        return raw_body, patched_count, missing_indexes, payload, None

    try:
        decoded = raw_body.decode("utf-8")
        loaded = json.loads(decoded)
    except Exception:
        return raw_body, patched_count, missing_indexes, payload, None

    if not isinstance(loaded, dict):
        return raw_body, patched_count, missing_indexes, payload, None

    payload = loaded
    payload, patched_count, missing_indexes = repair_request_payload(payload)

    if PATCH_MODE == "replay" and REPLAY_MISS_FALLBACK == "error" and missing_indexes:
        return raw_body, patched_count, missing_indexes, payload, JSONResponse(
            status_code=422,
            content={
                "error": "missing_reasoning_content_cache",
                "message": "assistant tool-call messages are missing reasoning_content, and no cached reasoning_content matched.",
                "missing_message_indexes": missing_indexes,
            },
        )

    new_body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return new_body, patched_count, missing_indexes, payload, None


def build_request_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}

    for key, value in request.headers.items():
        lower_key = key.lower()

        if lower_key in HOP_BY_HOP_HEADERS:
            continue

        if lower_key in DROP_REQUEST_HEADERS:
            continue

        headers[key] = value

    headers["accept-encoding"] = "identity"
    return headers


def filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    filtered: dict[str, str] = {}

    for key, value in headers.items():
        lower_key = key.lower()

        if lower_key in HOP_BY_HOP_HEADERS:
            continue

        if lower_key == "content-length":
            continue

        # 如果上游仍然压缩，代理解析后返回可能不是原始压缩体，避免错误声明。
        if lower_key == "content-encoding":
            continue

        filtered[key] = value

    return filtered


async def close_response_and_client(response: httpx.Response, client: httpx.AsyncClient):
    await response.aclose()
    await client.aclose()


@dataclass
class StreamChoiceState:
    reasoning_content: str = ""
    content: str = ""
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)


class SSEReasoningAccumulator:
    def __init__(self) -> None:
        self.buffer = b""
        self.choices: dict[int, StreamChoiceState] = {}

    def feed(self, chunk: bytes) -> None:
        self.buffer += chunk

        while True:
            event, rest = self._pop_event(self.buffer)
            if event is None:
                break
            self.buffer = rest
            self._handle_event(event)

    def _pop_event(self, data: bytes) -> tuple[bytes | None, bytes]:
        delimiters = []
        for delim in (b"\n\n", b"\r\n\r\n"):
            pos = data.find(delim)
            if pos != -1:
                delimiters.append((pos, delim))

        if not delimiters:
            return None, data

        pos, delim = min(delimiters, key=lambda item: item[0])
        return data[:pos], data[pos + len(delim):]

    def _handle_event(self, event: bytes) -> None:
        data_lines: list[bytes] = []
        for line in event.splitlines():
            stripped = line.strip()
            if stripped.startswith(b"data:"):
                data_lines.append(stripped[len(b"data:"):].strip())

        if not data_lines:
            return

        data = b"\n".join(data_lines).strip()
        if not data or data == b"[DONE]":
            return

        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception:
            return

        choices = payload.get("choices")
        if not isinstance(choices, list):
            return

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", 0)
            if not isinstance(index, int):
                index = 0

            state = self.choices.setdefault(index, StreamChoiceState())
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue

            reasoning_piece = delta.get("reasoning_content")
            if isinstance(reasoning_piece, str):
                state.reasoning_content += reasoning_piece

            content_piece = delta.get("content")
            if isinstance(content_piece, str):
                state.content += content_piece

            delta_tool_calls = delta.get("tool_calls")
            if isinstance(delta_tool_calls, list):
                for tc in delta_tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_index = tc.get("index", len(state.tool_calls))
                    if not isinstance(tc_index, int):
                        tc_index = len(state.tool_calls)

                    current = state.tool_calls.setdefault(
                        tc_index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )

                    tc_id = tc.get("id")
                    if isinstance(tc_id, str) and tc_id:
                        current["id"] = tc_id

                    tc_type = tc.get("type")
                    if isinstance(tc_type, str) and tc_type:
                        current["type"] = tc_type

                    function = tc.get("function")
                    if isinstance(function, dict):
                        current_function = current.setdefault("function", {"name": "", "arguments": ""})
                        name_piece = function.get("name")
                        if isinstance(name_piece, str):
                            current_function["name"] += name_piece
                        arguments_piece = function.get("arguments")
                        if isinstance(arguments_piece, str):
                            current_function["arguments"] += arguments_piece

    def store_all(self) -> int:
        stored = 0
        for state in self.choices.values():
            if not state.reasoning_content or not state.tool_calls:
                continue

            tool_calls = [state.tool_calls[index] for index in sorted(state.tool_calls)]
            message = {
                "role": "assistant",
                "content": state.content or "",
                "reasoning_content": state.reasoning_content,
                "tool_calls": tool_calls,
            }
            stored += store_assistant_message(message, source="stream")
        return stored


async def stream_and_store_response(response: httpx.Response, client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    accumulator = SSEReasoningAccumulator()

    try:
        async for chunk in response.aiter_raw():
            if chunk:
                accumulator.feed(chunk)
                yield chunk
    finally:
        try:
            stored = accumulator.store_all()
            if stored:
                logger.info("stored reasoning_content from stream keys=%s", stored)
        finally:
            await response.aclose()
            await client.aclose()


def response_is_json(headers: httpx.Headers) -> bool:
    content_type = headers.get("content-type", "")
    return "application/json" in content_type.lower()


def request_wants_stream(payload: dict[str, Any] | None) -> bool:
    return bool(isinstance(payload, dict) and payload.get("stream") is True)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(request: Request, path: str):
    base_url = request.headers.get("X-PX-BASE-URL", DEFAULT_BASE_URL).rstrip("/")

    if not check_base_url_allowed(base_url):
        return JSONResponse(
            status_code=403,
            content={
                "error": "base_url_not_allowed",
                "message": f"Base URL is not allowed: {base_url}",
                "allowed_base_hosts": ALLOWED_BASE_HOSTS,
            },
        )

    upstream_url = build_upstream_url(request, base_url)
    headers = build_request_headers(request)
    body, patched_count, missing_indexes, request_payload, early_error = await prepare_body(request)

    if early_error is not None:
        return early_error

    is_chat_completions = request.url.path.endswith("/chat/completions")
    wants_stream = request_wants_stream(request_payload)

    logger.info(
        "%s %s -> %s | patch_mode=%s patched=%s missing=%s stream=%s",
        request.method,
        request.url.path,
        upstream_url,
        PATCH_MODE,
        patched_count,
        missing_indexes,
        wants_stream,
    )

    client = httpx.AsyncClient(timeout=None)
    upstream_request = client.build_request(
        method=request.method,
        url=upstream_url,
        headers=headers,
        content=body,
    )

    try:
        upstream_response = await client.send(upstream_request, stream=True)
    except Exception as exc:
        await client.aclose()
        logger.exception("Upstream request failed.")
        return JSONResponse(
            status_code=502,
            content={
                "error": "upstream_request_failed",
                "message": str(exc),
            },
        )

    response_headers = filter_response_headers(upstream_response.headers)

    if is_chat_completions and wants_stream:
        return StreamingResponse(
            stream_and_store_response(upstream_response, client),
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    if is_chat_completions and response_is_json(upstream_response.headers):
        try:
            raw_response_body = await upstream_response.aread()
            try:
                response_payload = json.loads(raw_response_body.decode("utf-8"))
                if isinstance(response_payload, dict):
                    stored = store_from_response_payload(response_payload, source="json")
                    if stored:
                        logger.info("stored reasoning_content from json keys=%s", stored)
            except Exception:
                pass

            return Response(
                content=raw_response_body,
                status_code=upstream_response.status_code,
                headers=response_headers,
                media_type=upstream_response.headers.get("content-type"),
            )
        finally:
            await upstream_response.aclose()
            await client.aclose()

    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=response_headers,
        background=BackgroundTask(close_response_and_client, upstream_response, client),
    )
