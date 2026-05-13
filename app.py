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
#   off              不处理 reasoning_content，只做普通转发
PATCH_MODE = os.getenv("PATCH_MODE", "replay").lower()

# replay 模式找不到缓存时的兜底策略：
#   none  不兜底，保持原样
#   fake  填 FAKE_REASONING_CONTENT
#   error 直接返回 422，便于定位是哪条历史消息缺缓存
REPLAY_MISS_FALLBACK = os.getenv("REPLAY_MISS_FALLBACK", "none").lower()
FAKE_REASONING_CONTENT = os.getenv("FAKE_REASONING_CONTENT", "done")

# 修复 OpenAI tool_calls 调用链：
#   prune      默认，删除不完整的 assistant tool_calls 消息，以及它后面残留的 tool 消息
#   synthesize 给缺失的 tool_call_id 自动补一个 synthetic tool 消息
#   error      发现不完整调用链就返回 422，便于调试
#   off        不修复工具调用链
TOOL_CHAIN_FIX_MODE = os.getenv("TOOL_CHAIN_FIX_MODE", "prune").lower()
SYNTHETIC_TOOL_CONTENT = os.getenv(
    "SYNTHETIC_TOOL_CONTENT",
    '{"ok":true,"note":"synthetic tool result inserted by proxy because original tool response was missing"}',
)
DROP_ORPHAN_TOOL_MESSAGES = os.getenv("DROP_ORPHAN_TOOL_MESSAGES", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

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
                "CREATE INDEX IF NOT EXISTS idx_reasoning_cache_created_at "
                "ON reasoning_cache(created_at)"
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
                cur = conn.execute(
                    "DELETE FROM reasoning_cache WHERE created_at < ?",
                    (expire_before,),
                )
                return cur.rowcount

    def stats(self) -> dict[str, Any]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM reasoning_cache"
                ).fetchone()
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
    return [
        normalize_tool_call(tc, include_id=include_id)
        for tc in tool_calls
        if isinstance(tc, dict)
    ]


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
                keys.append(
                    f"function_sha256:{sha256_text(stable_json({'name': name, 'arguments': arguments}))}"
                )

    if normalized_with_id:
        keys.append(f"tool_calls_sha256:{sha256_text(stable_json(normalized_with_id))}")

    if normalized_without_id:
        keys.append(
            f"tool_calls_no_id_sha256:{sha256_text(stable_json(normalized_without_id))}"
        )

    message_signature = {
        "role": message.get("role"),
        "content": message.get("content", ""),
        "tool_calls": normalized_with_id,
    }
    keys.append(f"message_sha256:{sha256_text(stable_json(message_signature))}")

    message_signature_no_id = {
        "role": message.get("role"),
        "content": message.get("content", ""),
        "tool_calls": normalized_without_id,
    }
    keys.append(f"message_no_id_sha256:{sha256_text(stable_json(message_signature_no_id))}")

    return list(dict.fromkeys(keys))


def is_assistant_tool_call_message(message: Any) -> bool:
    return (
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("tool_calls"), list)
        and len(message.get("tool_calls")) > 0
    )


def is_tool_message(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "tool"


def ensure_tool_call_ids(message: dict[str, Any]) -> tuple[list[str], int]:
    """
    OpenAI 工具调用链要求每个 tool_call 有 id，后续 tool 消息用 tool_call_id 对应。
    有些客户端中转时可能丢 id，这里补一个稳定 id，避免上游直接拒绝。
    """
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return [], 0

    ids: list[str] = []
    changed = 0
    used: set[str] = set()

    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue

        tc_id = tool_call.get("id")
        if not isinstance(tc_id, str) or not tc_id:
            signature = {
                "index": index,
                "type": tool_call.get("type", "function"),
                "function": tool_call.get("function", {}),
            }
            tc_id = "call_px_" + sha256_text(stable_json(signature))[:24]
            tool_call["id"] = tc_id
            changed += 1

        # 防止同一条 assistant 消息里出现重复 id。
        if tc_id in used:
            tc_id = f"{tc_id}_{index}"
            tool_call["id"] = tc_id
            changed += 1

        used.add(tc_id)
        ids.append(tc_id)

    return ids, changed


def repair_tool_call_chain(messages: list[Any]) -> tuple[list[Any], int, list[str]]:
    """
    修复 DeepSeek/OpenAI 严格校验的 tool_calls 消息链。

    合法格式必须是：
      assistant(tool_calls=[id1,id2])
      tool(tool_call_id=id1)
      tool(tool_call_id=id2)

    如果 assistant 后面没有紧跟对应 tool 消息，就会出现：
      An assistant message with 'tool_calls' must be followed by tool messages...
    """
    if TOOL_CHAIN_FIX_MODE == "off":
        return messages, 0, []

    fixed_count = 0
    errors: list[str] = []
    repaired: list[Any] = []
    i = 0

    while i < len(messages):
        message = messages[i]

        if is_assistant_tool_call_message(message):
            tool_call_ids, id_fixed = ensure_tool_call_ids(message)
            fixed_count += id_fixed

            j = i + 1
            following_tools: list[dict[str, Any]] = []
            while j < len(messages) and is_tool_message(messages[j]):
                following_tools.append(messages[j])
                j += 1

            seen: set[str] = set()
            valid_tools: list[dict[str, Any]] = []
            required_ids = set(tool_call_ids)

            for tool_message in following_tools:
                tool_call_id = tool_message.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id in required_ids and tool_call_id not in seen:
                    valid_tools.append(tool_message)
                    seen.add(tool_call_id)
                else:
                    if DROP_ORPHAN_TOOL_MESSAGES:
                        fixed_count += 1
                    else:
                        valid_tools.append(tool_message)

            missing_ids = [tc_id for tc_id in tool_call_ids if tc_id not in seen]

            if missing_ids:
                message_text = (
                    f"messages[{i}] assistant tool_calls missing tool responses: "
                    f"{', '.join(missing_ids)}"
                )

                if TOOL_CHAIN_FIX_MODE == "error":
                    errors.append(message_text)
                    repaired.append(message)
                    repaired.extend(following_tools)
                elif TOOL_CHAIN_FIX_MODE == "synthesize":
                    repaired.append(message)
                    repaired.extend(valid_tools)
                    for missing_id in missing_ids:
                        repaired.append(
                            {
                                "role": "tool",
                                "tool_call_id": missing_id,
                                "content": SYNTHETIC_TOOL_CONTENT,
                            }
                        )
                    fixed_count += len(missing_ids)
                    logger.warning("%s; inserted synthetic tool messages", message_text)
                else:
                    # 默认 prune：删掉这条不完整的 assistant tool_calls，以及后面残留的 tool。
                    # 这样比伪造真实工具结果更安全，模型会基于剩余上下文重新决策。
                    fixed_count += 1 + len(following_tools)
                    logger.warning("%s; pruned invalid assistant/tool chain", message_text)

                i = j
                continue

            repaired.append(message)
            repaired.extend(valid_tools)
            i = j
            continue

        if is_tool_message(message):
            # 没有紧跟在 assistant tool_calls 后面的 tool 消息属于孤儿消息，很多上游也会拒绝。
            if DROP_ORPHAN_TOOL_MESSAGES:
                fixed_count += 1
                logger.warning("pruned orphan tool message at messages[%s]", i)
            else:
                repaired.append(message)
            i += 1
            continue

        repaired.append(message)
        i += 1

    return repaired, fixed_count, errors


def store_assistant_message(message: dict[str, Any], source: str) -> int:
    if not is_assistant_tool_call_message(message):
        return 0

    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning:
        return 0

    keys = build_reasoning_keys_from_message(message)
    count = store.put_many(keys, reasoning, source=source)
    logger.info(
        "stored reasoning_content source=%s keys=%s reasoning_chars=%s",
        source,
        count,
        len(reasoning),
    )
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


def repair_reasoning_content(payload: dict[str, Any]) -> tuple[int, list[int]]:
    patched_count = 0
    missing_indexes: list[int] = []

    if PATCH_MODE == "off":
        return patched_count, missing_indexes

    if PATCH_MODE == "disable_thinking":
        payload["thinking"] = {"type": "disabled"}
        return patched_count, missing_indexes

    messages = payload.get("messages")
    if not isinstance(messages, list):
        return patched_count, missing_indexes

    if PATCH_MODE == "fake":
        for index, message in enumerate(messages):
            if not is_assistant_tool_call_message(message):
                continue
            current = message.get("reasoning_content")
            if current is None or current == "":
                message["reasoning_content"] = FAKE_REASONING_CONTENT
                patched_count += 1
        return patched_count, missing_indexes

    if PATCH_MODE != "replay":
        logger.warning("Unknown PATCH_MODE=%s, skip reasoning patching.", PATCH_MODE)
        return patched_count, missing_indexes

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
            logger.info(
                "restored reasoning_content for messages[%s] by key=%s chars=%s",
                index,
                matched_key,
                len(restored),
            )
        else:
            missing_indexes.append(index)
            logger.warning("missing reasoning_content cache for messages[%s]", index)

            if REPLAY_MISS_FALLBACK == "fake":
                message["reasoning_content"] = FAKE_REASONING_CONTENT
                patched_count += 1
            elif REPLAY_MISS_FALLBACK == "error":
                # 由调用方返回 422
                pass

    return patched_count, missing_indexes


def repair_request_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    stats: dict[str, Any] = {
        "reasoning_patched_count": 0,
        "reasoning_missing_indexes": [],
        "tool_chain_fixed_count": 0,
        "tool_chain_errors": [],
    }

    messages = payload.get("messages")
    if isinstance(messages, list):
        repaired_messages, fixed_count, errors = repair_tool_call_chain(messages)
        payload["messages"] = repaired_messages
        stats["tool_chain_fixed_count"] = fixed_count
        stats["tool_chain_errors"] = errors

    patched_count, missing_indexes = repair_reasoning_content(payload)
    stats["reasoning_patched_count"] = patched_count
    stats["reasoning_missing_indexes"] = missing_indexes

    return payload, stats


@app.get("/__health")
async def health():
    return {
        "ok": True,
        "default_base_url": DEFAULT_BASE_URL,
        "patch_mode": PATCH_MODE,
        "replay_miss_fallback": REPLAY_MISS_FALLBACK,
        "tool_chain_fix_mode": TOOL_CHAIN_FIX_MODE,
        "drop_orphan_tool_messages": DROP_ORPHAN_TOOL_MESSAGES,
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
        path = path[len("/v1") :]

    upstream_url = base_url + path

    if request.url.query:
        upstream_url += "?" + request.url.query

    return upstream_url


async def prepare_body(
    request: Request,
) -> tuple[bytes, dict[str, Any], dict[str, Any] | None, JSONResponse | None]:
    raw_body = await request.body()
    stats: dict[str, Any] = {
        "reasoning_patched_count": 0,
        "reasoning_missing_indexes": [],
        "tool_chain_fixed_count": 0,
        "tool_chain_errors": [],
    }
    payload: dict[str, Any] | None = None

    if not raw_body or not is_json_request(request):
        return raw_body, stats, payload, None

    if not request.url.path.endswith("/chat/completions"):
        return raw_body, stats, payload, None

    try:
        decoded = raw_body.decode("utf-8")
        loaded = json.loads(decoded)
    except Exception:
        return raw_body, stats, payload, None

    if not isinstance(loaded, dict):
        return raw_body, stats, payload, None

    payload = loaded
    payload, stats = repair_request_payload(payload)

    if TOOL_CHAIN_FIX_MODE == "error" and stats["tool_chain_errors"]:
        return raw_body, stats, payload, JSONResponse(
            status_code=422,
            content={
                "error": "invalid_tool_call_chain",
                "message": "assistant tool_calls must be immediately followed by tool messages for every tool_call_id.",
                "details": stats["tool_chain_errors"],
            },
        )

    if (
        PATCH_MODE == "replay"
        and REPLAY_MISS_FALLBACK == "error"
        and stats["reasoning_missing_indexes"]
    ):
        return raw_body, stats, payload, JSONResponse(
            status_code=422,
            content={
                "error": "missing_reasoning_content_cache",
                "message": "assistant tool-call messages are missing reasoning_content, and no cached reasoning_content matched.",
                "missing_message_indexes": stats["reasoning_missing_indexes"],
            },
        )

    new_body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return new_body, stats, payload, None


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
        return data[:pos], data[pos + len(delim) :]

    def _handle_event(self, event: bytes) -> None:
        data_lines: list[bytes] = []
        for line in event.splitlines():
            stripped = line.strip()
            if stripped.startswith(b"data:"):
                data_lines.append(stripped[len(b"data:") :].strip())

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
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )

                    tc_id = tc.get("id")
                    if isinstance(tc_id, str) and tc_id:
                        current["id"] = tc_id

                    tc_type = tc.get("type")
                    if isinstance(tc_type, str) and tc_type:
                        current["type"] = tc_type

                    function = tc.get("function")
                    if isinstance(function, dict):
                        current_function = current.setdefault(
                            "function", {"name": "", "arguments": ""}
                        )
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
            ensure_tool_call_ids(message)
            stored += store_assistant_message(message, source="stream")
        return stored


async def stream_and_store_response(
    response: httpx.Response,
    client: httpx.AsyncClient,
) -> AsyncIterator[bytes]:
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
    body, repair_stats, request_payload, early_error = await prepare_body(request)

    if early_error is not None:
        return early_error

    is_chat_completions = request.url.path.endswith("/chat/completions")
    wants_stream = request_wants_stream(request_payload)

    logger.info(
        "%s %s -> %s | patch_mode=%s tool_chain_mode=%s reasoning_patched=%s "
        "reasoning_missing=%s tool_fixed=%s stream=%s",
        request.method,
        request.url.path,
        upstream_url,
        PATCH_MODE,
        TOOL_CHAIN_FIX_MODE,
        repair_stats.get("reasoning_patched_count"),
        repair_stats.get("reasoning_missing_indexes"),
        repair_stats.get("tool_chain_fixed_count"),
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
