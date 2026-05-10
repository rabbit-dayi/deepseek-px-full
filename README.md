# deepseek-px-full

DeepSeek API 转发代理，解决 DeepSeek `reasoning_content` 在 tool-call 多轮对话中丢失的问题，并提供 OpenAI/Claude 兼容性修复。

## 功能

- **reasoning_content 缓存回填**：自动缓存 DeepSeek 返回的 `reasoning_content`，下一轮 tool-call 请求缺字段时自动回填
- **X-PX-BASE-URL 动态上游**：通过 Header 动态指定上游 API 地址，支持任意 OpenAI 兼容端点
- **兼容性修复**：
  - `developer` role 自动转为 `system`
  - OpenAI/Claude content blocks `[{"type":"text","text":"..."}]` 转普通字符串
  - 清理 DeepSeek 不支持的扩展字段（store、metadata、audio 等）
- **流式响应支持**：SSE 流式响应中累积并缓存 reasoning_content
- **SQLite 持久化缓存**：基于 tool_call_id + 函数签名多键匹配

## 快速开始

### Docker Compose（推荐）

```bash
git clone git@github.com:rabbit-dayi/deepseek-px-full.git
cd deepseek-px-full
docker compose up -d --build
```

### Docker 运行

```bash
docker run -d \
  --name deepseek-px \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  -e DEFAULT_BASE_URL=https://api.deepseek.com \
  ghcr.io/rabbit-dayi/deepseek-px-full:latest
```

### 手动运行

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 客户端配置

**Base URL**: `http://127.0.0.1:8000/v1`

API Key 使用 DeepSeek 本身的 key。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEFAULT_BASE_URL` | `https://api.deepseek.com` | 默认上游地址 |
| `PATCH_MODE` | `replay` | 模式：replay / fake / disable_thinking / off |
| `REPLAY_MISS_FALLBACK` | `none` | replay 模式缓存缺失兜底：none / fake / error |
| `FAKE_REASONING_CONTENT` | `done` | fake 模式下的占位 reasoning 内容 |
| `NORMALIZE_DEVELOPER_ROLE` | `true` | developer role 转 system |
| `NORMALIZE_CONTENT_BLOCKS` | `true` | content blocks 转字符串 |
| `STRIP_UNSUPPORTED_PARAMS` | `true` | 清理不支持的参数字段 |
| `FORCE_MODEL` | (空) | 强制覆盖请求中的 model 字段 |
| `CACHE_DB_PATH` | `/data/reasoning_cache.sqlite3` | SQLite 缓存路径 |
| `CACHE_TTL_SECONDS` | `86400` | 缓存过期时间（秒） |
| `ALLOWED_BASE_HOSTS` | `api.deepseek.com` | 允许转发的上游 host 白名单 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## 模式说明

| 模式 | 行为 |
|------|------|
| `replay` | 推荐。缓存真实 reasoning_content，缺失时回填 |
| `fake` | 给 assistant tool call 填占位 reasoning_content |
| `disable_thinking` | 请求中设置 `thinking.type=disabled` |
| `off` | 不做 reasoning 处理，只做转发和兼容性清理 |

## API

### 健康检查

```bash
curl http://127.0.0.1:8000/__health
```

### 清理过期缓存

```bash
curl -X POST http://127.0.0.1:8000/__cache/cleanup
```

## License

MIT
