# rag_law · 法条知识库问答系统

面向给定法律文本（15 部中华人民共和国法律）的问答系统：根据用户描述检索相关法条，说明法律如何规定，并给出可核验的法条引用；证据不足时明确说明，不编造结论。

> 本系统提供法律信息检索与参考，不构成法律意见，不替代执业律师。

## 当前进度

**V1.0 已完成。** 阶段 5 的不可回答题行为准确率未达到原定目标，已作为后续优化代办保留，不阻塞 V1.0 收尾。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段 0 | 范围与基线（项目骨架、法律清单、评测集、环境验证） | 完成 |
| 阶段 1 | 文档解析与法条切分 | 完成 |
| 阶段 2 | BM25、Dense、RRF 混合检索与量化基线 | 完成（存在已记录限制） |
| 阶段 3 | Reranker 精排与可追溯上下文构建 | 完成（存在已记录限制） |
| 阶段 4 | 结构化答案生成、引用校验、拒答与脱敏日志 | 完成（存在已记录限制） |
| 阶段 5 | 独立 holdout 端到端评测与质量门槛 | 完成（1 项质量指标转后续代办） |
| 阶段 6 | API、界面与部署 | 完成 |

详见 [docs/plan/v1/](docs/plan/v1/)。

## 环境要求

- Python 3.11+（`.python-version` 已锁定 3.13）
- [uv](https://docs.astral.sh/uv/) 包管理工具
- Node.js 20.19+ 与 npm（仅开发或构建 Vue 前端时需要；Docker 镜像内由多阶段构建自动完成）
- 一个模型服务（本项目当前使用 SiliconFlow：`Qwen/Qwen3-8B`、`BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3`）
- PostgreSQL 11+（可选，仅启用页面登录时需要；未配置时 CLI 与 `/v1/health` 照常可用）

## 快速开始

```bash
# 1. 安装依赖（按 uv.lock 复现环境，自动创建 .venv）
uv sync

# 2. 配置环境变量
cp .env.example .env       # Windows PowerShell: Copy-Item .env.example .env
# 然后编辑 .env，填入模型服务的 BASE_URL / API_KEY / 模型名
# 需要页面登录时，再填 RAG_LAW_DATABASE_URL（PostgreSQL 连接串）

# 3. 初始化登录账号表（只用 CLI 可跳过，可重复执行）
uv run python scripts/init_auth_db.py

# 4. 验证三条链路：embedding / Chroma / LLM
uv run python scripts/verify_env.py

# 5. 校验法律清单
uv run python scripts/check_manifest.py

# 6. 执行完整问答（检索、精排、生成和引用校验）
uv run python -m rag_law.cli answer --query "公司一直没跟我签书面劳动合同，我可以要求赔偿吗？"
```

可添加 `--debug` 查看分阶段耗时和校验诊断，或添加 `--json` 输出完整结构化结果。脱敏运行日志默认写入 `logs/answer_runs.jsonl`，该目录不进入 Git。

### 运行阶段 5 系统评测

```bash
# 默认运行全部 development + holdout；正式验收以 holdout 为准
uv run python -m rag_law.cli evaluate-system

# 仅重跑独立 holdout，并在质量门槛失败时返回非零退出码
uv run python -m rag_law.cli evaluate-system --split holdout --fail-on-gates
```

命令同时生成 `data/eval/results/answer_quality_v1.json` 和
`data/eval/results/answer_quality_v1.md`。评测集共 100 题：原 30 题作为
development，新增 70 题作为独立 holdout。

### 启动 Web 界面（Vue + FastAPI）

界面位于 `web/`，技术栈为 Vue 3 + TypeScript + Vite + Element Plus。FastAPI 直接托管构建产物，
生产环境只有一个服务。RAG 编排由 LangGraph StateGraph 驱动：检索侧 BM25 与 Dense 并行召回后融合精排，
生成侧通过条件边完成证据闸门、结构化生成、引用校验与一次安全修复。

```bash
# 生产模式：构建静态页面，再由 FastAPI 同时提供页面与接口
cd web
npm install          # 首次执行
npm run build
cd ..
uv run uvicorn rag_law.api.main:app --host 127.0.0.1 --port 8000
# 打开 http://127.0.0.1:8000/
```

开发模式下前后端分别启动，Vite 会把 `/v1` 代理到本机 8000：

```bash
# 终端 1
uv run uvicorn rag_law.api.main:app --host 127.0.0.1 --port 8000

# 终端 2
cd web
npm run dev
# 打开 http://localhost:5173/
```

Vite 8 默认只监听 `localhost`（IPv6 的 `::1`），所以开发地址要用 `localhost`；
直接访问 `127.0.0.1:5173` 会被拒绝。如需固定 IPv4，可在 `web/vite.config.ts` 的
`server` 中设置 `host: "127.0.0.1"`。

Windows 下可用两条命令分别新开窗口启动前后端，结束后按端口停掉：

```powershell
# 启动（在项目根目录执行）
Start-Process powershell -ArgumentList '-NoExit', "-Command", "cd '$PWD'; uv run uvicorn rag_law.api.main:app --host 127.0.0.1 --port 8000"
Start-Process powershell -ArgumentList '-NoExit', "-Command", "cd '$PWD\web'; npm run dev"

# 停止
(Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force }
(Get-NetTCPConnection -LocalPort 5173 -State Listen).OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force }
```

页面支持连续提问和多轮上下文、按法律筛选、展示回答状态、适用条件、边界限制和已经过校验的法条引用。
最近几轮保留原文，更早对话自动压缩为摘要（增量合并后写回 `conversations.summary`，不重算已有部分）；
保留轮数可通过 `RAG_LAW_CONVERSATION_RECENT_ROUNDS` 配置。
历史对话保存在 PostgreSQL 里，刷新页面或重启服务后仍在：侧栏可列出、切换、删除，
页面靠 localStorage 里的会话 id 接回上一次的对话。会话内容不能作为法律证据。
侧栏可以查看法律清单、上传新法律并重建索引，也可以对已有语料做全量重建；
重建过程通过 SSE 实时显示进度，完成后 API 会自动丢弃旧流水线并加载新索引，无需重启进程。

### 登录与账号

页面与 HTTP 接口采用服务端会话 + Session Cookie：登录成功后下发 `rag_law_sid`（`httponly`、
`samesite=lax`），Cookie 里只有随机串，用户信息保存在服务进程内存中，登出或进程重启即失效。

账号库建表（读取 `.env` 的 `RAG_LAW_DATABASE_URL`，可重复执行）：

```bash
uv run python scripts/init_auth_db.py
```

历史对话（`conversations` / `conversation_messages`）与账号同库：服务第一次读写时会自动建表；
若数据库账号没有建表权限，先执行 `scripts/init_conversations_db.sql`。
对话按 `users.id` 归属，删账号时会话级联清理。

`conversations` 上的 `summary` / `summary_covered_seq` 两列是早期对话摘要的缓存：
超出最近若干轮的旧消息会被压缩成摘要，`summary_covered_seq` 记下摘要已覆盖到最后一条消息的序号，
下一轮只对新增的旧消息做增量合并，不必每轮都重算一遍历史。
旧库首次启动时由上述建表语句自动补列，也可手动执行 `scripts/init_conversations_db.sql` 中的 `ALTER TABLE`。

系统没有注册接口，账号由人工写入数据库（密码目前为明文，仅限内网或演示环境）：

```sql
INSERT INTO users (username, password, role)
VALUES ('your_name', 'your_password', 'user');
```

| 配置项 | 说明 |
|---|---|
| `RAG_LAW_DATABASE_URL` | 账号库 DSN，如 `postgresql://user:pass@host:5432/rag_law`；留空则登录接口返回 `503` |
| `RAG_LAW_SESSION_COOKIE_NAME` | 会话 Cookie 名称，默认 `rag_law_sid` |
| `RAG_LAW_SESSION_TTL_SECONDS` | 登录保持时长，默认 28800 秒 |
| `RAG_LAW_SESSION_COOKIE_SECURE` | HTTPS 部署时必须设为 `true`，否则浏览器不回传 Cookie |

状态码约定：未登录或会话过期 `401`；账号不存在与密码错误同为 `401` 且提示相同（避免账号枚举）；
账号被禁用 `403`；账号库未配置或连不上 `503`。

由于会话保存在进程内，服务必须以单进程运行（当前本地命令与 Compose 均为单 worker）；需要多实例
部署时，先把 `SessionStore` 换成 Redis 等共享存储，再提高 worker 数量。

上线前必须改进：`users.password` 改为哈希存储（argon2id / bcrypt），补注册与改密流程，
并收窄账号库的来源 IP 范围。

### 运行测试

```bash
# 后端测试
uv run pytest -q

# 前端类型检查与单元测试
cd web
npm run build      # 包含 vue-tsc 类型检查
npm test           # vitest run
```

### 启动 HTTP API

```bash
# 本机启动；生产环境应放在反向代理之后并限制访问来源
uv run uvicorn rag_law.api.main:app --host 127.0.0.1 --port 8000

# 查看 API schema 与交互式文档
# http://127.0.0.1:8000/docs
```

接口包括公开的 `GET /v1/health`，登录相关的 `POST /v1/auth/login`、`POST /v1/auth/logout`、
`GET /v1/auth/me`，需要登录的 `GET /v1/laws`、`POST /v1/retrieve`、`POST /v1/query`，
历史对话的 `GET /v1/conversations`、`POST /v1/conversations`、
`GET /v1/conversations/{id}`、`DELETE /v1/conversations/{id}`，
以及页面维护知识库使用的 `POST /v1/laws/upload`（multipart，字段 `file` / `law_id` / `law_name`）
与 `POST /v1/knowledge-base/rebuild`。两个知识库接口返回 `text/event-stream`，
以 `progress`、`complete`、`error` 三类事件推送进度；同一时间只允许一个重建任务，
重复提交会返回 `409`。`/v1/query` 可额外传入 `history`，例如 `{"query":"那赔偿怎么算？","history":[{"role":"user","content":"公司三个月没有和我签合同"}]}`；
再带上 `conversation_id` 时，这一轮问答会作为两条消息写入历史对话，不传则不记录；
同时，有 `conversation_id` 时多轮上下文以库里的历史为准（前端传来的 `history` 只作为库读不到时的兜底），
这样才能带上消息序号去复用早期对话摘要的缓存。
`/v1/retrieve` 的请求体仍为 `{"query":"问题","law_name":"可选法律名称"}`；
`/v1/retrieve` 还可接受 `top_k`（1–20）。`/v1/query` 返回完整、已校验的结构化回答。
若存在 `web/dist/`，FastAPI 会把页面挂载在根路径；`/v1` 与 `/docs` 保持可用。
前端单独部署时，可通过 `RAG_LAW_CORS_ORIGINS`（逗号分隔）放行来源。

除 `/v1/health` 与 `/v1/auth/*` 之外，业务接口都要求请求携带登录 Cookie，未登录或会话过期
返回 `401`；账号库未配置或连不上时登录接口返回 `503`。详见上文「登录与账号」。

API 单进程默认按客户端 IP 限制为每分钟 30 次请求，单次查询超时为 60 秒；超时返回
`504`，过于频繁返回 `429`，缺少索引或模型配置时返回 `503`。`/v1/health` 会逐项报告
manifest、法条数据、BM25 和 Chroma 索引是否就绪。多实例部署时请将内存限流替换为网关或共享存储限流。

### 使用 Docker Compose 部署

常规启动前需准备好 `.env`、`data/processed/`、`indexes/bm25/` 和 `indexes/chroma/`；
首次部署也可以只准备 `.env` 与仓库中的原始法律，再通过页面按钮建库。
Compose 只启动一个服务：多阶段构建先用 Node 打包 Vue，再由 FastAPI 同时提供页面与接口。
`.env` 通过 `env_file` 整体传入容器，因此 `RAG_LAW_DATABASE_URL` 里不能写 `localhost` 或
`127.0.0.1`——容器内的回环地址指向容器自身，应填宿主机可达的 IP 或 Compose 服务名。

```bash
docker compose config --quiet
docker compose build
docker compose up -d --wait
docker compose ps
```

国内 Linux 服务器无法直连 Docker Hub 时，可以任选一种方式完成构建：为 Docker 配置镜像加速
（在 `/etc/docker/daemon.json` 中设置 `registry-mirrors` 后重启 Docker），或指定可访问的
基础镜像地址（以实际可用的加速服务为准）：

```bash
export RAG_LAW_PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.13-slim
export RAG_LAW_NODE_IMAGE=docker.m.daocloud.io/library/node:22-alpine
docker compose build
```

镜像内的 `uv` 与 Python 依赖从阿里云 PyPI 镜像源安装，npm 依赖从 npmmirror 安装，
构建过程无需访问 ghcr.io。

默认地址：

- 页面与 API：<http://127.0.0.1:8000>，交互文档位于 `/docs`；
- 健康检查：<http://127.0.0.1:8000/v1/health>。

启动后的冒烟检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/health
Invoke-WebRequest http://127.0.0.1:8000/ -UseBasicParsing
```

如果服务器只有 GitHub 中的 `data/raw/` 和 `data/manifest.yaml`，尚无结构化语料和索引，
先执行 `docker compose up -d`，打开页面，在侧栏“重新构建现有知识库”中点击“开始重建”。
完成后健康检查四项均为 `true`，服务器首次建库即完成；API 会自动加载新索引，无需重启容器。

可通过环境变量修改宿主机端口，例如：

```powershell
$env:RAG_LAW_API_PORT = "18000"
docker compose up -d
```

`data/`、`indexes/` 和 `logs/` 挂载到宿主机，删除或重建容器不会删除语料、索引和日志。
API 容器对 `data/` 与 `indexes/` 保留读写权限，因为页面上的上传与重建需要写入新索引。

Compose 默认以 UID/GID `1000:1000` 访问挂载目录。Linux 上若当前用户不是 1000，可在启动前设置：

```bash
export RAG_LAW_UID=$(id -u)
export RAG_LAW_GID=$(id -g)
docker compose up -d
```

查看日志或停止服务：

```bash
docker compose logs -f api
docker compose down
```

### 重建索引与故障排查

新增法律可在页面侧栏“添加新法律”上传，也可使用管理命令：

```bash
uv run python -m rag_law.cli add-document --file path/to/law.md --law-id cn_example_law
```

若健康检查显示 `articles`、`bm25_index` 或 `chroma_index` 为 `false`，请先确认
`data/manifest.yaml` 和原始文件存在，再运行上述添加命令或按项目既有流程重建索引。若 API
返回 `503`，检查 `.env` 中三组模型配置并运行 `uv run python scripts/verify_env.py`。日志写入
`logs/answer_runs.jsonl` 前会脱敏身份证号、手机号、邮箱、银行卡号和座机号；API 错误日志不会记录请求原文。

回答日志（`schema_version` 1.1）除脱敏后的原始问题外，还固定记录三项定位信息：
`retrieval_query`（多轮追问时真正送去检索的拼接文本）、`reranked_article_nos`（精排命中的条号）
和 `context_article_nos` / `context_law_names`（模型实际看到的证据条号，含相邻条文）。
排查“为什么这一轮拒答”不必再复现请求：库里每轮 `payload` 已存完整结果，可直接打印对照：

```bash
uv run python scripts/show_conversation_trace.py --conversation-id 5 --user-id 1
```

输出会逐轮列出“用户输入 → 实际检索 query → 精排命中条号 → 模型看到的证据 → 模型引用 → 状态”，
并在 `insufficient_evidence` 时区分两种来源：证据闸门拦截（未调用模型）与模型判定证据不支持（已调用模型但未给出引用）。

容器异常时先运行 `docker compose ps` 和 `docker compose logs --tail=200 api`。健康状态为
`unhealthy` 通常表示挂载中缺少结构化法条或索引；出现 `PermissionError` 时检查上述 UID/GID
设置以及宿主机目录权限。

## 目录结构

```text
data/
  raw/                 # 15 部法律原始文件（pdf / docx / md）
  processed/           # 阶段 1 起产出的结构化法条数据（不入 Git）
  eval/                # 评测集
  manifest.yaml        # 法律清单：law_id / law_name / file
docs/
  plan/v1/             # V1 各阶段实施规划
  note/v1/             # 各阶段实施后的知识沉淀
scripts/               # 开发辅助脚本（环境自检、清单校验、账号库初始化、会话链路对照）
src/rag_law/
  api/                 # FastAPI 接口、SSE 进度流与静态页面托管
  auth/                # 登录接口、Session Cookie、进程内会话与账号库连接池
  catalog.py           # 法律清单读取
  evaluation/          # 离线评测：检索 / 重排上下文 / 答案质量指标
  config.py            # 全局配置对象（从 .env 读取）
  ingestion/           # 文档解析、归一化与法条切分
  retrieval/           # BM25、Dense、RRF 融合、重排与上下文构建
    retrieval_graph.py # LangGraph 检索编排：并行召回 → 融合 → 精排 → 上下文
  generation/          # 结构化生成、引用校验、拒答与脱敏日志
    answer_graph.py    # LangGraph 生成编排：证据闸门 → 生成 → 校验 → 修复
web/                   # Vue 3 + TypeScript + Vite + Element Plus 前端
  src/api/             # 接口与 SSE 客户端
  src/components/      # 页面组件（对话、回答卡片、上传与重建弹窗）
  src/composables/     # 对话与知识库状态
tests/                 # pytest 用例（接口、检索、生成、评测与鉴权）
indexes/               # 检索索引（不入 Git）
logs/                  # 本地脱敏运行日志（不入 Git）
```

## 数据说明

- 语料范围：当前仅回答 `data/manifest.yaml` 中 15 部法律范围内的问题，范围外明确说明；
- 评测集：`data/eval/questions_v1.yaml`，30 题（含 6 题拒答/追问），法条标注均与原文逐条核对；
- 原始文件以仓库内为准，不核验法律时效与历史版本。
