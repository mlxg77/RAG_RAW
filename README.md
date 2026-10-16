# rag_law · 法条知识库问答系统

面向给定法律文本（15 部中华人民共和国法律）的问答系统：根据用户描述检索相关法条，说明法律如何规定，并给出可核验的法条引用；证据不足时明确说明，不编造结论。

> 本系统提供法律信息检索与参考，不构成法律意见，不替代执业律师。

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段 0 | 范围与基线（项目骨架、法律清单、评测集、环境验证） | 完成 |
| 阶段 1 | 文档解析与法条切分 | 完成 |
| 阶段 2 | BM25、Dense、RRF 混合检索与量化基线 | 完成（存在已记录限制） |
| 阶段 3 | Reranker 精排与可追溯上下文构建 | 完成（存在已记录限制） |
| 阶段 4 | 结构化答案生成、引用校验、拒答与脱敏日志 | 完成（存在已记录限制） |
| 阶段 5 | 独立 holdout 端到端评测与质量门槛 | 已实现（1 项质量门槛未通过） |
| 阶段 6 | API、界面与部署 | 进行中（API 与演示界面已完成） |

详见 [docs/plan/v1/](docs/plan/v1/)。

## 环境要求

- Python 3.11+（`.python-version` 已锁定 3.13）
- [uv](https://docs.astral.sh/uv/) 包管理工具
- 一个模型服务（本项目当前使用 SiliconFlow：`Qwen/Qwen3-8B`、`BAAI/bge-m3` 和 `BAAI/bge-reranker-v2-m3`）

## 快速开始

```bash
# 1. 安装依赖（按 uv.lock 复现环境，自动创建 .venv）
uv sync

# 2. 配置环境变量
cp .env.example .env       # Windows PowerShell: Copy-Item .env.example .env
# 然后编辑 .env，填入模型服务的 BASE_URL / API_KEY / 模型名

# 3. 验证三条链路：embedding / Chroma / LLM
uv run python scripts/verify_env.py

# 4. 校验法律清单
uv run python scripts/check_manifest.py

# 5. 执行完整问答（检索、精排、生成和引用校验）
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

### 启动演示界面

```bash
# 启动演示界面（本次的方式）
uv run streamlit run src/rag_law/ui/app.py --server.port 8502
uv run streamlit run src/rag_law/ui/app.py
```

页面支持连续提问、按法律筛选、展示回答状态、适用条件、边界限制和已经过校验的法条引用。启动前需按上文完成环境变量配置，并确保 `data/processed/` 与 `indexes/` 已生成。

### 启动 HTTP API

```bash
# 本机启动；生产环境应放在反向代理之后并限制访问来源
uv run uvicorn rag_law.api.main:app --host 127.0.0.1 --port 8000

# 查看 API schema 与交互式文档
# http://127.0.0.1:8000/docs
```

接口包括 `GET /v1/health`、`GET /v1/laws`、`POST /v1/retrieve` 和
`POST /v1/query`。后两个接口的请求体为 `{"query":"问题","law_name":"可选法律名称"}`；
`/v1/retrieve` 还可接受 `top_k`（1–20）。`/v1/query` 返回完整、已校验的结构化回答。

API 单进程默认按客户端 IP 限制为每分钟 30 次请求，单次查询超时为 60 秒；超时返回
`504`，过于频繁返回 `429`，缺少索引或模型配置时返回 `503`。`/v1/health` 会逐项报告
manifest、法条数据、BM25 和 Chroma 索引是否就绪。多实例部署时请将内存限流替换为网关或共享存储限流。

### 重建索引与故障排查

新增法律可在演示页“添加新法律”上传，也可使用管理命令：

```bash
uv run python -m rag_law.cli add-document --file path/to/law.md --law-id cn_example_law
```

若健康检查显示 `articles`、`bm25_index` 或 `chroma_index` 为 `false`，请先确认
`data/manifest.yaml` 和原始文件存在，再运行上述添加命令或按项目既有流程重建索引。若 API
返回 `503`，检查 `.env` 中三组模型配置并运行 `uv run python scripts/verify_env.py`。日志写入
`logs/answer_runs.jsonl` 前会脱敏身份证号、手机号、邮箱、银行卡号和座机号；API 错误日志不会记录请求原文。

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
scripts/               # 开发辅助脚本（环境自检、清单校验）
src/rag_law/
  config.py            # 全局配置对象（从 .env 读取）
  generation/          # 结构化生成、引用校验、拒答与脱敏日志
  ui/                  # Streamlit 演示界面与展示规则
indexes/               # 检索索引（不入 Git）
logs/                  # 本地脱敏运行日志（不入 Git）
```

## 数据说明

- 语料范围：当前仅回答 `data/manifest.yaml` 中 15 部法律范围内的问题，范围外明确说明；
- 评测集：`data/eval/questions_v1.yaml`，30 题（含 6 题拒答/追问），法条标注均与原文逐条核对；
- 原始文件以仓库内为准，不核验法律时效与历史版本。
