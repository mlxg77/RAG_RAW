# rag_law · 法条知识库问答系统

面向给定法律文本（15 部中华人民共和国法律）的问答系统：根据用户描述检索相关法条，说明法律如何规定，并给出可核验的法条引用；证据不足时明确说明，不编造结论。

> 本系统提供法律信息检索与参考，不构成法律意见，不替代执业律师。

## 当前进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段 0 | 范围与基线（项目骨架、法律清单、评测集、环境验证） | 完成 |
| 阶段 1 | 文档解析与法条切分 | 未开始 |

详见 [docs/plan/v1/](docs/plan/v1/)。

## 环境要求

- Python 3.11+（`.python-version` 已锁定 3.13）
- [uv](https://docs.astral.sh/uv/) 包管理工具
- 一个 OpenAI 兼容接口的模型服务（本项目当前使用 SiliconFlow：`Qwen/Qwen3-8B` + `BAAI/bge-m3`）

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
```

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
indexes/               # 检索索引（不入 Git）
```

## 数据说明

- 语料范围：当前仅回答 `data/manifest.yaml` 中 15 部法律范围内的问题，范围外明确说明；
- 评测集：`data/eval/questions_v1.yaml`，30 题（含 6 题拒答/追问），法条标注均与原文逐条核对；
- 原始文件以仓库内为准，不核验法律时效与历史版本。