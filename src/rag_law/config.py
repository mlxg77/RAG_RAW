"""项目配置：从 .env 读取环境变量，生成全局 Settings 对象。

其他模块统一使用：from rag_law.config import settings
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录：config.py 位于 src/rag_law/ 下，向上三级即项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """项目配置，字段名与 .env 中 RAG_LAW_ 前缀之后的部分一一对应。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_LAW_",
        extra="ignore",
    )

    # 大模型（回答生成）
    llm_base_url: str
    llm_api_key: str
    llm_model: str

    # 向量模型（Embedding）
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str

    # 重排模型（Reranker）
    rerank_base_url: str
    rerank_api_key: str
    rerank_model: str

    # 多轮会话：最近若干轮保留原文，更早内容压缩为摘要。
    conversation_recent_rounds: int = Field(
        default=3,
        ge=1,
        le=20,
    )
    conversation_summary_max_chars: int = Field(
        default=1500,
        ge=200,
        le=6000,
    )

    # ---------- 登录与用户 ----------
    # 留空表示尚未接入账号库：/v1/auth/* 会返回 503，业务接口一律拒绝。
    database_url: str = ""
    session_cookie_name: str = "rag_law_sid"
    session_ttl_seconds: int = Field(
        default=8 * 3600,
        ge=60,
        le=30 * 24 * 3600,
    )
    # 生产环境走 HTTPS 时必须置为 true，否则浏览器不会回传 Cookie。
    session_cookie_secure: bool = False

    # ---------- 历史对话 ----------
    # 复用 database_url：留空时无法登录，历史对话也随之不可用（问答本身不受影响）。
    # 侧边栏一次最多列出多少个历史对话。
    conversation_history_limit: int = Field(
        default=50,
        ge=1,
        le=500,
    )

# 如果 .env 缺任何一个字段，ValidationError 会立刻带着缺的字段名抛出来——这叫 fail fast：
# 宁可在启动时秒挂，也不要跑到一半才发现配置没有。
settings = Settings()
