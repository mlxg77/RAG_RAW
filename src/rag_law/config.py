"""项目配置：从 .env 读取环境变量，生成全局 Settings 对象。

其他模块统一使用：from rag_law.config import settings
"""

from pathlib import Path

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

# 如果 .env 缺任何一个字段，ValidationError 会立刻带着缺的字段名抛出来——这叫 fail fast：
# 宁可在启动时秒挂，也不要跑到一半才发现配置没有。
settings = Settings()
