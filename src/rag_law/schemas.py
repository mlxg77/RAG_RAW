"""项目内部使用的数据结构。"""

from typing import Literal

from pydantic import BaseModel, Field


SourceFormat = Literal["pdf", "docx", "markdown"]


class SourceBlock(BaseModel):
    """Loader 提取出的最小原始文本块。

    PDF：一个 block 对应一页。
    DOCX：一个 block 对应一个非空段落。
    Markdown：一个 block 对应一个非空文本行。
    """

    text: str
    page: int | None = Field(default=None, ge=1)
    block_index: int = Field(ge=0)


class LoadedDocument(BaseModel):
    """一部法律经过 Loader 后的结果，尚未清洗和切分。"""

    law_id: str
    law_name: str
    source_file: str
    source_format: SourceFormat
    page_count: int | None = Field(default=None, ge=1)
    blocks: list[SourceBlock]


class ArticleChunk(BaseModel):
    """最终写入 articles.jsonl 的一条法条记录。

    现在先定义，后面的 splitter 和 pipeline 会使用。
    """

    chunk_id: str
    law_id: str
    law_name: str
    jurisdiction: str = "中华人民共和国"
    document_type: str = "法律"

    part: str | None = None
    chapter: str | None = None
    section: str | None = None
    article_no: str
    paragraph_no: int | None = None

    text: str
    source_file: str
    page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)
    text_hash: str

RetrievalMethod = Literal[
    "bm25",
    "dense",
    "hybrid",
    "rerank",
]

class RetrievalResult(BaseModel):
    """BM25、Dense 和 Hybrid 共用的检索结果结构。"""

    chunk_id: str
    law_id: str
    law_name: str
    article_no: str
    text: str

    part: str | None = None
    chapter: str | None = None
    section: str | None = None
    paragraph_no: int | None = None

    source_file: str
    page: int | None = Field(default=None, ge=1)
    end_page: int | None = Field(default=None, ge=1)

    score: float
    source: RetrievalMethod

    # Hybrid 阶段用来记录各检索器贡献的原始分数。
    component_scores: dict[str, float] = Field(
        default_factory=dict
    )

ContextRelation = Literal[
    "matched",
    "adjacent",
]


class ContextEvidence(BaseModel):
    """交给模型的一段可追溯证据。"""

    evidence_id: str = Field(
        pattern=r"^E\d{3}$"
    )

    chunk_id: str
    law_id: str
    law_name: str
    article_no: str

    text: str = Field(min_length=1)

    part: str | None = None
    chapter: str | None = None
    section: str | None = None

    source_file: str
    page: int | None = Field(
        default=None,
        ge=1,
    )
    end_page: int | None = Field(
        default=None,
        ge=1,
    )

    relation: ContextRelation
    is_excerpt: bool
    score: float


class BuiltContext(BaseModel):
    """完成去重、截断和编号后的模型上下文。"""

    text: str
    evidences: list[ContextEvidence]

    estimated_tokens: int = Field(ge=0)
    max_tokens: int = Field(gt=0)

    # 只要发生片段提取、预算淘汰或数量淘汰，就为 True。
    truncated: bool


class RetrievalContextResult(BaseModel):
    """阶段 3 完整流水线的输出。"""

    query: str = Field(min_length=1)
    reranked_results: list[RetrievalResult]
    context: BuiltContext

    retrieval_latency_ms: float = Field(ge=0)
    context_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)

class SplitResult(BaseModel):
    """一部法律切分后的结果和诊断信息。"""

    chunks: list[ArticleChunk]
    article_count: int
    empty_articles: list[str] = Field(default_factory=list)
    duplicate_articles: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
