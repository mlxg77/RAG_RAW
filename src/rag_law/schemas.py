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
    
AnswerStatus = Literal[
    "answered",
    "needs_clarification",
    "insufficient_evidence",
    "generation_failed",
]


class GeneratedCitation(BaseModel):
    """大模型选择的证据，展示字段由程序从证据中回填。"""

    evidence_id: str = Field(
        pattern=r"^E\d{3}$",
        description=(
            "本次检索上下文中的证据编号，例如 E001。"
            "不得自行编造。"
        ),
    )


class GeneratedAnswer(BaseModel):
    """大模型必须返回的结构化回答。"""

    summary: str = Field(
        min_length=1,
        description="面向用户的简明结论。",
    )
    analysis: str = Field(
        min_length=1,
        description="严格依据检索证据进行的分析。",
    )
    conditions: list[str] = Field(
        default_factory=list,
        description="规则适用条件或仍需确认的关键事实。",
    )
    citations: list[GeneratedCitation] = Field(
        default_factory=list,
        description="支持回答内容的法条引用。",
    )
    limitations: str = Field(
        min_length=1,
        description="证据边界、知识库边界和必要风险提示。",
    )
    follow_up_question: str | None = Field(
        default=None,
        description=(
            "缺少关键事实时，只提出一个最重要的追问；"
            "无需追问时为 null。"
        ),
    )


class ValidatedCitation(BaseModel):
    """通过校验后，允许展示给用户的引用。"""

    evidence_id: str = Field(
        pattern=r"^E\d{3}$"
    )
    chunk_id: str
    law_id: str
    law_name: str
    article_no: str
    quote: str
    source_file: str
    page: int | None = Field(
        default=None,
        ge=1,
    )
    end_page: int | None = Field(
        default=None,
        ge=1,
    )


class CitationValidationResult(BaseModel):
    """一次引用校验的完整结果。"""

    is_valid: bool
    citations: list[ValidatedCitation] = Field(
        default_factory=list
    )
    errors: list[str] = Field(
        default_factory=list
    )
    
class AnswerResponse(BaseModel):
    """最终允许返回给用户的回答。"""

    status: AnswerStatus
    summary: str
    analysis: str
    conditions: list[str] = Field(
        default_factory=list
    )
    citations: list[ValidatedCitation] = Field(
        default_factory=list
    )
    limitations: str
    follow_up_question: str | None = None


class AnswerPipelineResult(BaseModel):
    """阶段 4 完整回答流水线的输出。"""

    query: str = Field(min_length=1)
    answer: AnswerResponse
    retrieval: RetrievalContextResult

    generation_attempts: int = Field(
        ge=0,
        le=2,
    )
    top_relevance_score: float | None = None
    validation_errors: list[str] = Field(
        default_factory=list
    )

    prompt_version: str
    generation_latency_ms: float = Field(ge=0)
    total_latency_ms: float = Field(ge=0)
