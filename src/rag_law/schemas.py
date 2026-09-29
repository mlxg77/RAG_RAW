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

class SplitResult(BaseModel):
    """一部法律切分后的结果和诊断信息。"""

    chunks: list[ArticleChunk]
    article_count: int
    empty_articles: list[str] = Field(default_factory=list)
    duplicate_articles: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)