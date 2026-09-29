"""检索模块共用的数据加载和转换函数。"""

from pathlib import Path

from rag_law.schemas import (
    ArticleChunk,
    RetrievalMethod,
    RetrievalResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ARTICLES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "articles.jsonl"
)


def load_articles(
    path: Path = DEFAULT_ARTICLES_PATH,
) -> list[ArticleChunk]:
    """读取阶段 1 生成的 articles.jsonl。"""

    if not path.exists():
        raise FileNotFoundError(
            f"法条数据不存在：{path}\n"
            "请先运行阶段 1 pipeline。"
        )

    chunks: list[ArticleChunk] = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                chunk = ArticleChunk.model_validate_json(
                    stripped
                )
            except Exception as error:
                raise ValueError(
                    f"{path} 第 {line_number} 行格式错误"
                ) from error

            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"法条数据为空：{path}")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            f"法条数据包含重复 chunk_id：{path}"
        )

    return chunks


def build_search_text(
    chunk: ArticleChunk,
) -> str:
    """构造真正用于检索的文本。

    不仅检索正文，还加入法律名、编章节和条号，使类似
    “劳动合同法第八十七条”的查询可以直接命中。
    """

    fields = [
        chunk.law_name,
        chunk.part,
        chunk.chapter,
        chunk.section,
        chunk.article_no,
        chunk.text,
    ]

    return "\n".join(
        field.strip()
        for field in fields
        if field is not None and field.strip()
    )


def make_retrieval_result(
    chunk: ArticleChunk,
    *,
    score: float,
    source: RetrievalMethod,
    component_scores: dict[str, float] | None = None,
) -> RetrievalResult:
    """把 ArticleChunk 转换为统一检索结果。"""

    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        law_id=chunk.law_id,
        law_name=chunk.law_name,
        article_no=chunk.article_no,
        text=chunk.text,
        part=chunk.part,
        chapter=chunk.chapter,
        section=chunk.section,
        paragraph_no=chunk.paragraph_no,
        source_file=chunk.source_file,
        page=chunk.page,
        end_page=chunk.end_page,
        score=float(score),
        source=source,
        component_scores=component_scores or {},
    )