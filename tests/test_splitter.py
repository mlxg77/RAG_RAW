"""法条结构识别与切分测试。"""

from pathlib import Path

import pytest

from rag_law.ingestion.article_splitter import split_articles
from rag_law.ingestion.loaders import load_document
from rag_law.ingestion.normalizer import normalize_document
from rag_law.schemas import LoadedDocument, SourceBlock


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_document(
    texts: list[str],
    *,
    source_format: str = "markdown",
    pages: list[int | None] | None = None,
) -> LoadedDocument:
    """构造用于测试的已标准化文档。"""

    if pages is None:
        pages = [None] * len(texts)

    return LoadedDocument(
        law_id="test_law",
        law_name="测试法",
        source_file=f"data/raw/测试法.{source_format}",
        source_format=source_format,
        page_count=(
            max(page for page in pages if page is not None)
            if any(page is not None for page in pages)
            else None
        ),
        blocks=[
            SourceBlock(
                text=text,
                page=page,
                block_index=index,
            )
            for index, (text, page)
            in enumerate(zip(texts, pages, strict=True))
        ],
    )


def test_skip_table_of_contents_and_inherit_hierarchy() -> None:
    document = make_document(
        [
            "# 测试法",
            "# 目 录",
            "第一章 总则",
            "第一条 ……1",
            "第二条 ……2",
            "# 第一编 总则编",
            "# 第一章 基本规定",
            "## 第一节 一般规定",
            "第一条 正文第一条。",
            "第二条 正文第二条。",
        ]
    )

    result = split_articles(document)

    assert result.article_count == 2
    assert len(result.chunks) == 2

    first = result.chunks[0]
    second = result.chunks[1]

    assert first.article_no == "第一条"
    assert first.text == "正文第一条。"
    assert first.part == "第一编 总则编"
    assert first.chapter == "第一章 基本规定"
    assert first.section == "第一节 一般规定"

    assert second.article_no == "第二条"
    assert second.text == "正文第二条。"

    # 目录内容不得进入正文。
    assert all("……" not in chunk.text for chunk in result.chunks)


def test_recognize_long_chinese_number_and_article_suffix() -> None:
    document = make_document(
        [
            "第一百零一条 第一百零一条正文。",
            "第二百二十四条之一 插入条文正文。",
        ]
    )

    result = split_articles(document)

    assert [
        chunk.article_no
        for chunk in result.chunks
    ] == [
        "第一百零一条",
        "第二百二十四条之一",
    ]


def test_new_chapter_clears_previous_section() -> None:
    document = make_document(
        [
            "第一章 第一章标题",
            "第一节 第一节标题",
            "第一条 第一条正文。",
            "第二章 第二章标题",
            "第二条 第二条正文。",
        ]
    )

    result = split_articles(document)

    assert result.chunks[0].section == "第一节 第一节标题"

    assert result.chunks[1].chapter == "第二章 第二章标题"
    assert result.chunks[1].section is None


def test_join_article_across_pdf_pages() -> None:
    document = make_document(
        [
            "第一章 总则\n第一条 本条内容尚未结",
            "束。\n第二条 第二条正文。",
        ],
        source_format="pdf",
        pages=[1, 2],
    )

    result = split_articles(document)

    first = result.chunks[0]

    assert first.article_no == "第一条"
    assert first.text == "本条内容尚未结束。"
    assert first.page == 1
    assert first.end_page == 2

    second = result.chunks[1]

    assert second.article_no == "第二条"
    assert second.page == 2
    assert second.end_page == 2


def test_preserve_paragraph_boundary_across_pages() -> None:
    document = make_document(
        [
            "第一条 第一款。",
            "第二款。\n第二条 第二条正文。",
        ],
        source_format="pdf",
        pages=[1, 2],
    )

    result = split_articles(document)

    assert result.chunks[0].text == "第一款。\n第二款。"


def test_report_empty_and_duplicate_articles() -> None:
    document = make_document(
        [
            "第一条 第一条正文。",
            "第二条 第二条正文。",
            "第二条 重复的第二条正文。",
            "第三条",
            "第四条 第四条正文。",
        ]
    )

    result = split_articles(document)

    assert result.article_count == 5
    assert result.empty_articles == ["第三条"]
    assert result.duplicate_articles == ["第二条"]
    assert len(result.chunks) == 4
    assert len(result.warnings) == 2

    # 即使条号重复，chunk_id 也不能重复。
    chunk_ids = [
        chunk.chunk_id
        for chunk in result.chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_id_is_stable() -> None:
    document = make_document(
        [
            "第一条 第一条正文。",
            "第二条 第二条正文。",
        ]
    )

    first_run = split_articles(document)
    second_run = split_articles(document)

    assert [
        chunk.chunk_id
        for chunk in first_run.chunks
    ] == [
        chunk.chunk_id
        for chunk in second_run.chunks
    ]

    assert [
        chunk.text_hash
        for chunk in first_run.chunks
    ] == [
        chunk.text_hash
        for chunk in second_run.chunks
    ]

    assert all(
        chunk.chunk_id.startswith("sha256:")
        for chunk in first_run.chunks
    )

    assert all(
        chunk.text_hash.startswith("sha256:")
        for chunk in first_run.chunks
    )


def test_no_articles_returns_warning() -> None:
    document = make_document(
        [
            "测试法",
            "这里没有任何法条。",
        ]
    )

    result = split_articles(document)

    assert result.article_count == 0
    assert result.chunks == []
    assert len(result.warnings) == 1
    assert "未识别到任何法条" in result.warnings[0]


@pytest.mark.parametrize(
    (
        "relative_path",
        "law_id",
        "law_name",
        "minimum_article_count",
    ),
    [
        (
            "data/raw/中华人民共和国劳动合同法.pdf",
            "cn_labor_contract_law",
            "中华人民共和国劳动合同法",
            90,
        ),
        (
            "data/raw/中华人民共和国刑法.docx",
            "cn_criminal_law",
            "中华人民共和国刑法",
            400,
        ),
        (
            "data/raw/中华人民共和国道路交通安全法.md",
            "cn_road_traffic_safety_law",
            "中华人民共和国道路交通安全法",
            100,
        ),
    ],
)
def test_split_real_documents(
    relative_path: str,
    law_id: str,
    law_name: str,
    minimum_article_count: int,
) -> None:
    loaded = load_document(
        PROJECT_ROOT / relative_path,
        law_id=law_id,
        law_name=law_name,
        source_file=relative_path,
    )

    normalized = normalize_document(loaded)
    result = split_articles(normalized)

    assert result.article_count >= minimum_article_count
    assert len(result.chunks) >= minimum_article_count
    assert result.chunks[0].article_no == "第一条"

    assert all(
        chunk.text.strip()
        for chunk in result.chunks
    )

    assert all(
        chunk.law_id == law_id
        for chunk in result.chunks
    )