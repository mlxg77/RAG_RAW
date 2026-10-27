import pytest

from rag_law.retrieval.article_anchor import (
    ArticleAnchorIndex,
    extract_article_references,
    law_short_name,
    normalize_article_numbers,
    to_chinese_article_number,
)
from rag_law.schemas import ArticleChunk


def make_chunk(
    *,
    chunk_id: str,
    law_name: str,
    article_no: str,
    text: str = "测试条文内容。",
) -> ArticleChunk:
    return ArticleChunk(
        chunk_id=chunk_id,
        law_id=f"law-{chunk_id}",
        law_name=law_name,
        article_no=article_no,
        text=text,
        source_file="data/raw/test.md",
        text_hash=f"hash-{chunk_id}",
    )


def make_index() -> ArticleAnchorIndex:
    return ArticleAnchorIndex(
        [
            make_chunk(
                chunk_id="c1",
                law_name="中华人民共和国刑法",
                article_no="第二条",
            ),
            make_chunk(
                chunk_id="c2",
                law_name="中华人民共和国刑法",
                article_no="第一百二十条之三",
            ),
            make_chunk(
                chunk_id="c3",
                law_name="中华人民共和国民法典",
                article_no="第二条",
            ),
        ]
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("刑法第二条是什么", ["第二条"]),
        ("第2条呢", ["第二条"]),
        ("第 20 条怎么规定", ["第二十条"]),
        (
            "第一百二十条之三说了什么",
            ["第一百二十条之三"],
        ),
        ("刑法第二十条和第一条", ["第二十条", "第一条"]),
        ("公司一直没有和我签书面劳动合同", []),
    ],
)
def test_extract_article_references(
    text: str,
    expected: list[str],
) -> None:
    assert extract_article_references(text) == expected


def test_normalize_article_numbers_handles_suffix() -> None:
    assert (
        normalize_article_numbers("刑法第120条之3")
        == "刑法第一百二十条之三"
    )


def test_to_chinese_article_number_matches_index() -> None:
    assert to_chinese_article_number(2) == "二"
    assert to_chinese_article_number(105) == "一百零五"
    assert to_chinese_article_number(120) == "一百二十"


def test_law_short_name_drops_national_prefix() -> None:
    assert (
        law_short_name("中华人民共和国刑法") == "刑法"
    )
    assert (
        law_short_name("中华人民共和国民法典") == "民法典"
    )


def test_resolve_locates_article_from_query() -> None:
    results = make_index().resolve("刑法第二条是什么")

    assert [item.article_no for item in results] == [
        "第二条"
    ]
    assert results[0].law_name == "中华人民共和国刑法"
    assert results[0].chunk_id == "c1"


def test_resolve_accepts_arabic_number() -> None:
    results = make_index().resolve("民法典第2条")

    assert [item.chunk_id for item in results] == ["c3"]


def test_resolve_accepts_article_suffix() -> None:
    results = make_index().resolve(
        "刑法第一百二十条之三"
    )

    assert [item.chunk_id for item in results] == ["c2"]


def test_resolve_uses_law_filter_when_query_omits_law() -> None:
    results = make_index().resolve(
        "第二条呢",
        law_name="中华人民共和国民法典",
    )

    assert [item.chunk_id for item in results] == ["c3"]


def test_resolve_skips_ambiguous_law() -> None:
    # 同时提到两部法律时无法判断「第二条」是哪一部，宁可放弃。
    assert (
        make_index().resolve("刑法和民法典第二条")
        == []
    )


def test_resolve_skips_query_without_article_number() -> None:
    assert make_index().resolve("刑法怎么规定的") == []


def test_resolve_skips_unknown_article() -> None:
    assert make_index().resolve("刑法第九十九条") == []
