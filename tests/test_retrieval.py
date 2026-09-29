from pathlib import Path

from rag_law.retrieval.bm25 import (
    BM25Retriever,
    tokenize,
)
from rag_law.retrieval.common import (
    build_search_text,
)
from rag_law.schemas import ArticleChunk


def make_chunk(
    *,
    chunk_id: str,
    law_name: str,
    article_no: str,
    text: str,
) -> ArticleChunk:
    return ArticleChunk(
        chunk_id=chunk_id,
        law_id=f"law-{chunk_id}",
        law_name=law_name,
        article_no=article_no,
        text=text,
        source_file=f"data/raw/{law_name}.md",
        text_hash=f"hash-{chunk_id}",
    )


def build_test_retriever() -> BM25Retriever:
    chunks = [
        make_chunk(
            chunk_id="chunk-labor",
            law_name="中华人民共和国劳动合同法",
            article_no="第十条",
            text=(
                "建立劳动关系，应当订立"
                "书面劳动合同。"
            ),
        ),
        make_chunk(
            chunk_id="chunk-traffic",
            law_name="中华人民共和国道路交通安全法",
            article_no="第四十七条",
            text=(
                "机动车行经人行横道时，"
                "应当减速行驶。"
            ),
        ),
        make_chunk(
            chunk_id="chunk-consumer",
            law_name="中华人民共和国消费者权益保护法",
            article_no="第二十五条",
            text=(
                "经营者采用网络方式销售商品，"
                "消费者有权自收到商品之日起"
                "七日内退货。"
            ),
        ),
    ]

    tokenized_corpus = [
        tokenize(build_search_text(chunk))
        for chunk in chunks
    ]

    return BM25Retriever(
        chunks=chunks,
        tokenized_corpus=tokenized_corpus,
    )


def test_tokenize_returns_chinese_terms() -> None:
    tokens = tokenize(
        "公司应当签订书面劳动合同"
    )

    assert tokens
    assert all(token.strip() for token in tokens)
    assert " " not in tokens


def test_search_text_contains_metadata() -> None:
    chunk = make_chunk(
        chunk_id="chunk-1",
        law_name="中华人民共和国劳动合同法",
        article_no="第八十七条",
        text="违法解除劳动合同的，应当支付赔偿金。",
    )

    search_text = build_search_text(chunk)

    assert "中华人民共和国劳动合同法" in search_text
    assert "第八十七条" in search_text
    assert "赔偿金" in search_text


def test_bm25_returns_relevant_article_first() -> None:
    retriever = build_test_retriever()

    results = retriever.search(
        "公司没有和员工签书面劳动合同",
        top_k=3,
    )

    assert results[0].chunk_id == "chunk-labor"
    assert results[0].source == "bm25"
    assert results[0].score >= results[1].score


def test_bm25_supports_law_name_filter() -> None:
    retriever = build_test_retriever()

    results = retriever.search(
        "法律规定",
        top_k=10,
        law_name="中华人民共和国道路交通安全法",
    )

    assert len(results) == 1
    assert (
        results[0].law_name
        == "中华人民共和国道路交通安全法"
    )


def test_bm25_index_can_be_saved_and_loaded(
    tmp_path: Path,
) -> None:
    retriever = build_test_retriever()
    index_path = tmp_path / "bm25_index.json"

    retriever.save(index_path)
    loaded = BM25Retriever.load(index_path)

    original_results = retriever.search(
        "七日无理由退货",
        top_k=3,
    )
    loaded_results = loaded.search(
        "七日无理由退货",
        top_k=3,
    )

    assert [
        result.chunk_id
        for result in loaded_results
    ] == [
        result.chunk_id
        for result in original_results
    ]


def test_bm25_rejects_empty_query() -> None:
    retriever = build_test_retriever()

    try:
        retriever.search("   ")
    except ValueError as error:
        assert "不能为空" in str(error)
    else:
        raise AssertionError(
            "空问题应当抛出 ValueError"
        )