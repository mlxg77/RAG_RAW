from rag_law.retrieval.common import (
    make_retrieval_result,
)
from rag_law.retrieval.context_builder import (
    ContextBuilder,
    estimate_tokens,
    normalize_body,
    split_text_spans,
)
from rag_law.schemas import (
    ArticleChunk,
    RetrievalResult,
)


def make_chunk(
    *,
    chunk_id: str,
    law_id: str = "test-law",
    law_name: str = "中华人民共和国测试法",
    article_no: str,
    text: str,
) -> ArticleChunk:
    return ArticleChunk(
        chunk_id=chunk_id,
        law_id=law_id,
        law_name=law_name,
        article_no=article_no,
        text=text,
        source_file=(
            f"data/raw/{law_id}.md"
        ),
        text_hash=f"hash-{chunk_id}",
    )


def make_result(
    chunk: ArticleChunk,
    *,
    score: float = 0.9,
) -> RetrievalResult:
    return make_retrieval_result(
        chunk,
        score=score,
        source="rerank",
        component_scores={
            "rerank": score,
        },
    )


class FakeScorer:
    """根据关键词返回确定性测试分数。"""

    def score_documents(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        assert query

        scores: list[float] = []

        for document in documents:
            if "二倍工资" in document:
                scores.append(0.99)
            elif "后一条" in document:
                scores.append(0.80)
            elif "前一条" in document:
                scores.append(0.70)
            elif "劳动合同" in document:
                scores.append(0.60)
            else:
                scores.append(0.10)

        return scores


def test_token_estimator_ignores_whitespace() -> None:
    assert estimate_tokens(
        "劳动 合同\n法"
    ) == 5


def test_normalize_body_ignores_layout() -> None:
    assert normalize_body(
        "建立劳动关系，\n应当订立合同。"
    ) == normalize_body(
        "建立劳动关系，应当订立合同。"
    )


def test_split_spans_preserves_original_text() -> None:
    text = (
        "第一款规定。\n"
        "第二款规定；第三款规定。"
    )

    spans = split_text_spans(text)

    units = [
        text[start:end]
        for start, end in spans
    ]

    assert units == [
        "第一款规定。",
        "第二款规定；",
        "第三款规定。",
    ]


def test_context_assigns_evidence_ids_and_maps_chunks() -> None:
    first = make_chunk(
        chunk_id="chunk-1",
        article_no="第一条",
        text="建立劳动关系，应当订立劳动合同。",
    )
    second = make_chunk(
        chunk_id="chunk-2",
        article_no="第二条",
        text="未订立书面劳动合同的，应当支付二倍工资。",
    )

    builder = ContextBuilder(
        chunks=[first, second],
        scorer=FakeScorer(),
        max_context_tokens=1000,
        max_evidences=2,
        max_evidence_tokens=200,
        adjacent_for_top_n=0,
    )

    context = builder.build(
        query="没有签劳动合同怎么办",
        results=[
            make_result(second, score=0.95),
            make_result(first, score=0.80),
        ],
    )

    assert [
        evidence.evidence_id
        for evidence in context.evidences
    ] == ["E001", "E002"]

    assert [
        evidence.chunk_id
        for evidence in context.evidences
    ] == ["chunk-2", "chunk-1"]

    assert "[E001]" in context.text
    assert "第二条" in context.text

    assert (
        context.estimated_tokens
        <= context.max_tokens
    )


def test_context_deduplicates_article_and_body() -> None:
    first = make_chunk(
        chunk_id="chunk-1",
        article_no="第一条",
        text="相同法条正文。",
    )

    same_article = make_chunk(
        chunk_id="chunk-2",
        article_no="第一条",
        text="同一条号的另一个块。",
    )

    same_body = make_chunk(
        chunk_id="chunk-3",
        article_no="第二条",
        text="相同法条正文。",
    )

    unique = make_chunk(
        chunk_id="chunk-4",
        article_no="第三条",
        text="完全不同的正文。",
    )

    builder = ContextBuilder(
        chunks=[
            first,
            same_article,
            same_body,
            unique,
        ],
        scorer=FakeScorer(),
        max_context_tokens=1000,
        max_evidences=6,
        max_evidence_tokens=200,
        adjacent_for_top_n=0,
    )

    context = builder.build(
        query="测试去重",
        results=[
            make_result(first),
            make_result(same_article),
            make_result(same_body),
            make_result(unique),
        ],
    )

    assert [
        evidence.chunk_id
        for evidence in context.evidences
    ] == [
        "chunk-1",
        "chunk-4",
    ]


def test_long_article_uses_contiguous_complete_excerpt() -> None:
    text = (
        "第一款是与问题无关的规定。"
        "第二款仍然与问题无关。"
        "用人单位未订立书面劳动合同的，"
        "应当向劳动者支付二倍工资。"
        "最后一款也是其他规定。"
    )

    chunk = make_chunk(
        chunk_id="long-chunk",
        article_no="第八十二条",
        text=text,
    )

    builder = ContextBuilder(
        chunks=[chunk],
        scorer=FakeScorer(),
        max_context_tokens=500,
        max_evidences=1,
        max_evidence_tokens=35,
        adjacent_for_top_n=0,
    )

    context = builder.build(
        query="公司没有签劳动合同能要二倍工资吗",
        results=[make_result(chunk)],
    )

    assert len(context.evidences) == 1

    evidence = context.evidences[0]

    assert evidence.is_excerpt is True
    assert "二倍工资" in evidence.text

    # 必须是原文连续子串，不能把几个离散句子重新拼接。
    assert evidence.text in text

    # 不能从句子中间截断。
    assert evidence.text.endswith(
        ("。", "！", "？", "；")
    )

    assert (
        estimate_tokens(evidence.text)
        <= 35
    )

    assert context.truncated is True


def test_adjacent_articles_stay_in_same_law() -> None:
    previous = make_chunk(
        chunk_id="previous",
        article_no="第一条",
        text="这是前一条规定。",
    )

    matched = make_chunk(
        chunk_id="matched",
        article_no="第二条",
        text="这是直接命中的劳动合同规定。",
    )

    following = make_chunk(
        chunk_id="following",
        article_no="第三条",
        text="这是后一条规定。",
    )

    other_law = make_chunk(
        chunk_id="other-law",
        law_id="other-law",
        law_name="中华人民共和国其他法",
        article_no="第一条",
        text="另一部法律的第一条。",
    )

    builder = ContextBuilder(
        chunks=[
            previous,
            matched,
            following,
            other_law,
        ],
        scorer=FakeScorer(),
        max_context_tokens=1000,
        max_evidences=3,
        max_evidence_tokens=200,
        adjacent_for_top_n=1,
    )

    context = builder.build(
        query="劳动合同有什么规定",
        results=[make_result(matched)],
    )

    assert [
        evidence.chunk_id
        for evidence in context.evidences
    ] == [
        "matched",
        "following",
        "previous",
    ]

    assert (
        context.evidences[0].relation
        == "matched"
    )
    assert all(
        evidence.relation == "adjacent"
        for evidence in context.evidences[1:]
    )

    assert all(
        evidence.law_id == "test-law"
        for evidence in context.evidences
    )


def test_context_never_exceeds_total_budget() -> None:
    first = make_chunk(
        chunk_id="chunk-1",
        article_no="第一条",
        text=(
            "第一条是一段相对较长但完整的法条正文。"
        ),
    )

    second = make_chunk(
        chunk_id="chunk-2",
        article_no="第二条",
        text=(
            "第二条也是一段相对较长但完整的法条正文。"
        ),
    )

    builder = ContextBuilder(
        chunks=[first, second],
        scorer=FakeScorer(),
        max_context_tokens=65,
        max_evidences=2,
        max_evidence_tokens=100,
        adjacent_for_top_n=0,
    )

    context = builder.build(
        query="测试总预算",
        results=[
            make_result(first),
            make_result(second),
        ],
    )

    assert (
        context.estimated_tokens
        <= 65
    )

    # 至少有一条因为总预算不足而被整体省略。
    assert context.truncated is True


def test_low_score_adjacent_article_is_excluded() -> None:
    previous = make_chunk(
        chunk_id="previous-low",
        article_no="第一条",
        text="这是完全无关的内容。",
    )

    matched = make_chunk(
        chunk_id="matched-main",
        article_no="第二条",
        text="这是直接命中的劳动合同规定。",
    )

    builder = ContextBuilder(
        chunks=[previous, matched],
        scorer=FakeScorer(),
        max_context_tokens=1000,
        max_evidences=3,
        max_evidence_tokens=200,
        adjacent_for_top_n=1,
        min_adjacent_score=0.35,
    )

    context = builder.build(
        query="劳动合同有什么规定",
        results=[make_result(matched)],
    )

    assert [
        evidence.chunk_id
        for evidence in context.evidences
    ] == ["matched-main"]
