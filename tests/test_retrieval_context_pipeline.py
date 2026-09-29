from rag_law.retrieval.common import (
    make_retrieval_result,
)
from rag_law.retrieval.context_builder import (
    ContextBuilder,
)
from rag_law.retrieval.reranker import (
    RerankedRetriever,
)
from rag_law.retrieval.retrieval_context_pipeline import (
    RetrievalContextPipeline,
)
from rag_law.schemas import (
    ArticleChunk,
    RetrievalResult,
)


def make_chunk(
    *,
    chunk_id: str,
    article_no: str,
    text: str,
) -> ArticleChunk:
    return ArticleChunk(
        chunk_id=chunk_id,
        law_id="test-law",
        law_name="中华人民共和国测试法",
        article_no=article_no,
        text=text,
        source_file="data/raw/test.md",
        text_hash=f"hash-{chunk_id}",
    )


def make_hybrid_result(
    chunk: ArticleChunk,
    *,
    score: float,
) -> RetrievalResult:
    return make_retrieval_result(
        chunk,
        score=score,
        source="hybrid",
        component_scores={"rrf": score},
    )


class FakeScorer:
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
            elif "书面劳动合同" in document:
                scores.append(0.80)
            else:
                scores.append(0.10)

        return scores


class StubHybridRetriever:
    def __init__(
        self,
        results: list[RetrievalResult],
    ) -> None:
        self.results = results
        self.calls: list[
            tuple[str, int, str | None]
        ] = []

    def search(
        self,
        query: str,
        *,
        top_k: int,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append(
            (query, top_k, law_name)
        )
        return self.results[:top_k]


def test_pipeline_connects_rerank_and_context() -> None:
    irrelevant = make_chunk(
        chunk_id="irrelevant",
        article_no="第一条",
        text="这是与问题无关的规定。",
    )
    contract = make_chunk(
        chunk_id="contract",
        article_no="第十条",
        text="建立劳动关系，应当订立书面劳动合同。",
    )
    compensation = make_chunk(
        chunk_id="compensation",
        article_no="第八十二条",
        text="未订立书面劳动合同的，应当支付二倍工资。",
    )

    hybrid = StubHybridRetriever(
        [
            make_hybrid_result(
                irrelevant,
                score=0.05,
            ),
            make_hybrid_result(
                contract,
                score=0.04,
            ),
            make_hybrid_result(
                compensation,
                score=0.03,
            ),
        ]
    )

    scorer = FakeScorer()
    reranked = RerankedRetriever(
        base_retriever=hybrid,
        scorer=scorer,
        candidate_top_k=3,
    )
    context_builder = ContextBuilder(
        chunks=[
            irrelevant,
            contract,
            compensation,
        ],
        scorer=scorer,
        max_context_tokens=1000,
        max_evidences=2,
        max_evidence_tokens=200,
        adjacent_for_top_n=0,
        min_adjacent_score=0.35,
    )
    pipeline = RetrievalContextPipeline(
        retriever=reranked,
        context_builder=context_builder,
        rerank_top_n=2,
    )

    output = pipeline.run(
        "公司一直没有和我签书面劳动合同",
        law_name="中华人民共和国测试法",
    )

    assert hybrid.calls == [
        (
            "公司一直没有和我签书面劳动合同",
            3,
            "中华人民共和国测试法",
        )
    ]
    assert [
        result.chunk_id
        for result in output.reranked_results
    ] == ["compensation", "contract"]
    assert [
        evidence.chunk_id
        for evidence in output.context.evidences
    ] == ["compensation", "contract"]
    assert (
        output.context.evidences[0].evidence_id
        == "E001"
    )
    assert "二倍工资" in output.context.text
    assert (
        output.context.estimated_tokens
        <= output.context.max_tokens
    )
    assert output.retrieval_latency_ms >= 0
    assert output.context_latency_ms >= 0
    assert output.total_latency_ms >= 0


def test_pipeline_rejects_empty_query() -> None:
    chunk = make_chunk(
        chunk_id="chunk-1",
        article_no="第一条",
        text="测试正文。",
    )
    hybrid = StubHybridRetriever(
        [make_hybrid_result(chunk, score=0.05)]
    )
    scorer = FakeScorer()
    pipeline = RetrievalContextPipeline(
        retriever=RerankedRetriever(
            base_retriever=hybrid,
            scorer=scorer,
            candidate_top_k=1,
        ),
        context_builder=ContextBuilder(
            chunks=[chunk],
            scorer=scorer,
            adjacent_for_top_n=0,
            min_adjacent_score=0.35,
        ),
        rerank_top_n=1,
    )

    try:
        pipeline.run("   ")
    except ValueError as error:
        assert "不能为空" in str(error)
    else:
        raise AssertionError(
            "空查询应当抛出 ValueError"
        )


def test_pipeline_supports_empty_results() -> None:
    chunk = make_chunk(
        chunk_id="chunk-1",
        article_no="第一条",
        text="测试正文。",
    )
    hybrid = StubHybridRetriever([])
    scorer = FakeScorer()
    pipeline = RetrievalContextPipeline(
        retriever=RerankedRetriever(
            base_retriever=hybrid,
            scorer=scorer,
            candidate_top_k=1,
        ),
        context_builder=ContextBuilder(
            chunks=[chunk],
            scorer=scorer,
            adjacent_for_top_n=0,
            min_adjacent_score=0.35,
        ),
        rerank_top_n=1,
    )

    output = pipeline.run(
        "没有检索结果的问题"
    )

    assert output.reranked_results == []
    assert output.context.evidences == []
    assert output.context.text == ""
