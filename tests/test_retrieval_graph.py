import time

from rag_law.retrieval.common import (
    make_retrieval_result,
)
from rag_law.retrieval.context_builder import (
    ContextBuilder,
)
from rag_law.retrieval.retrieval_context_pipeline import (
    RetrievalContextPipeline,
)
from rag_law.retrieval.retrieval_graph import (
    build_retrieval_graph,
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


class RecordingRetriever:
    """按固定结果返回，并记录每次调用的检索替身。"""

    def __init__(
        self,
        results: list[RetrievalResult],
        *,
        source: str,
    ) -> None:
        self.results = results
        self.source = source
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


def make_results(
    chunk: ArticleChunk,
    *,
    source: str,
    score: float,
) -> RetrievalResult:
    return make_retrieval_result(
        chunk,
        score=score,
        source=source,
        component_scores={source: score},
    )


def make_chunks() -> tuple[
    ArticleChunk,
    ArticleChunk,
    ArticleChunk,
]:
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
        text=(
            "未订立书面劳动合同的，"
            "应当支付二倍工资。"
        ),
    )
    return irrelevant, contract, compensation


def make_state(
    *,
    query: str,
    candidate_top_k: int,
) -> dict[str, object]:
    return {
        "query": query,
        "law_name": "中华人民共和国测试法",
        "candidate_top_k": candidate_top_k,
        "rrf_k": 60,
        "bm25_weight": 1.0,
        "dense_weight": 2.0,
        "rerank_top_n": 2,
        "retrieval_started_at": time.perf_counter(),
        "progress_callback": None,
    }


def test_graph_parallel_recall_fuses_and_reranks() -> None:
    irrelevant, contract, compensation = (
        make_chunks()
    )

    bm25 = RecordingRetriever(
        [
            make_results(
                irrelevant, source="bm25", score=0.5
            ),
            make_results(
                contract, source="bm25", score=0.4
            ),
            make_results(
                compensation,
                source="bm25",
                score=0.3,
            ),
        ],
        source="bm25",
    )
    dense = RecordingRetriever(
        [
            make_results(
                contract, source="dense", score=0.6
            ),
            make_results(
                compensation,
                source="dense",
                score=0.5,
            ),
            make_results(
                irrelevant,
                source="dense",
                score=0.1,
            ),
        ],
        source="dense",
    )
    scorer = FakeScorer()

    graph = build_retrieval_graph(
        bm25_retriever=bm25,
        dense_retriever=dense,
        scorer=scorer,
        context_builder=ContextBuilder(
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
        ),
        candidate_top_k=3,
        rrf_k=60,
        bm25_weight=1.0,
        dense_weight=2.0,
        rerank_top_n=2,
    )

    query = "公司一直没有和我签书面劳动合同"
    output = graph.invoke(
        make_state(
            query=query,
            candidate_top_k=3,
        )
    )

    assert bm25.calls == [
        (query, 3, "中华人民共和国测试法")
    ]
    assert dense.calls == [
        (query, 3, "中华人民共和国测试法")
    ]

    assert [
        result.chunk_id
        for result in output["reranked_results"]
    ] == ["compensation", "contract"]

    assert [
        evidence.chunk_id
        for evidence in output["context"].evidences
    ] == ["compensation", "contract"]

    assert "二倍工资" in output["context"].text
    assert output["retrieval_latency_ms"] >= 0
    assert output["context_latency_ms"] >= 0


def test_graph_supports_empty_recall() -> None:
    irrelevant, _, _ = make_chunks()
    scorer = FakeScorer()

    graph = build_retrieval_graph(
        bm25_retriever=RecordingRetriever(
            [], source="bm25"
        ),
        dense_retriever=RecordingRetriever(
            [], source="dense"
        ),
        scorer=scorer,
        context_builder=ContextBuilder(
            chunks=[irrelevant],
            scorer=scorer,
            adjacent_for_top_n=0,
            min_adjacent_score=0.35,
        ),
        candidate_top_k=3,
        rrf_k=60,
        bm25_weight=1.0,
        dense_weight=2.0,
        rerank_top_n=2,
    )

    output = graph.invoke(
        make_state(
            query="没有检索结果的问题",
            candidate_top_k=3,
        )
    )

    assert output["reranked_results"] == []
    assert output["context"].evidences == []
    assert output["context"].text == ""


def test_pipeline_uses_parallel_graph_in_load_path() -> None:
    irrelevant, contract, compensation = (
        make_chunks()
    )

    bm25 = RecordingRetriever(
        [
            make_results(
                contract, source="bm25", score=0.5
            ),
            make_results(
                compensation,
                source="bm25",
                score=0.4,
            ),
        ],
        source="bm25",
    )
    dense = RecordingRetriever(
        [
            make_results(
                compensation,
                source="dense",
                score=0.6,
            ),
            make_results(
                contract, source="dense", score=0.5
            ),
        ],
        source="dense",
    )
    scorer = FakeScorer()

    pipeline = RetrievalContextPipeline(
        context_builder=ContextBuilder(
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
        ),
        rerank_top_n=2,
        bm25_retriever=bm25,
        dense_retriever=dense,
        scorer=scorer,
        candidate_top_k=3,
    )

    events: list[str] = []
    output = pipeline.run(
        "公司一直没有和我签书面劳动合同",
        law_name="中华人民共和国测试法",
        progress_callback=(
            lambda stage, message: events.append(
                stage
            )
        ),
    )

    assert events == [
        "retrieval",
        "context_building",
    ]
    assert len(bm25.calls) == 1
    assert len(dense.calls) == 1
    assert len(output.reranked_results) == 2
    assert output.total_latency_ms >= 0
    assert output.retrieval_latency_ms >= 0
    assert output.context_latency_ms >= 0
