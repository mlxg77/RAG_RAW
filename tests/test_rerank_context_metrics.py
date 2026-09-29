from rag_law.evaluation.rerank_context_metrics import (
    audit_context,
    evaluate_stage3,
    top_k_average_relevance,
)
from rag_law.evaluation.retrieval_metrics import (
    EvaluationQuestion,
    ExpectedCitation,
)
from rag_law.retrieval.common import (
    make_retrieval_result,
)
from rag_law.retrieval.context_builder import (
    ContextBuilder,
)
from rag_law.schemas import (
    ArticleChunk,
    BuiltContext,
    ContextEvidence,
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


def make_result(
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
        return [
            0.99 if "目标法条" in document
            else 0.10
            for document in documents
        ]


def test_top3_average_relevance_uses_fixed_denominator() -> None:
    chunks = [
        make_chunk(
            chunk_id=f"chunk-{index}",
            article_no=f"第{index}条",
            text=f"正文 {index}",
        )
        for index in range(1, 4)
    ]
    results = [
        make_result(chunk, score=1.0)
        for chunk in chunks
    ]

    relevance = top_k_average_relevance(
        results,
        {
            (
                "中华人民共和国测试法",
                "第2条",
            )
        },
    )

    assert relevance == 1.0 / 3.0


def test_stage3_evaluation_detects_rerank_improvement() -> None:
    chunks = [
        make_chunk(
            chunk_id=f"chunk-{index}",
            article_no=f"第{index}条",
            text=(
                "这是目标法条。"
                if index == 6
                else f"无关正文 {index}。"
            ),
        )
        for index in range(1, 7)
    ]
    hybrid_results = [
        make_result(
            chunk,
            score=1.0 / index,
        )
        for index, chunk in enumerate(
            chunks,
            start=1,
        )
    ]
    question = EvaluationQuestion(
        id="q001",
        type="测试",
        question="目标法条是什么？",
        should_answer=True,
        expected=[
            ExpectedCitation(
                law="中华人民共和国测试法",
                article="第6条",
            )
        ],
    )
    scorer = FakeScorer()
    builder = ContextBuilder(
        chunks=chunks,
        scorer=scorer,
        adjacent_for_top_n=0,
    )

    report = evaluate_stage3(
        questions=[question],
        hybrid_results_by_question={
            "q001": hybrid_results
        },
        hybrid_latency_by_question={
            "q001": 10.0
        },
        scorer=scorer,
        context_builder=builder,
        chunks=chunks,
        candidate_top_k=6,
        context_top_n=5,
    )

    assert (
        report["methods"]["hybrid"][
            "recall_at_5"
        ]
        == 0.0
    )
    assert (
        report["methods"]["rerank"][
            "recall_at_5"
        ]
        == 1.0
    )
    assert (
        report["quality_gates"][
            "recall_at_5_improved"
        ]
        is True
    )
    assert (
        report["quality_gates"][
            "top3_average_relevance_improved"
        ]
        is True
    )
    assert (
        report["quality_gates"][
            "all_passed"
        ]
        is True
    )


def test_context_audit_detects_mapping_error() -> None:
    chunk = make_chunk(
        chunk_id="chunk-1",
        article_no="第一条",
        text="完整的原始正文。",
    )
    context = BuiltContext(
        text="",
        evidences=[
            ContextEvidence(
                evidence_id="E001",
                chunk_id="missing",
                law_id="test-law",
                law_name="中华人民共和国测试法",
                article_no="第一条",
                text="原始正文。",
                source_file="data/raw/test.md",
                relation="matched",
                is_excerpt=True,
                score=0.9,
            )
        ],
        estimated_tokens=0,
        max_tokens=100,
        truncated=False,
    )

    audit = audit_context(
        context,
        chunks_by_id={
            chunk.chunk_id: chunk
        },
    )

    assert audit["unmapped_evidence_count"] == 1
