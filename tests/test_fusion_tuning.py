from rag_law.evaluation.fusion_tuning import (
    CachedQuestion,
    evaluate_fusion_configuration,
    summarize_rankings,
)
from rag_law.retrieval.common import (
    make_retrieval_result,
)
from rag_law.schemas import ArticleChunk

from rag_law.schemas import RetrievalResult

def make_result(
    *,
    chunk_id: str,
    article_no: str,
    source: str,
) -> RetrievalResult:
    chunk = ArticleChunk(
        chunk_id=chunk_id,
        law_id="test-law",
        law_name="中华人民共和国测试法",
        article_no=article_no,
        text=f"{article_no}正文",
        source_file="data/raw/test.md",
        text_hash=f"hash-{chunk_id}",
    )

    return make_retrieval_result(
        chunk,
        score=1.0,
        source=source,
        component_scores={
            source: 1.0,
        },
    )


def test_summarize_rankings_handles_two_expected() -> None:
    cached = CachedQuestion(
        question_id="q001",
        question="测试问题",
        expected={
            (
                "中华人民共和国测试法",
                "第一条",
            ),
            (
                "中华人民共和国测试法",
                "第二条",
            ),
        },
        bm25_results=[],
        dense_results=[],
    )

    results = [
        make_result(
            chunk_id="one",
            article_no="第一条",
            source="dense",
        ),
        make_result(
            chunk_id="other",
            article_no="第三条",
            source="dense",
        ),
    ]

    summary = summarize_rankings(
        [(cached, results)]
    )

    assert summary["recall_at_1"] == 0.5
    assert summary["recall_at_10"] == 0.5
    assert summary["mrr"] == 1.0


def test_dense_weight_can_preserve_dense_result() -> None:
    relevant_bm25 = make_result(
        chunk_id="relevant",
        article_no="第一条",
        source="bm25",
    )
    irrelevant_bm25 = make_result(
        chunk_id="irrelevant-bm25",
        article_no="第二条",
        source="bm25",
    )

    relevant_dense = make_result(
        chunk_id="relevant",
        article_no="第一条",
        source="dense",
    )
    irrelevant_dense = make_result(
        chunk_id="irrelevant-dense",
        article_no="第三条",
        source="dense",
    )

    cached = CachedQuestion(
        question_id="q001",
        question="测试问题",
        expected={
            (
                "中华人民共和国测试法",
                "第一条",
            )
        },
        bm25_results=[
            irrelevant_bm25,
            relevant_bm25,
        ],
        dense_results=[
            relevant_dense,
            irrelevant_dense,
        ],
    )

    result = evaluate_fusion_configuration(
        [cached],
        candidate_top_k=2,
        rrf_k=60,
        bm25_weight=1.0,
        dense_weight=2.0,
    )

    assert result["recall_at_1"] == 1.0
    assert result["recall_at_10"] == 1.0