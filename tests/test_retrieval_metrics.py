from rag_law.evaluation.retrieval_metrics import (
    EvaluationQuestion,
    ExpectedCitation,
    evaluate_retriever,
    nearest_rank_percentile,
    recall_at_k,
    reciprocal_rank,
)
from rag_law.schemas import RetrievalResult


def make_result(
    law_name: str,
    article_no: str,
    *,
    rank_score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{law_name}-{article_no}",
        law_id="test-law",
        law_name=law_name,
        article_no=article_no,
        text="测试正文",
        source_file="data/raw/test.md",
        score=rank_score,
        source="bm25",
    )


def test_recall_at_k_supports_multiple_expected() -> None:
    expected = {
        ("测试法", "第一条"),
        ("测试法", "第二条"),
    }

    results = [
        make_result("测试法", "第一条"),
        make_result("其他法", "第三条"),
        make_result("测试法", "第二条"),
    ]

    assert recall_at_k(
        results,
        expected,
        1,
    ) == 0.5

    assert recall_at_k(
        results,
        expected,
        3,
    ) == 1.0


def test_reciprocal_rank_uses_first_relevant() -> None:
    expected = {
        ("测试法", "第二条"),
    }

    results = [
        make_result("其他法", "第一条"),
        make_result("测试法", "第二条"),
        make_result("测试法", "第三条"),
    ]

    assert reciprocal_rank(
        results,
        expected,
    ) == 0.5


def test_reciprocal_rank_returns_zero_for_miss() -> None:
    results = [
        make_result("其他法", "第一条"),
    ]

    assert reciprocal_rank(
        results,
        {("测试法", "第二条")},
    ) == 0.0


def test_nearest_rank_percentile() -> None:
    values = [
        float(value)
        for value in range(1, 21)
    ]

    assert nearest_rank_percentile(
        values,
        0.50,
    ) == 10.0

    assert nearest_rank_percentile(
        values,
        0.95,
    ) == 19.0


class StubRetriever:
    def __init__(self) -> None:
        self.call_count = 0

    def search(
        self,
        query: str,
        *,
        top_k: int,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        self.call_count += 1

        return [
            make_result(
                "测试法",
                "第一条",
            )
        ]


def test_evaluation_excludes_unanswerable_questions() -> None:
    answerable = EvaluationQuestion(
        id="q001",
        type="直接查询",
        question="第一条是什么？",
        should_answer=True,
        expected=[
            ExpectedCitation(
                law="测试法",
                article="第一条",
            )
        ],
    )

    unanswerable = EvaluationQuestion(
        id="q002",
        type="超范围拒答",
        question="范围外问题",
        should_answer=False,
        expected=[],
    )

    retriever = StubRetriever()

    report = evaluate_retriever(
        method_name="stub",
        retriever=retriever,
        questions=[
            answerable,
            unanswerable,
        ],
        max_k=10,
    )

    summary = report["summary"]

    assert retriever.call_count == 1
    assert summary["question_count"] == 1
    assert (
        summary["excluded_question_count"]
        == 1
    )
    assert summary["recall_at_10"] == 1.0
    assert summary["mrr"] == 1.0