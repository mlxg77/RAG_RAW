from pathlib import Path

from rag_law.evaluation.answer_metrics import (
    evaluate_answer_result,
    run_answer_evaluation,
)
from rag_law.evaluation.retrieval_metrics import (
    EvaluationQuestion,
    ExpectedCitation,
    ndcg_at_k,
)
from rag_law.schemas import (
    AnswerPipelineResult,
    AnswerResponse,
    BuiltContext,
    ContextEvidence,
    RetrievalContextResult,
    RetrievalResult,
    ValidatedCitation,
)


LAW_NAME = "中华人民共和国测试法"


def make_result(
    *,
    status: str = "answered",
    include_citation: bool = True,
) -> AnswerPipelineResult:
    retrieval_result = RetrievalResult(
        chunk_id="chunk-1",
        law_id="cn_test_law",
        law_name=LAW_NAME,
        article_no="第一条",
        text="第一条　测试规定。",
        source_file="data/raw/test.md",
        score=0.9,
        source="rerank",
    )
    evidence = ContextEvidence(
        evidence_id="E001",
        chunk_id="chunk-1",
        law_id="cn_test_law",
        law_name=LAW_NAME,
        article_no="第一条",
        text="第一条　测试规定。",
        source_file="data/raw/test.md",
        relation="matched",
        is_excerpt=False,
        score=0.9,
    )
    citations = []

    if include_citation:
        citations.append(
            ValidatedCitation(
                evidence_id="E001",
                chunk_id="chunk-1",
                law_id="cn_test_law",
                law_name=LAW_NAME,
                article_no="第一条",
                quote="第一条　测试规定。",
                source_file="data/raw/test.md",
            )
        )

    retrieval = RetrievalContextResult(
        query="测试问题",
        reranked_results=[retrieval_result],
        context=BuiltContext(
            text="[E001] 第一条　测试规定。",
            evidences=[evidence],
            estimated_tokens=10,
            max_tokens=100,
            truncated=False,
        ),
        retrieval_latency_ms=10,
        context_latency_ms=2,
        total_latency_ms=12,
    )

    return AnswerPipelineResult(
        query="测试问题",
        answer=AnswerResponse(
            status=status,
            summary="测试结论",
            analysis="测试分析",
            citations=citations,
            limitations="测试限制",
            follow_up_question=(
                "还需要什么事实？"
                if status == "needs_clarification"
                else None
            ),
        ),
        retrieval=retrieval,
        generation_attempts=1,
        top_relevance_score=0.9,
        prompt_version="test-v1",
        generation_latency_ms=20,
        total_latency_ms=32,
    )


def test_ndcg_at_k_uses_binary_relevance() -> None:
    irrelevant = RetrievalResult(
        chunk_id="other",
        law_id="other",
        law_name="其他法",
        article_no="第二条",
        text="其他",
        source_file="other.md",
        score=1,
        source="rerank",
    )
    relevant = make_result().retrieval.reranked_results[0]

    value = ndcg_at_k(
        [irrelevant, relevant],
        {(LAW_NAME, "第一条")},
        10,
    )

    assert round(value, 6) == 0.63093


def test_answer_result_scores_behavior_and_citations() -> None:
    question = EvaluationQuestion(
        id="q001",
        type="直接查询",
        question="测试问题",
        should_answer=True,
        expected=[
            ExpectedCitation(
                law=LAW_NAME,
                article="第一条",
            )
        ],
        split="holdout",
    )

    record = evaluate_answer_result(
        question=question,
        result=make_result(),
    )

    assert record["behavior_correct"] is True
    assert record["citation_exact_match"] is True
    assert record["faithful_citation_count"] == 1
    assert record["retrieval"]["recall_at_10"] == 1.0
    assert record["retrieval"]["ndcg_at_10"] == 1.0


class StubPipeline:
    def run(self, query: str, **kwargs):
        if "事实" in query:
            return make_result(
                status="needs_clarification",
                include_citation=False,
            )

        return make_result()


def test_run_answer_evaluation_writes_both_reports(
    tmp_path: Path,
) -> None:
    evaluation_path = tmp_path / "questions.yaml"
    evaluation_path.write_text(
        """
- id: q001
  type: 直接查询
  question: 测试规定是什么？
  should_answer: true
  split: holdout
  expected:
    - law: 中华人民共和国测试法
      article: 第一条
- id: q002
  type: 事实不足
  question: 事实不足怎么办？
  should_answer: false
  split: holdout
  expected_behavior: needs_clarification
  expected: []
""".strip(),
        encoding="utf-8",
    )
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    report = run_answer_evaluation(
        evaluation_path=evaluation_path,
        report_path=json_path,
        markdown_path=markdown_path,
        show_progress=False,
        pipeline=StubPipeline(),
    )

    summary = report["summary"]["gate_scope"]

    assert summary["retrieval"]["recall_at_10"] == 1.0
    assert summary["citation_precision"] == 1.0
    assert summary["unanswerable_behavior_accuracy"] == 1.0
    assert report["quality_gates"]["all_met"] is True
    assert json_path.exists()
    assert markdown_path.exists()
    assert "端到端评测报告" in markdown_path.read_text(
        encoding="utf-8"
    )
