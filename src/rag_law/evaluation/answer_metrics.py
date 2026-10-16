"""阶段 5 端到端评测：检索、回答行为、引用与延迟。"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Protocol

from rag_law.evaluation.retrieval_metrics import (
    DEFAULT_EVALUATION_PATH,
    EvaluationQuestion,
    load_evaluation_questions,
    ndcg_at_k,
    nearest_rank_percentile,
    recall_at_k,
    reciprocal_rank,
    sha256_file,
)
from rag_law.generation.answer_pipeline import (
    DEFAULT_MIN_ANSWER_SCORE,
    AnswerPipeline,
)
from rag_law.retrieval.common import (
    DEFAULT_ARTICLES_PATH,
    PROJECT_ROOT,
)
from rag_law.schemas import AnswerPipelineResult


DEFAULT_ANSWER_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "results"
    / "answer_quality_v1.json"
)
DEFAULT_ANSWER_MARKDOWN_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "results"
    / "answer_quality_v1.md"
)

RECALL_K_VALUES = (1, 3, 5, 10)


class AnswerPipelineProtocol(Protocol):
    """端到端评测器依赖的最小流水线接口。"""

    def run(
        self,
        query: str,
        *,
        law_name: str | None = None,
        progress_callback=None,
    ) -> AnswerPipelineResult:
        ...


def _average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def _ratio(
    numerator: int,
    denominator: int,
    *,
    empty: float = 0.0,
) -> float:
    if denominator == 0:
        return empty

    return numerator / denominator


def _citation_key(citation) -> tuple[str, str]:
    return citation.law_name, citation.article_no


def _is_faithful_citation(
    citation,
    result: AnswerPipelineResult,
) -> bool:
    evidences = {
        evidence.evidence_id: evidence
        for evidence in result.retrieval.context.evidences
    }
    evidence = evidences.get(citation.evidence_id)

    if evidence is None:
        return False

    return (
        citation.chunk_id == evidence.chunk_id
        and citation.law_id == evidence.law_id
        and citation.law_name == evidence.law_name
        and citation.article_no == evidence.article_no
        and citation.quote == evidence.text.strip()
    )


def evaluate_answer_result(
    *,
    question: EvaluationQuestion,
    result: AnswerPipelineResult,
) -> dict[str, object]:
    """把一次真实流水线结果转换成可聚合的逐题指标。"""

    expected = {
        citation.as_key()
        for citation in question.expected
    }
    retrieved = result.retrieval.reranked_results
    citations = result.answer.citations
    returned = {
        _citation_key(citation)
        for citation in citations
    }

    correct_citation_count = sum(
        1
        for citation in citations
        if _citation_key(citation) in expected
    )
    faithful_citation_count = sum(
        1
        for citation in citations
        if _is_faithful_citation(
            citation,
            result,
        )
    )

    expected_behavior = (
        question.resolved_expected_behavior()
    )
    behavior_correct = (
        result.answer.status == expected_behavior
    )

    retrieval_metrics: dict[str, float | None]

    if expected:
        retrieval_metrics = {
            f"recall_at_{k}": recall_at_k(
                retrieved,
                expected,
                k,
            )
            for k in RECALL_K_VALUES
        }
        retrieval_metrics.update(
            {
                "reciprocal_rank": reciprocal_rank(
                    retrieved,
                    expected,
                ),
                "ndcg_at_10": ndcg_at_k(
                    retrieved,
                    expected,
                    10,
                ),
            }
        )
    else:
        retrieval_metrics = {
            f"recall_at_{k}": None
            for k in RECALL_K_VALUES
        }
        retrieval_metrics.update(
            {
                "reciprocal_rank": None,
                "ndcg_at_10": None,
            }
        )

    return {
        "id": question.id,
        "split": question.split,
        "type": question.question_type,
        "question": question.question,
        "should_answer": question.should_answer,
        "expected_behavior": expected_behavior,
        "actual_behavior": result.answer.status,
        "behavior_correct": behavior_correct,
        "expected": [
            {
                "law": citation.law,
                "article": citation.article,
            }
            for citation in question.expected
        ],
        "expected_citation_count": len(expected),
        "returned_citation_count": len(citations),
        "correct_citation_count": (
            correct_citation_count
        ),
        "faithful_citation_count": (
            faithful_citation_count
        ),
        "citation_exact_match": (
            returned == expected
            if question.should_answer
            else not returned
        ),
        "retrieval": retrieval_metrics,
        "citations": [
            {
                "evidence_id": citation.evidence_id,
                "chunk_id": citation.chunk_id,
                "law": citation.law_name,
                "article": citation.article_no,
                "matches_expected": (
                    _citation_key(citation)
                    in expected
                ),
                "faithful_to_context": (
                    _is_faithful_citation(
                        citation,
                        result,
                    )
                ),
            }
            for citation in citations
        ],
        "generation_attempts": (
            result.generation_attempts
        ),
        "validation_error_count": len(
            result.validation_errors
        ),
        "top_relevance_score": (
            result.top_relevance_score
        ),
        "prompt_version": result.prompt_version,
        "latency_ms": {
            "retrieval": (
                result.retrieval.retrieval_latency_ms
            ),
            "context": (
                result.retrieval.context_latency_ms
            ),
            "generation": (
                result.generation_latency_ms
            ),
            "total": result.total_latency_ms,
        },
        "pipeline_error": None,
    }


def _error_record(
    question: EvaluationQuestion,
    error: Exception,
) -> dict[str, object]:
    expected_behavior = (
        question.resolved_expected_behavior()
    )

    return {
        "id": question.id,
        "split": question.split,
        "type": question.question_type,
        "question": question.question,
        "should_answer": question.should_answer,
        "expected_behavior": expected_behavior,
        "actual_behavior": "pipeline_error",
        "behavior_correct": False,
        "expected": [
            {
                "law": citation.law,
                "article": citation.article,
            }
            for citation in question.expected
        ],
        "expected_citation_count": len(
            {
                citation.as_key()
                for citation in question.expected
            }
        ),
        "returned_citation_count": 0,
        "correct_citation_count": 0,
        "faithful_citation_count": 0,
        "citation_exact_match": False,
        "retrieval": {
            **{
                f"recall_at_{k}": None
                for k in RECALL_K_VALUES
            },
            "reciprocal_rank": None,
            "ndcg_at_10": None,
        },
        "citations": [],
        "generation_attempts": 0,
        "validation_error_count": 0,
        "top_relevance_score": None,
        "prompt_version": None,
        "latency_ms": {
            "retrieval": None,
            "context": None,
            "generation": None,
            "total": None,
        },
        "pipeline_error": (
            f"{type(error).__name__}: {error}"
        ),
    }


def summarize_records(
    records: list[dict[str, object]],
) -> dict[str, object]:
    """聚合逐题记录；可用于 overall、split 和题型切片。"""

    if not records:
        return {"question_count": 0}

    answerable = [
        record
        for record in records
        if record["should_answer"]
    ]
    unanswerable = [
        record
        for record in records
        if not record["should_answer"]
    ]
    retrieval_records = [
        record
        for record in answerable
        if record["retrieval"]["recall_at_10"]
        is not None
    ]

    expected_citation_count = sum(
        int(record["expected_citation_count"])
        for record in answerable
    )
    returned_citation_count = sum(
        int(record["returned_citation_count"])
        for record in answerable
    )
    correct_citation_count = sum(
        int(record["correct_citation_count"])
        for record in answerable
    )
    all_returned_citation_count = sum(
        int(record["returned_citation_count"])
        for record in records
    )
    faithful_citation_count = sum(
        int(record["faithful_citation_count"])
        for record in records
    )

    total_latencies = [
        float(record["latency_ms"]["total"])
        for record in records
        if record["latency_ms"]["total"]
        is not None
    ]

    retrieval_summary = {
        "question_count": len(
            retrieval_records
        ),
        **{
            f"recall_at_{k}": _average(
                [
                    float(
                        record["retrieval"][
                            f"recall_at_{k}"
                        ]
                    )
                    for record in retrieval_records
                ]
            )
            for k in RECALL_K_VALUES
        },
        "mrr": _average(
            [
                float(
                    record["retrieval"][
                        "reciprocal_rank"
                    ]
                )
                for record in retrieval_records
            ]
        ),
        "ndcg_at_10": _average(
            [
                float(
                    record["retrieval"][
                        "ndcg_at_10"
                    ]
                )
                for record in retrieval_records
            ]
        ),
    }

    behavior_accuracy = _ratio(
        sum(
            bool(record["behavior_correct"])
            for record in records
        ),
        len(records),
    )
    unanswerable_accuracy = _ratio(
        sum(
            bool(record["behavior_correct"])
            for record in unanswerable
        ),
        len(unanswerable),
        empty=1.0,
    )

    return {
        "question_count": len(records),
        "answerable_question_count": len(
            answerable
        ),
        "unanswerable_question_count": len(
            unanswerable
        ),
        "pipeline_error_count": sum(
            record["pipeline_error"] is not None
            for record in records
        ),
        "behavior_accuracy": behavior_accuracy,
        "unanswerable_behavior_accuracy": (
            unanswerable_accuracy
        ),
        "answerable_behavior_accuracy": _ratio(
            sum(
                bool(record["behavior_correct"])
                for record in answerable
            ),
            len(answerable),
            empty=1.0,
        ),
        # expected 标注用于衡量核心法条覆盖；它不是穷尽式相关性标注，
        # 因此额外但忠实的辅助法条不应被判定为“伪造引用”。
        "expected_citation_precision": _ratio(
            correct_citation_count,
            returned_citation_count,
        ),
        "citation_precision": _ratio(
            faithful_citation_count,
            all_returned_citation_count,
            empty=1.0,
        ),
        "citation_recall": _ratio(
            correct_citation_count,
            expected_citation_count,
        ),
        "citation_exact_match_rate": _ratio(
            sum(
                bool(
                    record["citation_exact_match"]
                )
                for record in answerable
            ),
            len(answerable),
            empty=1.0,
        ),
        "evidence_faithfulness_rate": _ratio(
            faithful_citation_count,
            all_returned_citation_count,
            empty=1.0,
        ),
        "fabricated_citation_rate": _ratio(
            (
                all_returned_citation_count
                - faithful_citation_count
            ),
            all_returned_citation_count,
            empty=0.0,
        ),
        "retrieval": retrieval_summary,
        "latency_ms": {
            "mean": _average(total_latencies),
            "p50": (
                nearest_rank_percentile(
                    total_latencies,
                    0.50,
                )
                if total_latencies
                else 0.0
            ),
            "p95": (
                nearest_rank_percentile(
                    total_latencies,
                    0.95,
                )
                if total_latencies
                else 0.0
            ),
        },
    }


def build_quality_gates(
    summary: dict[str, object],
) -> dict[str, object]:
    retrieval = summary["retrieval"]

    gates = {
        "retrieval_recall_at_10": {
            "target": 0.90,
            "actual": retrieval["recall_at_10"],
            "met": retrieval["recall_at_10"] >= 0.90,
        },
        "citation_precision": {
            "target": 1.0,
            "actual": summary["citation_precision"],
            "met": summary["citation_precision"] == 1.0,
        },
        "fabricated_citation_rate": {
            "target": 0.0,
            "actual": summary[
                "fabricated_citation_rate"
            ],
            "met": (
                summary["fabricated_citation_rate"]
                == 0.0
            ),
        },
        "unanswerable_behavior_accuracy": {
            "target": 0.90,
            "actual": summary[
                "unanswerable_behavior_accuracy"
            ],
            "met": (
                summary[
                    "unanswerable_behavior_accuracy"
                ]
                >= 0.90
            ),
        },
        "pipeline_error_count": {
            "target": 0,
            "actual": summary[
                "pipeline_error_count"
            ],
            "met": summary[
                "pipeline_error_count"
            ] == 0,
        },
    }
    gates["all_met"] = all(
        gate["met"]
        for gate in gates.values()
    )

    return gates


def _write_json_atomic(
    path: Path,
    value: dict[str, object],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_path.replace(path)


def render_markdown_report(
    report: dict[str, object],
) -> str:
    """生成便于代码审查和人工验收的 Markdown 报告。"""

    metadata = report["metadata"]
    summary = report["summary"]
    gates = report["quality_gates"]
    gate_summary = summary["gate_scope"]

    lines = [
        "# RAG Law V1 端到端评测报告",
        "",
        f"- 生成时间：{metadata['generated_at']}",
        f"- 评测集：`{metadata['evaluation_path']}`",
        f"- 题目数量：{metadata['question_count']}",
        f"- 质量门槛口径：`{metadata['gate_scope']}`",
        f"- LLM：`{metadata['llm_model']}`",
        f"- Prompt：`{', '.join(metadata['prompt_versions']) or '-'}`",
        "",
        "## 核心指标",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        (
            "| Recall@10 | "
            f"{gate_summary['retrieval']['recall_at_10']:.3f} |"
        ),
        (
            "| MRR | "
            f"{gate_summary['retrieval']['mrr']:.3f} |"
        ),
        (
            "| nDCG@10 | "
            f"{gate_summary['retrieval']['ndcg_at_10']:.3f} |"
        ),
        (
            "| 引用精确率 | "
            f"{gate_summary['citation_precision']:.3f} |"
        ),
        (
            "| 核心法条标注精确率 | "
            f"{gate_summary['expected_citation_precision']:.3f} |"
        ),
        (
            "| 引用完整率 | "
            f"{gate_summary['citation_recall']:.3f} |"
        ),
        (
            "| 证据忠实度 | "
            f"{gate_summary['evidence_faithfulness_rate']:.3f} |"
        ),
        (
            "| 不可回答题行为准确率 | "
            f"{gate_summary['unanswerable_behavior_accuracy']:.3f} |"
        ),
        (
            "| P95 总延迟 | "
            f"{gate_summary['latency_ms']['p95']:.1f} ms |"
        ),
        "",
        "## 质量门槛",
        "",
        "| 门槛 | 目标 | 实际 | 通过 |",
        "|---|---:|---:|:---:|",
    ]

    for name, gate in gates.items():
        if name == "all_met":
            continue

        lines.append(
            f"| {name} | {gate['target']} | "
            f"{gate['actual']} | "
            f"{'是' if gate['met'] else '否'} |"
        )

    failures = [
        record
        for record in report["questions"]
        if (
            not record["behavior_correct"]
            or (
                record["should_answer"]
                and not record[
                    "citation_exact_match"
                ]
            )
            or record["pipeline_error"] is not None
        )
    ]
    lines.extend(
        [
            "",
            "## 待处理题目",
            "",
        ]
    )

    if not failures:
        lines.append("无。")
    else:
        lines.extend(
            [
                "| ID | Split | 题型 | 预期行为 | 实际行为 | 引用完全匹配 | 错误 |",
                "|---|---|---|---|---|:---:|---|",
            ]
        )

        for record in failures:
            error = (
                str(record["pipeline_error"] or "-")
                .replace("|", "\\|")
                .replace("\n", " ")
            )
            lines.append(
                f"| {record['id']} | {record['split']} | "
                f"{record['type']} | {record['expected_behavior']} | "
                f"{record['actual_behavior']} | "
                f"{'是' if record['citation_exact_match'] else '否'} | "
                f"{error} |"
            )

    lines.extend(
        [
            "",
            (
                "最终结论："
                + (
                    "全部质量门槛通过。"
                    if gates["all_met"]
                    else "仍有质量门槛未通过。"
                )
            ),
            "",
        ]
    )

    return "\n".join(lines)


def _write_text_atomic(
    path: Path,
    value: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = path.with_name(
        f".{path.name}.tmp"
    )
    temporary_path.write_text(
        value,
        encoding="utf-8",
        newline="\n",
    )
    temporary_path.replace(path)


def _filter_questions(
    questions: list[EvaluationQuestion],
    *,
    split: str,
    question_ids: set[str] | None,
    limit: int | None,
) -> list[EvaluationQuestion]:
    if split != "all":
        questions = [
            question
            for question in questions
            if question.split == split
        ]

    if question_ids:
        known_ids = {
            question.id
            for question in questions
        }
        missing_ids = question_ids - known_ids

        if missing_ids:
            raise ValueError(
                "评测题不存在或不属于所选 split："
                + ", ".join(sorted(missing_ids))
            )

        questions = [
            question
            for question in questions
            if question.id in question_ids
        ]

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")
        questions = questions[:limit]

    if not questions:
        raise ValueError("筛选后没有可评测题目")

    return questions


def _metadata(
    *,
    evaluation_path: Path,
    questions: list[EvaluationQuestion],
    records: list[dict[str, object]],
    min_answer_score: float,
    gate_scope: str,
) -> dict[str, object]:
    from rag_law.config import settings

    prompt_versions = sorted(
        {
            str(record["prompt_version"])
            for record in records
            if record["prompt_version"]
        }
    )
    split_counts = Counter(
        question.split
        for question in questions
    )

    return {
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "evaluation_path": str(
            evaluation_path
        ),
        "evaluation_sha256": sha256_file(
            evaluation_path
        ),
        "articles_path": str(
            DEFAULT_ARTICLES_PATH
        ),
        "articles_sha256": (
            sha256_file(DEFAULT_ARTICLES_PATH)
            if DEFAULT_ARTICLES_PATH.exists()
            else None
        ),
        "question_count": len(questions),
        "split_counts": dict(split_counts),
        "gate_scope": gate_scope,
        "llm_model": settings.llm_model,
        "embedding_model": (
            settings.embedding_model
        ),
        "rerank_model": settings.rerank_model,
        "prompt_versions": prompt_versions,
        "min_answer_score": min_answer_score,
        "report_schema_version": (
            "answer-quality-v1"
        ),
    }


def run_answer_evaluation(
    *,
    evaluation_path: Path = (
        DEFAULT_EVALUATION_PATH
    ),
    report_path: Path = (
        DEFAULT_ANSWER_REPORT_PATH
    ),
    markdown_path: Path = (
        DEFAULT_ANSWER_MARKDOWN_PATH
    ),
    min_answer_score: float = (
        DEFAULT_MIN_ANSWER_SCORE
    ),
    split: str = "all",
    question_ids: set[str] | None = None,
    limit: int | None = None,
    show_progress: bool = True,
    pipeline: AnswerPipelineProtocol | None = None,
) -> dict[str, object]:
    """运行一次端到端评测并同时写出 JSON 与 Markdown。"""

    if split not in {
        "all",
        "development",
        "holdout",
    }:
        raise ValueError(
            "split 必须是 all、development 或 holdout"
        )

    questions = _filter_questions(
        load_evaluation_questions(
            evaluation_path
        ),
        split=split,
        question_ids=question_ids,
        limit=limit,
    )

    if pipeline is None:
        pipeline = AnswerPipeline.load(
            min_answer_score=min_answer_score
        )

    records: list[dict[str, object]] = []

    for index, question in enumerate(
        questions,
        start=1,
    ):
        if show_progress:
            print(
                f"[answer] {index}/{len(questions)} "
                f"{question.id} ({question.split})"
            )

        try:
            result = pipeline.run(
                question.question
            )
            record = evaluate_answer_result(
                question=question,
                result=result,
            )
        except Exception as error:
            record = _error_record(
                question,
                error,
            )

        records.append(record)

    overall = summarize_records(records)
    by_split = {
        selected_split: summarize_records(
            [
                record
                for record in records
                if record["split"]
                == selected_split
            ]
        )
        for selected_split in (
            "development",
            "holdout",
        )
        if any(
            record["split"] == selected_split
            for record in records
        )
    }
    by_type = {
        question_type: summarize_records(
            [
                record
                for record in records
                if record["type"]
                == question_type
            ]
        )
        for question_type in sorted(
            {
                str(record["type"])
                for record in records
            }
        )
    }

    gate_scope = (
        "holdout"
        if "holdout" in by_split
        else "overall"
    )
    gate_summary = (
        by_split["holdout"]
        if gate_scope == "holdout"
        else overall
    )
    summary = {
        "overall": overall,
        "by_split": by_split,
        "by_type": by_type,
        "gate_scope": gate_summary,
    }
    quality_gates = build_quality_gates(
        gate_summary
    )

    report: dict[str, object] = {
        "metadata": _metadata(
            evaluation_path=evaluation_path,
            questions=questions,
            records=records,
            min_answer_score=min_answer_score,
            gate_scope=gate_scope,
        ),
        "summary": summary,
        "quality_gates": quality_gates,
        "questions": records,
    }

    _write_json_atomic(report_path, report)
    _write_text_atomic(
        markdown_path,
        render_markdown_report(report),
    )

    if show_progress:
        print()
        print(
            "阶段 5 端到端评测完成："
            f"{len(records)} 题，"
            f"质量门槛口径={gate_scope}，"
            f"全部通过={quality_gates['all_met']}"
        )
        print(f"JSON 报告：{report_path}")
        print(f"Markdown 报告：{markdown_path}")

    return report
