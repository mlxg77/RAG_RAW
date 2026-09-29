"""阶段 3 评测：Rerank 排序质量与上下文结构质量。"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from rag_law.evaluation.retrieval_metrics import (
    DEFAULT_EVALUATION_PATH,
    EvaluationQuestion,
    load_evaluation_questions,
    nearest_rank_percentile,
    package_version,
    recall_at_k,
    reciprocal_rank,
    result_key,
    sha256_file,
)
from rag_law.retrieval.common import (
    DEFAULT_ARTICLES_PATH,
    PROJECT_ROOT,
    load_articles,
    make_retrieval_result,
)
from rag_law.retrieval.context_builder import (
    DEFAULT_ADJACENT_FOR_TOP_N,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_EVIDENCES,
    DEFAULT_MAX_EVIDENCE_TOKENS,
    DEFAULT_MIN_ADJACENT_SCORE,
    ContextBuilder,
    normalize_body,
    split_text_spans,
)
from rag_law.retrieval.reranker import (
    RerankScorerProtocol,
    SiliconFlowReranker,
    rerank_results,
)
from rag_law.schemas import (
    ArticleChunk,
    BuiltContext,
    RetrievalResult,
)


DEFAULT_HYBRID_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "results"
    / "retrieval_rebuilt_v1.json"
)

DEFAULT_STAGE3_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "results"
    / "rerank_context_v1.json"
)

DEFAULT_CANDIDATE_TOP_K = 10
DEFAULT_CONTEXT_TOP_N = 5
TOP_RELEVANCE_K = 3


def _average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def top_k_average_relevance(
    results: list[RetrievalResult],
    expected: set[tuple[str, str]],
    *,
    k: int = TOP_RELEVANCE_K,
) -> float:
    """计算前 K 条的二元平均相关性。

    一条结果的法律名和条号命中 expected 时记为 1，否则为 0；
    固定除以 K，使不同问题之间可直接求平均。
    """

    if not expected:
        raise ValueError(
            "计算平均相关性时 expected 不能为空"
        )

    if k <= 0:
        raise ValueError("k 必须大于 0")

    relevant_count = sum(
        1
        for result in results[:k]
        if result_key(result) in expected
    )

    return relevant_count / k


def load_cached_hybrid_results(
    *,
    report_path: Path,
    articles_path: Path,
    chunks: list[ArticleChunk],
) -> tuple[
    dict[str, list[RetrievalResult]],
    dict[str, float],
    dict[str, object],
]:
    """从阶段 2 报告恢复 Hybrid Top10 候选。"""

    if not report_path.exists():
        raise FileNotFoundError(
            f"阶段 2 报告不存在：{report_path}"
        )

    with report_path.open(
        encoding="utf-8"
    ) as file:
        report = json.load(file)

    metadata = report.get("metadata")

    if not isinstance(metadata, dict):
        raise ValueError(
            "阶段 2 报告缺少 metadata"
        )

    actual_articles_hash = sha256_file(
        articles_path
    )

    if (
        metadata.get("articles_sha256")
        != actual_articles_hash
    ):
        raise ValueError(
            "阶段 2 报告与当前 articles.jsonl "
            "数据哈希不一致"
        )

    methods = report.get("methods")

    if not isinstance(methods, dict):
        raise ValueError(
            "阶段 2 报告缺少 methods"
        )

    hybrid = methods.get("hybrid")

    if not isinstance(hybrid, dict):
        raise ValueError(
            "阶段 2 报告缺少 hybrid 结果"
        )

    raw_questions = hybrid.get("questions")

    if not isinstance(raw_questions, list):
        raise ValueError(
            "阶段 2 Hybrid 报告缺少逐题结果"
        )

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    results_by_question: dict[
        str,
        list[RetrievalResult],
    ] = {}
    latency_by_question: dict[str, float] = {}

    for raw_question in raw_questions:
        if not isinstance(raw_question, dict):
            raise ValueError(
                "阶段 2 逐题结果格式错误"
            )

        question_id = raw_question.get("id")

        if not isinstance(question_id, str):
            raise ValueError(
                "阶段 2 逐题结果缺少 id"
            )

        raw_retrieved = raw_question.get(
            "retrieved"
        )

        if not isinstance(raw_retrieved, list):
            raise ValueError(
                f"{question_id} 缺少 retrieved"
            )

        restored_results: list[
            RetrievalResult
        ] = []

        for raw_result in raw_retrieved:
            if not isinstance(raw_result, dict):
                raise ValueError(
                    f"{question_id} retrieved 格式错误"
                )

            chunk_id = raw_result.get(
                "chunk_id"
            )
            chunk = chunks_by_id.get(chunk_id)

            if chunk is None:
                raise ValueError(
                    f"{question_id} 的候选无法映射："
                    f"{chunk_id}"
                )

            if (
                raw_result.get("law")
                != chunk.law_name
                or raw_result.get("article")
                != chunk.article_no
            ):
                raise ValueError(
                    f"{question_id} 的候选元数据冲突："
                    f"{chunk_id}"
                )

            restored_results.append(
                make_retrieval_result(
                    chunk,
                    score=float(
                        raw_result["score"]
                    ),
                    source="hybrid",
                    component_scores={
                        "rrf": float(
                            raw_result["score"]
                        )
                    },
                )
            )

        results_by_question[question_id] = (
            restored_results
        )
        latency_by_question[question_id] = float(
            raw_question.get("latency_ms", 0.0)
        )

    return (
        results_by_question,
        latency_by_question,
        metadata,
    )


def _excerpt_has_valid_boundaries(
    evidence_text: str,
    original_text: str,
) -> bool:
    """确认片段起止点都落在自然语义单元边界。"""

    start = original_text.find(evidence_text)

    if start < 0:
        return False

    end = start + len(evidence_text)
    spans = split_text_spans(original_text)

    valid_starts = {
        span_start
        for span_start, _ in spans
    }
    valid_ends = {
        span_end
        for _, span_end in spans
    }

    return (
        start in valid_starts
        and end in valid_ends
    )


def audit_context(
    context: BuiltContext,
    *,
    chunks_by_id: dict[str, ArticleChunk],
) -> dict[str, int]:
    """检查上下文的去重、映射、预算和片段边界。"""

    duplicate_article_count = 0
    duplicate_body_count = 0
    unmapped_evidence_count = 0
    metadata_mismatch_count = 0
    non_contiguous_excerpt_count = 0
    excerpt_boundary_violation_count = 0
    evidence_id_sequence_violation_count = 0

    seen_articles: set[tuple[str, str]] = set()
    seen_bodies: set[str] = set()

    for index, evidence in enumerate(
        context.evidences,
        start=1,
    ):
        if evidence.evidence_id != f"E{index:03d}":
            evidence_id_sequence_violation_count += 1

        article_key = (
            evidence.law_id,
            evidence.article_no,
        )
        body_key = normalize_body(evidence.text)

        if article_key in seen_articles:
            duplicate_article_count += 1
        else:
            seen_articles.add(article_key)

        if body_key in seen_bodies:
            duplicate_body_count += 1
        else:
            seen_bodies.add(body_key)

        original = chunks_by_id.get(
            evidence.chunk_id
        )

        if original is None:
            unmapped_evidence_count += 1
            continue

        if (
            evidence.law_id != original.law_id
            or evidence.law_name != original.law_name
            or evidence.article_no
            != original.article_no
            or evidence.source_file
            != original.source_file
        ):
            metadata_mismatch_count += 1

        if evidence.text not in original.text:
            non_contiguous_excerpt_count += 1
            continue

        if (
            evidence.is_excerpt
            and not _excerpt_has_valid_boundaries(
                evidence.text,
                original.text,
            )
        ):
            excerpt_boundary_violation_count += 1

    return {
        "duplicate_article_count": (
            duplicate_article_count
        ),
        "duplicate_body_count": (
            duplicate_body_count
        ),
        "unmapped_evidence_count": (
            unmapped_evidence_count
        ),
        "metadata_mismatch_count": (
            metadata_mismatch_count
        ),
        "non_contiguous_excerpt_count": (
            non_contiguous_excerpt_count
        ),
        "excerpt_boundary_violation_count": (
            excerpt_boundary_violation_count
        ),
        "evidence_id_sequence_violation_count": (
            evidence_id_sequence_violation_count
        ),
        "budget_violation_count": int(
            context.estimated_tokens
            > context.max_tokens
        ),
    }


def _method_summary(
    *,
    method: str,
    recall_values: dict[int, list[float]],
    reciprocal_ranks: list[float],
    top3_relevance_values: list[float],
    latency_values: list[float],
) -> dict[str, object]:
    return {
        "method": method,
        "question_count": len(
            top3_relevance_values
        ),
        "recall_at_1": _average(
            recall_values[1]
        ),
        "recall_at_3": _average(
            recall_values[3]
        ),
        "recall_at_5": _average(
            recall_values[5]
        ),
        "recall_at_10": _average(
            recall_values[10]
        ),
        "mrr": _average(reciprocal_ranks),
        "top3_average_relevance": _average(
            top3_relevance_values
        ),
        "mean_latency_ms": _average(
            latency_values
        ),
        "p50_latency_ms": (
            nearest_rank_percentile(
                latency_values,
                0.50,
            )
        ),
        "p95_latency_ms": (
            nearest_rank_percentile(
                latency_values,
                0.95,
            )
        ),
    }


def evaluate_stage3(
    *,
    questions: list[EvaluationQuestion],
    hybrid_results_by_question: dict[
        str,
        list[RetrievalResult],
    ],
    hybrid_latency_by_question: dict[str, float],
    scorer: RerankScorerProtocol,
    context_builder: ContextBuilder,
    chunks: list[ArticleChunk],
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    context_top_n: int = DEFAULT_CONTEXT_TOP_N,
    show_progress: bool = False,
) -> dict[str, object]:
    """对同一组 Hybrid 候选比较精排前后的质量。"""

    if candidate_top_k <= 0:
        raise ValueError(
            "candidate_top_k 必须大于 0"
        )

    if context_top_n <= 0:
        raise ValueError(
            "context_top_n 必须大于 0"
        )

    answerable_questions = [
        question
        for question in questions
        if question.should_answer
        and question.expected
    ]

    if not answerable_questions:
        raise ValueError(
            "评测集中没有可回答问题"
        )

    recall_ks = (1, 3, 5, 10)
    hybrid_recall = {
        k: []
        for k in recall_ks
    }
    rerank_recall = {
        k: []
        for k in recall_ks
    }
    hybrid_rr: list[float] = []
    rerank_rr: list[float] = []
    hybrid_top3: list[float] = []
    rerank_top3: list[float] = []
    hybrid_latencies: list[float] = []
    rerank_incremental_latencies: list[float] = []
    rerank_total_latencies: list[float] = []
    context_latencies: list[float] = []
    context_token_values: list[float] = []
    evidence_count_values: list[float] = []

    audit_totals = {
        "duplicate_article_count": 0,
        "duplicate_body_count": 0,
        "unmapped_evidence_count": 0,
        "metadata_mismatch_count": 0,
        "non_contiguous_excerpt_count": 0,
        "excerpt_boundary_violation_count": 0,
        "evidence_id_sequence_violation_count": 0,
        "budget_violation_count": 0,
    }

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }
    question_reports: list[dict[str, object]] = []

    for question_index, question in enumerate(
        answerable_questions,
        start=1,
    ):
        if show_progress:
            print(
                f"[stage3] {question_index}/"
                f"{len(answerable_questions)} "
                f"{question.id}",
                flush=True,
            )

        if question.id not in hybrid_results_by_question:
            raise ValueError(
                f"缺少 {question.id} 的 Hybrid 候选"
            )

        hybrid_results = hybrid_results_by_question[
            question.id
        ][:candidate_top_k]

        expected = {
            citation.as_key()
            for citation in question.expected
        }

        rerank_started_at = time.perf_counter()
        reranked_results = rerank_results(
            query=question.question,
            candidates=hybrid_results,
            scorer=scorer,
            top_n=len(hybrid_results),
        )
        rerank_incremental_latency = (
            time.perf_counter()
            - rerank_started_at
        ) * 1000.0

        hybrid_latency = (
            hybrid_latency_by_question.get(
                question.id,
                0.0,
            )
        )
        rerank_total_latency = (
            hybrid_latency
            + rerank_incremental_latency
        )

        context_started_at = time.perf_counter()
        context = context_builder.build(
            query=question.question,
            results=reranked_results[
                :context_top_n
            ],
        )
        context_latency = (
            time.perf_counter()
            - context_started_at
        ) * 1000.0

        hybrid_latencies.append(hybrid_latency)
        rerank_incremental_latencies.append(
            rerank_incremental_latency
        )
        rerank_total_latencies.append(
            rerank_total_latency
        )
        context_latencies.append(context_latency)
        context_token_values.append(
            float(context.estimated_tokens)
        )
        evidence_count_values.append(
            float(len(context.evidences))
        )

        hybrid_question_recall = {
            k: recall_at_k(
                hybrid_results,
                expected,
                k,
            )
            for k in recall_ks
        }
        rerank_question_recall = {
            k: recall_at_k(
                reranked_results,
                expected,
                k,
            )
            for k in recall_ks
        }

        for k in recall_ks:
            hybrid_recall[k].append(
                hybrid_question_recall[k]
            )
            rerank_recall[k].append(
                rerank_question_recall[k]
            )

        hybrid_question_rr = reciprocal_rank(
            hybrid_results,
            expected,
        )
        rerank_question_rr = reciprocal_rank(
            reranked_results,
            expected,
        )
        hybrid_question_top3 = (
            top_k_average_relevance(
                hybrid_results,
                expected,
            )
        )
        rerank_question_top3 = (
            top_k_average_relevance(
                reranked_results,
                expected,
            )
        )

        hybrid_rr.append(hybrid_question_rr)
        rerank_rr.append(rerank_question_rr)
        hybrid_top3.append(
            hybrid_question_top3
        )
        rerank_top3.append(
            rerank_question_top3
        )

        context_audit = audit_context(
            context,
            chunks_by_id=chunks_by_id,
        )

        for key, value in context_audit.items():
            audit_totals[key] += value

        question_reports.append(
            {
                "id": question.id,
                "type": question.question_type,
                "question": question.question,
                "expected": [
                    {
                        "law": citation.law,
                        "article": citation.article,
                    }
                    for citation in question.expected
                ],
                "hybrid": {
                    "recall_at_5": (
                        hybrid_question_recall[5]
                    ),
                    "recall_at_10": (
                        hybrid_question_recall[10]
                    ),
                    "top3_average_relevance": (
                        hybrid_question_top3
                    ),
                    "reciprocal_rank": (
                        hybrid_question_rr
                    ),
                    "latency_ms": hybrid_latency,
                    "retrieved": (
                        _serialize_results(
                            hybrid_results
                        )
                    ),
                },
                "rerank": {
                    "recall_at_5": (
                        rerank_question_recall[5]
                    ),
                    "recall_at_10": (
                        rerank_question_recall[10]
                    ),
                    "top3_average_relevance": (
                        rerank_question_top3
                    ),
                    "reciprocal_rank": (
                        rerank_question_rr
                    ),
                    "incremental_latency_ms": (
                        rerank_incremental_latency
                    ),
                    "total_retrieval_latency_ms": (
                        rerank_total_latency
                    ),
                    "retrieved": (
                        _serialize_results(
                            reranked_results
                        )
                    ),
                },
                "context": {
                    "evidence_count": len(
                        context.evidences
                    ),
                    "estimated_tokens": (
                        context.estimated_tokens
                    ),
                    "max_tokens": (
                        context.max_tokens
                    ),
                    "truncated": context.truncated,
                    "latency_ms": context_latency,
                    "audit": context_audit,
                    "evidences": [
                        {
                            "evidence_id": (
                                evidence.evidence_id
                            ),
                            "chunk_id": (
                                evidence.chunk_id
                            ),
                            "law": evidence.law_name,
                            "article": (
                                evidence.article_no
                            ),
                            "relation": (
                                evidence.relation
                            ),
                            "is_excerpt": (
                                evidence.is_excerpt
                            ),
                            "score": evidence.score,
                        }
                        for evidence in (
                            context.evidences
                        )
                    ],
                },
            }
        )

    hybrid_summary = _method_summary(
        method="hybrid",
        recall_values=hybrid_recall,
        reciprocal_ranks=hybrid_rr,
        top3_relevance_values=hybrid_top3,
        latency_values=hybrid_latencies,
    )
    rerank_summary = _method_summary(
        method="rerank",
        recall_values=rerank_recall,
        reciprocal_ranks=rerank_rr,
        top3_relevance_values=rerank_top3,
        latency_values=rerank_total_latencies,
    )
    rerank_summary[
        "mean_incremental_rerank_latency_ms"
    ] = _average(
        rerank_incremental_latencies
    )
    rerank_summary[
        "p95_incremental_rerank_latency_ms"
    ] = nearest_rank_percentile(
        rerank_incremental_latencies,
        0.95,
    )

    context_summary: dict[str, object] = {
        "question_count": len(
            answerable_questions
        ),
        "total_evidence_count": int(
            sum(evidence_count_values)
        ),
        "mean_evidence_count": _average(
            evidence_count_values
        ),
        "mean_estimated_tokens": _average(
            context_token_values
        ),
        "p95_context_latency_ms": (
            nearest_rank_percentile(
                context_latencies,
                0.95,
            )
        ),
        **audit_totals,
    }

    improvements = {
        "recall_at_5_delta": (
            rerank_summary["recall_at_5"]
            - hybrid_summary["recall_at_5"]
        ),
        "top3_average_relevance_delta": (
            rerank_summary[
                "top3_average_relevance"
            ]
            - hybrid_summary[
                "top3_average_relevance"
            ]
        ),
        "mrr_delta": (
            rerank_summary["mrr"]
            - hybrid_summary["mrr"]
        ),
    }

    quality_gates = {
        "recall_at_5_improved": (
            improvements["recall_at_5_delta"]
            > 0
        ),
        "top3_average_relevance_improved": (
            improvements[
                "top3_average_relevance_delta"
            ]
            > 0
        ),
        "recall_at_10_preserved": (
            rerank_summary["recall_at_10"]
            == hybrid_summary["recall_at_10"]
        ),
        "no_duplicate_articles": (
            audit_totals[
                "duplicate_article_count"
            ]
            == 0
        ),
        "no_duplicate_bodies": (
            audit_totals[
                "duplicate_body_count"
            ]
            == 0
        ),
        "all_evidence_mapped": (
            audit_totals[
                "unmapped_evidence_count"
            ]
            == 0
            and audit_totals[
                "metadata_mismatch_count"
            ]
            == 0
        ),
        "no_budget_violations": (
            audit_totals[
                "budget_violation_count"
            ]
            == 0
        ),
        "all_excerpts_contiguous": (
            audit_totals[
                "non_contiguous_excerpt_count"
            ]
            == 0
        ),
        "all_excerpt_boundaries_valid": (
            audit_totals[
                "excerpt_boundary_violation_count"
            ]
            == 0
        ),
        "evidence_ids_sequential": (
            audit_totals[
                "evidence_id_sequence_violation_count"
            ]
            == 0
        ),
    }
    quality_gates["all_passed"] = all(
        quality_gates.values()
    )

    return {
        "quality_gates": quality_gates,
        "improvements": improvements,
        "methods": {
            "hybrid": hybrid_summary,
            "rerank": rerank_summary,
        },
        "context": context_summary,
        "questions": question_reports,
    }


def _serialize_results(
    results: list[RetrievalResult],
) -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "law": result.law_name,
            "article": result.article_no,
            "chunk_id": result.chunk_id,
            "score": result.score,
        }
        for rank, result in enumerate(
            results,
            start=1,
        )
    ]


def _write_json_atomic(
    path: Path,
    report: dict[str, object],
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
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_path.replace(path)


def print_stage3_summary(
    report: dict[str, object],
) -> None:
    methods = report["methods"]
    hybrid = methods["hybrid"]
    rerank = methods["rerank"]
    context = report["context"]
    gates = report["quality_gates"]

    print()
    print(
        "方法      R@5     R@10    Top3Rel  "
        "MRR     P95(ms)"
    )
    print("-" * 62)

    for name, summary in (
        ("hybrid", hybrid),
        ("rerank", rerank),
    ):
        print(
            f"{name:<9} "
            f"{summary['recall_at_5']:.4f}  "
            f"{summary['recall_at_10']:.4f}  "
            f"{summary['top3_average_relevance']:.4f}   "
            f"{summary['mrr']:.4f}  "
            f"{summary['p95_latency_ms']:.1f}"
        )

    print()
    print(
        "上下文结构问题数："
        f"重复条文={context['duplicate_article_count']}，"
        f"重复正文={context['duplicate_body_count']}，"
        f"映射失败={context['unmapped_evidence_count']}，"
        f"预算超限={context['budget_violation_count']}，"
        "片段边界错误="
        f"{context['excerpt_boundary_violation_count']}"
    )
    print(
        "全部质量门槛通过："
        f"{gates['all_passed']}"
    )


def run_stage3_evaluation(
    *,
    evaluation_path: Path = DEFAULT_EVALUATION_PATH,
    articles_path: Path = DEFAULT_ARTICLES_PATH,
    hybrid_report_path: Path = DEFAULT_HYBRID_REPORT_PATH,
    report_path: Path = DEFAULT_STAGE3_REPORT_PATH,
    candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
    context_top_n: int = DEFAULT_CONTEXT_TOP_N,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    max_evidences: int = DEFAULT_MAX_EVIDENCES,
    max_evidence_tokens: int = DEFAULT_MAX_EVIDENCE_TOKENS,
    adjacent_for_top_n: int = DEFAULT_ADJACENT_FOR_TOP_N,
    min_adjacent_score: float = DEFAULT_MIN_ADJACENT_SCORE,
    show_progress: bool = True,
) -> dict[str, object]:
    """加载真实数据并运行阶段 3 正式评测。"""

    from rag_law.config import settings

    questions = load_evaluation_questions(
        evaluation_path
    )
    chunks = load_articles(articles_path)
    (
        hybrid_results_by_question,
        hybrid_latency_by_question,
        hybrid_metadata,
    ) = load_cached_hybrid_results(
        report_path=hybrid_report_path,
        articles_path=articles_path,
        chunks=chunks,
    )

    scorer = SiliconFlowReranker.from_settings()
    context_builder = ContextBuilder(
        chunks=chunks,
        scorer=scorer,
        max_context_tokens=max_context_tokens,
        max_evidences=max_evidences,
        max_evidence_tokens=max_evidence_tokens,
        adjacent_for_top_n=adjacent_for_top_n,
        min_adjacent_score=min_adjacent_score,
    )

    evaluation = evaluate_stage3(
        questions=questions,
        hybrid_results_by_question=(
            hybrid_results_by_question
        ),
        hybrid_latency_by_question=(
            hybrid_latency_by_question
        ),
        scorer=scorer,
        context_builder=context_builder,
        chunks=chunks,
        candidate_top_k=candidate_top_k,
        context_top_n=context_top_n,
        show_progress=show_progress,
    )

    metadata: dict[str, object] = {
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
        "articles_path": str(articles_path),
        "articles_sha256": sha256_file(
            articles_path
        ),
        "hybrid_report_path": str(
            hybrid_report_path
        ),
        "hybrid_report_sha256": sha256_file(
            hybrid_report_path
        ),
        "hybrid_index_version": (
            hybrid_metadata.get("index_version")
        ),
        "rerank_model": settings.rerank_model,
        "rerank_base_url": (
            settings.rerank_base_url
        ),
        "candidate_top_k": candidate_top_k,
        "context_top_n": context_top_n,
        "max_context_tokens": (
            max_context_tokens
        ),
        "max_evidences": max_evidences,
        "max_evidence_tokens": (
            max_evidence_tokens
        ),
        "adjacent_for_top_n": (
            adjacent_for_top_n
        ),
        "min_adjacent_score": (
            min_adjacent_score
        ),
        "top3_relevance_definition": (
            "前3条中命中标注法律名与条号的数量 / 3"
        ),
        "package_versions": {
            "httpx": package_version("httpx"),
            "pydantic": package_version(
                "pydantic"
            ),
        },
    }

    report = {
        "metadata": metadata,
        **evaluation,
    }

    _write_json_atomic(report_path, report)
    print_stage3_summary(report)
    print()
    print(f"评测报告已写入：{report_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行阶段 3 重排与上下文评测"
    )
    parser.add_argument(
        "--questions",
        default=str(DEFAULT_EVALUATION_PATH),
    )
    parser.add_argument(
        "--articles",
        default=str(DEFAULT_ARTICLES_PATH),
    )
    parser.add_argument(
        "--hybrid-report",
        default=str(DEFAULT_HYBRID_REPORT_PATH),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_STAGE3_REPORT_PATH),
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=DEFAULT_CANDIDATE_TOP_K,
    )
    parser.add_argument(
        "--context-top-n",
        type=int,
        default=DEFAULT_CONTEXT_TOP_N,
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
    )
    parser.add_argument(
        "--max-evidences",
        type=int,
        default=DEFAULT_MAX_EVIDENCES,
    )
    parser.add_argument(
        "--max-evidence-tokens",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_TOKENS,
    )
    parser.add_argument(
        "--adjacent-for-top-n",
        type=int,
        default=DEFAULT_ADJACENT_FOR_TOP_N,
    )
    parser.add_argument(
        "--min-adjacent-score",
        type=float,
        default=DEFAULT_MIN_ADJACENT_SCORE,
    )

    arguments = parser.parse_args()

    run_stage3_evaluation(
        evaluation_path=Path(
            arguments.questions
        ),
        articles_path=Path(
            arguments.articles
        ),
        hybrid_report_path=Path(
            arguments.hybrid_report
        ),
        report_path=Path(arguments.output),
        candidate_top_k=(
            arguments.candidate_top_k
        ),
        context_top_n=arguments.context_top_n,
        max_context_tokens=(
            arguments.max_context_tokens
        ),
        max_evidences=arguments.max_evidences,
        max_evidence_tokens=(
            arguments.max_evidence_tokens
        ),
        adjacent_for_top_n=(
            arguments.adjacent_for_top_n
        ),
        min_adjacent_score=(
            arguments.min_adjacent_score
        ),
        show_progress=True,
    )


if __name__ == "__main__":
    main()
