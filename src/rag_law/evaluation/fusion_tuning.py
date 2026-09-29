"""对缓存的 BM25/Dense 候选执行离线 RRF 参数搜索。"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rag_law.evaluation.retrieval_metrics import (
    DEFAULT_EVALUATION_PATH,
    EvaluationQuestion,
    load_evaluation_questions,
    recall_at_k,
    reciprocal_rank,
)
from rag_law.retrieval.bm25 import (
    BM25Retriever,
)
from rag_law.retrieval.common import (
    PROJECT_ROOT,
)
from rag_law.retrieval.dense import (
    DenseRetriever,
)
from rag_law.retrieval.fusion import (
    reciprocal_rank_fusion,
)
from rag_law.schemas import RetrievalResult


DEFAULT_TUNING_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "results"
    / "fusion_tuning_v1.json"
)

CANDIDATE_TOP_K_VALUES = (
    10,
    20,
    30,
    50,
)

RRF_K_VALUES = (
    30,
    60,
    90,
)

DENSE_WEIGHT_VALUES = (
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
)

BM25_WEIGHT = 1.0


@dataclass
class CachedQuestion:
    """一道题及两路已经排好序的候选。"""

    question_id: str
    question: str
    expected: set[tuple[str, str]]
    bm25_results: list[RetrievalResult]
    dense_results: list[RetrievalResult]


def _result_key(
    result: RetrievalResult,
) -> tuple[str, str]:
    return result.law_name, result.article_no


def _find_expected_rank(
    results: list[RetrievalResult],
    expected: tuple[str, str],
) -> int | None:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        if _result_key(result) == expected:
            return rank

    return None


def collect_candidate_cache(
    *,
    questions: list[EvaluationQuestion],
    maximum_candidate_top_k: int = 50,
    show_progress: bool = True,
) -> list[CachedQuestion]:
    """每道题只调用一次 Dense，然后缓存前 50 名。

    后续参数组合只重新计算 RRF，不重复调用 Embedding API。
    """

    if maximum_candidate_top_k < 10:
        raise ValueError(
            "maximum_candidate_top_k 不能小于 10"
        )

    answerable_questions = [
        question
        for question in questions
        if (
            question.should_answer
            and question.expected
        )
    ]

    bm25_retriever = BM25Retriever.load()
    dense_retriever = DenseRetriever.load()

    cache: list[CachedQuestion] = []

    for index, question in enumerate(
        answerable_questions,
        start=1,
    ):
        if show_progress:
            print(
                f"[候选缓存] "
                f"{index}/"
                f"{len(answerable_questions)} "
                f"{question.id}"
            )

        bm25_results = (
            bm25_retriever.search(
                question.question,
                top_k=maximum_candidate_top_k,
            )
        )

        dense_results = (
            dense_retriever.search(
                question.question,
                top_k=maximum_candidate_top_k,
            )
        )

        expected = {
            citation.as_key()
            for citation in question.expected
        }

        cache.append(
            CachedQuestion(
                question_id=question.id,
                question=question.question,
                expected=expected,
                bm25_results=bm25_results,
                dense_results=dense_results,
            )
        )

    return cache


def summarize_rankings(
    rankings: list[
        tuple[
            CachedQuestion,
            list[RetrievalResult],
        ]
    ],
) -> dict[str, object]:
    """汇总一组逐题排名。"""

    recall_1_values: list[float] = []
    recall_3_values: list[float] = []
    recall_5_values: list[float] = []
    recall_10_values: list[float] = []
    reciprocal_ranks: list[float] = []

    misses: list[dict[str, object]] = []

    for cached, results in rankings:
        recall_1 = recall_at_k(
            results,
            cached.expected,
            1,
        )
        recall_3 = recall_at_k(
            results,
            cached.expected,
            3,
        )
        recall_5 = recall_at_k(
            results,
            cached.expected,
            5,
        )
        recall_10 = recall_at_k(
            results,
            cached.expected,
            10,
        )
        rr = reciprocal_rank(
            results,
            cached.expected,
        )

        recall_1_values.append(recall_1)
        recall_3_values.append(recall_3)
        recall_5_values.append(recall_5)
        recall_10_values.append(recall_10)
        reciprocal_ranks.append(rr)

        if recall_10 < 1.0:
            misses.append(
                {
                    "id": cached.question_id,
                    "recall_at_10": recall_10,
                    "expected": [
                        {
                            "law": law,
                            "article": article,
                        }
                        for law, article in sorted(
                            cached.expected
                        )
                    ],
                    "retrieved": [
                        {
                            "rank": rank,
                            "law": result.law_name,
                            "article": (
                                result.article_no
                            ),
                        }
                        for rank, result in enumerate(
                            results[:10],
                            start=1,
                        )
                    ],
                }
            )

    question_count = len(rankings)

    if question_count == 0:
        raise ValueError(
            "没有可用于汇总的问题"
        )

    def average(
        values: list[float],
    ) -> float:
        return sum(values) / len(values)

    return {
        "question_count": question_count,
        "recall_at_1": average(
            recall_1_values
        ),
        "recall_at_3": average(
            recall_3_values
        ),
        "recall_at_5": average(
            recall_5_values
        ),
        "recall_at_10": average(
            recall_10_values
        ),
        "mrr": average(reciprocal_ranks),
        "misses_at_10": misses,
    }


def evaluate_single_route(
    cache: list[CachedQuestion],
    *,
    route: str,
) -> dict[str, object]:
    """从缓存中重新计算单路前 10 基线。"""

    if route == "bm25":
        rankings = [
            (
                cached,
                cached.bm25_results[:10],
            )
            for cached in cache
        ]
    elif route == "dense":
        rankings = [
            (
                cached,
                cached.dense_results[:10],
            )
            for cached in cache
        ]
    else:
        raise ValueError(
            f"未知单路检索器：{route}"
        )

    return summarize_rankings(rankings)


def evaluate_fusion_configuration(
    cache: list[CachedQuestion],
    *,
    candidate_top_k: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
) -> dict[str, object]:
    """评测一组 RRF 参数。"""

    rankings: list[
        tuple[
            CachedQuestion,
            list[RetrievalResult],
        ]
    ] = []

    for cached in cache:
        fused_results = (
            reciprocal_rank_fusion(
                cached.bm25_results[
                    :candidate_top_k
                ],
                cached.dense_results[
                    :candidate_top_k
                ],
                top_k=10,
                rrf_k=rrf_k,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
            )
        )

        rankings.append(
            (cached, fused_results)
        )

    summary = summarize_rankings(
        rankings
    )

    return {
        "candidate_top_k": candidate_top_k,
        "rrf_k": rrf_k,
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        **summary,
    }


def build_candidate_diagnostics(
    cache: list[CachedQuestion],
) -> list[dict[str, object]]:
    """记录每个标准法条在两路候选中的位置。"""

    diagnostics: list[
        dict[str, object]
    ] = []

    for cached in cache:
        expected_ranks = []

        for expected in sorted(
            cached.expected
        ):
            law, article = expected

            expected_ranks.append(
                {
                    "law": law,
                    "article": article,
                    "bm25_rank": (
                        _find_expected_rank(
                            cached.bm25_results,
                            expected,
                        )
                    ),
                    "dense_rank": (
                        _find_expected_rank(
                            cached.dense_results,
                            expected,
                        )
                    ),
                }
            )

        diagnostics.append(
            {
                "id": cached.question_id,
                "question": cached.question,
                "expected_ranks": (
                    expected_ranks
                ),
            }
        )

    return diagnostics


def configuration_sort_key(
    configuration: dict[str, object],
) -> tuple[float, ...]:
    """优先 Recall@10，其次 MRR 和较小配置。"""

    return (
        float(
            configuration["recall_at_10"]
        ),
        float(configuration["mrr"]),
        float(
            configuration["recall_at_5"]
        ),
        float(
            configuration["recall_at_3"]
        ),
        -float(
            configuration[
                "candidate_top_k"
            ]
        ),
        -abs(
            float(configuration["rrf_k"])
            - 60.0
        ),
        -abs(
            float(
                configuration[
                    "dense_weight"
                ]
            )
            - 1.0
        ),
    )


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


def run_fusion_tuning(
    *,
    evaluation_path: Path = (
        DEFAULT_EVALUATION_PATH
    ),
    output_path: Path = (
        DEFAULT_TUNING_REPORT_PATH
    ),
    maximum_candidate_top_k: int = 50,
) -> dict[str, object]:
    """缓存候选并搜索一组小而明确的参数网格。"""

    questions = load_evaluation_questions(
        evaluation_path
    )

    cache = collect_candidate_cache(
        questions=questions,
        maximum_candidate_top_k=(
            maximum_candidate_top_k
        ),
        show_progress=True,
    )

    bm25_baseline = evaluate_single_route(
        cache,
        route="bm25",
    )
    dense_baseline = evaluate_single_route(
        cache,
        route="dense",
    )

    configurations: list[
        dict[str, object]
    ] = []

    for candidate_top_k in (
        CANDIDATE_TOP_K_VALUES
    ):
        if (
            candidate_top_k
            > maximum_candidate_top_k
        ):
            continue

        for rrf_k in RRF_K_VALUES:
            for dense_weight in (
                DENSE_WEIGHT_VALUES
            ):
                configuration = (
                    evaluate_fusion_configuration(
                        cache,
                        candidate_top_k=(
                            candidate_top_k
                        ),
                        rrf_k=rrf_k,
                        bm25_weight=(
                            BM25_WEIGHT
                        ),
                        dense_weight=(
                            dense_weight
                        ),
                    )
                )

                configurations.append(
                    configuration
                )

    configurations.sort(
        key=configuration_sort_key,
        reverse=True,
    )

    selected = configurations[0]

    report: dict[str, object] = {
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "warning": (
            "参数搜索和最终指标使用同一份小型评测集，"
            "结果属于开发集表现，不是独立测试集表现。"
        ),
        "maximum_candidate_top_k": (
            maximum_candidate_top_k
        ),
        "grid": {
            "candidate_top_k": list(
                CANDIDATE_TOP_K_VALUES
            ),
            "rrf_k": list(RRF_K_VALUES),
            "bm25_weight": BM25_WEIGHT,
            "dense_weight": list(
                DENSE_WEIGHT_VALUES
            ),
        },
        "single_baselines": {
            "bm25": bm25_baseline,
            "dense": dense_baseline,
        },
        "selected_configuration": (
            selected
        ),
        "configurations": configurations,
        "candidate_diagnostics": (
            build_candidate_diagnostics(
                cache
            )
        ),
    }

    _write_json_atomic(
        output_path,
        report,
    )

    print()
    print(
        "排名  cand  rrf  bm25_w  "
        "dense_w  R@5    R@10   MRR"
    )
    print("-" * 67)

    for rank, configuration in enumerate(
        configurations[:10],
        start=1,
    ):
        print(
            f"{rank:<5} "
            f"{configuration['candidate_top_k']:<5} "
            f"{configuration['rrf_k']:<4} "
            f"{configuration['bm25_weight']:<7.2f} "
            f"{configuration['dense_weight']:<8.2f} "
            f"{configuration['recall_at_5']:.4f} "
            f"{configuration['recall_at_10']:.4f} "
            f"{configuration['mrr']:.4f}"
        )

    print()
    print(
        "Dense 单路："
        f"R@10="
        f"{dense_baseline['recall_at_10']:.4f}，"
        f"MRR={dense_baseline['mrr']:.4f}"
    )
    print(
        "选中配置："
        f"candidate_top_k="
        f"{selected['candidate_top_k']}，"
        f"rrf_k={selected['rrf_k']}，"
        f"bm25_weight="
        f"{selected['bm25_weight']}，"
        f"dense_weight="
        f"{selected['dense_weight']}"
    )
    print(
        "选中配置结果："
        f"R@10="
        f"{selected['recall_at_10']:.4f}，"
        f"MRR={selected['mrr']:.4f}"
    )
    print()
    print(f"调参报告已写入：{output_path}")

    return report


def main() -> None:
    run_fusion_tuning()


if __name__ == "__main__":
    main()