"""检索基线评测：Recall@K、MRR 和延迟。"""

import hashlib
import json
import math
import time
from datetime import datetime
from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, Field

from rag_law.retrieval.bm25 import (
    BM25Retriever,
)
from rag_law.retrieval.common import (
    DEFAULT_ARTICLES_PATH,
    PROJECT_ROOT,
)
from rag_law.retrieval.dense import (
    DenseRetriever,
)
from rag_law.retrieval.fusion import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_RRF_K,
    HybridRetriever,
)
from rag_law.schemas import RetrievalResult


DEFAULT_EVALUATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "questions_v1.yaml"
)

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "results"
    / "retrieval_baseline_v1.json"
)

RECALL_K_VALUES = (1, 3, 5, 10)


class ExpectedCitation(BaseModel):
    """一道题预期召回的法律和条号。"""

    law: str = Field(min_length=1)
    article: str = Field(min_length=1)

    def as_key(self) -> tuple[str, str]:
        return self.law, self.article


class EvaluationQuestion(BaseModel):
    """评测集中的一道问题。"""

    id: str = Field(min_length=1)
    question_type: str = Field(
        alias="type",
        min_length=1,
    )
    question: str = Field(min_length=1)
    should_answer: bool
    expected: list[ExpectedCitation]
    note: str = ""


class RetrieverProtocol(Protocol):
    """评测器依赖的最小检索接口。"""

    def search(
        self,
        query: str,
        *,
        top_k: int,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        ...


def load_evaluation_questions(
    path: Path = DEFAULT_EVALUATION_PATH,
) -> list[EvaluationQuestion]:
    """读取并校验 YAML 评测集。"""

    if not path.exists():
        raise FileNotFoundError(
            f"评测集不存在：{path}"
        )

    with path.open(encoding="utf-8") as file:
        raw_questions = yaml.safe_load(file)

    if not isinstance(raw_questions, list):
        raise ValueError(
            "评测集顶层必须是列表"
        )

    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()

    for index, raw_question in enumerate(
        raw_questions,
        start=1,
    ):
        try:
            question = (
                EvaluationQuestion.model_validate(
                    raw_question
                )
            )
        except Exception as error:
            raise ValueError(
                f"评测集第 {index} 项格式错误"
            ) from error

        if question.id in seen_ids:
            raise ValueError(
                f"评测题 ID 重复：{question.id}"
            )

        if (
            question.should_answer
            and not question.expected
        ):
            raise ValueError(
                f"{question.id} 标记为可回答，"
                "但没有 expected 法条"
            )

        seen_ids.add(question.id)
        questions.append(question)

    return questions


def result_key(
    result: RetrievalResult,
) -> tuple[str, str]:
    """把检索结果转成与标注一致的比较键。"""

    return result.law_name, result.article_no


def recall_at_k(
    results: list[RetrievalResult],
    expected: set[tuple[str, str]],
    k: int,
) -> float:
    """计算单道题的 Recall@K。

    多法条问题按召回比例计算。例如预期两条，只命中一条，
    Recall 就是 0.5，而不是 1。
    """

    if not expected:
        raise ValueError(
            "计算 Recall 时 expected 不能为空"
        )

    if k <= 0:
        raise ValueError("k 必须大于 0")

    retrieved = {
        result_key(result)
        for result in results[:k]
    }

    hit_count = len(
        retrieved.intersection(expected)
    )

    return hit_count / len(expected)


def reciprocal_rank(
    results: list[RetrievalResult],
    expected: set[tuple[str, str]],
) -> float:
    """计算第一条相关结果的倒数排名。"""

    if not expected:
        raise ValueError(
            "计算 MRR 时 expected 不能为空"
        )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result_key(result) in expected:
            return 1.0 / rank

    return 0.0


def nearest_rank_percentile(
    values: list[float],
    percentile: float,
) -> float:
    """使用 nearest-rank 方法计算百分位数。"""

    if not values:
        raise ValueError(
            "计算百分位数时数据不能为空"
        )

    if not 0 < percentile <= 1:
        raise ValueError(
            "percentile 必须在 (0, 1] 范围内"
        )

    ordered = sorted(values)

    rank = math.ceil(
        percentile * len(ordered)
    )

    return ordered[rank - 1]


def _average(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def evaluate_retriever(
    *,
    method_name: str,
    retriever: RetrieverProtocol,
    questions: list[EvaluationQuestion],
    max_k: int = 10,
    show_progress: bool = False,
) -> dict[str, object]:
    """评测一个检索器并返回汇总和逐题结果。"""

    if max_k < max(RECALL_K_VALUES):
        raise ValueError(
            "max_k 不能小于 10"
        )

    answerable_questions = [
        question
        for question in questions
        if (
            question.should_answer
            and question.expected
        )
    ]

    if not answerable_questions:
        raise ValueError(
            "评测集中没有可回答的问题"
        )

    latency_values: list[float] = []
    reciprocal_ranks: list[float] = []

    recall_values: dict[int, list[float]] = {
        k: []
        for k in RECALL_K_VALUES
    }

    question_results: list[
        dict[str, object]
    ] = []

    for question_index, question in enumerate(
        answerable_questions,
        start=1,
    ):
        if show_progress:
            print(
                f"[{method_name}] "
                f"{question_index}/"
                f"{len(answerable_questions)} "
                f"{question.id}"
            )

        started_at = time.perf_counter()

        results = retriever.search(
            question.question,
            top_k=max_k,
        )

        latency_ms = (
            time.perf_counter() - started_at
        ) * 1000.0

        latency_values.append(latency_ms)

        expected = {
            citation.as_key()
            for citation in question.expected
        }

        per_question_recall = {
            k: recall_at_k(
                results,
                expected,
                k,
            )
            for k in RECALL_K_VALUES
        }

        for k, value in (
            per_question_recall.items()
        ):
            recall_values[k].append(value)

        rr = reciprocal_rank(
            results,
            expected,
        )
        reciprocal_ranks.append(rr)

        question_results.append(
            {
                "id": question.id,
                "type": (
                    question.question_type
                ),
                "question": question.question,
                "expected": [
                    {
                        "law": citation.law,
                        "article": (
                            citation.article
                        ),
                    }
                    for citation in (
                        question.expected
                    )
                ],
                "recall_at_1": (
                    per_question_recall[1]
                ),
                "recall_at_3": (
                    per_question_recall[3]
                ),
                "recall_at_5": (
                    per_question_recall[5]
                ),
                "recall_at_10": (
                    per_question_recall[10]
                ),
                "reciprocal_rank": rr,
                "latency_ms": latency_ms,
                "retrieved": [
                    {
                        "rank": rank,
                        "law": result.law_name,
                        "article": (
                            result.article_no
                        ),
                        "chunk_id": (
                            result.chunk_id
                        ),
                        "score": result.score,
                    }
                    for rank, result in enumerate(
                        results,
                        start=1,
                    )
                ],
            }
        )

    summary: dict[str, object] = {
        "method": method_name,
        "question_count": len(
            answerable_questions
        ),
        "excluded_question_count": (
            len(questions)
            - len(answerable_questions)
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

    return {
        "summary": summary,
        "questions": question_results,
    }


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def package_version(
    distribution_name: str,
) -> str | None:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None


def build_run_metadata(
    *,
    articles_path: Path,
    question_count: int,
    candidate_top_k: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
) -> dict[str, object]:
    """生成可复现实验所需的元数据。"""

    from rag_law.config import settings

    articles_sha256 = sha256_file(
        articles_path
    )

    fingerprint_source = "|".join(
        [
            articles_sha256,
            settings.embedding_model,
            "search-text-v1",
            "jieba-accurate-v1",
        ]
    )

    index_version = hashlib.sha256(
        fingerprint_source.encode("utf-8")
    ).hexdigest()[:16]

    return {
        "generated_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "index_version": index_version,
        "articles_path": str(articles_path),
        "articles_sha256": articles_sha256,
        "question_count": question_count,
        "embedding_model": (
            settings.embedding_model
        ),
        "embedding_base_url": (
            settings.embedding_base_url
        ),
        "search_text_version": (
            "search-text-v1"
        ),
        "bm25_tokenizer": (
            "jieba-accurate-v1"
        ),
        "collection_name": "law_articles",
        "candidate_top_k": (
            candidate_top_k
        ),
        "rrf_k": rrf_k,
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        "package_versions": {
            "jieba": package_version("jieba"),
            "rank-bm25": package_version(
                "rank-bm25"
            ),
            "langchain-core": (
                package_version(
                    "langchain-core"
                )
            ),
            "langchain-chroma": (
                package_version(
                    "langchain-chroma"
                )
            ),
            "langchain-openai": (
                package_version(
                    "langchain-openai"
                )
            ),
            "chromadb": package_version(
                "chromadb"
            ),
        },
    }


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


def print_summary_table(
    report: dict[str, object],
) -> None:
    """在终端输出三种方法的对比表。"""

    methods = report["methods"]

    print()
    print(
        "方法     R@1     R@3     R@5     "
        "R@10    MRR     P95(ms)"
    )
    print("-" * 64)

    for method_name in (
        "bm25",
        "dense",
        "hybrid",
    ):
        method_report = methods[method_name]
        summary = method_report["summary"]

        print(
            f"{method_name:<8} "
            f"{summary['recall_at_1']:.3f}   "
            f"{summary['recall_at_3']:.3f}   "
            f"{summary['recall_at_5']:.3f}   "
            f"{summary['recall_at_10']:.3f}   "
            f"{summary['mrr']:.3f}   "
            f"{summary['p95_latency_ms']:.1f}"
        )

    gates = report["quality_gates"]

    print()
    print("质量门槛：")
    print(
        "  Hybrid Recall@10 >= 0.90："
        f"{gates['hybrid_recall_at_10_met']}"
    )
    print(
        "  Hybrid Recall@10 不低于最佳单路："
        f"{gates['hybrid_recall_not_below_single']}"
    )
    print(
        "  Hybrid MRR 不低于最佳单路："
        f"{gates['hybrid_mrr_not_below_single']}"
    )


def run_baseline_evaluation(
    *,
    evaluation_path: Path = (
        DEFAULT_EVALUATION_PATH
    ),
    articles_path: Path = (
        DEFAULT_ARTICLES_PATH
    ),
    report_path: Path = DEFAULT_REPORT_PATH,
    candidate_top_k: int = (
        DEFAULT_CANDIDATE_TOP_K
    ),
    rrf_k: int = DEFAULT_RRF_K,
    bm25_weight: float = (
        DEFAULT_BM25_WEIGHT
    ),
    dense_weight: float = (
        DEFAULT_DENSE_WEIGHT
    ),
    show_progress: bool = True,
) -> dict[str, object]:
    """加载三个检索器并运行完整基线评测。"""

    questions = load_evaluation_questions(
        evaluation_path
    )

    bm25_retriever = BM25Retriever.load()
    dense_retriever = DenseRetriever.load(
        articles_path=articles_path
    )

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        candidate_top_k=candidate_top_k,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )

    methods: dict[str, object] = {}

    for method_name, retriever in (
        ("bm25", bm25_retriever),
        ("dense", dense_retriever),
        ("hybrid", hybrid_retriever),
    ):
        methods[method_name] = (
            evaluate_retriever(
                method_name=method_name,
                retriever=retriever,
                questions=questions,
                max_k=10,
                show_progress=show_progress,
            )
        )

    bm25_summary = methods["bm25"][
        "summary"
    ]
    dense_summary = methods["dense"][
        "summary"
    ]
    hybrid_summary = methods["hybrid"][
        "summary"
    ]

    best_single_recall = max(
        bm25_summary["recall_at_10"],
        dense_summary["recall_at_10"],
    )
    best_single_mrr = max(
        bm25_summary["mrr"],
        dense_summary["mrr"],
    )

    quality_gates = {
        "hybrid_recall_at_10_target": 0.90,
        "hybrid_recall_at_10_met": (
            hybrid_summary["recall_at_10"]
            >= 0.90
        ),
        "hybrid_recall_not_below_single": (
            hybrid_summary["recall_at_10"]
            >= best_single_recall
        ),
        "hybrid_mrr_not_below_single": (
            hybrid_summary["mrr"]
            >= best_single_mrr
        ),
        "hybrid_recall_delta_vs_best_single": (
            hybrid_summary["recall_at_10"]
            - best_single_recall
        ),
        "hybrid_mrr_delta_vs_best_single": (
            hybrid_summary["mrr"]
            - best_single_mrr
        ),
    }

    metadata = build_run_metadata(
        articles_path=articles_path,
        question_count=len(questions),
        candidate_top_k=candidate_top_k,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )

    report: dict[str, object] = {
        "metadata": metadata,
        "quality_gates": quality_gates,
        "methods": methods,
    }

    _write_json_atomic(
        report_path,
        report,
    )

    print_summary_table(report)
    print()
    print(f"评测报告已写入：{report_path}")

    return report