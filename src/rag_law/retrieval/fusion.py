"""BM25 与 Dense 的 RRF 排名融合。"""

import argparse
from dataclasses import dataclass, field

from rag_law.retrieval.bm25 import (
    BM25Retriever,
)
from rag_law.retrieval.dense import (
    DenseRetriever,
)
from rag_law.schemas import RetrievalResult


DEFAULT_CANDIDATE_TOP_K = 10
DEFAULT_RRF_K = 60
DEFAULT_BM25_WEIGHT = 1.0
DEFAULT_DENSE_WEIGHT = 2.0


@dataclass
class _FusionEntry:
    """RRF 融合过程中的内部累加记录。"""

    result: RetrievalResult
    rrf_score: float = 0.0
    best_rank: int = 0
    component_scores: dict[str, float] = field(
        default_factory=dict
    )


def _validate_same_chunk(
    first: RetrievalResult,
    second: RetrievalResult,
) -> None:
    """确认相同 chunk_id 对应的是同一条法条。"""

    fields = (
        "law_id",
        "law_name",
        "article_no",
        "text",
        "source_file",
    )

    mismatched_fields = [
        field_name
        for field_name in fields
        if (
            getattr(first, field_name)
            != getattr(second, field_name)
        )
    ]

    if mismatched_fields:
        raise ValueError(
            "相同 chunk_id 对应了不同内容："
            f"{first.chunk_id}，"
            f"冲突字段：{mismatched_fields}"
        )


def reciprocal_rank_fusion(
    bm25_results: list[RetrievalResult],
    dense_results: list[RetrievalResult],
    *,
    top_k: int = 10,
    rrf_k: int = DEFAULT_RRF_K,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
) -> list[RetrievalResult]:
    """使用 RRF 融合两路已经排好序的结果。

    两个输入列表都必须按照相关性从高到低排列。
    相同 chunk_id 只保留一条，RRF 分数累加。
    """

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    if rrf_k <= 0:
        raise ValueError("rrf_k 必须大于 0")

    if bm25_weight <= 0:
        raise ValueError(
            "bm25_weight 必须大于 0"
        )

    if dense_weight <= 0:
        raise ValueError(
            "dense_weight 必须大于 0"
        )

    routes = (
        (
            "bm25",
            bm25_results,
            bm25_weight,
        ),
        (
            "dense",
            dense_results,
            dense_weight,
        ),
    )

    entries: dict[str, _FusionEntry] = {}

    for route_name, results, weight in routes:
        seen_in_route: set[str] = set()

        for rank, result in enumerate(
            results,
            start=1,
        ):
            # 防止单个检索器意外返回重复 chunk。
            if result.chunk_id in seen_in_route:
                continue

            seen_in_route.add(result.chunk_id)

            if result.source != route_name:
                raise ValueError(
                    f"{route_name} 结果中出现了 "
                    f"source={result.source}"
                )

            contribution = weight / (
                rrf_k + rank
            )

            entry = entries.get(result.chunk_id)

            if entry is None:
                entry = _FusionEntry(
                    result=result,
                    best_rank=rank,
                )
                entries[result.chunk_id] = entry
            else:
                _validate_same_chunk(
                    entry.result,
                    result,
                )
                entry.best_rank = min(
                    entry.best_rank,
                    rank,
                )

            entry.rrf_score += contribution

            # 保留原始检索器提供的分数。
            entry.component_scores.update(
                result.component_scores
            )

            # 同时记录该路的排名和 RRF 贡献。
            entry.component_scores[
                f"{route_name}_rank"
            ] = float(rank)

            entry.component_scores[
                f"{route_name}_rrf"
            ] = float(contribution)

    ordered_entries = sorted(
        entries.values(),
        key=lambda entry: (
            -entry.rrf_score,
            entry.best_rank,
            entry.result.chunk_id,
        ),
    )

    fused_results: list[RetrievalResult] = []

    for entry in ordered_entries[:top_k]:
        component_scores = dict(
            entry.component_scores
        )
        component_scores["rrf"] = (
            entry.rrf_score
        )

        fused_results.append(
            entry.result.model_copy(
                update={
                    "score": entry.rrf_score,
                    "source": "hybrid",
                    "component_scores": (
                        component_scores
                    ),
                }
            )
        )

    return fused_results


class HybridRetriever:
    """组合 BM25 和 Dense 的混合检索器。"""

    def __init__(
        self,
        *,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
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
    ) -> None:
        if candidate_top_k <= 0:
            raise ValueError(
                "candidate_top_k 必须大于 0"
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k 必须大于 0"
            )

        self.bm25_retriever = (
            bm25_retriever
        )
        self.dense_retriever = (
            dense_retriever
        )
        self.candidate_top_k = (
            candidate_top_k
        )
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    @classmethod
    def load(
        cls,
        *,
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
    ) -> "HybridRetriever":
        """加载默认 BM25 和 Chroma 索引。"""

        return cls(
            bm25_retriever=(
                BM25Retriever.load()
            ),
            dense_retriever=(
                DenseRetriever.load()
            ),
            candidate_top_k=(
                candidate_top_k
            ),
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        """分别召回候选，然后执行 RRF。"""

        if not query.strip():
            raise ValueError("检索问题不能为空")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        # 如果调用方要求超过默认候选数量的结果，
        # 两路至少需要召回同样多的候选。
        candidate_limit = max(
            self.candidate_top_k,
            top_k,
        )

        bm25_results = (
            self.bm25_retriever.search(
                query,
                top_k=candidate_limit,
                law_name=law_name,
            )
        )

        dense_results = (
            self.dense_retriever.search(
                query,
                top_k=candidate_limit,
                law_name=law_name,
            )
        )

        return reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            top_k=top_k,
            rrf_k=self.rrf_k,
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
        )


def _format_rank(
    value: float | None,
) -> str:
    if value is None:
        return "-"

    return str(int(value))


def _print_results(
    results: list[RetrievalResult],
) -> None:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        components = result.component_scores

        bm25_rank = _format_rank(
            components.get("bm25_rank")
        )
        dense_rank = _format_rank(
            components.get("dense_rank")
        )

        preview = result.text.replace(
            "\n",
            " ",
        )[:160]

        print(
            f"{rank:>2}. "
            f"{result.law_name}"
            f"{result.article_no} "
            f"rrf={result.score:.8f} "
            f"bm25_rank={bm25_rank} "
            f"dense_rank={dense_rank}"
        )
        print(f"    chunk_id={result.chunk_id}")
        print(f"    {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BM25 + Dense RRF 混合检索"
    )

    parser.add_argument(
        "--query",
        required=True,
        help="用户问题",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="最终返回数量",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=DEFAULT_CANDIDATE_TOP_K,
        help="每一路的候选召回数量",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=DEFAULT_BM25_WEIGHT,
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
    )
    parser.add_argument(
        "--law-name",
        help="使用正式法律名称过滤",
    )

    arguments = parser.parse_args()

    retriever = HybridRetriever.load(
        candidate_top_k=(
            arguments.candidate_top_k
        ),
        rrf_k=arguments.rrf_k,
        bm25_weight=arguments.bm25_weight,
        dense_weight=arguments.dense_weight,
    )

    results = retriever.search(
        arguments.query,
        top_k=arguments.top_k,
        law_name=arguments.law_name,
    )

    _print_results(results)


if __name__ == "__main__":
    main()