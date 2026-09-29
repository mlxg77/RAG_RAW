"""项目统一命令行入口。"""

import argparse
from pathlib import Path

from rag_law.evaluation.retrieval_metrics import (
    DEFAULT_EVALUATION_PATH,
    DEFAULT_REPORT_PATH,
    run_baseline_evaluation,
)
from rag_law.retrieval.bm25 import (
    BM25Retriever,
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


def load_retriever(
    method: str,
    *,
    candidate_top_k: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
):
    """按照命令行参数加载指定检索器。"""

    if method == "bm25":
        return BM25Retriever.load()

    if method == "dense":
        return DenseRetriever.load()

    if method == "hybrid":
        bm25_retriever = (
            BM25Retriever.load()
        )
        dense_retriever = (
            DenseRetriever.load()
        )

        return HybridRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            candidate_top_k=(
                candidate_top_k
            ),
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    raise ValueError(
        f"未知检索方法：{method}"
    )


def print_results(
    results: list[RetrievalResult],
) -> None:
    if not results:
        print("没有检索到结果。")
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print()
        print(
            f"[{rank}] "
            f"《{result.law_name}》"
            f"{result.article_no}"
        )
        print(
            f"方法：{result.source} "
            f"分数：{result.score:.8f}"
        )
        print(
            f"chunk_id：{result.chunk_id}"
        )

        if result.chapter:
            print(f"章节：{result.chapter}")

        if result.page is not None:
            if (
                result.end_page is not None
                and result.end_page
                != result.page
            ):
                page_text = (
                    f"{result.page}-"
                    f"{result.end_page}"
                )
            else:
                page_text = str(result.page)

            print(f"页码：{page_text}")

        if result.source == "hybrid":
            bm25_rank = (
                result.component_scores.get(
                    "bm25_rank"
                )
            )
            dense_rank = (
                result.component_scores.get(
                    "dense_rank"
                )
            )

            print(
                "分路排名："
                f"BM25={format_rank(bm25_rank)}，"
                f"Dense={format_rank(dense_rank)}"
            )

        print("正文：")
        print(result.text)


def format_rank(
    rank: float | None,
) -> str:
    if rank is None:
        return "-"

    return str(int(rank))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="法条知识库检索工具"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    retrieve_parser = (
        subparsers.add_parser(
            "retrieve",
            help="检索法条",
        )
    )

    retrieve_parser.add_argument(
        "--method",
        choices=(
            "bm25",
            "dense",
            "hybrid",
        ),
        default="hybrid",
    )
    retrieve_parser.add_argument(
        "--query",
        help="用户问题；省略时进入交互输入",
    )
    retrieve_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    retrieve_parser.add_argument(
        "--law-name",
        help="按正式法律名称过滤",
    )
    retrieve_parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=DEFAULT_CANDIDATE_TOP_K,
    )
    retrieve_parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
    )
    retrieve_parser.add_argument(
        "--bm25-weight",
        type=float,
        default=DEFAULT_BM25_WEIGHT,
    )
    retrieve_parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
    )

    evaluate_parser = (
        subparsers.add_parser(
            "evaluate",
            help="运行完整检索基线评测",
        )
    )

    evaluate_parser.add_argument(
        "--questions",
        default=str(
            DEFAULT_EVALUATION_PATH
        ),
    )
    evaluate_parser.add_argument(
        "--output",
        default=str(
            DEFAULT_REPORT_PATH
        ),
    )
    evaluate_parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=DEFAULT_CANDIDATE_TOP_K,
    )
    evaluate_parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
    )
    evaluate_parser.add_argument(
        "--bm25-weight",
        type=float,
        default=DEFAULT_BM25_WEIGHT,
    )
    evaluate_parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
    )

    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.command == "retrieve":
        query = arguments.query

        if query is None:
            query = input(
                "请输入法律问题："
            ).strip()

        if not query:
            parser.error("问题不能为空")

        retriever = load_retriever(
            arguments.method,
            candidate_top_k=(
                arguments.candidate_top_k
            ),
            rrf_k=arguments.rrf_k,
            bm25_weight=(
                arguments.bm25_weight
            ),
            dense_weight=(
                arguments.dense_weight
            ),
        )

        results = retriever.search(
            query,
            top_k=arguments.top_k,
            law_name=arguments.law_name,
        )

        print_results(results)
        return

    if arguments.command == "evaluate":
        run_baseline_evaluation(
            evaluation_path=Path(
                arguments.questions
            ),
            report_path=Path(
                arguments.output
            ),
            candidate_top_k=(
                arguments.candidate_top_k
            ),
            rrf_k=arguments.rrf_k,
            bm25_weight=(
                arguments.bm25_weight
            ),
            dense_weight=(
                arguments.dense_weight
            ),
            show_progress=True,
        )
        return

    parser.error(
        f"未知命令：{arguments.command}"
    )


if __name__ == "__main__":
    main()