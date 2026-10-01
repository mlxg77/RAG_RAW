"""项目统一命令行入口。"""

import argparse
from pathlib import Path

from rag_law.evaluation.retrieval_metrics import (
    DEFAULT_EVALUATION_PATH,
    DEFAULT_REPORT_PATH,
    run_baseline_evaluation,
)
from rag_law.generation.answer_pipeline import (
    DEFAULT_MIN_ANSWER_SCORE,
    AnswerPipeline,
)
from rag_law.retrieval.bm25 import (
    BM25Retriever,
)
from rag_law.retrieval.context_builder import (
    DEFAULT_ADJACENT_FOR_TOP_N,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_EVIDENCES,
    DEFAULT_MAX_EVIDENCE_TOKENS,
    DEFAULT_MIN_ADJACENT_SCORE,
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
from rag_law.retrieval.reranker import (
    DEFAULT_RERANK_TOP_N,
)
from rag_law.retrieval.retrieval_context_pipeline import (
    RetrievalContextPipeline,
)
from rag_law.schemas import (
    AnswerPipelineResult,
    RetrievalContextResult,
    RetrievalResult,
)


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


def print_context_result(
    output: RetrievalContextResult,
) -> None:
    """打印阶段 3 的精排结果和最终上下文。"""

    print()
    print("=== 阶段 3 执行结果 ===")
    print(f"问题：{output.query}")
    print(
        "精排检索耗时："
        f"{output.retrieval_latency_ms:.1f} ms"
    )
    print(
        "上下文构建耗时："
        f"{output.context_latency_ms:.1f} ms"
    )
    print(
        "总耗时："
        f"{output.total_latency_ms:.1f} ms"
    )

    print()
    print("=== Rerank 排名 ===")

    if not output.reranked_results:
        print("没有检索到候选法条。")
    else:
        for rank, result in enumerate(
            output.reranked_results,
            start=1,
        ):
            hybrid_rank = (
                result.component_scores.get(
                    "rerank_input_rank"
                )
            )

            print(
                f"{rank}. "
                f"《{result.law_name}》"
                f"{result.article_no} "
                f"rerank={result.score:.6f} "
                "hybrid_rank="
                f"{format_rank(hybrid_rank)}"
            )

    print()
    print("=== 上下文摘要 ===")
    print(
        "证据数量："
        f"{len(output.context.evidences)}"
    )
    print(
        "估算 token："
        f"{output.context.estimated_tokens}"
        f"/{output.context.max_tokens}"
    )
    print(
        "发生截断或省略："
        f"{output.context.truncated}"
    )

    print()
    print("=== 证据映射 ===")

    if not output.context.evidences:
        print("没有可用证据。")
    else:
        for evidence in output.context.evidences:
            print(
                f"{evidence.evidence_id} -> "
                f"{evidence.chunk_id} | "
                f"《{evidence.law_name}》"
                f"{evidence.article_no} | "
                f"{evidence.relation} | "
                f"excerpt={evidence.is_excerpt}"
            )

    print()
    print("=== 模型上下文 ===")

    if output.context.text:
        print(output.context.text)
    else:
        print("当前没有可提供给模型的上下文。")


ANSWER_STATUS_LABELS = {
    "answered": "已回答",
    "needs_clarification": "需要补充事实",
    "insufficient_evidence": "证据不足",
    "generation_failed": "生成或校验失败",
}


def format_source_location(
    *,
    source_file: str,
    page: int | None,
    end_page: int | None,
) -> str:
    """格式化引用的原始文件位置。"""

    if page is None:
        return source_file

    if (
        end_page is not None
        and end_page != page
    ):
        page_text = f"{page}-{end_page}"
    else:
        page_text = str(page)

    return f"{source_file}，第 {page_text} 页"


def print_answer_result(
    output: AnswerPipelineResult,
    *,
    debug: bool = False,
) -> None:
    """打印阶段 4 的最终回答。"""

    answer = output.answer

    print()
    print("=== 阶段 4 回答结果 ===")
    print(f"问题：{output.query}")
    print(
        "状态："
        f"{ANSWER_STATUS_LABELS[answer.status]}"
        f"（{answer.status}）"
    )

    print()
    print("=== 简明结论 ===")
    print(answer.summary)

    print()
    print("=== 分析 ===")
    print(answer.analysis)

    print()
    print("=== 适用条件或待确认事实 ===")

    if answer.conditions:
        for condition in answer.conditions:
            print(f"- {condition}")
    else:
        print("- 无")

    print()
    print("=== 法条引用 ===")

    if not answer.citations:
        print("没有可展示的已校验引用。")
    else:
        for index, citation in enumerate(
            answer.citations,
            start=1,
        ):
            print()
            print(
                f"[{index}] "
                f"《{citation.law_name}》"
                f"{citation.article_no}"
            )
            print(
                "证据映射："
                f"{citation.evidence_id} -> "
                f"{citation.chunk_id}"
            )
            print(
                "来源："
                + format_source_location(
                    source_file=(
                        citation.source_file
                    ),
                    page=citation.page,
                    end_page=citation.end_page,
                )
            )
            print("原文：")
            print(citation.quote)

    print()
    print("=== 回答限制 ===")
    print(answer.limitations)

    if answer.follow_up_question is not None:
        print()
        print("=== 需要补充的问题 ===")
        print(answer.follow_up_question)

    print()
    print("=== 运行摘要 ===")

    if output.top_relevance_score is None:
        score_text = "-"
    else:
        score_text = (
            f"{output.top_relevance_score:.6f}"
        )

    print(f"最高相关度：{score_text}")
    print(
        "模型生成次数："
        f"{output.generation_attempts}"
    )
    print(
        "生成耗时："
        f"{output.generation_latency_ms:.1f} ms"
    )
    print(
        "总耗时："
        f"{output.total_latency_ms:.1f} ms"
    )

    if not debug:
        return

    print()
    print("=== 调试信息 ===")
    print(
        "Prompt 版本："
        f"{output.prompt_version}"
    )
    print(
        "检索耗时："
        f"{output.retrieval.retrieval_latency_ms:.1f} ms"
    )
    print(
        "上下文构建耗时："
        f"{output.retrieval.context_latency_ms:.1f} ms"
    )
    print(
        "上下文证据数："
        f"{len(output.retrieval.context.evidences)}"
    )

    if output.validation_errors:
        print("引用或生成诊断：")

        for error in output.validation_errors:
            print(f"- {error}")
    else:
        print("引用或生成诊断：无")


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

    context_parser = (
        subparsers.add_parser(
            "context",
            help="执行混合检索、精排和上下文构建",
        )
    )

    context_parser.add_argument(
        "--query",
        help="用户问题；省略时进入交互输入",
    )
    context_parser.add_argument(
        "--law-name",
        help="按正式法律名称过滤",
    )
    context_parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=DEFAULT_CANDIDATE_TOP_K,
    )
    context_parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=DEFAULT_RERANK_TOP_N,
    )
    context_parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
    )
    context_parser.add_argument(
        "--bm25-weight",
        type=float,
        default=DEFAULT_BM25_WEIGHT,
    )
    context_parser.add_argument(
        "--dense-weight",
        type=float,
        default=DEFAULT_DENSE_WEIGHT,
    )
    context_parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
    )
    context_parser.add_argument(
        "--max-evidences",
        type=int,
        default=DEFAULT_MAX_EVIDENCES,
    )
    context_parser.add_argument(
        "--max-evidence-tokens",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_TOKENS,
    )
    context_parser.add_argument(
        "--adjacent-for-top-n",
        type=int,
        default=DEFAULT_ADJACENT_FOR_TOP_N,
    )
    context_parser.add_argument(
        "--min-adjacent-score",
        type=float,
        default=DEFAULT_MIN_ADJACENT_SCORE,
    )

    answer_parser = (
        subparsers.add_parser(
            "answer",
            help="执行检索、生成和引用校验",
        )
    )

    answer_parser.add_argument(
        "--query",
        help="用户问题；省略时进入交互输入",
    )
    answer_parser.add_argument(
        "--law-name",
        help="按正式法律名称过滤",
    )
    answer_parser.add_argument(
        "--min-answer-score",
        type=float,
        default=DEFAULT_MIN_ANSWER_SCORE,
        help=(
            "允许调用回答模型的最低 Top1 "
            "Rerank 分数"
        ),
    )
    answer_parser.add_argument(
        "--debug",
        action="store_true",
        help="显示内部诊断和分阶段耗时",
    )
    answer_parser.add_argument(
        "--json",
        action="store_true",
        help="输出完整 JSON 结果",
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

    if arguments.command == "context":
        query = arguments.query

        if query is None:
            query = input(
                "请输入法律问题："
            ).strip()

        if not query:
            parser.error("问题不能为空")

        try:
            pipeline = RetrievalContextPipeline.load(
                candidate_top_k=(
                    arguments.candidate_top_k
                ),
                rerank_top_n=(
                    arguments.rerank_top_n
                ),
                rrf_k=arguments.rrf_k,
                bm25_weight=(
                    arguments.bm25_weight
                ),
                dense_weight=(
                    arguments.dense_weight
                ),
                max_context_tokens=(
                    arguments.max_context_tokens
                ),
                max_evidences=(
                    arguments.max_evidences
                ),
                max_evidence_tokens=(
                    arguments.max_evidence_tokens
                ),
                adjacent_for_top_n=(
                    arguments.adjacent_for_top_n
                ),
                min_adjacent_score=(
                    arguments.min_adjacent_score
                ),
            )

            output = pipeline.run(
                query,
                law_name=arguments.law_name,
            )
        except ValueError as error:
            parser.error(str(error))

        print_context_result(output)
        return

    if arguments.command == "answer":
        query = arguments.query

        if query is None:
            query = input(
                "请输入法律问题："
            ).strip()

        if not query:
            parser.error("问题不能为空")

        try:
            pipeline = AnswerPipeline.load(
                min_answer_score=(
                    arguments.min_answer_score
                )
            )

            output = pipeline.run(
                query,
                law_name=arguments.law_name,
            )
        except ValueError as error:
            parser.error(str(error))

        if arguments.json:
            print(
                output.model_dump_json(
                    indent=2,
                )
            )
        else:
            print_answer_result(
                output,
                debug=arguments.debug,
            )

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
