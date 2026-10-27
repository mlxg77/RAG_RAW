"""基于 LangGraph 的检索编排图。

BM25 与 Dense 并行召回，再依次执行 RRF 融合、条号补召回、精排
和上下文构建。图只负责编排，具体能力仍由现有检索器和上下文
构建器提供。
"""

import logging
import time
from collections.abc import Callable
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from rag_law.retrieval.fusion import (
    reciprocal_rank_fusion,
)
from rag_law.retrieval.reranker import (
    rerank_results,
)
from rag_law.schemas import (
    BuiltContext,
    RetrievalResult,
)


ProgressCallback = Callable[[str, str], None]


logger = logging.getLogger(__name__)


class SearchRetrieverProtocol(Protocol):
    """图中单个召回分支依赖的检索接口。"""

    def search(
        self,
        query: str,
        *,
        top_k: int,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        ...


class ScorerProtocol(Protocol):
    """精排节点依赖的最小打分接口。"""

    def score_documents(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        ...


class ContextBuilderProtocol(Protocol):
    """上下文节点依赖的构建接口。"""

    def build(
        self,
        *,
        query: str,
        results: list[RetrievalResult],
    ) -> BuiltContext:
        ...


class ArticleAnchorProtocol(Protocol):
    """按条号补召回的最小接口。"""

    def resolve(
        self,
        query: str,
        *,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        ...


class RetrievalGraphState(TypedDict, total=False):
    """检索图在一次请求中流转的全部状态。"""

    query: str
    law_name: str | None
    candidate_top_k: int
    rrf_k: int
    bm25_weight: float
    dense_weight: float
    rerank_top_n: int

    bm25_results: list[RetrievalResult]
    dense_results: list[RetrievalResult]
    fused_results: list[RetrievalResult]
    reranked_results: list[RetrievalResult]
    context: BuiltContext

    retrieval_started_at: float
    retrieval_latency_ms: float
    context_latency_ms: float
    progress_callback: ProgressCallback | None


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
) -> None:
    if callback is not None:
        callback(stage, message)


def build_retrieval_graph(
    *,
    bm25_retriever: SearchRetrieverProtocol,
    dense_retriever: SearchRetrieverProtocol,
    scorer: ScorerProtocol,
    context_builder: ContextBuilderProtocol,
    candidate_top_k: int,
    rrf_k: int,
    bm25_weight: float,
    dense_weight: float,
    rerank_top_n: int,
    article_anchor: ArticleAnchorProtocol | None = None,
):
    """编译并行召回 -> 融合 -> 补召回 -> 精排 -> 上下文 检索图。"""

    def bm25_search(
        state: RetrievalGraphState,
    ) -> dict[str, object]:
        return {
            "bm25_results": bm25_retriever.search(
                state["query"],
                top_k=state["candidate_top_k"],
                law_name=state.get("law_name"),
            )
        }

    def dense_search(
        state: RetrievalGraphState,
    ) -> dict[str, object]:
        return {
            "dense_results": dense_retriever.search(
                state["query"],
                top_k=state["candidate_top_k"],
                law_name=state.get("law_name"),
            )
        }

    def rrf_fuse(
        state: RetrievalGraphState,
    ) -> dict[str, object]:
        return {
            "fused_results": reciprocal_rank_fusion(
                state.get("bm25_results", []),
                state.get("dense_results", []),
                top_k=state["candidate_top_k"],
                rrf_k=state["rrf_k"],
                bm25_weight=state["bm25_weight"],
                dense_weight=state["dense_weight"],
            )
        }

    def anchor_articles(
        state: RetrievalGraphState,
    ) -> dict[str, object]:
        """查询点名了「第几条」时，把那条法条直接补进候选集。"""

        if article_anchor is None:
            return {}

        fused = state.get("fused_results", [])

        extra = article_anchor.resolve(
            state["query"],
            law_name=state.get("law_name"),
        )

        if not extra:
            return {}

        existing = {item.chunk_id for item in fused}
        fresh = [
            item
            for item in extra
            if item.chunk_id not in existing
        ]

        if not fresh:
            return {}

        logger.info(
            "[RAG] anchored candidates=%s",
            [item.article_no for item in fresh],
        )

        # 放在最前面：候选顺序不参与精排打分，
        # 但保留「补召回优先入选」的语义。
        return {"fused_results": fresh + list(fused)}

    def rerank(
        state: RetrievalGraphState,
    ) -> dict[str, object]:
        return {
            "reranked_results": rerank_results(
                query=state["query"],
                candidates=state.get("fused_results", []),
                scorer=scorer,
                top_n=state["rerank_top_n"],
            )
        }

    def build_context(
        state: RetrievalGraphState,
    ) -> dict[str, object]:
        _emit_progress(
            state.get("progress_callback"),
            "context_building",
            "正在筛选相邻法条并构建证据上下文…",
        )

        started_at = time.perf_counter()

        context = context_builder.build(
            query=state["query"],
            results=state.get("reranked_results", []),
        )

        return {
            "context": context,
            "retrieval_latency_ms": (
                started_at - state["retrieval_started_at"]
            )
            * 1000.0,
            "context_latency_ms": (
                time.perf_counter() - started_at
            )
            * 1000.0,
        }

    builder = StateGraph(RetrievalGraphState)
    builder.add_node("bm25_search", bm25_search)
    builder.add_node("dense_search", dense_search)
    builder.add_node("rrf_fuse", rrf_fuse)
    builder.add_node(
        "anchor_articles", anchor_articles
    )
    builder.add_node("rerank", rerank)
    builder.add_node("build_context", build_context)

    builder.add_edge(START, "bm25_search")
    builder.add_edge(START, "dense_search")
    builder.add_edge(
        ["bm25_search", "dense_search"],
        "rrf_fuse",
    )
    builder.add_edge("rrf_fuse", "anchor_articles")
    builder.add_edge("anchor_articles", "rerank")
    builder.add_edge("rerank", "build_context")
    builder.add_edge("build_context", END)

    return builder.compile()


def build_single_retriever_graph(
    *,
    retriever: SearchRetrieverProtocol,
    context_builder: ContextBuilderProtocol,
):
    """编译只依赖单个检索器的兼容图。

    用于测试替身注入等场景：检索器内部可以自行组合召回与精排，
    图只负责“检索 -> 上下文”两步。
    """

    def retrieve(
        state: RetrievalGraphState,
    ) -> dict[str, object]:
        return {
            "reranked_results": retriever.search(
                state["query"],
                top_k=state["rerank_top_n"],
                law_name=state.get("law_name"),
            )
        }

    def build_context(
        state: RetrievalGraphState,
    ) -> dict[str, object]:
        _emit_progress(
            state.get("progress_callback"),
            "context_building",
            "正在筛选相邻法条并构建证据上下文…",
        )

        started_at = time.perf_counter()

        context = context_builder.build(
            query=state["query"],
            results=state.get("reranked_results", []),
        )

        return {
            "context": context,
            "retrieval_latency_ms": (
                started_at - state["retrieval_started_at"]
            )
            * 1000.0,
            "context_latency_ms": (
                time.perf_counter() - started_at
            )
            * 1000.0,
        }

    builder = StateGraph(RetrievalGraphState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("build_context", build_context)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "build_context")
    builder.add_edge("build_context", END)

    return builder.compile()
