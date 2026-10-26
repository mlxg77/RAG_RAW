"""检索上下文流水线：基于 LangGraph 的混合检索、精排和上下文构建。"""

import time
from collections.abc import Callable
from typing import Protocol

from rag_law.retrieval.bm25 import (
    BM25Retriever,
)
from rag_law.retrieval.context_builder import (
    DEFAULT_ADJACENT_FOR_TOP_N,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_EVIDENCES,
    DEFAULT_MAX_EVIDENCE_TOKENS,
    DEFAULT_MIN_ADJACENT_SCORE,
    ContextBuilder,
)
from rag_law.retrieval.dense import (
    DenseRetriever,
)
from rag_law.retrieval.fusion import (
    DEFAULT_BM25_WEIGHT,
    DEFAULT_CANDIDATE_TOP_K,
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_RRF_K,
)
from rag_law.retrieval.reranker import (
    DEFAULT_RERANK_TOP_N,
    SiliconFlowReranker,
)
from rag_law.retrieval.retrieval_graph import (
    build_retrieval_graph,
    build_single_retriever_graph,
)
from rag_law.schemas import (
    BuiltContext,
    RetrievalContextResult,
    RetrievalResult,
)


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


# 旧名称保留，避免外部引用失效。
RerankedRetrieverProtocol = SearchRetrieverProtocol


class ScorerProtocol(Protocol):
    """精排节点依赖的最小打分接口。"""

    def score_documents(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        ...


class ContextBuilderProtocol(Protocol):
    """流水线依赖的上下文构建接口。"""

    def build(
        self,
        *,
        query: str,
        results: list[RetrievalResult],
    ) -> BuiltContext:
        ...


class RetrievalContextPipeline:
    """组合检索召回、RRF 融合、精排和上下文构建。

    真实运行时使用 BM25 与 Dense 并行召回的图；
    只注入单个检索器时退化为“检索 -> 上下文”两步图，
    方便单元测试替换底层组件。
    """

    def __init__(
        self,
        *,
        retriever: SearchRetrieverProtocol | None = None,
        context_builder: ContextBuilderProtocol,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
        bm25_retriever: SearchRetrieverProtocol | None = None,
        dense_retriever: SearchRetrieverProtocol | None = None,
        scorer: ScorerProtocol | None = None,
        candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
    ) -> None:
        if rerank_top_n <= 0:
            raise ValueError(
                "rerank_top_n 必须大于 0"
            )

        self.retriever = retriever
        self.context_builder = context_builder
        self.rerank_top_n = rerank_top_n
        self.candidate_top_k = candidate_top_k
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        parallel_ready = (
            bm25_retriever is not None
            and dense_retriever is not None
            and scorer is not None
        )

        if parallel_ready:
            if candidate_top_k <= 0:
                raise ValueError(
                    "candidate_top_k 必须大于 0"
                )

            if rerank_top_n > candidate_top_k:
                raise ValueError(
                    "rerank_top_n 不能大于 candidate_top_k"
                )

            self._graph = build_retrieval_graph(
                bm25_retriever=bm25_retriever,
                dense_retriever=dense_retriever,
                scorer=scorer,
                context_builder=context_builder,
                candidate_top_k=candidate_top_k,
                rrf_k=rrf_k,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                rerank_top_n=rerank_top_n,
            )
            return

        if retriever is None:
            raise ValueError(
                "必须提供 retriever，"
                "或同时提供 bm25_retriever、"
                "dense_retriever 和 scorer"
            )

        self._graph = build_single_retriever_graph(
            retriever=retriever,
            context_builder=context_builder,
        )

    @classmethod
    def load(
        cls,
        *,
        candidate_top_k: int = DEFAULT_CANDIDATE_TOP_K,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
        rrf_k: int = DEFAULT_RRF_K,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        dense_weight: float = DEFAULT_DENSE_WEIGHT,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_evidences: int = DEFAULT_MAX_EVIDENCES,
        max_evidence_tokens: int = DEFAULT_MAX_EVIDENCE_TOKENS,
        adjacent_for_top_n: int = DEFAULT_ADJACENT_FOR_TOP_N,
        min_adjacent_score: float = DEFAULT_MIN_ADJACENT_SCORE,
    ) -> "RetrievalContextPipeline":
        """加载阶段 3 所需的全部真实组件。"""

        if candidate_top_k <= 0:
            raise ValueError(
                "candidate_top_k 必须大于 0"
            )

        if rerank_top_n > candidate_top_k:
            raise ValueError(
                "rerank_top_n 不能大于 candidate_top_k"
            )

        # 检索精排、邻条精排和长法条片段选择共用同一客户端配置。
        scorer = SiliconFlowReranker.from_settings()

        context_builder = ContextBuilder.load(
            scorer=scorer,
            max_context_tokens=max_context_tokens,
            max_evidences=max_evidences,
            max_evidence_tokens=max_evidence_tokens,
            adjacent_for_top_n=adjacent_for_top_n,
            min_adjacent_score=min_adjacent_score,
        )

        return cls(
            context_builder=context_builder,
            rerank_top_n=rerank_top_n,
            bm25_retriever=BM25Retriever.load(),
            dense_retriever=DenseRetriever.load(),
            scorer=scorer,
            candidate_top_k=candidate_top_k,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    def run(
        self,
        query: str,
        *,
        law_name: str | None = None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> RetrievalContextResult:
        """执行完整的阶段 3 流水线。"""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("查询问题不能为空")

        started_at = time.perf_counter()

        if progress_callback is not None:
            progress_callback(
                "retrieval",
                "正在进行混合检索与法条重排…",
            )

        state = self._graph.invoke(
            {
                "query": normalized_query,
                "law_name": law_name,
                "candidate_top_k": self.candidate_top_k,
                "rrf_k": self.rrf_k,
                "bm25_weight": self.bm25_weight,
                "dense_weight": self.dense_weight,
                "rerank_top_n": self.rerank_top_n,
                "retrieval_started_at": started_at,
                "progress_callback": progress_callback,
            }
        )

        return RetrievalContextResult(
            query=normalized_query,
            reranked_results=state.get(
                "reranked_results", []
            ),
            context=state["context"],
            retrieval_latency_ms=state.get(
                "retrieval_latency_ms", 0.0
            ),
            context_latency_ms=state.get(
                "context_latency_ms", 0.0
            ),
            total_latency_ms=(
                time.perf_counter() - started_at
            )
            * 1000.0,
        )
