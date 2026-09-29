"""检索上下文流水线：混合检索、精排和上下文构建。"""

import time
from typing import Protocol

from rag_law.retrieval.context_builder import (
    DEFAULT_ADJACENT_FOR_TOP_N,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_MAX_EVIDENCES,
    DEFAULT_MAX_EVIDENCE_TOKENS,
    DEFAULT_MIN_ADJACENT_SCORE,
    ContextBuilder,
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
    RerankedRetriever,
    SiliconFlowReranker,
)
from rag_law.schemas import (
    BuiltContext,
    RetrievalContextResult,
    RetrievalResult,
)


class RerankedRetrieverProtocol(Protocol):
    """流水线依赖的精排检索接口。"""

    def search(
        self,
        query: str,
        *,
        top_k: int,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
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
    """组合精排检索器和上下文构建器。"""

    def __init__(
        self,
        *,
        retriever: RerankedRetrieverProtocol,
        context_builder: ContextBuilderProtocol,
        rerank_top_n: int = DEFAULT_RERANK_TOP_N,
    ) -> None:
        if rerank_top_n <= 0:
            raise ValueError(
                "rerank_top_n 必须大于 0"
            )

        self.retriever = retriever
        self.context_builder = context_builder
        self.rerank_top_n = rerank_top_n

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

        hybrid_retriever = HybridRetriever.load(
            candidate_top_k=candidate_top_k,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

        reranked_retriever = RerankedRetriever(
            base_retriever=hybrid_retriever,
            scorer=scorer,
            candidate_top_k=candidate_top_k,
        )

        context_builder = ContextBuilder.load(
            scorer=scorer,
            max_context_tokens=max_context_tokens,
            max_evidences=max_evidences,
            max_evidence_tokens=max_evidence_tokens,
            adjacent_for_top_n=adjacent_for_top_n,
            min_adjacent_score=min_adjacent_score,
        )

        return cls(
            retriever=reranked_retriever,
            context_builder=context_builder,
            rerank_top_n=rerank_top_n,
        )

    def run(
        self,
        query: str,
        *,
        law_name: str | None = None,
    ) -> RetrievalContextResult:
        """执行完整的阶段 3 流水线。"""

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("查询问题不能为空")

        total_started_at = time.perf_counter()
        retrieval_started_at = time.perf_counter()

        reranked_results = self.retriever.search(
            normalized_query,
            top_k=self.rerank_top_n,
            law_name=law_name,
        )

        retrieval_latency_ms = (
            time.perf_counter()
            - retrieval_started_at
        ) * 1000.0

        context_started_at = time.perf_counter()

        context = self.context_builder.build(
            query=normalized_query,
            results=reranked_results,
        )

        context_latency_ms = (
            time.perf_counter()
            - context_started_at
        ) * 1000.0

        total_latency_ms = (
            time.perf_counter()
            - total_started_at
        ) * 1000.0

        return RetrievalContextResult(
            query=normalized_query,
            reranked_results=reranked_results,
            context=context,
            retrieval_latency_ms=retrieval_latency_ms,
            context_latency_ms=context_latency_ms,
            total_latency_ms=total_latency_ms,
        )
