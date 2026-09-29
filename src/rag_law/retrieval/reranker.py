"""使用远程 Cross-Encoder 对混合检索候选进行精排。"""

import argparse
from typing import Protocol

import httpx

from rag_law.retrieval.fusion import (
    DEFAULT_CANDIDATE_TOP_K,
    HybridRetriever,
)
from rag_law.schemas import RetrievalResult


DEFAULT_RERANK_TOP_N = 5
DEFAULT_RERANK_TIMEOUT_SECONDS = 30.0


class RerankScorerProtocol(Protocol):
    """上下文构建和测试依赖的最小重排接口。"""

    def score_documents(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        ...


class RetrieverProtocol(Protocol):
    """RerankedRetriever 依赖的基础检索接口。"""

    def search(
        self,
        query: str,
        *,
        top_k: int,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        ...


def build_rerank_text(
    result: RetrievalResult,
) -> str:
    """构造传给 Reranker 的候选文本。

    Reranker 应同时看到法律名、编章节、条号和正文，
    不能只看到正文。
    """

    fields = [
        result.law_name,
        result.part,
        result.chapter,
        result.section,
        result.article_no,
        result.text,
    ]

    return "\n".join(
        field.strip()
        for field in fields
        if field is not None and field.strip()
    )


class SiliconFlowReranker:
    """调用 SiliconFlow OpenAI 风格的 Rerank API。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = (
            DEFAULT_RERANK_TIMEOUT_SECONDS
        ),
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError(
                "Rerank base_url 不能为空"
            )

        if not api_key.strip():
            raise ValueError(
                "Rerank api_key 不能为空"
            )

        if not model.strip():
            raise ValueError(
                "Rerank model 不能为空"
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds 必须大于 0"
            )

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.client = client

    @classmethod
    def from_settings(
        cls,
    ) -> "SiliconFlowReranker":
        """根据项目配置创建远程 Reranker。

        config 放在函数内部导入，使单元测试不依赖真实环境变量。
        """

        from rag_law.config import settings

        return cls(
            base_url=settings.rerank_base_url,
            api_key=settings.rerank_api_key,
            model=settings.rerank_model,
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/rerank"

    def _post(
        self,
        payload: dict[str, object],
    ) -> httpx.Response:
        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
        }

        if self.client is not None:
            return self.client.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )

        with httpx.Client(
            timeout=self.timeout_seconds
        ) as client:
            return client.post(
                self.endpoint,
                headers=headers,
                json=payload,
            )

    def score_documents(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """返回与 documents 原始顺序一一对应的相关性分数。"""

        if not query.strip():
            raise ValueError(
                "重排问题不能为空"
            )

        if not documents:
            return []

        if any(
            not document.strip()
            for document in documents
        ):
            raise ValueError(
                "重排候选文本不能为空"
            )

        payload: dict[str, object] = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }

        response = self._post(payload)
        response.raise_for_status()

        try:
            response_data = response.json()
        except ValueError as error:
            raise ValueError(
                "Rerank API 返回的不是合法 JSON"
            ) from error

        raw_results = response_data.get(
            "results"
        )

        if not isinstance(raw_results, list):
            raise ValueError(
                "Rerank API 响应缺少 results 列表"
            )

        scores: list[float | None] = [
            None
        ] * len(documents)

        for item in raw_results:
            if not isinstance(item, dict):
                raise ValueError(
                    "Rerank results 项格式错误"
                )

            index = item.get("index")
            relevance_score = item.get(
                "relevance_score"
            )

            if (
                not isinstance(index, int)
                or index < 0
                or index >= len(documents)
            ):
                raise ValueError(
                    "Rerank API 返回了非法 index"
                )

            if scores[index] is not None:
                raise ValueError(
                    "Rerank API 返回了重复 index"
                )

            try:
                scores[index] = float(
                    relevance_score
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Rerank API 返回了非法分数"
                ) from error

        if any(
            score is None
            for score in scores
        ):
            raise ValueError(
                "Rerank API 没有返回全部候选的分数"
            )

        return [
            float(score)
            for score in scores
            if score is not None
        ]


def rerank_results(
    *,
    query: str,
    candidates: list[RetrievalResult],
    scorer: RerankScorerProtocol,
    top_n: int = DEFAULT_RERANK_TOP_N,
) -> list[RetrievalResult]:
    """根据 Cross-Encoder 分数重新排列候选结果。"""

    if not query.strip():
        raise ValueError(
            "重排问题不能为空"
        )

    if top_n <= 0:
        raise ValueError(
            "top_n 必须大于 0"
        )

    if not candidates:
        return []

    seen_chunk_ids: set[str] = set()

    for candidate in candidates:
        if candidate.chunk_id in seen_chunk_ids:
            raise ValueError(
                "重排候选包含重复 chunk_id："
                f"{candidate.chunk_id}"
            )

        seen_chunk_ids.add(
            candidate.chunk_id
        )

    documents = [
        build_rerank_text(candidate)
        for candidate in candidates
    ]

    scores = scorer.score_documents(
        query,
        documents,
    )

    if len(scores) != len(candidates):
        raise ValueError(
            "Reranker 返回的分数数量与候选数量不一致"
        )

    reranked: list[
        tuple[int, RetrievalResult]
    ] = []

    for input_rank, (
        candidate,
        rerank_score,
    ) in enumerate(
        zip(candidates, scores, strict=True),
        start=1,
    ):
        component_scores = dict(
            candidate.component_scores
        )

        # 原始 RRF 分数已位于 component_scores["rrf"]。
        # 这里追加重排信息，不覆盖原始检索证据。
        component_scores["rerank"] = float(
            rerank_score
        )
        component_scores[
            "rerank_input_rank"
        ] = float(input_rank)

        reranked_result = candidate.model_copy(
            update={
                "score": float(rerank_score),
                "source": "rerank",
                "component_scores": (
                    component_scores
                ),
            }
        )

        reranked.append(
            (input_rank, reranked_result)
        )

    # 分数相同时保持原始 Hybrid 顺序；
    # chunk_id 是最后的确定性排序条件。
    reranked.sort(
        key=lambda item: (
            -item[1].score,
            item[0],
            item[1].chunk_id,
        )
    )

    return [
        result
        for _, result in reranked[:top_n]
    ]


class RerankedRetriever:
    """组合 HybridRetriever 和 Reranker。"""

    def __init__(
        self,
        *,
        base_retriever: RetrieverProtocol,
        scorer: RerankScorerProtocol,
        candidate_top_k: int = (
            DEFAULT_CANDIDATE_TOP_K
        ),
    ) -> None:
        if candidate_top_k <= 0:
            raise ValueError(
                "candidate_top_k 必须大于 0"
            )

        self.base_retriever = (
            base_retriever
        )
        self.scorer = scorer
        self.candidate_top_k = (
            candidate_top_k
        )

    @classmethod
    def load(
        cls,
        *,
        candidate_top_k: int = (
            DEFAULT_CANDIDATE_TOP_K
        ),
    ) -> "RerankedRetriever":
        """加载默认 HybridRetriever 和远程 Reranker。"""

        return cls(
            base_retriever=(
                HybridRetriever.load(
                    candidate_top_k=(
                        candidate_top_k
                    )
                )
            ),
            scorer=(
                SiliconFlowReranker.from_settings()
            ),
            candidate_top_k=candidate_top_k,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_RERANK_TOP_N,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        """召回 Hybrid 前 10，再返回精排后的前 top_k。"""

        if not query.strip():
            raise ValueError(
                "检索问题不能为空"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k 必须大于 0"
            )

        if top_k > self.candidate_top_k:
            raise ValueError(
                "top_k 不能大于重排候选数量"
            )

        candidates = (
            self.base_retriever.search(
                query,
                top_k=self.candidate_top_k,
                law_name=law_name,
            )
        )

        return rerank_results(
            query=query,
            candidates=candidates,
            scorer=self.scorer,
            top_n=top_k,
        )


def print_results(
    results: list[RetrievalResult],
) -> None:
    for rank, result in enumerate(
        results,
        start=1,
    ):
        input_rank = (
            result.component_scores.get(
                "rerank_input_rank"
            )
        )

        preview = result.text.replace(
            "\n",
            " ",
        )[:160]

        print()
        print(
            f"{rank:>2}. "
            f"《{result.law_name}》"
            f"{result.article_no}"
        )
        print(
            f"    rerank={result.score:.6f} "
            f"hybrid_rank={int(input_rank or 0)}"
        )
        print(
            f"    chunk_id={result.chunk_id}"
        )
        print(f"    {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "对 Hybrid 候选法条进行精排"
        )
    )

    parser.add_argument(
        "--query",
        required=True,
        help="用户法律问题",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_RERANK_TOP_N,
        help="精排后返回数量",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=DEFAULT_CANDIDATE_TOP_K,
        help="传给 Reranker 的 Hybrid 候选数量",
    )
    parser.add_argument(
        "--law-name",
        help="按正式法律名称过滤",
    )

    arguments = parser.parse_args()

    retriever = RerankedRetriever.load(
        candidate_top_k=(
            arguments.candidate_top_k
        )
    )

    results = retriever.search(
        arguments.query,
        top_k=arguments.top_n,
        law_name=arguments.law_name,
    )

    print_results(results)


if __name__ == "__main__":
    main()