import json

import httpx

from rag_law.retrieval.reranker import (
    RerankedRetriever,
    SiliconFlowReranker,
    build_rerank_text,
    rerank_results,
)
from rag_law.schemas import RetrievalResult


def make_result(
    *,
    chunk_id: str,
    article_no: str,
    text: str,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        law_id="test-law",
        law_name="中华人民共和国测试法",
        article_no=article_no,
        text=text,
        source_file="data/raw/test.md",
        score=score,
        source="hybrid",
        component_scores={
            "rrf": score,
        },
    )


class FakeScorer:
    """根据测试文本中的关键词返回固定分数。"""

    def score_documents(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        assert query

        scores: list[float] = []

        for document in documents:
            if "书面劳动合同" in document:
                scores.append(0.95)
            elif "七日内退货" in document:
                scores.append(0.70)
            else:
                scores.append(0.10)

        return scores


def test_build_rerank_text_contains_metadata() -> None:
    result = make_result(
        chunk_id="chunk-1",
        article_no="第八十二条",
        text="未订立书面劳动合同的，应当支付二倍工资。",
        score=0.03,
    )

    text = build_rerank_text(result)

    assert "中华人民共和国测试法" in text
    assert "第八十二条" in text
    assert "书面劳动合同" in text


def test_rerank_results_changes_order() -> None:
    candidates = [
        make_result(
            chunk_id="irrelevant",
            article_no="第一条",
            text="这是无关法条。",
            score=0.05,
        ),
        make_result(
            chunk_id="labor",
            article_no="第二条",
            text="建立劳动关系，应当订立书面劳动合同。",
            score=0.04,
        ),
        make_result(
            chunk_id="consumer",
            article_no="第三条",
            text="消费者有权在七日内退货。",
            score=0.03,
        ),
    ]

    results = rerank_results(
        query="公司一直没有和我签书面劳动合同",
        candidates=candidates,
        scorer=FakeScorer(),
        top_n=2,
    )

    assert [
        result.chunk_id
        for result in results
    ] == ["labor", "consumer"]

    assert results[0].source == "rerank"
    assert results[0].score == 0.95

    assert (
        results[0].component_scores["rrf"]
        == 0.04
    )
    assert (
        results[0].component_scores[
            "rerank_input_rank"
        ]
        == 2.0
    )


def test_rerank_rejects_duplicate_chunk_id() -> None:
    duplicated = make_result(
        chunk_id="duplicate",
        article_no="第一条",
        text="重复法条。",
        score=0.05,
    )

    try:
        rerank_results(
            query="测试问题",
            candidates=[
                duplicated,
                duplicated,
            ],
            scorer=FakeScorer(),
        )
    except ValueError as error:
        assert "重复 chunk_id" in str(error)
    else:
        raise AssertionError(
            "重复 chunk_id 应当抛出错误"
        )


class StubRetriever:
    def __init__(
        self,
        results: list[RetrievalResult],
    ) -> None:
        self.results = results
        self.calls: list[
            tuple[str, int, str | None]
        ] = []

    def search(
        self,
        query: str,
        *,
        top_k: int,
        law_name: str | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append(
            (query, top_k, law_name)
        )
        return self.results[:top_k]


def test_reranked_retriever_requests_ten_candidates() -> None:
    base = StubRetriever(
        [
            make_result(
                chunk_id="labor",
                article_no="第二条",
                text="应当订立书面劳动合同。",
                score=0.04,
            ),
            make_result(
                chunk_id="other",
                article_no="第三条",
                text="其他法条。",
                score=0.03,
            ),
        ]
    )

    retriever = RerankedRetriever(
        base_retriever=base,
        scorer=FakeScorer(),
        candidate_top_k=10,
    )

    results = retriever.search(
        "公司没有签书面劳动合同",
        top_k=2,
        law_name="中华人民共和国测试法",
    )

    assert base.calls == [
        (
            "公司没有签书面劳动合同",
            10,
            "中华人民共和国测试法",
        )
    ]

    assert results[0].chunk_id == "labor"


def test_siliconflow_response_is_mapped_by_index() -> None:
    def handle_request(
        request: httpx.Request,
    ) -> httpx.Response:
        payload = json.loads(
            request.content.decode("utf-8")
        )

        assert payload["model"] == "test-reranker"
        assert payload["top_n"] == 2

        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "index": 1,
                        "relevance_score": 0.9,
                    },
                    {
                        "index": 0,
                        "relevance_score": 0.2,
                    },
                ]
            },
        )

    transport = httpx.MockTransport(
        handle_request
    )

    with httpx.Client(
        transport=transport
    ) as client:
        reranker = SiliconFlowReranker(
            base_url="https://example.com/v1",
            api_key="test-key",
            model="test-reranker",
            client=client,
        )

        scores = reranker.score_documents(
            "测试问题",
            [
                "候选一",
                "候选二",
            ],
        )

    assert scores == [0.2, 0.9]