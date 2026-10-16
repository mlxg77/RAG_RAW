import time

from fastapi.testclient import TestClient

from rag_law.api.main import ApiServices, create_app
from rag_law.schemas import (
    BuiltContext,
    RetrievalContextResult,
    RetrievalResult,
)


def _retrieval_result() -> RetrievalContextResult:
    item = RetrievalResult(
        chunk_id="C001",
        law_id="cn_test",
        law_name="中华人民共和国测试法",
        article_no="第一条",
        text="测试法条正文。",
        source_file="data/raw/test.md",
        score=0.9,
        source="rerank",
    )
    return RetrievalContextResult(
        query="测试问题",
        reranked_results=[item],
        context=BuiltContext(
            text="", evidences=[], estimated_tokens=0, max_tokens=100, truncated=False
        ),
        retrieval_latency_ms=1,
        context_latency_ms=1,
        total_latency_ms=2,
    )


class StubRetrievalPipeline:
    def run(self, query, *, law_name=None):
        result = _retrieval_result()
        return result.model_copy(update={"query": query})


class SlowRetrievalPipeline:
    def run(self, query, *, law_name=None):
        time.sleep(0.1)
        return _retrieval_result()


def _client(*, retrieval_pipeline, **kwargs) -> TestClient:
    services = ApiServices()
    services.retrieval_pipeline = retrieval_pipeline
    return TestClient(create_app(services=services, **kwargs))


def test_health_reports_index_readiness() -> None:
    response = _client(retrieval_pipeline=StubRetrievalPipeline()).get("/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}
    assert set(response.json()["checks"]) == {
        "manifest", "articles", "bm25_index", "chroma_index"
    }


def test_laws_returns_manifest_entries() -> None:
    response = _client(retrieval_pipeline=StubRetrievalPipeline()).get("/v1/laws")

    assert response.status_code == 200
    assert response.json()[0]["law_id"]
    assert response.json()[0]["law_name"]


def test_retrieve_uses_stable_response_schema() -> None:
    response = _client(retrieval_pipeline=StubRetrievalPipeline()).post(
        "/v1/retrieve", json={"query": "测试问题", "top_k": 1}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "测试问题"
    assert body["results"][0]["chunk_id"] == "C001"
    assert body["results"][0]["text"] == "测试法条正文。"


def test_retrieve_rejects_unknown_law() -> None:
    response = _client(retrieval_pipeline=StubRetrievalPipeline()).post(
        "/v1/retrieve", json={"query": "测试问题", "law_name": "不存在的法律"}
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "指定法律不在当前知识库中"


def test_retrieve_returns_timeout() -> None:
    response = _client(
        retrieval_pipeline=SlowRetrievalPipeline(), timeout_seconds=0.01
    ).post("/v1/retrieve", json={"query": "测试问题"})

    assert response.status_code == 504


def test_rate_limit_rejects_excess_request() -> None:
    client = _client(retrieval_pipeline=StubRetrievalPipeline(), rate_limit=1)

    first = client.post("/v1/retrieve", json={"query": "测试问题"})
    second = client.post("/v1/retrieve", json={"query": "另一个问题"})

    assert first.status_code == 200
    assert second.status_code == 429
