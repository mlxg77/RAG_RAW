import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from rag_law.api.main import ApiServices, QueryRequest, create_app
from rag_law.knowledge_base import (
    AddDocumentResult,
    RebuildKnowledgeBaseResult,
)
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


class StubServices(ApiServices):
    """记录知识库维护调用，并返回固定结果。"""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []

    def get_add_document(self):
        def add_document(
            path, *, law_id, law_name=None, progress_callback=None
        ):
            self.calls.append(("add_document", law_id))
            assert Path(path).is_file()
            if progress_callback:
                progress_callback("validate", "校验通过")
            return AddDocumentResult(
                law_id=law_id,
                law_name=law_name or "测试法",
                file=f"data/raw/{Path(path).name}",
                article_count=3,
                chunk_count=3,
                law_count=4,
                chunk_total=30,
                manifest_updated=True,
            )

        return add_document

    def get_rebuild_knowledge_base(self):
        def rebuild(*, progress_callback=None):
            self.calls.append(("rebuild",))
            if progress_callback:
                progress_callback("complete", "现有知识库重建全部完成。")
            return RebuildKnowledgeBaseResult(law_count=4, chunk_total=30)

        return rebuild


def _sse_events(body: str) -> list[tuple[str, dict]]:
    """把响应文本拆回 (event, data) 列表。"""

    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        event = "message"
        data: dict | None = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = json.loads(line.removeprefix("data:").strip())
        if data is not None:
            events.append((event, data))
    return events


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


def test_query_request_accepts_explicit_conversation_history() -> None:
    request = QueryRequest.model_validate(
        {
            "query": "那赔偿怎么算？",
            "history": [
                {
                    "role": "user",
                    "content": "公司三个月没有和我签合同。",
                },
                {
                    "role": "assistant",
                    "content": "需要结合工作时间继续判断。",
                },
            ],
        }
    )

    assert request.history[0].role == "user"
    assert request.history[1].role == "assistant"


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


def test_upload_law_streams_progress_and_complete() -> None:
    services = StubServices()
    client = TestClient(create_app(services=services))

    response = client.post(
        "/v1/laws/upload",
        files={"file": ("测试法.md", "# 测试".encode(), "text/markdown")},
        data={"law_id": "cn_test_law", "law_name": "测试法"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _sse_events(response.text)
    assert events[0][0] == "progress"
    assert events[-1][0] == "complete"
    assert events[-1][1]["law_count"] == 4
    assert services.calls == [("add_document", "cn_test_law")]


def test_upload_rejects_unsupported_suffix() -> None:
    client = TestClient(create_app(services=StubServices()))

    response = client.post(
        "/v1/laws/upload",
        files={"file": ("law.exe", b"binary", "application/octet-stream")},
        data={"law_id": "cn_test_law"},
    )

    assert response.status_code == 422
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_invalid_law_id() -> None:
    client = TestClient(create_app(services=StubServices()))

    response = client.post(
        "/v1/laws/upload",
        files={"file": ("law.md", "# 测试".encode(), "text/markdown")},
        data={"law_id": "TestLaw"},
    )

    assert response.status_code == 422
    assert "cn_" in response.json()["detail"]


def test_upload_reports_failure_as_error_event() -> None:
    class FailingServices(StubServices):
        def get_add_document(self):
            def add_document(*args, **kwargs):
                raise ValueError("未识别到任何法条")

            return add_document

    client = TestClient(create_app(services=FailingServices()))

    response = client.post(
        "/v1/laws/upload",
        files={"file": ("law.md", "# 测试".encode(), "text/markdown")},
        data={"law_id": "cn_test_law"},
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["message"] == "文件或参数不符合要求，请检查后重试。"
    assert "ValueError" in events[-1][1]["detail"]


def test_rebuild_streams_completion_result() -> None:
    services = StubServices()
    client = TestClient(create_app(services=services))

    response = client.post("/v1/knowledge-base/rebuild")

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert events[-1][0] == "complete"
    assert events[-1][1] == {"law_count": 4, "chunk_total": 30}
    assert services.calls == [("rebuild",)]


def test_knowledge_update_rejects_concurrent_task() -> None:
    app = create_app(services=StubServices())
    app.state.knowledge_lock.acquire()
    try:
        response = TestClient(app).post("/v1/knowledge-base/rebuild")
        assert response.status_code == 409
    finally:
        app.state.knowledge_lock.release()


def test_static_web_page_is_served(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text(
        "<!doctype html><title>RAG Law</title>",
        encoding="utf-8",
    )
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("console.log(1)", encoding="utf-8")

    client = TestClient(create_app(services=StubServices(), web_dist=tmp_path))

    page = client.get("/")
    assert page.status_code == 200
    assert "RAG Law" in page.text
    assert client.get("/assets/app.js").status_code == 200
    assert client.get("/v1/health").status_code == 200


def test_cors_allows_configured_origin() -> None:
    client = TestClient(
        create_app(
            services=StubServices(),
            cors_origins=["http://localhost:5173"],
        )
    )

    response = client.options(
        "/v1/query",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "http://localhost:5173"
    )
