"""历史对话接口：创建、落库、隔离与删除。"""

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from rag_law.api.main import ApiServices, ConversationSummaryCache, create_app
from rag_law.auth.state import AuthState
from rag_law.schemas import (
    AnswerPipelineResult,
    AnswerResponse,
    BuiltContext,
    RetrievalContextResult,
)


def _answer(query: str) -> AnswerPipelineResult:
    return AnswerPipelineResult(
        query=query,
        answer=AnswerResponse(
            status="answered",
            summary="公司应当补签书面劳动合同。",
            analysis="依据劳动合同法第十条。",
            conditions=["存在事实劳动关系"],
            limitations="",
        ),
        retrieval=RetrievalContextResult(
            query=query,
            reranked_results=[],
            context=BuiltContext(
                text="",
                evidences=[],
                estimated_tokens=0,
                max_tokens=100,
                truncated=False,
            ),
            retrieval_latency_ms=1,
            context_latency_ms=1,
            total_latency_ms=2,
        ),
        generation_attempts=1,
        prompt_version="v1",
        generation_latency_ms=1,
        total_latency_ms=2,
    )


class StubAnswerPipeline:
    """不调用大模型，直接返回固定回答，并记录收到的上下文。"""

    def __init__(self) -> None:
        self.heard: list[dict] = []

    def run(self, query, *, law_name=None, history=(), summary_cache=None):
        self.heard.append(
            {
                "query": query,
                "history": list(history),
                "summary_cache": summary_cache,
            }
        )
        return _answer(query)


class StubAuthState(AuthState):
    """内存账号：user_id 可指定，用来验证会话隔离。"""

    def __init__(self, user_id: int = 1) -> None:
        super().__init__(
            dsn="",
            cookie_name="rag_law_sid",
            session_ttl_seconds=600,
            cookie_secure=False,
        )
        self._user_id = user_id

    def authenticate(self, username: str, password: str) -> dict | None:
        if username.lower() != "tester" or password != "secret":
            return None
        return {
            "id": self._user_id,
            "username": "tester",
            "role": "user",
            "status": 1,
        }


class FakeConversationStore:
    """内存版会话库，接口与 ConversationStore 一致。"""

    def __init__(self) -> None:
        self.rows: dict[int, dict] = {}
        self.messages: dict[int, list[dict]] = {}
        self._next_conversation_id = 1
        self._next_message_id = 1

    def create(self, *, user_id: int, law_name: str | None = None) -> dict:
        conversation_id = self._next_conversation_id
        self._next_conversation_id += 1
        now = datetime.now(timezone.utc)
        self.rows[conversation_id] = {
            "id": conversation_id,
            "user_id": user_id,
            "title": None,
            "law_name": law_name,
            "summary": None,
            "summary_covered_seq": 0,
            "created_at": now,
            "updated_at": now,
        }
        self.messages[conversation_id] = []
        return self._summary(self.rows[conversation_id])

    def append_exchange(
        self,
        *,
        conversation_id: int,
        user_id: int,
        query: str,
        law_name: str | None,
        result: AnswerPipelineResult,
    ) -> bool:
        row = self.rows.get(conversation_id)
        if row is None or row["user_id"] != user_id:
            return False

        row["title"] = row["title"] or query[:24]
        row["law_name"] = row["law_name"] or law_name
        row["updated_at"] = datetime.now(timezone.utc)

        self.messages[conversation_id].extend(
            [
                self._message("user", query, result=None),
                self._message(
                    "assistant",
                    f"{result.answer.summary}\n{result.answer.analysis}",
                    result=result.model_dump(mode="json"),
                ),
            ]
        )
        return True

    def load_messages(
        self, *, conversation_id: int, user_id: int, limit: int = 100
    ) -> list[dict]:
        row = self.rows.get(conversation_id)
        if row is None or row["user_id"] != user_id:
            return []
        return [
            {
                "seq": item["seq"],
                "role": item["role"],
                "content": item["content"],
            }
            for item in self.messages[conversation_id][-limit:]
        ]

    def load_summary(self, *, conversation_id: int, user_id: int) -> tuple[str, int]:
        row = self.rows.get(conversation_id)
        if row is None or row["user_id"] != user_id:
            return "", 0
        return row["summary"] or "", row["summary_covered_seq"]

    def save_summary(
        self,
        *,
        conversation_id: int,
        user_id: int,
        summary: str,
        covered_seq: int,
    ) -> bool:
        row = self.rows.get(conversation_id)
        if row is None or row["user_id"] != user_id:
            return False
        row["summary"] = summary
        row["summary_covered_seq"] = covered_seq
        return True

    def list_for_user(self, *, user_id: int, limit: int = 50) -> list[dict]:
        rows = [
            self._summary(row)
            for row in self.rows.values()
            if row["user_id"] == user_id and self.messages[row["id"]]
        ]
        rows.sort(key=lambda item: item["updated_at"], reverse=True)
        return rows[:limit]

    def get(self, *, conversation_id: int, user_id: int) -> dict | None:
        row = self.rows.get(conversation_id)
        if row is None or row["user_id"] != user_id:
            return None
        detail = self._summary(row)
        detail["messages"] = self.messages[conversation_id]
        return detail

    def delete(self, *, conversation_id: int, user_id: int) -> bool:
        row = self.rows.get(conversation_id)
        if row is None or row["user_id"] != user_id:
            return False
        self.rows.pop(conversation_id)
        self.messages.pop(conversation_id, None)
        return True

    def close(self) -> None:
        pass

    def _summary(self, row: dict) -> dict:
        return {
            "id": row["id"],
            "title": row["title"] or "新对话",
            "law_name": row["law_name"],
            "message_count": len(self.messages[row["id"]]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _message(self, role: str, content: str, *, result) -> dict:
        message = {
            "id": self._next_message_id,
            # 与真实表一致：seq 从 1 开始连续递增，摘要游标依赖它。
            "seq": self._next_message_id,
            "role": role,
            "content": content,
            "result": result,
            "created_at": datetime.now(timezone.utc),
        }
        self._next_message_id += 1
        return message


def _client(*, user_id: int = 1, store: FakeConversationStore | None = None):
    services = ApiServices()
    services.answer_pipeline = StubAnswerPipeline()
    app = create_app(
        services=services,
        auth=StubAuthState(user_id),
        conversations=store or FakeConversationStore(),
    )
    client = TestClient(app)
    response = client.post(
        "/v1/auth/login", json={"username": "tester", "password": "secret"}
    )
    assert response.status_code == 200
    return client, app


def test_history_requires_login() -> None:
    services = ApiServices()
    services.answer_pipeline = StubAnswerPipeline()
    client = TestClient(
        create_app(
            services=services,
            auth=StubAuthState(),
            conversations=FakeConversationStore(),
        )
    )

    assert client.get("/v1/conversations").status_code == 401


def test_new_conversation_is_hidden_until_it_has_messages() -> None:
    client, _ = _client()

    created = client.post("/v1/conversations", json={"law_name": None})

    assert created.status_code == 200
    assert created.json()["title"] == "新对话"
    assert client.get("/v1/conversations").json() == []


def test_query_writes_exchange_into_history() -> None:
    client, _ = _client()
    conversation_id = client.post("/v1/conversations", json={}).json()["id"]

    answer = client.post(
        "/v1/query",
        json={
            "query": "公司不签劳动合同怎么办？",
            "history": [],
            "conversation_id": conversation_id,
        },
    )
    detail = client.get(f"/v1/conversations/{conversation_id}")

    assert answer.status_code == 200
    assert detail.status_code == 200
    body = detail.json()
    assert body["title"] == "公司不签劳动合同怎么办？"
    assert body["message_count"] == 2
    assert [item["role"] for item in body["messages"]] == ["user", "assistant"]
    assert body["messages"][1]["result"]["answer"]["status"] == "answered"

    listed = client.get("/v1/conversations").json()
    assert [item["id"] for item in listed] == [conversation_id]


def test_query_without_conversation_id_keeps_no_history() -> None:
    client, app = _client()

    response = client.post("/v1/query", json={"query": "只问一次", "history": []})

    assert response.status_code == 200
    assert app.state.conversations.rows == {}
    assert client.get("/v1/conversations").json() == []


def test_conversation_is_isolated_between_users() -> None:
    alice, _ = _client(user_id=1)
    conversation_id = alice.post("/v1/conversations", json={}).json()["id"]
    alice.post(
        "/v1/query",
        json={"query": "我的问题", "history": [], "conversation_id": conversation_id},
    )

    bob, _ = _client(user_id=2)

    assert bob.get("/v1/conversations").json() == []
    assert bob.get(f"/v1/conversations/{conversation_id}").status_code == 404
    assert bob.delete(f"/v1/conversations/{conversation_id}").status_code == 404


def test_delete_conversation_removes_it_from_list() -> None:
    client, _ = _client()
    conversation_id = client.post("/v1/conversations", json={}).json()["id"]
    client.post(
        "/v1/query",
        json={"query": "待删除", "history": [], "conversation_id": conversation_id},
    )

    response = client.delete(f"/v1/conversations/{conversation_id}")

    assert response.status_code == 200
    assert client.get("/v1/conversations").json() == []
    assert client.get(f"/v1/conversations/{conversation_id}").status_code == 404


def test_repeated_query_replays_history_from_database() -> None:
    client, app = _client()
    conversation_id = client.post("/v1/conversations", json={}).json()["id"]

    client.post(
        "/v1/query",
        json={
            "query": "公司不签劳动合同怎么办？",
            "history": [],
            "conversation_id": conversation_id,
        },
    )
    client.post(
        "/v1/query",
        json={
            "query": "那赔偿怎么算？",
            # 前端即使传了 history，有会话 id 时也以库里的为准。
            "history": [{"role": "user", "content": "前端自带的旧问题"}],
            "conversation_id": conversation_id,
        },
    )

    pipeline = app.state.services.answer_pipeline
    first, second = pipeline.heard
    assert first["history"] == []
    assert first["summary_cache"] is None
    assert [turn.role for turn in second["history"]] == ["user", "assistant"]
    assert [turn.seq for turn in second["history"]] == [1, 2]
    assert second["history"][0].content == "公司不签劳动合同怎么办？"
    assert isinstance(second["summary_cache"], ConversationSummaryCache)


def test_summary_cache_round_trip_and_gap_protection() -> None:
    store = FakeConversationStore()
    conversation_id = store.create(user_id=1)["id"]
    cache = ConversationSummaryCache(
        store, conversation_id=conversation_id, user_id=1, first_seq=1
    )

    assert cache.load() == ("", 0)
    cache.save("早期摘要", 4)
    assert cache.load() == ("早期摘要", 4)

    # 加载窗口没从第一条消息开始、缓存也没覆盖窗口开头：那一段没人记过，
    # 本轮只能用空摘要，避免增量合并漏掉中间内容。
    late = ConversationSummaryCache(
        store, conversation_id=conversation_id, user_id=1, first_seq=50
    )
    assert late.load() == ("", 0)


def test_summary_cache_failure_degrades_to_no_cache() -> None:
    class BrokenStore:
        def load_summary(self, **kwargs):
            raise RuntimeError("库挂了")

        def save_summary(self, **kwargs):
            raise RuntimeError("库挂了")

    cache = ConversationSummaryCache(
        BrokenStore(), conversation_id=1, user_id=1, first_seq=1
    )

    assert cache.load() == ("", 0)
    # 写失败不抛异常，回答流转继续。
    cache.save("任意摘要", 2)
