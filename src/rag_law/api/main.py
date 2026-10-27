"""阶段 6 HTTP API。

运行：``uv run uvicorn rag_law.api.main:app --host 127.0.0.1 --port 8000``。
接口只在真正处理请求时加载模型与索引，因而 ``/v1/health`` 可用于定位配置
或索引缺失，而不会在导入模块时读取密钥。
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_law.catalog import LawOption, load_law_options
from rag_law.ingestion.loaders import SUPPORTED_SUFFIXES
from rag_law.knowledge_base import LAW_ID_PATTERN, PROJECT_ROOT
from rag_law.retrieval.bm25 import DEFAULT_BM25_INDEX_PATH
from rag_law.retrieval.dense import DEFAULT_CHROMA_DIRECTORY
from rag_law.schemas import (
    AnswerPipelineResult,
    ConversationRole,
    ConversationTurn,
    RetrievalResult,
)

if TYPE_CHECKING:
    from rag_law.auth import AuthState
    from rag_law.history import ConversationStore



logger = logging.getLogger(__name__)
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.yaml"
ARTICLES_PATH = PROJECT_ROOT / "data" / "processed" / "articles.jsonl"
WEB_DIST_DIRECTORY = PROJECT_ROOT / "web" / "dist"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_RATE_LIMIT = 30
DEFAULT_RATE_WINDOW_SECONDS = 60.0
DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# 单次回填的历史消息条数上限（约 50 轮）；更早的内容已进摘要。
HISTORY_MESSAGE_LIMIT = 100
# 与 ConversationTurn.content 的上限保持一致：库里存原文，回灌时要截断。
MAX_HISTORY_CONTENT_CHARS = 2000


class BaseQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    law_name: str | None = Field(default=None, max_length=200)


class QueryRequest(BaseQueryRequest):
    history: list[ConversationTurn] = Field(
        default_factory=list,
        max_length=50,
        description="当前问题之前的对话历史；同时会随回答落到历史对话表。",
    )
    # 为空表示这一轮不记历史（比如调试或未登录场景）。
    conversation_id: int | None = Field(default=None, ge=1)


class RetrieveRequest(BaseQueryRequest):
    top_k: int = Field(default=5, ge=1, le=20)


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
    total_latency_ms: float = Field(ge=0)


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, bool]


class ConversationMessage(BaseModel):
    """历史对话中的一条消息；result 只在助手消息上出现。"""

    id: int
    role: ConversationRole
    content: str
    result: AnswerPipelineResult | None = None
    created_at: datetime


class ConversationSummary(BaseModel):
    id: int
    title: str
    law_name: str | None = None
    message_count: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[ConversationMessage] = Field(default_factory=list)


class CreateConversationRequest(BaseModel):
    law_name: str | None = Field(default=None, max_length=200)


class ConversationSummaryCache:
    """把早期对话摘要缓存在 conversations 表里。

    任何一步失败都退化成「这一轮没有缓存」：摘要最多多算一次，
    不能让历史存储的问题影响回答本身。
    """

    def __init__(
        self,
        store: Any,
        *,
        conversation_id: int,
        user_id: int,
        first_seq: int,
    ) -> None:
        self._store = store
        self._conversation_id = conversation_id
        self._user_id = user_id
        self._first_seq = first_seq

    def load(self) -> tuple[str, int]:
        try:
            summary, covered_seq = self._store.load_summary(
                conversation_id=self._conversation_id,
                user_id=self._user_id,
            )
        except Exception as error:
            logger.warning(
                "读取对话摘要缓存失败：%s", type(error).__name__
            )
            return "", 0

        if covered_seq < self._first_seq - 1:
            # 加载窗口没从第一条消息开始，摘要也没覆盖到窗口开头：
            # 中间那段谁都没记过，增量合并会把它漏掉，这一轮不用缓存。
            return "", 0
        return summary, covered_seq

    def save(self, summary: str, covered_seq: int) -> None:
        try:
            self._store.save_summary(
                conversation_id=self._conversation_id,
                user_id=self._user_id,
                summary=summary,
                covered_seq=covered_seq,
            )
        except Exception as error:
            logger.warning(
                "写入对话摘要缓存失败：%s", type(error).__name__
            )


class SlidingWindowRateLimiter:
    """进程内的轻量限流器；多实例部署时应换成共享存储实现。"""

    def __init__(self, *, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._timestamps: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, *, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        with self._lock:
            entries = self._timestamps[key]
            cutoff = current - self.window_seconds
            while entries and entries[0] <= cutoff:
                entries.popleft()
            if len(entries) >= self.limit:
                return False
            entries.append(current)
            return True


class ApiServices:
    """延迟创建、可替换的流水线容器，便于测试且避免启动即加载密钥。"""

    def __init__(self) -> None:
        self.answer_pipeline: Any | None = None
        self.retrieval_pipeline: Any | None = None
        self._lock = Lock()

    def get_answer_pipeline(self) -> Any:
        with self._lock:
            if self.answer_pipeline is None:
                from rag_law.generation.answer_pipeline import AnswerPipeline
                self.answer_pipeline = AnswerPipeline.load()
            return self.answer_pipeline

    def get_retrieval_pipeline(self) -> Any:
        with self._lock:
            if self.retrieval_pipeline is None:
                from rag_law.retrieval.retrieval_context_pipeline import (
                    RetrievalContextPipeline,
                )
                self.retrieval_pipeline = RetrievalContextPipeline.load()
            return self.retrieval_pipeline

    def get_add_document(self) -> Callable[..., Any]:
        """返回追加上传法律的实现；测试可覆写此方法。"""

        from rag_law.knowledge_base import add_document

        return add_document

    def get_rebuild_knowledge_base(self) -> Callable[..., Any]:
        """返回全量重建知识库的实现；测试可覆写此方法。"""

        from rag_law.knowledge_base import rebuild_knowledge_base

        return rebuild_knowledge_base

    def reset(self) -> None:
        """丢弃内存中的流水线，使后续请求重新加载最新索引。"""

        with self._lock:
            self.answer_pipeline = None
            self.retrieval_pipeline = None


def _health_checks() -> dict[str, bool]:
    return {
        "manifest": MANIFEST_PATH.is_file(),
        "articles": ARTICLES_PATH.is_file(),
        "bm25_index": DEFAULT_BM25_INDEX_PATH.is_file(),
        "chroma_index": DEFAULT_CHROMA_DIRECTORY.is_dir(),
    }


def _validate_law_name(law_name: str | None) -> str | None:
    if law_name is None:
        return None
    normalized = law_name.strip()
    if not normalized:
        return None
    options = load_law_options(MANIFEST_PATH)
    if normalized not in {item.law_name for item in options}:
        raise HTTPException(status_code=422, detail="指定法律不在当前知识库中")
    return normalized


async def _run_with_timeout(
    operation: Callable[[], Any], *, timeout_seconds: float
) -> Any:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(operation), timeout=timeout_seconds
        )
    except TimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="查询超时，请稍后重试或缩短问题描述",
        ) from error
    except HTTPException:
        raise
    except (FileNotFoundError, OSError) as error:
        logger.warning("API dependency unavailable: %s", type(error).__name__)
        raise HTTPException(
            status_code=503,
            detail="索引或模型服务暂不可用，请检查环境配置和索引",
        ) from error
    except Exception as error:
        # 请求正文可能包含敏感信息，日志只记录异常类型，绝不写入 query。
        logger.exception("API request failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="查询处理失败，请稍后重试",
        ) from error


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    """把一次进度或结果编码为一帧 SSE。"""

    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _knowledge_error_payload(error: Exception) -> dict[str, str]:
    """把知识库维护异常拆成用户文案与技术细节。"""

    if isinstance(error, ValueError):
        message = "文件或参数不符合要求，请检查后重试。"
    elif isinstance(error, (FileNotFoundError, OSError)):
        message = "读写法律文件或索引时出问题了，请检查目录权限。"
    else:
        message = "这次处理没有完成，请稍后重试。"

    return {
        "message": message,
        "detail": f"{type(error).__name__}: {error}",
    }


def _query_error_payload(error: Exception) -> dict[str, str]:
    """把问答异常拆成用户能看懂的一句话。"""

    if isinstance(error, TimeoutError):
        message = "这次回答超时了，可以稍后重试，或把问题说得更简短一些。"
    elif isinstance(error, (FileNotFoundError, OSError)):
        message = "索引或法律文件读取失败了，请确认服务端文件完好后重试。"
    else:
        message = "这次回答没有生成成功，请稍后再试一次。"

    return {
        "message": message,
        "detail": f"{type(error).__name__}: {error}",
    }


def _stream_payload(value: Any) -> Any:
    """把流水线结果转成可以直接序列化的结构。"""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


async def _stream_operation(
    operation: Callable[[Callable[[str, Any], None]], Any],
    *,
    error_handler: Callable[[Exception], dict[str, Any]],
    failure_label: str,
    on_finish: Callable[[], None] | None = None,
):
    """在线程中执行耗时任务，并把事件逐条推给浏览器。

    ``operation`` 拿到的 ``emit(event, payload)`` 可以在任意线程调用；
    一次请求只允许以 complete 或 error 收尾。
    """

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    def emit(event: str, payload: Any) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event, payload))

    async def worker() -> None:
        try:
            result = await asyncio.to_thread(operation, emit)
        except Exception as error:
            # 请求正文与上传文件可能敏感，日志只记录异常类型。
            logger.exception(
                "%s failed: %s", failure_label, type(error).__name__
            )
            await queue.put(("error", error_handler(error)))
        else:
            await queue.put(("complete", _stream_payload(result)))
        finally:
            if on_finish is not None:
                on_finish()

    task = asyncio.create_task(worker())
    try:
        while True:
            event, payload = await queue.get()
            yield _sse_event(event, payload)
            if event in {"complete", "error"}:
                break
    finally:
        if not task.done():
            task.cancel()


async def _knowledge_event_stream(
    operation: Callable[[Callable[[str, Any], None]], dict[str, Any]],
    *,
    services: "ApiServices",
):
    """知识库维护：线程内执行，结束后丢弃旧的流水线实例。"""

    async for frame in _stream_operation(
        operation,
        error_handler=_knowledge_error_payload,
        failure_label="Knowledge base update",
        # 无论成败都丢弃旧流水线，避免继续使用过期索引。
        on_finish=services.reset,
    ):
        yield frame


def _resolve_cors_origins(configured: Sequence[str] | None) -> list[str]:
    """优先使用显式配置，其次环境变量，最后是本地开发地址。"""

    if configured is not None:
        return [origin.strip() for origin in configured if origin.strip()]

    raw = os.getenv("RAG_LAW_CORS_ORIGINS", "")
    if raw.strip():
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    return list(DEFAULT_CORS_ORIGINS)


def create_app(
    *,
    services: ApiServices | None = None,
    auth: "AuthState | None" = None,
    conversations: "ConversationStore | None" = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    rate_limit: int = DEFAULT_RATE_LIMIT,
    rate_window_seconds: float = DEFAULT_RATE_WINDOW_SECONDS,
    web_dist: Path | None = None,
    cors_origins: Sequence[str] | None = None,
) -> FastAPI:
    """创建应用；测试可传入短超时、预置 services 或临时静态目录。"""

    # 延迟导入：账号库与对话库驱动只在实际启用登录时才需要。
    from rag_law.auth import AuthState, Session, require_user
    from rag_law.auth import router as auth_router
    from rag_law.config import settings
    from rag_law.history import ConversationStore, ConversationStoreUnavailable

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            yield
        finally:
            application.state.auth.close()
            application.state.conversations.close()

    app = FastAPI(title="RAG Law API", version="v1", lifespan=lifespan)
    app.state.services = services or ApiServices()
    app.state.auth = auth or AuthState(
        dsn=settings.database_url,
        cookie_name=settings.session_cookie_name,
        session_ttl_seconds=settings.session_ttl_seconds,
        cookie_secure=settings.session_cookie_secure,
    )
    # 与账号库共用同一个 DSN，会话按 users.id 归属。
    app.state.conversations = conversations or ConversationStore(
        dsn=settings.database_url,
    )
    app.state.timeout_seconds = timeout_seconds
    app.state.rate_limiter = SlidingWindowRateLimiter(
        limit=rate_limit, window_seconds=rate_window_seconds
    )
    # 同一时间只允许一个重建任务，避免并发重写同一套索引。
    app.state.knowledge_lock = Lock()

    resolved_origins = _resolve_cors_origins(cors_origins)
    if resolved_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    # 登录相关接口自身不做登录校验，其余业务接口统一挂 require_user。
    app.include_router(auth_router)

    async def enforce_rate_limit(request: Request) -> None:
        client = request.client.host if request.client else "unknown"
        if not app.state.rate_limiter.allow(client):
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")

    @app.get("/v1/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        checks = _health_checks()
        return HealthResponse(
            status="ok" if all(checks.values()) else "degraded", checks=checks
        )

    @app.get(
        "/v1/laws",
        response_model=list[LawOption],
        tags=["knowledge-base"],
        dependencies=[Depends(require_user)],
    )
    def laws() -> list[LawOption]:
        try:
            return load_law_options(MANIFEST_PATH)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=503, detail="法律清单不可用") from error

    @app.post(
        "/v1/query",
        response_model=AnswerPipelineResult,
        tags=["query"],
    )
    async def query(
        payload: QueryRequest,
        request: Request,
        session: Session = Depends(require_user),
    ) -> AnswerPipelineResult:
        await enforce_rate_limit(request)
        law_name = _validate_law_name(payload.law_name)
        normalized_query = payload.query.strip()
        if not normalized_query:
            raise HTTPException(status_code=422, detail="查询问题不能为空")
        history, summary_cache = await _load_conversation_context(
            conversation_id=payload.conversation_id,
            user_id=session.user_id,
            fallback=payload.history,
        )
        result = await _run_with_timeout(
            lambda: app.state.services.get_answer_pipeline().run(
                normalized_query,
                law_name=law_name,
                history=history,
                summary_cache=summary_cache,
            ),
            timeout_seconds=app.state.timeout_seconds,
        )
        if payload.conversation_id is not None:
            await _persist_exchange(
                conversation_id=payload.conversation_id,
                user_id=session.user_id,
                query=normalized_query,
                law_name=law_name,
                result=result,
            )
        return result

    @app.post(
        "/v1/query/stream",
        tags=["query"],
    )
    async def query_stream(
        payload: QueryRequest,
        request: Request,
        session: Session = Depends(require_user),
    ) -> StreamingResponse:
        """与 /v1/query 结果相同，但把思考过程与正文逐段推给浏览器。

        参数校验全部在开流前完成，所以这些错误仍能用正常状态码返回；
        一旦开始推送，后续失败只能以 error 事件下发。
        """

        await enforce_rate_limit(request)
        law_name = _validate_law_name(payload.law_name)
        normalized_query = payload.query.strip()
        if not normalized_query:
            raise HTTPException(status_code=422, detail="查询问题不能为空")

        history, summary_cache = await _load_conversation_context(
            conversation_id=payload.conversation_id,
            user_id=session.user_id,
            fallback=payload.history,
        )

        from rag_law.generation.chain import GenerationStreamHandlers

        def operation(
            emit: Callable[[str, Any], None],
        ) -> AnswerPipelineResult:
            handlers = GenerationStreamHandlers(
                on_reasoning=lambda text: emit(
                    "reasoning", {"text": text}
                ),
                on_field=lambda field, text: emit(
                    "answer", {"field": field, "text": text}
                ),
            )

            result = app.state.services.get_answer_pipeline().run(
                normalized_query,
                law_name=law_name,
                history=history,
                summary_cache=summary_cache,
                progress_callback=lambda stage, message: emit(
                    "progress", {"stage": stage, "message": message}
                ),
                stream_handlers=handlers,
            )

            if payload.conversation_id is not None:
                # 在推送完最终结果之前落库，浏览器中途断开也不丢这一轮。
                _store_exchange(
                    conversation_id=payload.conversation_id,
                    user_id=session.user_id,
                    query=normalized_query,
                    law_name=law_name,
                    result=result,
                )

            return result

        return StreamingResponse(
            _stream_operation(
                operation,
                error_handler=_query_error_payload,
                failure_label="Answer stream",
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def _persist_exchange(
        *,
        conversation_id: int,
        user_id: int,
        query: str,
        law_name: str | None,
        result: AnswerPipelineResult,
    ) -> None:
        """异步接口里落库；真正写入的实现在 _store_exchange。"""

        await asyncio.to_thread(
            _store_exchange,
            conversation_id=conversation_id,
            user_id=user_id,
            query=query,
            law_name=law_name,
            result=result,
        )

    def _store_exchange(
        *,
        conversation_id: int,
        user_id: int,
        query: str,
        law_name: str | None,
        result: AnswerPipelineResult,
    ) -> None:
        """历史落库是附属动作：失败只记日志，不能反噬已经生成的回答。"""

        try:
            stored = app.state.conversations.append_exchange(
                conversation_id=conversation_id,
                user_id=user_id,
                query=query,
                law_name=law_name,
                result=result,
            )
        except ConversationStoreUnavailable:
            logger.warning("历史对话没能写入：会话库暂不可用")
            return
        except Exception:
            logger.exception("历史对话写入异常")
            return

        if not stored:
            logger.warning(
                "会话 %s 不属于当前用户，本次未写入历史", conversation_id
            )

    async def _load_conversation_context(
        *,
        conversation_id: int | None,
        user_id: int,
        fallback: Sequence[ConversationTurn],
    ) -> tuple[Sequence[ConversationTurn], ConversationSummaryCache | None]:
        """有会话 id 时以库里的历史为准，这样才能用摘要缓存。

        库里读不到（新会话、已删除、会话库不可用）就退回前端传来的 history，
        保证即使历史存储全挂，多轮问答也能照旧进行。
        """

        if conversation_id is None:
            return fallback, None

        try:
            rows = await asyncio.to_thread(
                app.state.conversations.load_messages,
                conversation_id=conversation_id,
                user_id=user_id,
                limit=HISTORY_MESSAGE_LIMIT,
            )
        except ConversationStoreUnavailable:
            logger.warning("历史消息读取失败，改用前端传入的 history")
            return fallback, None

        if not rows:
            return fallback, None

        turns = [
            ConversationTurn(
                role=row["role"],
                content=row["content"][:MAX_HISTORY_CONTENT_CHARS],
                seq=row["seq"],
            )
            for row in rows
        ]
        return turns, ConversationSummaryCache(
            app.state.conversations,
            conversation_id=conversation_id,
            user_id=user_id,
            first_seq=rows[0]["seq"],
        )

    @app.post(
        "/v1/retrieve",
        response_model=RetrieveResponse,
        tags=["query"],
        dependencies=[Depends(require_user)],
    )
    async def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
        await enforce_rate_limit(request)
        law_name = _validate_law_name(payload.law_name)
        normalized_query = payload.query.strip()
        if not normalized_query:
            raise HTTPException(status_code=422, detail="查询问题不能为空")
        result = await _run_with_timeout(
            lambda: app.state.services.get_retrieval_pipeline().run(
                normalized_query, law_name=law_name
            ),
            timeout_seconds=app.state.timeout_seconds,
        )
        return RetrieveResponse(
            query=result.query,
            results=result.reranked_results[: payload.top_k],
            total_latency_ms=result.total_latency_ms,
        )

    # ---------- 历史对话：全部按当前登录用户隔离 ----------

    def _conversation_store_error(error: Exception) -> HTTPException:
        logger.warning("会话库不可用：%s", type(error).__name__)
        return HTTPException(
            status_code=503, detail="历史对话暂时打不开，请稍后再试"
        )

    @app.get(
        "/v1/conversations",
        response_model=list[ConversationSummary],
        tags=["conversations"],
    )
    def list_conversations(
        session: Session = Depends(require_user),
    ) -> list[ConversationSummary]:
        try:
            rows = app.state.conversations.list_for_user(
                user_id=session.user_id,
                limit=settings.conversation_history_limit,
            )
        except ConversationStoreUnavailable as error:
            raise _conversation_store_error(error) from error
        return [ConversationSummary(**row) for row in rows]

    @app.post(
        "/v1/conversations",
        response_model=ConversationSummary,
        tags=["conversations"],
    )
    def create_conversation(
        session: Session = Depends(require_user),
        payload: CreateConversationRequest | None = None,
    ) -> ConversationSummary:
        law_name = _validate_law_name(payload.law_name if payload else None)
        try:
            row = app.state.conversations.create(
                user_id=session.user_id,
                law_name=law_name,
            )
        except ConversationStoreUnavailable as error:
            raise _conversation_store_error(error) from error
        return ConversationSummary(**row)

    @app.get(
        "/v1/conversations/{conversation_id}",
        response_model=ConversationDetail,
        tags=["conversations"],
    )
    def get_conversation(
        conversation_id: int,
        session: Session = Depends(require_user),
    ) -> ConversationDetail:
        try:
            row = app.state.conversations.get(
                conversation_id=conversation_id,
                user_id=session.user_id,
            )
        except ConversationStoreUnavailable as error:
            raise _conversation_store_error(error) from error
        if row is None:
            # 不是本人的会话与不存在的会话返回同一句话，避免枚举 id。
            raise HTTPException(status_code=404, detail="这个对话不存在或已删除")
        return ConversationDetail(**row)

    @app.delete(
        "/v1/conversations/{conversation_id}",
        tags=["conversations"],
    )
    def delete_conversation(
        conversation_id: int,
        session: Session = Depends(require_user),
    ) -> dict[str, str]:
        try:
            deleted = app.state.conversations.delete(
                conversation_id=conversation_id,
                user_id=session.user_id,
            )
        except ConversationStoreUnavailable as error:
            raise _conversation_store_error(error) from error
        if not deleted:
            raise HTTPException(status_code=404, detail="这个对话不存在或已删除")
        return {"detail": "对话已删除"}

    def start_knowledge_stream(
        operation: Callable[[Callable[[str, str], None]], dict[str, Any]],
    ) -> StreamingResponse:
        """占用维护锁后启动事件流，结束时释放锁并重建服务。"""

        if not app.state.knowledge_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409,
                detail="知识库正在更新，请等当前任务结束后再试",
            )

        def locked_operation(
            report: Callable[[str, str], None],
        ) -> dict[str, Any]:
            try:
                return operation(report)
            finally:
                app.state.knowledge_lock.release()

        return StreamingResponse(
            _knowledge_event_stream(
                locked_operation,
                services=app.state.services,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post(
        "/v1/laws/upload",
        tags=["knowledge-base"],
        dependencies=[Depends(require_user)],
    )
    async def upload_law(
        request: Request,
        file: UploadFile = File(...),
        law_id: str = Form(...),
        law_name: str = Form(""),
    ) -> StreamingResponse:
        await enforce_rate_limit(request)

        filename = Path(file.filename or "").name
        if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=422,
                detail="只支持 PDF、DOCX 和 Markdown 文件",
            )

        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="文件超过 50 MB，请换一个更小的文件",
            )

        cleaned_law_id = law_id.strip()
        if not LAW_ID_PATTERN.fullmatch(cleaned_law_id):
            raise HTTPException(
                status_code=422,
                detail=(
                    "law_id 要用 cn_ 开头，后面只能是小写字母、"
                    "数字或下划线"
                ),
            )

        add_document = app.state.services.get_add_document()
        final_name = law_name.strip() or None

        def operation(emit: Callable[[str, Any], None]) -> dict[str, Any]:
            def report(stage: str, message: str) -> None:
                emit("progress", {"stage": stage, "message": message})

            with tempfile.TemporaryDirectory() as temp_dir:
                uploaded_path = Path(temp_dir) / filename
                uploaded_path.write_bytes(content)
                result = add_document(
                    uploaded_path,
                    law_id=cleaned_law_id,
                    law_name=final_name,
                    progress_callback=report,
                )

            return {
                "law_id": result.law_id,
                "law_name": result.law_name,
                "article_count": result.article_count,
                "chunk_count": result.chunk_count,
                "law_count": result.law_count,
                "chunk_total": result.chunk_total,
                "manifest_updated": result.manifest_updated,
            }

        return start_knowledge_stream(operation)

    @app.post(
        "/v1/knowledge-base/rebuild",
        tags=["knowledge-base"],
        dependencies=[Depends(require_user)],
    )
    async def rebuild_knowledge_base_endpoint(
        request: Request,
    ) -> StreamingResponse:
        await enforce_rate_limit(request)
        rebuild = app.state.services.get_rebuild_knowledge_base()

        def operation(emit: Callable[[str, Any], None]) -> dict[str, Any]:
            def report(stage: str, message: str) -> None:
                emit("progress", {"stage": stage, "message": message})

            result = rebuild(progress_callback=report)
            return {
                "law_count": result.law_count,
                "chunk_total": result.chunk_total,
            }

        return start_knowledge_stream(operation)

    resolved_web_dist = (
        Path(web_dist)
        if web_dist is not None
        else Path(
            os.getenv("RAG_LAW_WEB_DIST", str(WEB_DIST_DIRECTORY))
        )
    )

    # 静态页面最后挂载，避免抢走 /v1 与 /docs 路由。
    if (resolved_web_dist / "index.html").is_file():
        app.mount(
            "/",
            StaticFiles(directory=resolved_web_dist, html=True),
            name="web",
        )
    else:
        logger.info(
            "Web 构建产物不存在，仅提供 API：%s", resolved_web_dist
        )

    return app


app = create_app()
