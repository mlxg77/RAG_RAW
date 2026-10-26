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
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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
    ConversationTurn,
    RetrievalResult,
)


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


class BaseQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    law_name: str | None = Field(default=None, max_length=200)


class QueryRequest(BaseQueryRequest):
    history: list[ConversationTurn] = Field(
        default_factory=list,
        max_length=50,
        description="当前问题之前的对话历史；服务端不保存会话。",
    )


class RetrieveRequest(BaseQueryRequest):
    top_k: int = Field(default=5, ge=1, le=20)


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievalResult]
    total_latency_ms: float = Field(ge=0)


class HealthResponse(BaseModel):
    status: str
    checks: dict[str, bool]


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


async def _knowledge_event_stream(
    operation: Callable[[Callable[[str, str], None]], dict[str, Any]],
    *,
    services: "ApiServices",
):
    """在线程中执行耗时任务，并把进度逐条推给浏览器。"""

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

    def report(stage: str, message: str) -> None:
        loop.call_soon_threadsafe(
            queue.put_nowait,
            ("progress", {"stage": stage, "message": message}),
        )

    async def worker() -> None:
        try:
            result = await asyncio.to_thread(operation, report)
        except Exception as error:
            # 上传文件或查询正文可能敏感，日志只记录异常类型。
            logger.exception(
                "Knowledge base update failed: %s", type(error).__name__
            )
            await queue.put(("error", _knowledge_error_payload(error)))
        else:
            await queue.put(("complete", result))
        finally:
            # 无论成败都丢弃旧流水线，避免继续使用过期索引。
            services.reset()

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
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    rate_limit: int = DEFAULT_RATE_LIMIT,
    rate_window_seconds: float = DEFAULT_RATE_WINDOW_SECONDS,
    web_dist: Path | None = None,
    cors_origins: Sequence[str] | None = None,
) -> FastAPI:
    """创建应用；测试可传入短超时、预置 services 或临时静态目录。"""

    app = FastAPI(title="RAG Law API", version="v1")
    app.state.services = services or ApiServices()
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
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

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

    @app.get("/v1/laws", response_model=list[LawOption], tags=["knowledge-base"])
    def laws() -> list[LawOption]:
        try:
            return load_law_options(MANIFEST_PATH)
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=503, detail="法律清单不可用") from error

    @app.post("/v1/query", response_model=AnswerPipelineResult, tags=["query"])
    async def query(payload: QueryRequest, request: Request) -> AnswerPipelineResult:
        await enforce_rate_limit(request)
        law_name = _validate_law_name(payload.law_name)
        normalized_query = payload.query.strip()
        if not normalized_query:
            raise HTTPException(status_code=422, detail="查询问题不能为空")
        return await _run_with_timeout(
            lambda: app.state.services.get_answer_pipeline().run(
                normalized_query,
                law_name=law_name,
                history=payload.history,
            ),
            timeout_seconds=app.state.timeout_seconds,
        )

    @app.post("/v1/retrieve", response_model=RetrieveResponse, tags=["query"])
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

    @app.post("/v1/laws/upload", tags=["knowledge-base"])
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

        def operation(
            report: Callable[[str, str], None],
        ) -> dict[str, Any]:
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

    @app.post("/v1/knowledge-base/rebuild", tags=["knowledge-base"])
    async def rebuild_knowledge_base_endpoint(
        request: Request,
    ) -> StreamingResponse:
        await enforce_rate_limit(request)
        rebuild = app.state.services.get_rebuild_knowledge_base()

        def operation(
            report: Callable[[str, str], None],
        ) -> dict[str, Any]:
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
