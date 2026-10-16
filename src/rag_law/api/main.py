"""阶段 6 HTTP API。

运行：``uv run uvicorn rag_law.api.main:app --host 127.0.0.1 --port 8000``。
接口只在真正处理请求时加载模型与索引，因而 ``/v1/health`` 可用于定位配置
或索引缺失，而不会在导入模块时读取密钥。
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag_law.knowledge_base import PROJECT_ROOT
from rag_law.retrieval.bm25 import DEFAULT_BM25_INDEX_PATH
from rag_law.retrieval.dense import DEFAULT_CHROMA_DIRECTORY
from rag_law.schemas import AnswerPipelineResult, RetrievalResult
from rag_law.ui.presentation import LawOption, load_law_options


logger = logging.getLogger(__name__)
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.yaml"
ARTICLES_PATH = PROJECT_ROOT / "data" / "processed" / "articles.jsonl"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_RATE_LIMIT = 30
DEFAULT_RATE_WINDOW_SECONDS = 60.0


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    law_name: str | None = Field(default=None, max_length=200)


class RetrieveRequest(QueryRequest):
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


def create_app(
    *,
    services: ApiServices | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    rate_limit: int = DEFAULT_RATE_LIMIT,
    rate_window_seconds: float = DEFAULT_RATE_WINDOW_SECONDS,
) -> FastAPI:
    """创建应用；测试可传入短超时或预置 services。"""

    app = FastAPI(title="RAG Law API", version="v1")
    app.state.services = services or ApiServices()
    app.state.timeout_seconds = timeout_seconds
    app.state.rate_limiter = SlidingWindowRateLimiter(
        limit=rate_limit, window_seconds=rate_window_seconds
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
                normalized_query, law_name=law_name
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

    return app


app = create_app()
