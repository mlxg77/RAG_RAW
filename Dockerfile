# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.11.5 AS uv

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./

# 先安装锁定依赖，源代码变化时可以复用依赖层缓存。
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 raglaw \
    && mkdir -p /app/data /app/indexes /app/logs \
    && chown -R raglaw:raglaw /app

USER raglaw

EXPOSE 8000 8501

CMD ["uvicorn", "rag_law.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
