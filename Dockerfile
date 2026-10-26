# syntax=docker/dockerfile:1

# 默认使用 Docker Hub 官方镜像；国内服务器无法直连 Docker Hub 时，
# 可通过构建参数替换为可访问的镜像地址，例如：
#   --build-arg PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.13-slim
#   --build-arg NODE_IMAGE=docker.m.daocloud.io/library/node:22-alpine
ARG PYTHON_IMAGE=python:3.13-slim
ARG NODE_IMAGE=node:22-alpine

# ---------- 构建 Vue 前端 ----------
FROM ${NODE_IMAGE} AS web-build

WORKDIR /web

# 使用国内 npm 镜像，构建过程无需访问默认 registry。
ENV npm_config_registry=https://registry.npmmirror.com

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
RUN npm run build

# ---------- 运行 API 与静态页面 ----------
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/ \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# 从 PyPI 镜像源安装与锁定文件匹配的 uv，构建过程不依赖 ghcr.io。
RUN pip install --no-cache-dir "uv==0.11.5"

COPY pyproject.toml uv.lock README.md ./

# 先安装锁定依赖，源代码变化时可以复用依赖层缓存。
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# 前端构建产物由 FastAPI 从 /app/web/dist 直接托管。
COPY --from=web-build /web/dist ./web/dist

RUN useradd --create-home --uid 10001 raglaw \
    && mkdir -p /app/data /app/indexes /app/logs \
    && chown -R raglaw:raglaw /app

USER raglaw

EXPOSE 8000

CMD ["uvicorn", "rag_law.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
