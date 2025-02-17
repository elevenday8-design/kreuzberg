FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libpq-dev \
    pandoc \
    tesseract-ocr \
    tesseract-ocr-deu \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

FROM base AS install
WORKDIR /app/
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --verbose --frozen
ENV PATH="/app/.venv/bin:$PATH"
