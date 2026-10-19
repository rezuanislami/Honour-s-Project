FROM node:22-bookworm-slim AS frontend
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@10.18.0 --activate
COPY package.json pnpm-lock.yaml ./
COPY patches ./patches
RUN CI=1 pnpm install --frozen-lockfile
COPY . .
RUN pnpm build:static

FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend/src \
    PORT=3000 \
    CONSEARCH_DATA=/app/backend/data/documents.jsonl \
    CONSEARCH_STATIC_DIR=/app/dist/public \
    HF_HOME=/tmp/huggingface
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/backend/requirements.txt
COPY backend/src /app/backend/src
COPY backend/data /app/backend/data
COPY --from=frontend /app/dist/public /app/dist/public
EXPOSE 3000
CMD ["sh", "-c", "exec python -m uvicorn consearch.api:app --host 0.0.0.0 --port ${PORT:-3000} --workers 1"]
