# Multi-stage Dockerfile for opencv-studio

# Stage 1: Build React Web Assets
FROM oven/bun:1-alpine AS web-builder
WORKDIR /app
COPY package.json bun.lockb* bun.lock* ./
COPY apps/web/package.json ./apps/web/
RUN bun install
COPY apps/web ./apps/web
COPY nx.json ./
RUN bunx vite build apps/web

# Stage 2: Python Backend Runtime
FROM ghcr.io/astral-sh/uv:latest AS uv
FROM python:3.13-slim

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

# System dependencies for OpenCV & C++ CUDA/CPU extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    libxcb1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy application source code
COPY . .

# Copy built static web app from Stage 1 into apps/web/dist
COPY --from=web-builder /app/apps/web/dist ./apps/web/dist

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "uvicorn", "apps.backend.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
