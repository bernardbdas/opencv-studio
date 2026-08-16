# Generate env.json files from local.env
generate-env:
    uv run python scripts/generate_env_json.py

# Build all monorepo dependencies and packages
build: generate-env
    uv sync
    bun install

# Run backend FastAPI, web React studio, and mobile Expo app concurrently
start: generate-env
    nx run-many --target=serve --all --parallel

# Run web studio frontend
start-web: generate-env
    bunx vite apps/web --host 0.0.0.0 --force

# Run mobile Expo app
start-mobile: generate-env
    cd apps/mobile && bun run start

# Start the backend server
server:
    just start-backend

# Start web and mobile clients concurrently
client: generate-env
    bunx nx run-many --target=serve --projects=web,mobile --parallel


# Run backend server directly
start-backend:
    uv run python -m uvicorn apps.backend.src.main:app --host 0.0.0.0 --port 8000 --reload

# Build production assets for web application
build-web:
    bunx vite build apps/web

# Build Docker container for opencv-studio
docker-build:
    docker compose build

# Run opencv-studio inside Docker
docker-run:
    docker compose up

# Stop running docker containers
docker-stop:
    docker compose down

# Clean temporary folders, compile outputs, and build artifacts
clean:
    rm -rf dist node_modules/.cache .pytest_cache .ruff_cache
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find libs/shared/src -type f \( -name "*.js" -o -name "*.d.ts" \) -delete
    rm -f apps/mobile/env.json apps/web/src/env.json

# Run syntax check and linting across the monorepo
lint:
    bun run lint

# Download all MediaPipe models
download-mediapipe:
    uv run python -m apps.backend.src.utils.download_models mediapipe

# Download all YOLO models
download-yolo:
    uv run python -m apps.backend.src.utils.download_models yolo

# Download all MiDaS depth models
download-depth:
    uv run python -m apps.backend.src.utils.download_models depth

# Download all models together (MediaPipe, YOLO, MiDaS Depth)
download-all-models:
    uv run python -m apps.backend.src.utils.download_models all
