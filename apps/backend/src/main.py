import os
import logging
import dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Resolve workspace root dynamically
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Load dotenv
env_path = os.path.join(WORKSPACE_DIR, "local.env")
if not os.path.exists(env_path):
    env_path = os.path.join(WORKSPACE_DIR, ".env")
dotenv.load_dotenv(env_path)

import asyncio
from contextlib import asynccontextmanager
from apps.backend.src.routers import unet, vision, text_audio, yolo, knowledge, depth, models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("opencv-studio")

async def prewarm_models():
    """Background startup task to pre-download baseline models."""
    # Add a short delay to allow Uvicorn to bind and start printing startup logs
    await asyncio.sleep(2)
    logger.info("Initializing background pre-warming of model assets...")
    
    # 1. MediaPipe models
    try:
        from apps.backend.src.utils.model_downloader import get_model_path_by_config
        from apps.backend.src.utils import models
        mediapipe_configs = [
            models.OBJECT_DETECTION,
            models.IMAGE_CLASSIFICATION,
            models.IMAGE_SEGMENTATION,
            models.GESTURE_RECOGNIZER,
            models.HAND_LANDMARKER,
            models.FACE_DETECTION,
            models.FACE_LANDMARKER,
            models.POSE_LANDMARKER,
            models.HOLISTIC_LANDMARKER,
            models.IMAGE_EMBEDDING,
            models.LANGUAGE_DETECTION
        ]
        for config in mediapipe_configs:
            logger.info(f"Checking MediaPipe model cache for: {config.filename}")
            await asyncio.to_thread(get_model_path_by_config, config)
    except Exception as e:
        logger.warning(f"Error pre-warming MediaPipe models: {e}")

    # 2. YOLO models
    try:
        from apps.backend.src.services.yolo_service import YoloService
        yolo_service = YoloService()
        # Pre-download default YOLOv8 detector, segmenter, and pose models
        for task in ["detect", "segment", "pose"]:
            logger.info(f"Checking YOLO model cache for: yolov8n ({task})")
            await asyncio.to_thread(yolo_service.get_model, "v8", task)
    except Exception as e:
        logger.warning(f"Error pre-warming YOLOv8 models: {e}")

    # 3. Depth estimation model (MiDaS small)
    try:
        from apps.backend.src.services.depth_service import DepthService
        depth_s = DepthService()
        logger.info("Checking Depth estimation model cache for: MiDaS_small")
        await asyncio.to_thread(depth_s._load_model, "MiDaS_small")
    except Exception as e:
        logger.warning(f"Error pre-warming Depth model: {e}")

    logger.info("Model pre-warming background task completed! All baseline assets cached successfully.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Launch model pre-warming in background thread to avoid blocking server start
    asyncio.create_task(prewarm_models())
    yield

app = FastAPI(
    title="OpenCV Studio API",
    description="Next-generation Computer Vision, Custom CUDA Kernels & U-Net Studio Backend",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS for frontend web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import HTMLResponse, FileResponse
from apps.backend.src.services.metrics_service import metrics_service

# Include Routers
app.include_router(unet.router)
app.include_router(vision.router)
app.include_router(text_audio.router)
app.include_router(yolo.router)
app.include_router(knowledge.router)
app.include_router(depth.router)
app.include_router(models.router)

@app.get("/api/performance/chart", response_class=HTMLResponse)
def get_performance_chart():
    """Serves interactive real-time performance line chart generated using xy."""
    try:
        return metrics_service.get_chart_html()
    except Exception as e:
        return f"<h3>Error generating chart: {str(e)}</h3>"

@app.get("/api/health")
def health_check():
    """Health check endpoint and active model cache status."""
    import torch
    
    # Check downloaded status of models in apps/backend/src/models/
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    
    status_summary = {}
    
    # YOLO checks
    yolo_models = ["yolov8n.pt", "yolov8n-seg.pt", "yolov8n-pose.pt"]
    status_summary["yolo"] = {
        m: os.path.exists(os.path.join(models_dir, "yolo", m)) for m in yolo_models
    }
    
    # MediaPipe checks
    mp_models = {
        "gesture": ("gesture_recognizer", "gesture_recognizer.task"),
        "hand": ("hand_landmarker", "hand_landmarker.task"),
        "face": ("face_landmarker", "face_landmarker.task"),
        "pose": ("pose_landmarker", "pose_landmarker_full.task"),
        "holistic": ("holistic_landmarker", "holistic_landmarker.task")
    }
    status_summary["mediapipe"] = {
        k: os.path.exists(os.path.join(models_dir, "mediapipe", v[0], v[1]))
        for k, v in mp_models.items()
    }
    
    # Depth checks
    status_summary["depth"] = {
        "MiDaS_small": os.path.exists(os.path.join(models_dir, "depth", "hub", "checkpoints", "hubconf.py")) or os.path.exists(os.path.join(models_dir, "depth", "hub", "checkpoints", "intel-isl_MiDaS_master"))
    }
    
    return {
        "status": "online",
        "app": "opencv-studio",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "models_cached": status_summary
    }

# Mount demo assets directory for streaming videos and other test files
demo_assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "demo"))
os.makedirs(demo_assets_dir, exist_ok=True)
app.mount("/api/demo-files", StaticFiles(directory=demo_assets_dir), name="demo-files")

# Mount static frontend build if dist directory exists
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "web", "dist"))
if os.path.exists(frontend_dist):
    logger.info(f"Mounting static frontend build from {frontend_dist}")
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="static")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api"):
            return None
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))

