import asyncio
import os
import requests
from fastapi import APIRouter, UploadFile, File, Query, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/api/depth", tags=["depth"])

class VideoItem(BaseModel):
    id: str
    title: str
    description: str
    url: str
    filename: str
    cached: bool
    status: str # "ready", "missing", "downloading", "error"

# In-memory tracking of currently downloading videos
video_download_status = {}

VIDEOS_REGISTRY = {
    "drone": {
        "title": "Drone Coast Flyover",
        "description": "Aerial drone flyover video over a coastal road, ideal for terrain and landscape depth reconstruction.",
        "url": "https://shotstack-assets.s3-ap-southeast-2.amazonaws.com/footage/drone.mp4",
        "filename": "drone.mp4"
    },
    "road": {
        "title": "Highway Car Driving",
        "description": "Dashcam video of car driving along a highway, ideal for road surface and vehicle depth tracking.",
        "url": "https://shotstack-assets.s3-ap-southeast-2.amazonaws.com/footage/road.mp4",
        "filename": "road.mp4"
    },
    "traffic": {
        "title": "Traffic Junction",
        "description": "High-angle stationary video of a traffic junction with moving cars, ideal for vehicle segmentation and depth estimation.",
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/traffic.mp4",
        "filename": "traffic.mp4"
    }
}

# Resolve target directory (assets/demo)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
assets_demo_dir = os.path.join(root_dir, "assets", "demo")

def download_video_in_background(video_id: str, download_url: str, dest_path: str):
    video_download_status[video_id] = "downloading"
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        r = requests.get(download_url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        video_download_status[video_id] = "ready"
    except Exception as e:
        video_download_status[video_id] = "error"
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except:
                pass

@router.get("/videos", response_model=List[VideoItem])
def get_videos():
    videos_list = []
    for vid_id, info in VIDEOS_REGISTRY.items():
        path = os.path.join(assets_demo_dir, info["filename"])
        cached = os.path.exists(path) and os.path.getsize(path) > 100000
        
        status = "ready" if cached else "missing"
        if vid_id in video_download_status:
            status = video_download_status[vid_id]
            if status == "downloading":
                cached = False
                
        videos_list.append({
            "id": vid_id,
            "title": info["title"],
            "description": info["description"],
            "url": info["url"],
            "filename": info["filename"],
            "cached": cached,
            "status": status
        })
    return videos_list

@router.post("/videos/{video_id}/download")
def download_video(video_id: str, background_tasks: BackgroundTasks):
    if video_id not in VIDEOS_REGISTRY:
        raise HTTPException(status_code=404, detail="Video ID not found in registry")
        
    info = VIDEOS_REGISTRY[video_id]
    dest_path = os.path.join(assets_demo_dir, info["filename"])
    
    # Check if already cached
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 100000:
        return {"status": "ready", "message": f"{video_id} is already cached locally."}
        
    # Check if downloading
    if video_download_status.get(video_id) == "downloading":
        return {"status": "downloading", "message": f"{video_id} is currently downloading."}
        
    # Queue background task
    background_tasks.add_task(
        download_video_in_background,
        video_id=video_id,
        download_url=info["url"],
        dest_path=dest_path
    )
    
    video_download_status[video_id] = "downloading"
    return {"status": "started", "message": f"Started download of {video_id} in background."}

# Lazy singleton
_depth_service = None


def _get_service():
    global _depth_service
    if _depth_service is None:
        from apps.backend.src.services.depth_service import DepthService
        _depth_service = DepthService()
    return _depth_service


import time
from apps.backend.src.services.metrics_service import metrics_service

@router.post("/estimate")
async def estimate_depth(
    file: UploadFile = File(...),
    model_type: str = Query("MiDaS_small", description="MiDaS model variant: DPT_Large, DPT_Hybrid, MiDaS_small"),
):
    """Runs monocular depth estimation and returns a colormapped depth heatmap."""
    image_bytes = await file.read()
    service = _get_service()
    start = time.time()
    # Run the CPU/IO blocking download & inference in a thread pool to avoid blocking the main event loop
    res = await asyncio.to_thread(service.estimate_depth, image_bytes, model_type=model_type)
    latency = (time.time() - start) * 1000
    metrics_service.record("3D Depth Estimation", latency)
    return res


@router.post("/pointcloud")
async def generate_point_cloud(
    file: UploadFile = File(...),
    model_type: str = Query("MiDaS_small", description="MiDaS model variant"),
    downsample: int = Query(2, description="Sample every Nth pixel (1=full, 2=quarter, 4=sparse)"),
):
    """Runs depth estimation and projects the result into a 3D point cloud."""
    image_bytes = await file.read()
    service = _get_service()
    start = time.time()
    # Run in a thread pool to prevent blocking while calculating 3D projection or downloading model weights
    res = await asyncio.to_thread(service.generate_point_cloud, image_bytes, model_type=model_type, downsample=downsample)
    latency = (time.time() - start) * 1000
    metrics_service.record("3D Depth Estimation", latency)
    return res

