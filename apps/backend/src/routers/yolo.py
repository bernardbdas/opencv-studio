import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from apps.backend.src.services.yolo_service import YoloService
from apps.backend.src.services.metrics_service import metrics_service

router = APIRouter(prefix="/api/yolo", tags=["YOLO Tasks"])
yolo_service = YoloService()

@router.post("/detect")
async def yolo_detect(version: str = "v8", conf: float = 0.25, iou: float = 0.45, show_labels: bool = True, file: UploadFile = File(...)):
    """Runs YOLO object detection."""
    try:
        contents = await file.read()
        start = time.time()
        res = yolo_service.detect_objects(contents, version=version, conf=conf, iou=iou, show_labels=show_labels)
        latency = (time.time() - start) * 1000
        metrics_service.record("YOLO Studio", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO Object Detection failed: {str(e)}")

@router.post("/segment")
async def yolo_segment(version: str = "v8", conf: float = 0.25, iou: float = 0.45, file: UploadFile = File(...)):
    """Runs YOLO instance segmentation."""
    try:
        contents = await file.read()
        start = time.time()
        res = yolo_service.segment_objects(contents, version=version, conf=conf, iou=iou)
        latency = (time.time() - start) * 1000
        metrics_service.record("YOLO Studio", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO Instance Segmentation failed: {str(e)}")

@router.post("/pose")
async def yolo_pose(version: str = "v8", conf: float = 0.25, iou: float = 0.45, file: UploadFile = File(...)):
    """Runs YOLO pose estimation."""
    try:
        contents = await file.read()
        start = time.time()
        res = yolo_service.estimate_pose(contents, version=version, conf=conf, iou=iou)
        latency = (time.time() - start) * 1000
        metrics_service.record("YOLO Studio", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO Pose Estimation failed: {str(e)}")
