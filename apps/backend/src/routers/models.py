"""FastAPI Router for integrated Model Repository management in opencv-studio."""

import os
import shutil
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter(prefix="/api/models", tags=["Model Repository"])
logger = logging.getLogger("opencv-studio.models-router")

# Base directory for models
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
models_dir = os.path.join(src_dir, "models")

class ModelItem(BaseModel):
    id: str
    name: str
    filename: str
    category: str
    description: str
    cached: bool
    size_mb: float
    download_url: str
    status: str # "ready", "missing", "downloading", "error"

# Cache in-memory tracking of currently downloading models
downloading_status = {}

# Registry mapping model ID to metadata and local path
MODELS_REGISTRY = {
    # 1. MediaPipe
    "mp_object_detector": {
        "name": "MediaPipe Object Detector (EfficientDet-Lite0)",
        "filename": "efficientdet_lite0.tflite",
        "category": "MediaPipe",
        "description": "On-device object detection model trained on COCO dataset capable of identifying 80 common object classes.",
        "path": os.path.join(models_dir, "mediapipe", "object_detector", "efficientdet_lite0.tflite"),
        "download_url": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"
    },
    "mp_image_classifier": {
        "name": "MediaPipe Image Classifier (EfficientNet-Lite0)",
        "filename": "efficientnet_lite0.tflite",
        "category": "MediaPipe",
        "description": "Classifies images into 1000 categories from the ImageNet database.",
        "path": os.path.join(models_dir, "mediapipe", "image_classifier", "efficientnet_lite0.tflite"),
        "download_url": "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite"
    },
    "mp_image_segmenter": {
        "name": "MediaPipe Selfie Segmenter",
        "filename": "selfie_segmenter.tflite",
        "category": "MediaPipe",
        "description": "High-speed portrait segmentation model designed for separating human subjects from arbitrary backgrounds.",
        "path": os.path.join(models_dir, "mediapipe", "image_segmenter", "selfie_segmenter.tflite"),
        "download_url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    },
    "mp_gesture_recognizer": {
        "name": "MediaPipe Gesture Recognizer",
        "filename": "gesture_recognizer.task",
        "category": "MediaPipe",
        "description": "Recognizes hand gestures in real-time, enabling gesture-based control interfaces.",
        "path": os.path.join(models_dir, "mediapipe", "gesture_recognizer", "gesture_recognizer.task"),
        "download_url": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
    },
    "mp_hand_landmarker": {
        "name": "MediaPipe Hand Landmarker",
        "filename": "hand_landmarker.task",
        "category": "MediaPipe",
        "description": "Tracks 21 3D landmarks of hands in real-time for gesture detection.",
        "path": os.path.join(models_dir, "mediapipe", "hand_landmarker", "hand_landmarker.task"),
        "download_url": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    },
    "mp_face_detector": {
        "name": "MediaPipe Face Detector",
        "filename": "blaze_face_short_range.tflite",
        "category": "MediaPipe",
        "description": "Detects bounding boxes of faces in images and video streams.",
        "path": os.path.join(models_dir, "mediapipe", "face_detector", "blaze_face_short_range.tflite"),
        "download_url": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    },
    "mp_face_landmarker": {
        "name": "MediaPipe Face Landmarker",
        "filename": "face_landmarker.task",
        "category": "MediaPipe",
        "description": "Estimates 468 3D facial landmarks in real-time for detailed mapping.",
        "path": os.path.join(models_dir, "mediapipe", "face_landmarker", "face_landmarker.task"),
        "download_url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    },
    "mp_pose_landmarker": {
        "name": "MediaPipe Pose Landmarker",
        "filename": "pose_landmarker_full.task",
        "category": "MediaPipe",
        "description": "Tracks 33 body pose landmarks in real-time for full body posture analysis.",
        "path": os.path.join(models_dir, "mediapipe", "pose_landmarker", "pose_landmarker_full.task"),
        "download_url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    },
    "mp_holistic_landmarker": {
        "name": "MediaPipe Holistic Landmarker",
        "filename": "holistic_landmarker.task",
        "category": "MediaPipe",
        "description": "Combines pose, face, and hand landmarkers into a single holistic tracking model.",
        "path": os.path.join(models_dir, "mediapipe", "holistic_landmarker", "holistic_landmarker.task"),
        "download_url": "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
    },
    "mp_image_embedder": {
        "name": "MediaPipe Image Embedder",
        "filename": "mobilenet_v3_small.tflite",
        "category": "MediaPipe",
        "description": "Generates high-dimensional feature embeddings representing the visual content of images.",
        "path": os.path.join(models_dir, "mediapipe", "image_embedder", "mobilenet_v3_small.tflite"),
        "download_url": "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite"
    },
    "mp_language_detector": {
        "name": "MediaPipe Language Detector",
        "filename": "language_detector.tflite",
        "category": "MediaPipe",
        "description": "Identifies the language of text input sequences.",
        "path": os.path.join(models_dir, "mediapipe", "language_detector", "language_detector.tflite"),
        "download_url": "https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/latest/language_detector.tflite"
    },
    # 2. YOLO
    "yolo_v5nu": {
        "name": "YOLOv5nu Object Detector (Nano)",
        "filename": "yolov5nu.pt",
        "category": "YOLO",
        "description": "Ultra-lightweight YOLOv5 object detector.",
        "path": os.path.join(models_dir, "yolo", "yolov5nu.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5nu.pt"
    },
    "yolo_v5n_seg": {
        "name": "YOLOv5n-seg Instance Segmenter (Nano)",
        "filename": "yolov5n-seg.pt",
        "category": "YOLO",
        "description": "Lightweight instance segmentation using YOLOv5.",
        "path": os.path.join(models_dir, "yolo", "yolov5n-seg.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov5n-seg.pt"
    },
    "yolo_v6n": {
        "name": "YOLOv6n Object Detector (Nano)",
        "filename": "yolov6n.pt",
        "category": "YOLO",
        "description": "Real-time object detector built by Meituan.",
        "path": os.path.join(models_dir, "yolo", "yolov6n.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov6n.pt"
    },
    "yolo_v8n": {
        "name": "YOLOv8n Object Detector (Nano)",
        "filename": "yolov8n.pt",
        "category": "YOLO",
        "description": "Fast and lightweight real-time object detection model.",
        "path": os.path.join(models_dir, "yolo", "yolov8n.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
    },
    "yolo_v8n_seg": {
        "name": "YOLOv8n-seg Instance Segmenter (Nano)",
        "filename": "yolov8n-seg.pt",
        "category": "YOLO",
        "description": "High-speed instance segmentation model identifying class boundaries.",
        "path": os.path.join(models_dir, "yolo", "yolov8n-seg.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt"
    },
    "yolo_v8n_pose": {
        "name": "YOLOv8n-pose Keypoint Tracker (Nano)",
        "filename": "yolov8n-pose.pt",
        "category": "YOLO",
        "description": "Computes keypoint pose annotations for human subjects.",
        "path": os.path.join(models_dir, "yolo", "yolov8n-pose.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"
    },
    "yolo_v8s_worldv2": {
        "name": "YOLOv8s-worldv2 Open-Vocabulary Detector",
        "filename": "yolov8s-worldv2.pt",
        "category": "YOLO",
        "description": "Detects objects dynamically based on any arbitrary user-defined text labels.",
        "path": os.path.join(models_dir, "yolo", "yolov8s-worldv2.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt"
    },
    "yolo_v9c": {
        "name": "YOLOv9c Object Detector (Compact)",
        "filename": "yolov9c.pt",
        "category": "YOLO",
        "description": "Object detection weights utilizing programmable gradient information.",
        "path": os.path.join(models_dir, "yolo", "yolov9c.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c.pt"
    },
    "yolo_v9c_seg": {
        "name": "YOLOv9c-seg Instance Segmenter (Compact)",
        "filename": "yolov9c-seg.pt",
        "category": "YOLO",
        "description": "Compact instance segmentation using YOLOv9 architecture.",
        "path": os.path.join(models_dir, "yolo", "yolov9c-seg.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c-seg.pt"
    },
    "yolo_v10n": {
        "name": "YOLOv10n Object Detector (Nano)",
        "filename": "yolov10n.pt",
        "category": "YOLO",
        "description": "NMS-free real-time object detection model for lower inference latency.",
        "path": os.path.join(models_dir, "yolo", "yolov10n.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt"
    },
    "yolo_v11n": {
        "name": "YOLO11n Object Detector (Nano)",
        "filename": "yolo11n.pt",
        "category": "YOLO",
        "description": "Next-generation YOLO model for high-speed object detection.",
        "path": os.path.join(models_dir, "yolo", "yolo11n.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    },
    "yolo_v11n_seg": {
        "name": "YOLO11n-seg Instance Segmenter (Nano)",
        "filename": "yolo11n-seg.pt",
        "category": "YOLO",
        "description": "High-accuracy instance segmentation via YOLO11 architecture.",
        "path": os.path.join(models_dir, "yolo", "yolo11n-seg.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"
    },
    "yolo_v11n_pose": {
        "name": "YOLO11n-pose Keypoint Tracker (Nano)",
        "filename": "yolo11n-pose.pt",
        "category": "YOLO",
        "description": "Human skeletal keypoint poses via YOLO11 nano weights.",
        "path": os.path.join(models_dir, "yolo", "yolo11n-pose.pt"),
        "download_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
    },
    # 3. Depth
    "depth_midas_small": {
        "name": "MiDaS Small (Depth Map)",
        "filename": "midas_v21_small_256.pt",
        "category": "Depth Lab",
        "description": "Efficient, CPU-friendly monocular depth estimation model.",
        "path": os.path.join(models_dir, "depth", "checkpoints", "midas_v21_small_256.pt"),
        "download_url": "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
    },
    "depth_dpt_hybrid": {
        "name": "MiDaS DPT Hybrid",
        "filename": "dpt_hybrid_384.pt",
        "category": "Depth Lab",
        "description": "Balanced depth accuracy using hybrid transformer backbone.",
        "path": os.path.join(models_dir, "depth", "checkpoints", "dpt_hybrid_384.pt"),
        "download_url": "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt"
    },
    "depth_dpt_large": {
        "name": "MiDaS DPT Large",
        "filename": "dpt_large_384.pt",
        "category": "Depth Lab",
        "description": "Highest accuracy monocular depth estimation for high-end GPUs.",
        "path": os.path.join(models_dir, "depth", "checkpoints", "dpt_large_384.pt"),
        "download_url": "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt"
    }
}

def download_model_in_background(model_id: str, download_url: str, dest_path: str):
    downloading_status[model_id] = "downloading"
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if model_id.startswith("yolo_"):
            from ultralytics import YOLO
            YOLO(dest_path)
        elif model_id.startswith("depth_"):
            import torch
            torch.hub.set_dir(os.path.dirname(os.path.dirname(dest_path)))
            torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: None
            model_type_map = {
                "depth_midas_small": "MiDaS_small",
                "depth_dpt_hybrid": "DPT_Hybrid",
                "depth_dpt_large": "DPT_Large"
            }
            m_type = model_type_map.get(model_id, "MiDaS_small")
            torch.hub.load("intel-isl/MiDaS", m_type, trust_repo=True, skip_validation=True)
            torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True, skip_validation=True)
        else:
            import requests
            r = requests.get(download_url, stream=True, timeout=45)
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        
        downloading_status[model_id] = "ready"
        logger.info(f"Background download of {model_id} complete.")
    except Exception as e:
        downloading_status[model_id] = "error"
        logger.error(f"Error downloading model {model_id}: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except:
                pass

@router.get("", response_model=List[ModelItem])
def get_models():
    models_list = []
    for model_id, info in MODELS_REGISTRY.items():
        path = info["path"]
        cached = os.path.exists(path) and os.path.getsize(path) > 0
        size_mb = round(os.path.getsize(path) / (1024 * 1024), 2) if cached else 0.0
        
        # Check current downloading status
        status = "ready" if cached else "missing"
        if model_id in downloading_status:
            status = downloading_status[model_id]
            if status == "downloading":
                cached = False
        
        models_list.append({
            "id": model_id,
            "name": info["name"],
            "filename": info["filename"],
            "category": info["category"],
            "description": info["description"],
            "cached": cached,
            "size_mb": size_mb,
            "download_url": info["download_url"],
            "status": status
        })
    return models_list

@router.post("/{model_id}/download")
def download_model(model_id: str, background_tasks: BackgroundTasks):
    if model_id not in MODELS_REGISTRY:
        raise HTTPException(status_code=404, detail="Model ID not found in registry")
        
    info = MODELS_REGISTRY[model_id]
    path = info["path"]
    
    # Check if already cached
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return {"status": "ready", "message": f"{model_id} is already cached locally."}
        
    # Check if downloading
    if downloading_status.get(model_id) == "downloading":
        return {"status": "downloading", "message": f"{model_id} is currently downloading."}
        
    # Queue background task
    background_tasks.add_task(
        download_model_in_background,
        model_id=model_id,
        download_url=info["download_url"],
        dest_path=path
    )
    
    downloading_status[model_id] = "downloading"
    return {"status": "started", "message": f"Started download of {model_id} in background."}

@router.delete("/{model_id}")
def delete_model(model_id: str):
    if model_id not in MODELS_REGISTRY:
        raise HTTPException(status_code=404, detail="Model ID not found in registry")
        
    info = MODELS_REGISTRY[model_id]
    path = info["path"]
    
    if not os.path.exists(path):
        return {"status": "not_found", "message": "Model is not cached locally."}
        
    try:
        os.remove(path)
        if model_id in downloading_status:
            del downloading_status[model_id]
        return {"status": "deleted", "message": f"Successfully deleted local cache for {model_id}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model file: {e}")
