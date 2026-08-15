"""Script to pre-download model weights for opencv-studio tasks."""

import os
import sys
import argparse
import logging

# Ensure project root is in python path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("opencv-studio.download-models")

def download_mediapipe():
    logger.info("=== Downloading MediaPipe Models ===")
    from apps.backend.src.utils.model_downloader import get_model_path_by_config
    from apps.backend.src.utils import models as mp_models
    
    configs = [
        mp_models.OBJECT_DETECTION,
        mp_models.IMAGE_CLASSIFICATION,
        mp_models.IMAGE_SEGMENTATION,
        mp_models.GESTURE_RECOGNIZER,
        mp_models.HAND_LANDMARKER,
        mp_models.FACE_DETECTION,
        mp_models.FACE_LANDMARKER,
        mp_models.POSE_LANDMARKER,
        mp_models.HOLISTIC_LANDMARKER,
        mp_models.IMAGE_EMBEDDING,
        mp_models.LANGUAGE_DETECTION
    ]
    for config in configs:
        logger.info(f"Downloading {config.filename} for {config.task_name}...")
        try:
            get_model_path_by_config(config)
        except Exception as e:
            logger.error(f"Failed to download MediaPipe model {config.filename}: {e}")
    logger.info("MediaPipe models download task complete.\n")

def download_yolo():
    logger.info("=== Downloading YOLO Models ===")
    from ultralytics import YOLO
    
    yolo_dir = os.path.join(src_dir, "apps", "backend", "src", "models", "yolo")
    os.makedirs(yolo_dir, exist_ok=True)
    
    yolo_model_names = [
        "yolov5nu.pt",
        "yolov5n-seg.pt",
        "yolov8n-pose.pt",
        "yolov6n.pt",
        "yolov8s-worldv2.pt",
        "yolov8n.pt",
        "yolov8n-seg.pt",
        "yolov9c.pt",
        "yolov9c-seg.pt",
        "yolov10n.pt",
        "yolo11n.pt",
        "yolo11n-seg.pt",
        "yolo11n-pose.pt"
    ]
    for name in yolo_model_names:
        model_path = os.path.join(yolo_dir, name)
        logger.info(f"Downloading YOLO model: {name} to {model_path}...")
        try:
            YOLO(model_path)
        except Exception as e:
            logger.error(f"Failed to download YOLO model {name}: {e}")
    logger.info("YOLO models download task complete.\n")

def download_depth():
    logger.info("=== Downloading Depth Estimation Models ===")
    import torch
    
    depth_dir = os.path.join(src_dir, "apps", "backend", "src", "models", "depth")
    os.makedirs(depth_dir, exist_ok=True)
    
    # Configure PyTorch Hub cache path
    torch.hub.set_dir(depth_dir)
    
    # Monkey-patch torch.hub to skip GitHub rate-limit validation
    torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: None
    
    models = ["MiDaS_small", "DPT_Large", "DPT_Hybrid"]
    for model_type in models:
        logger.info(f"Downloading MiDaS model {model_type} to {depth_dir}...")
        try:
            torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True, skip_validation=True)
        except Exception as e:
            logger.error(f"Failed to download MiDaS model {model_type}: {e}")
            
    logger.info("Downloading MiDaS transforms...")
    try:
        torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True, skip_validation=True)
    except Exception as e:
        logger.error(f"Failed to download MiDaS transforms: {e}")
    logger.info("Depth models download task complete.\n")

def main():
    parser = argparse.ArgumentParser(description="Pre-download model weights for opencv-studio.")
    parser.add_argument("tab", choices=["mediapipe", "yolo", "depth", "all"], help="Model family to download (or 'all' for all model families)")
    args = parser.parse_args()
    
    if args.tab == "mediapipe":
        download_mediapipe()
    elif args.tab == "yolo":
        download_yolo()
    elif args.tab == "depth":
        download_depth()
    elif args.tab == "all":
        download_mediapipe()
        download_yolo()
        download_depth()

if __name__ == "__main__":
    main()
