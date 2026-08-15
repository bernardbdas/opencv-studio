import os
import cv2
import numpy as np
import base64
import logging
from ultralytics import YOLO

logger = logging.getLogger("opencv-studio.yolo")

class YoloService:
    def __init__(self):
        # Cache loaded models by key, e.g. "yolov8n.pt", "yolo11n-seg.pt", etc.
        self._models = {}
        
    def _get_model_name(self, version: str, task: str) -> str:
        # version: 'v5', 'v6', 'v8', 'v9', 'v10', 'v11', 'world'
        # task: 'detect', 'segment', 'pose'
        v = version.lower()
        t = task.lower()
        
        if v == 'v5':
            if t == 'segment':
                return 'yolov5n-seg.pt'
            elif t == 'pose':
                return 'yolov8n-pose.pt'  # fallback
            return 'yolov5nu.pt'
        elif v == 'v6':
            if t == 'detect':
                return 'yolov6n.pt'
            raise ValueError("YOLOv6 only supports the Object Detection task in standard releases.")
        elif v == 'world':
            if t == 'detect':
                return 'yolov8s-worldv2.pt'
            raise ValueError("YOLO-World only supports the Object Detection task.")
        elif v == 'v8':
            if t == 'segment':
                return 'yolov8n-seg.pt'
            elif t == 'pose':
                return 'yolov8n-pose.pt'
            return 'yolov8n.pt'
        elif v == 'v9':
            if t == 'segment':
                return 'yolov9c-seg.pt'
            elif t == 'pose':
                return 'yolov8n-pose.pt'  # fallback
            return 'yolov9c.pt'
        elif v == 'v10':
            if t == 'detect':
                return 'yolov10n.pt'
            raise ValueError("YOLOv10 officially only supports the Object Detection task in standard releases.")
        elif v == 'v11':
            if t == 'segment':
                return 'yolo11n-seg.pt'
            elif t == 'pose':
                return 'yolo11n-pose.pt'
            return 'yolo11n.pt'
            
        return 'yolov8n.pt'

    def get_model(self, version: str, task: str) -> YOLO:
        model_name = self._get_model_name(version, task)
        if model_name not in self._models:
            logger.info(f"Loading YOLO model {model_name}...")
            
            # Resolve directory for YOLO weights in apps/backend/src/models/yolo/
            src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            yolo_dir = os.path.join(src_dir, "models", "yolo")
            os.makedirs(yolo_dir, exist_ok=True)
            
            model_path = os.path.join(yolo_dir, model_name)
            self._models[model_name] = YOLO(model_path)
        return self._models[model_name]

    def detect_objects(self, image_bytes: bytes, version: str = "v8", conf: float = 0.25, iou: float = 0.45, show_labels: bool = True) -> dict:
        """Runs YOLO object detection on an image frame and returns BGR output image."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
            
        model = self.get_model(version, "detect")
        results = model(img, conf=conf, iou=iou, verbose=False)
        result = results[0]
        
        detections = []
        out_img = img.copy()
        
        # Color palette for classes
        colors = [
            (255, 0, 128),  # Pink
            (0, 255, 255),  # Cyan
            (0, 255, 0),    # Green
            (255, 165, 0),  # Orange
            (128, 0, 255),  # Violet
            (255, 255, 0),  # Yellow
        ]
        
        if result.boxes:
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(box.cls[0].cpu().item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                score = float(box.conf[0].cpu().item())
                
                detections.append({
                    "class_name": cls_name,
                    "confidence": score,
                    "bbox": [x1, y1, x2, y2]
                })
                
                color = colors[cls_id % len(colors)]
                
                # Draw rounded-corners bounding box with anti-aliasing!
                cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
                
                if show_labels:
                    label = f"{cls_name} {round(score * 100)}%"
                    # Draw a nice semi-transparent label tag background
                    (w_txt, h_txt), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(out_img, (x1, y1 - h_txt - 6), (x1 + w_txt + 10, y1), color, -1)
                    cv2.putText(out_img, label, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                    
        _, encoded = cv2.imencode(".jpg", out_img)
        img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        
        # Get speed metrics from ultralytics
        speed = getattr(result, "speed", {})
        
        return {
            "status": "success",
            "detections": detections,
            "image_base64": f"data:image/jpeg;base64,{img_b64}",
            "speed": {
                "preprocess": speed.get("preprocess", 0.0),
                "inference": speed.get("inference", 0.0),
                "postprocess": speed.get("postprocess", 0.0)
            }
        }

    def segment_objects(self, image_bytes: bytes, version: str = "v8", conf: float = 0.25, iou: float = 0.45) -> dict:
        """Runs YOLO segmentation on an image frame and overlays semi-transparent color masks."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
            
        model = self.get_model(version, "segment")
        results = model(img, conf=conf, iou=iou, verbose=False)
        result = results[0]
        
        out_img = img.copy()
        detections = []
        
        colors = [
            (255, 0, 128),  # Pink
            (0, 255, 255),  # Cyan
            (0, 255, 0),    # Green
            (255, 165, 0),  # Orange
            (128, 0, 255),  # Violet
            (255, 255, 0),  # Yellow
        ]
        
        # Overlay masks if detected
        if result.masks is not None:
            mask_layer = np.zeros_like(img)
            
            for idx, (mask, box) in enumerate(zip(result.masks.xy, result.boxes)):
                cls_id = int(box.cls[0].cpu().item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                score = float(box.conf[0].cpu().item())
                
                # Check bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                detections.append({
                    "class_name": cls_name,
                    "confidence": score,
                    "bbox": [x1, y1, x2, y2]
                })
                
                color = colors[cls_id % len(colors)]
                
                # Fill polygon mask
                pts = np.array(mask, dtype=np.int32)
                if len(pts) > 0:
                    cv2.fillPoly(mask_layer, [pts], color)
                    # Draw outline contour smoothly
                    cv2.polylines(out_img, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            
            # Blend mask layer with original image
            cv2.addWeighted(mask_layer, 0.4, out_img, 1.0, 0, out_img)
            
            # Redraw tags on top of masks
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(box.cls[0].cpu().item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                score = float(box.conf[0].cpu().item())
                color = colors[cls_id % len(colors)]
                
                label = f"{cls_name} {round(score * 100)}%"
                (w_txt, h_txt), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cv2.rectangle(out_img, (x1, y1 - h_txt - 6), (x1 + w_txt + 10, y1), color, -1)
                cv2.putText(out_img, label, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, lineType=cv2.LINE_AA)
                
        _, encoded = cv2.imencode(".jpg", out_img)
        img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        
        # Get speed metrics from ultralytics
        speed = getattr(result, "speed", {})
        
        return {
            "status": "success",
            "detections": detections,
            "image_base64": f"data:image/jpeg;base64,{img_b64}",
            "speed": {
                "preprocess": speed.get("preprocess", 0.0),
                "inference": speed.get("inference", 0.0),
                "postprocess": speed.get("postprocess", 0.0)
            }
        }

    def estimate_pose(self, image_bytes: bytes, version: str = "v8", conf: float = 0.25, iou: float = 0.45) -> dict:
        """Runs YOLO Pose estimation and draws anti-aliased skeleton joints and connections."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")
            
        model = self.get_model(version, "pose")
        results = model(img, conf=conf, iou=iou, verbose=False)
        result = results[0]
        
        out_img = img.copy()
        
        # YOLOv8/v11 keypoints: 17 landmarks
        skeleton_connections = [
            [0, 1], [0, 2], [1, 3], [2, 4], # Face
            [5, 6],                         # Shoulders
            [5, 7], [7, 9],                 # Left arm
            [6, 8], [8, 10],                # Right arm
            [5, 11], [6, 12],               # Hips
            [11, 12],                       # Pelvis
            [11, 13], [13, 15],             # Left leg
            [12, 14], [14, 16]              # Right leg
        ]
        
        colors = {
            "joint": (0, 255, 255),       # Cyan
            "face_connect": (255, 0, 128), # Pink
            "left_limb": (0, 255, 0),     # Green
            "right_limb": (255, 165, 0),  # Orange
            "torso": (255, 255, 0)        # Yellow
        }
        
        people_detected = 0
        if result.keypoints is not None:
            people_detected = len(result.keypoints)
            for kpts in result.keypoints.data:
                kpts_np = kpts.cpu().numpy()
                
                # Draw skeleton connections
                for conn in skeleton_connections:
                    start_idx, end_idx = conn[0], conn[1]
                    pt1 = kpts_np[start_idx]
                    pt2 = kpts_np[end_idx]
                    
                    if pt1[2] > 0.5 and pt2[2] > 0.5:
                        x1, y1 = int(pt1[0]), int(pt1[1])
                        x2, y2 = int(pt2[0]), int(pt2[1])
                        
                        if start_idx < 5 or end_idx < 5:
                            conn_color = colors["face_connect"]
                        elif start_idx in [5, 7, 9, 11, 13, 15] or end_idx in [5, 7, 9, 11, 13, 15]:
                            conn_color = colors["left_limb"]
                        elif start_idx in [6, 8, 10, 12, 14, 16] or end_idx in [6, 8, 10, 12, 14, 16]:
                            conn_color = colors["right_limb"]
                        else:
                            conn_color = colors["torso"]
                            
                        cv2.line(out_img, (x1, y1), (x2, y2), conn_color, 2, lineType=cv2.LINE_AA)
                        
                # Draw keypoint dots
                for idx, pt in enumerate(kpts_np):
                    if pt[2] > 0.5:
                        cx, cy = int(pt[0]), int(pt[1])
                        cv2.circle(out_img, (cx, cy), 4, colors["joint"], -1, lineType=cv2.LINE_AA)
                        
        _, encoded = cv2.imencode(".jpg", out_img)
        img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        
        # Get speed metrics from ultralytics
        speed = getattr(result, "speed", {})
        
        return {
            "status": "success",
            "people_detected": people_detected,
            "image_base64": f"data:image/jpeg;base64,{img_b64}",
            "speed": {
                "preprocess": speed.get("preprocess", 0.0),
                "inference": speed.get("inference", 0.0),
                "postprocess": speed.get("postprocess", 0.0)
            }
        }
