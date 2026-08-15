import os
import cv2
import numpy as np
import base64
import torch
import time

# Monkey-patch torch.hub to skip GitHub rate-limited validation checks
import torch.hub
torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: None

# Configure PyTorch Hub cache path to apps/backend/src/models/depth/
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
depth_dir = os.path.join(src_dir, "models", "depth")
os.makedirs(depth_dir, exist_ok=True)
torch.hub.set_dir(depth_dir)


class DepthService:
    """Manages MiDaS depth estimation models and point cloud generation."""

    def __init__(self):
        self._models = {}
        self._transforms = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import threading
        self._lock = threading.Lock()

    def _load_model(self, model_type: str):
        """Lazy-load and cache MiDaS model + transforms."""
        if model_type in self._models:
            return self._models[model_type], self._transforms[model_type]

        with self._lock:
            if model_type in self._models:
                return self._models[model_type], self._transforms[model_type]

            model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True, skip_validation=True)
            model.to(self._device)
            model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True, skip_validation=True)
            if model_type in ("DPT_Large", "DPT_Hybrid"):
                transform = midas_transforms.dpt_transform
            else:
                transform = midas_transforms.small_transform

            self._models[model_type] = model
            self._transforms[model_type] = transform
            return model, transform

    def estimate_depth(self, image_bytes: bytes, model_type: str = "MiDaS_small") -> dict:
        """
        Estimates a depth map from a single RGB image.

        Returns:
            - depth heatmap as base64 JPEG
            - raw depth values as a flattened float32 list
            - speed metrics (preprocess, inference, postprocess in ms)
        """
        t_start = time.perf_counter()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        model, transform = self._load_model(model_type)

        t_pre = time.perf_counter()
        input_batch = transform(img_rgb).to(self._device)
        preprocess_ms = (time.perf_counter() - t_pre) * 1000

        t_inf = time.perf_counter()
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        inference_ms = (time.perf_counter() - t_inf) * 1000

        t_post = time.perf_counter()
        depth_np = prediction.cpu().numpy()

        # Normalize to 0–255 for heatmap
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        depth_u8 = depth_norm.astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

        _, encoded = cv2.imencode(".jpg", depth_colormap, [cv2.IMWRITE_JPEG_QUALITY, 90])
        heatmap_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        # Also encode source image for side-by-side display
        _, src_encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        source_b64 = base64.b64encode(src_encoded.tobytes()).decode("utf-8")

        postprocess_ms = (time.perf_counter() - t_post) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        return {
            "status": "success",
            "depth_heatmap": f"data:image/jpeg;base64,{heatmap_b64}",
            "source_image": f"data:image/jpeg;base64,{source_b64}",
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "speed": {
                "preprocess": round(preprocess_ms, 2),
                "inference": round(inference_ms, 2),
                "postprocess": round(postprocess_ms, 2),
            },
        }

    def generate_point_cloud(
        self, image_bytes: bytes, model_type: str = "MiDaS_small", downsample: int = 2
    ) -> dict:
        """
        Generates a 3D point cloud from an RGB image by running depth estimation
        and projecting pixels into 3D space using pinhole camera intrinsics.

        Args:
            image_bytes: Raw image bytes.
            model_type: MiDaS model variant.
            downsample: Sample every Nth pixel (1 = full res, 2 = 1/4 points, etc.).

        Returns:
            - vertices: list of {x, y, z, r, g, b} dicts
            - speed metrics
        """
        t_start = time.perf_counter()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        model, transform = self._load_model(model_type)

        t_pre = time.perf_counter()
        input_batch = transform(img_rgb).to(self._device)
        preprocess_ms = (time.perf_counter() - t_pre) * 1000

        t_inf = time.perf_counter()
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        inference_ms = (time.perf_counter() - t_inf) * 1000

        t_post = time.perf_counter()
        depth_np = prediction.cpu().numpy()

        # Normalize depth to a reasonable range (0 to 10 "units")
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 0:
            depth_normalized = (depth_np - d_min) / (d_max - d_min) * 10.0
        else:
            depth_normalized = np.zeros_like(depth_np)

        # Synthetic pinhole camera intrinsics
        fx = w * 0.8
        fy = w * 0.8
        cx = w / 2.0
        cy = h / 2.0

        # Build point cloud with downsampling
        ds = max(1, int(downsample))
        vertices = []

        for v in range(0, h, ds):
            for u in range(0, w, ds):
                z = float(depth_normalized[v, u])
                if z < 0.01:
                    continue
                x = float((u - cx) * z / fx)
                y = float((v - cy) * z / fy)
                r, g, b = int(img_rgb[v, u, 0]), int(img_rgb[v, u, 1]), int(img_rgb[v, u, 2])
                vertices.append({"x": round(x, 4), "y": round(-y, 4), "z": round(-z, 4), "r": r, "g": g, "b": b})

        postprocess_ms = (time.perf_counter() - t_post) * 1000

        return {
            "status": "success",
            "vertex_count": len(vertices),
            "vertices": vertices,
            "speed": {
                "preprocess": round(preprocess_ms, 2),
                "inference": round(inference_ms, 2),
                "postprocess": round(postprocess_ms, 2),
            },
        }
