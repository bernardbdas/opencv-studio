import os
import io
import time
import base64
import torch
import numpy as np
from PIL import Image
from apps.backend.src.architectures.unet_model import ConfigurableUNet
from apps.backend.src.cuda.cuda_extension import benchmark_cuda_vs_cpu

class UNetService:
    def __init__(self):
        self.config = {
            "in_channels": 3,
            "out_channels": 1,
            "depth": 4,
            "init_features": 32,
            "use_attention": True,
            "use_cuda_kernel": True,
            "custom_weights_loaded": False,
            "weights_filename": "default_initialized"
        }
        self.model: ConfigurableUNet = self._build_model()

    def _build_model(self) -> ConfigurableUNet:
        model = ConfigurableUNet(
            in_channels=self.config["in_channels"],
            out_channels=self.config["out_channels"],
            depth=self.config["depth"],
            init_features=self.config["init_features"],
            use_attention=self.config["use_attention"],
            use_cuda_kernel=self.config["use_cuda_kernel"]
        )
        model.eval()
        return model

    def get_config_and_summary(self):
        summary = self.model.get_summary()
        return {
            "config": self.config,
            "summary": summary,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }

    def update_config(self, in_channels: int, out_channels: int, depth: int, init_features: int, use_attention: bool, use_cuda_kernel: bool):
        self.config.update({
            "in_channels": in_channels,
            "out_channels": out_channels,
            "depth": depth,
            "init_features": init_features,
            "use_attention": use_attention,
            "use_cuda_kernel": use_cuda_kernel
        })
        self.model = self._build_model()
        return self.get_config_and_summary()

    def load_custom_weights(self, file_bytes: bytes, filename: str) -> dict:
        # Resolve target directory for U-Net weights in apps/backend/src/models/unet/
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        unet_dir = os.path.join(src_dir, "models", "unet")
        os.makedirs(unet_dir, exist_ok=True)
        
        # Save custom weights file to directory
        weight_path = os.path.join(unet_dir, filename)
        with open(weight_path, "wb") as f:
            f.write(file_bytes)

        res = self.model.load_custom_weights(file_bytes)
        self.config["custom_weights_loaded"] = True
        self.config["weights_filename"] = filename
        return {
            "status": "success",
            "filename": filename,
            "details": res,
            "summary": self.model.get_summary()
        }

    def export_weights(self) -> bytes:
        return self.model.export_weights_bytes()

    def run_benchmark(self, batch_size: int, channels: int, height: int, width: int, iterations: int):
        shape = (batch_size, channels, height, width)
        return benchmark_cuda_vs_cpu(tensor_shape=shape, iterations=iterations)

    def predict_segmentation(self, image_bytes: bytes) -> dict:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Preprocess
        resized = pil_img.resize((256, 256))
        img_np = np.array(resized).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() and self.config["use_cuda_kernel"] else "cpu")
        model = self.model.to(device)
        input_tensor = input_tensor.to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            output_logits = model(input_tensor)
            mask_tensor = torch.sigmoid(output_logits)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Postprocess Mask Overlay
        mask_np = mask_tensor.squeeze().cpu().numpy()
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

        mask_pil = Image.fromarray(mask_binary, mode="L").resize((orig_w, orig_h))
        overlay = Image.new("RGBA", (orig_w, orig_h), (0, 255, 128, 0))
        mask_rgba = Image.new("RGBA", (orig_w, orig_h), (0, 240, 255, 140))
        overlay.paste(mask_rgba, (0, 0), mask_pil)

        combined = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
        
        buffered = io.BytesIO()
        combined.save(buffered, format="PNG")
        overlay_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "latency_ms": round(latency_ms, 2),
            "device": str(device),
            "custom_weights_loaded": self.config["custom_weights_loaded"],
            "weights_filename": self.config["weights_filename"],
            "overlay_base64": f"data:image/png;base64,{overlay_b64}"
        }

# Global Singleton Instance
unet_service = UNetService()
