"""FastAPI Router for Custom U-Net Architecture, Custom Weights, and CUDA Benchmarks."""

from pydantic import BaseModel
from fastapi import APIRouter, File, UploadFile, HTTPException, Response
from apps.backend.src.services.unet_service import unet_service

router = APIRouter(prefix="/api/unet", tags=["U-Net & CUDA Studio"])

class UNetConfigUpdate(BaseModel):
    in_channels: int = 3
    out_channels: int = 1
    depth: int = 4
    init_features: int = 32
    use_attention: bool = True
    use_cuda_kernel: bool = True

class BenchmarkRequest(BaseModel):
    batch_size: int = 1
    channels: int = 64
    height: int = 256
    width: int = 256
    iterations: int = 50

@router.get("/config")
def get_unet_config():
    """Get active U-Net architecture configuration, parameter metadata, and CUDA hardware status."""
    return unet_service.get_config_and_summary()

@router.post("/config")
def update_unet_config(req: UNetConfigUpdate):
    """Reconfigure U-Net architecture dynamically."""
    return unet_service.update_config(
        in_channels=req.in_channels,
        out_channels=req.out_channels,
        depth=req.depth,
        init_features=req.init_features,
        use_attention=req.use_attention,
        use_cuda_kernel=req.use_cuda_kernel
    )

@router.post("/weights/upload")
async def upload_custom_weights(file: UploadFile = File(...)):
    """Upload custom PyTorch weights (.pt, .pth) into active U-Net model instance."""
    try:
        contents = await file.read()
        return unet_service.load_custom_weights(contents, file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load weight checkpoint: {str(e)}")

@router.get("/weights/export")
def export_weights():
    """Download active U-Net weights as a PyTorch state_dict binary (.pt)."""
    weights_bytes = unet_service.export_weights()
    return Response(
        content=weights_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=unet_custom_weights.pt"}
    )

@router.post("/benchmark")
def run_cuda_benchmark(req: BenchmarkRequest):
    """Benchmark Custom CUDA Kernel vs CPU baseline execution performance."""
    return unet_service.run_benchmark(
        batch_size=req.batch_size,
        channels=req.channels,
        height=req.height,
        width=req.width,
        iterations=req.iterations
    )

import time
from apps.backend.src.services.metrics_service import metrics_service

@router.post("/predict")
async def predict_segmentation(file: UploadFile = File(...)):
    """Perform U-Net image segmentation with custom CUDA/CPU kernel pipeline."""
    try:
        contents = await file.read()
        start = time.time()
        res = unet_service.predict_segmentation(contents)
        latency = (time.time() - start) * 1000
        metrics_service.record("U-Net Segmentation", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")
