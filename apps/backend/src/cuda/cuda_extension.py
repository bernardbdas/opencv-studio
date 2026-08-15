"""CUDA Extension loader and fallback dispatcher for opencv-studio."""

import os
import time
import logging
import torch
from torch.utils.cpp_extension import load_inline

logger = logging.getLogger("opencv-studio.cuda")

_CUDA_EXTENSION = None
_CUDA_AVAILABLE = False

CUDA_SOURCE = """
#include <torch/extension.h>
#include <cmath>

__global__ void fused_unet_activation_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int num_elements,
    const float alpha,
    const float beta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = input[idx];
        float sig = 1.0f / (1.0f + expf(-alpha * val));
        output[idx] = (val * sig) * beta;
    }
}

torch::Tensor fused_unet_activation_cuda(
    torch::Tensor input,
    float alpha,
    float beta
) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    auto output = torch::empty_like(input);
    const int num_elements = input.numel();

    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    fused_unet_activation_kernel<<<blocks_per_grid, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_elements,
        alpha,
        beta
    );

    return output;
}

torch::Tensor fused_unet_activation_cpu(
    torch::Tensor input,
    float alpha,
    float beta
) {
    auto sig = torch::sigmoid(alpha * input);
    return (input * sig) * beta;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_unet_activation_cuda", &fused_unet_activation_cuda, "Fused U-Net Activation CUDA");
    m.def("fused_unet_activation_cpu", &fused_unet_activation_cpu, "Fused U-Net Activation CPU");
}
"""

CPP_SOURCE = """
#include <torch/extension.h>

torch::Tensor fused_unet_activation_cpu(
    torch::Tensor input,
    float alpha,
    float beta
) {
    auto sig = torch::sigmoid(alpha * input);
    return (input * sig) * beta;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_unet_activation_cpu", &fused_unet_activation_cpu, "Fused U-Net Activation CPU");
}
"""

def init_cuda_extension():
    global _CUDA_EXTENSION, _CUDA_AVAILABLE
    if _CUDA_EXTENSION is not None:
        return _CUDA_AVAILABLE

    if torch.cuda.is_available():
        try:
            logger.info("Compiling CUDA extension module for PyTorch...")
            _CUDA_EXTENSION = load_inline(
                name="opencv_studio_cuda_ext",
                cpp_sources=CPP_SOURCE,
                cuda_sources=CUDA_SOURCE,
                functions=["fused_unet_activation_cuda", "fused_unet_activation_cpu"],
                verbose=False,
            )
            _CUDA_AVAILABLE = True
            logger.info("CUDA extension module loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Failed to load CUDA extension, using CPU fallback: {e}")
            _CUDA_AVAILABLE = False
            return False
    else:
        logger.info("CUDA is not available on this device. Using PyTorch CPU fallback.")
        _CUDA_AVAILABLE = False
        return False

def run_fused_unet_activation(x: torch.Tensor, alpha: float = 1.0, beta: float = 1.0) -> torch.Tensor:
    """Execute fused U-Net activation using CUDA kernel if available, else PyTorch CPU fallback."""
    init_cuda_extension()
    if _CUDA_AVAILABLE and x.is_cuda and _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION.fused_unet_activation_cuda(x.contiguous(), alpha, beta)
    else:
        sig = torch.sigmoid(alpha * x)
        return (x * sig) * beta

def benchmark_cuda_vs_cpu(tensor_shape=(1, 64, 256, 256), iterations=50):
    """Benchmark execution latency between PyTorch CPU baseline and Custom CUDA Kernel."""
    init_cuda_extension()
    
    # 1. CPU Execution Benchmark
    x_cpu = torch.randn(*tensor_shape, dtype=torch.float32)
    start_cpu = time.perf_counter()
    for _ in range(iterations):
        sig = torch.sigmoid(1.0 * x_cpu)
        res_cpu = (x_cpu * sig) * 1.0
    end_cpu = time.perf_counter()
    cpu_ms = ((end_cpu - start_cpu) / iterations) * 1000.0

    # 2. CUDA / Accelerated Benchmark
    cuda_active = torch.cuda.is_available() and _CUDA_AVAILABLE
    if cuda_active:
        x_cuda = x_cpu.cuda()
        # Warmup
        for _ in range(5):
            _ = run_fused_unet_activation(x_cuda)
        torch.cuda.synchronize()
        
        start_cuda = time.perf_counter()
        for _ in range(iterations):
            _ = run_fused_unet_activation(x_cuda)
        torch.cuda.synchronize()
        end_cuda = time.perf_counter()
        cuda_ms = ((end_cuda - start_cuda) / iterations) * 1000.0
        speedup = cpu_ms / max(cuda_ms, 1e-5)
    else:
        # Optimized PyTorch vectorized CPU benchmark
        start_py = time.perf_counter()
        for _ in range(iterations):
            _ = run_fused_unet_activation(x_cpu)
        end_py = time.perf_counter()
        cuda_ms = ((end_py - start_py) / iterations) * 1000.0
        speedup = 1.0

    return {
        "cuda_available": _CUDA_AVAILABLE,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (Host System)",
        "tensor_shape": list(tensor_shape),
        "total_elements": int(x_cpu.numel()),
        "cpu_latency_ms": round(cpu_ms, 4),
        "cuda_latency_ms": round(cuda_ms, 4),
        "speedup": round(speedup, 2)
    }
