#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA Kernel: Fused Swish Activation with Spatial Feature Scaling for U-Net
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
        // Swish / SiLU: val * sigmoid(alpha * val) + scaling factor beta
        float sig = 1.0f / (1.0f + expf(-alpha * val));
        output[idx] = (val * sig) * beta;
    }
}

// Host C++ wrapper to launch CUDA kernel
torch::Tensor fused_unet_activation_cuda(
    torch::Tensor input,
    float alpha,
    float beta
) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
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

// CPU Fallback Implementation
torch::Tensor fused_unet_activation_cpu(
    torch::Tensor input,
    float alpha,
    float beta
) {
    auto sig = torch::sigmoid(alpha * input);
    return (input * sig) * beta;
}

// PyTorch extension binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_unet_activation_cuda", &fused_unet_activation_cuda, "Fused U-Net Activation CUDA");
    m.def("fused_unet_activation_cpu", &fused_unet_activation_cpu, "Fused U-Net Activation CPU");
}
