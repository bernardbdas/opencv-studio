# OpenCV Studio 🚀

**Github Repo:** [https://github.com/bernardbdas/opencv-studio](https://github.com/bernardbdas/opencv-studio)

<div align="center">
  <a href="https://nx.dev/"><img src="https://img.shields.io/badge/Nx-Monorepo-14171F?style=for-the-badge&logo=nx&logoColor=white" alt="Nx Monorepo"></a>
  <a href="https://react.dev/"><img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React"></a>
  <a href="https://radix-ui.com/"><img src="https://img.shields.io/badge/Radix_UI-Primitives-161618?style=for-the-badge&logo=radixui&logoColor=white" alt="Radix UI"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-CUDA_Kernels-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"></a>
</div>

**OpenCV Studio** is a next-generation Computer Vision workspace refactored into an **Nx-managed monorepo**. It features a modern React (Vite, TypeScript, Radix UI primitives, Tailwind CSS) frontend and a high-performance Python FastAPI backend supporting customizable **PyTorch U-Net neural architectures**, **custom C++/CUDA kernel acceleration**, and on-device MediaPipe vision tasks.

---

## 🌟 Key Features & Architecture

### ⚡ Nx Monorepo Architecture
- **`apps/frontend`**: React 18 SPA styled with Radix UI, Tailwind CSS, and Lucide icons.
- **`apps/backend`**: FastAPI backend exposing REST & WebSockets for real-time computer vision inference, benchmark labs, and weights checkpoints.

### 🧬 Custom U-Net & CUDA Acceleration
- **Configurable U-Net**: Dynamic downsampling depth (2–5 levels), feature channel dimensions (16, 32, 64), and Attention Gates.
- **Custom C++/CUDA Fused Activation**: Fused elementwise activation & spatial scaling CUDA kernel compiled JIT via `torch.utils.cpp_extension` with automatic PyTorch CPU fallback.
- **Weights Management**: Live uploading of custom `.pt` / `.pth` checkpoints and state_dict export.
- **CUDA vs CPU Benchmark**: Real-time performance benchmark measuring millisecond latency speedup multipliers.

### 🧪 Experiments & Research
- **`experiments/`**: Organized Jupyter Notebook experiments:
  - `experiments/vision/face_tracking/`
  - `experiments/vision/gesture_recognition/`
  - `experiments/vision/hand_tracking/`
  - `experiments/unet_customization/`


---

## 🛠️ Quick Start

### Local Development (Nx)
```bash
# Install dependencies
just build

# Start both Frontend & Backend concurrently
just start
```
- **Frontend Studio**: `http://localhost:3000`
- **FastAPI API & Docs**: `http://localhost:8000/docs`

### Running via Docker & Docker Compose
```bash
just docker-build
just docker-run
```
Access the application at `http://localhost:8000`.
