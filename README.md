# OpenCV Studio — Custom CUDA Kernels & U-Net Workspace

[![Nx Monorepo](https://img.shields.io/badge/Nx-Monorepo-14171F?style=flat&logo=nx&logoColor=white)](https://nx.dev)
[![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)](https://react.dev)
[![Radix UI](https://img.shields.io/badge/Radix_UI-161618?style=flat&logo=radix-ui&logoColor=white)](https://radix-ui.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat&logo=tailwind-css)](https://tailwindcss.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)

A computer vision research workspace built as an Nx monorepo. It features a React + Radix UI frontend and a high-performance Python FastAPI backend supporting customizable PyTorch U-Net architectures, JIT-compiled C++/CUDA kernel acceleration, and an on-device MediaPipe vision suite.

---

## Repository Structure

- `apps/web/`: React 18 single-page application built with Vite, TypeScript, Tailwind CSS, and Radix UI primitives.
- `apps/mobile/`: React Native Expo mobile client for camera stream inspection and edge testing.
- `apps/backend/`: FastAPI backend exposing REST and WebSocket endpoints for live inference, custom kernel benchmarks, and model management.
- `libs/shared/`: Shared TypeScript data contracts, interfaces, and UI utility modules.
- `experiments/`: Organized Jupyter notebooks covering vision tasks, face/hand tracking, and U-Net customization.
- `.obsidian/`: Technical knowledge hub and wiki notes formatted according to Google Open Knowledge Format (OKF) standards.

---

## Quick Start Guide

### Prerequisites
- [Bun](https://bun.sh/)
- [UV](https://docs.astral.sh/uv/)
- [Just](https://github.com/casey/just)
- NVIDIA GPU with CUDA Toolkit installed (optional; automatic CPU fallback is supported)

### 1. Install Dependencies
```bash
just install
```

### 2. Launch Local Development
Start the frontend and backend services concurrently:
```bash
just start
```
- **Web Studio**: `http://localhost:3000`
- **FastAPI Documentation**: `http://localhost:8000/docs`

You can also run components separately:
```bash
# Start backend only
just start-backend

# Start web client only
just start-web
```

### 3. Docker Deployment
To run the full stack containerized:
```bash
# Build the Docker image
just docker-build

# Run the container
just docker-run
```
Open `http://localhost:8000` to access the application.

---

## Key Features

### 1. Custom C++/CUDA Fused Kernel Acceleration
- Features a custom JIT-compiled CUDA kernel (`torch.utils.cpp_extension`) fusing activation and spatial scaling operations.
- Automatically falls back to standard PyTorch CPU tensors when running on hardware without dedicated NVIDIA GPUs.
- Built-in benchmark harness to measure millisecond latency and throughput differences between CPU and CUDA execution paths.

### 2. Configurable U-Net Architecture
- Parameterize network depth (2–5 stages), base feature channel dimensions (16, 32, 64), and optional Attention Gates.
- Live checkpoint management allowing runtime upload and inference with custom `.pt` / `.pth` state dictionaries.

### 3. MediaPipe Vision Suite
Integrated tasks for edge vision pipelines including:
- Face Detection & 468-point Face Landmarks
- Hand Tracking & Gesture Recognition
- Holistic Pose & Landmark Estimation
- Interactive Object & Image Segmentation
- Multi-language Optical Character Recognition (OCR)

---

## Visual Demo Gallery

| Task | Sample Preview |
| :--- | :--- |
| **Interactive Segmentation** | ![Interactive Segmentation](assets/demo/interactive_segmentation.jpg) |
| **Face Landmark Estimation** | ![Face Landmark](assets/demo/face_landmark.jpg) |
| **Hand Tracking & Gestures** | ![Hand Landmark](assets/demo/hand_landmark.jpg) |
| **Classic Image Processing** | ![Classic CV](assets/demo/classic_cv.jpg) |

---

## Code Quality & Maintenance

```bash
# Typecheck and lint all packages
just lint

# Remove build artifacts and temporary files
just clean
```
