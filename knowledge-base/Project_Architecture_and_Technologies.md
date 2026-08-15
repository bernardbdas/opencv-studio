---
type: article
title: "Project Architecture & Technologies"
description: "Detailed specification of the tools powering opencv-studio: uv, nx, bun, xy, fastapi, react, and three.js."
resource: "opencv-studio"
tags: ["architecture", "bundling", "uv", "nx", "bun", "xy", "fastapi", "threejs"]
timestamp: 2026-08-07T15:00:00Z
---

# OpenCV Studio Core Architecture & Technologies

This document outlines the advanced, next-generation development stack and orchestration architecture that powers **OpenCV Studio**. The project implements a hybrid multi-language monorepo that seamlessly coordinates compiled C++/CUDA extensions, an asynchronous Python FastAPI backend, and high-performance WebGL-based frontend applications.

---

## 🚀 Package & Dependency Management

### 🐍 uv (Python)
**uv** is an extremely fast Python package installer and resolver, written in Rust. It serves as a drop-in replacement for `pip`, `pip-tools`, and `virtualenv`.
*   **Role in Project:** 
    *   Resolves and locks all backend dependencies in [uv.lock](file:///home/galahad/Developer/DATA-SCIENCE/PRACTICE/opencv-studio/uv.lock).
    *   Synchronizes and isolates the virtual environment (`.venv/`) via `uv sync`.
    *   Runs the backend uvicorn server in a reproducible state via `uv run`.
*   **Performance Advantage:** Reduces dependency installation and synchronization times from minutes to milliseconds, utilizing global package caching and hard-linking.

### 🍞 Bun (JavaScript/TypeScript)
**Bun** is an all-in-one JavaScript runtime, bundler, transpiler, and package manager designed for speed and compatibility.
*   **Role in Project:**
    *   Replaces Node.js and npm as the default package manager for the frontend workspace, creating the unified text-based lockfile `bun.lock`.
    *   Speeds up package installation times significantly through highly optimized caching.
    *   Executes build and local development binaries via `bunx` (e.g., `bunx vite`, `bunx expo`).

---

## 📦 Monorepo Orchestration

### ⚡ Nx (Workspace Scheduler)
**Nx** is a smart, fast, and extensible build system with first-class monorepo support. It manages the dependencies between the frontend applications and shared library structures.
*   **Role in Project:**
    *   Configures task graphs and targets (e.g., `build`, `serve`, `lint`) via individual `project.json` files for each package.
    *   Runs multiple applications concurrently in parallel via:
        `nx run-many --target=serve --all --parallel`
    *   Implements local and remote computation caching to ensure that tasks are only rebuilt when source files change.

---

## 🎨 Metrics & Performance Rendering

### 📈 xy (Python Charting)
**xy** is a lightweight, performance-focused charting and visualization library for Python.
*   **Role in Project:**
    *   Aggregates live latency metrics from YOLO, MediaPipe, and U-Net service models.
    *   Compiles dense metric arrays into interactive SVG line charts directly on the backend.
    *   Generates standalone HTML elements served dynamically at the performance endpoint `/api/performance/chart` which are rendered side-by-side in the Web Studio frontend.

---

## 🖥️ Backend Runtime

### ⚡ FastAPI (Asynchronous Python Web Server)
**FastAPI** is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.
*   **Role in Project:**
    *   Serves as the primary API orchestrator.
    *   Utilizes Python's `asyncio` loop to handle high-concurrency requests, live web socket streams, and parallel background tasks.
    *   Controls model lifecycle pre-warming in background worker threads on application startup to ensure instant response on the first client request.
    *   Exposes endpoints for MediaPipe tasks, YOLO detection, Custom CUDA benchmarking, and MiDaS Depth estimation.

---

## 💻 Web & 3D Interactive Client

### ⚛️ React & Vite (Frontend App)
The web client (`apps/web`) is built using **React** and bundled via **Vite** for near-instantaneous Hot Module Replacement (HMR).
*   **Role in Project:**
    *   Presents a sleek, dark-themed, glassmorphic dashboard that displays video feeds, parameter sliders, and performance metrics.
    *   Uses React's hook-based state management to cycle through tabs, stream canvas images to the backend, and overlay segmentation masks.

### 🧊 Three.js & React Three Fiber (3D Engine)
**Three.js** is a lightweight 3D library which renders GPU-accelerated 3D graphics in the browser using WebGL.
*   **Role in Project:**
    *   Powers the **3D Depth Lab** tab.
    *   Takes 2D depth estimations from the MiDaS backend, projects the pixels into 3D camera space via coordinate homography, and renders an interactive, rotating 3D Point Cloud.
    *   Uses `@react-three/fiber` and `@react-three/drei` to bind the WebGL canvas natively into the React rendering tree.
