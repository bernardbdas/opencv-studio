import { UNetConfig, UNetSummary, CudaBenchmarkResult } from '../types';

export class OpenCVStudioClient {
  public baseUrl: string;

  constructor(baseUrl?: string) {
    if (baseUrl) {
      this.baseUrl = baseUrl;
    } else if (typeof window !== 'undefined' && window.location && window.location.hostname) {
      this.baseUrl = `${window.location.protocol}//${window.location.hostname}:8000`;
    } else {
      this.baseUrl = 'http://localhost:8000';
    }
  }

  async getHealth() {
    const res = await fetch(`${this.baseUrl}/api/health`);
    return await res.json();
  }

  async getUNetConfig(): Promise<{ config: UNetConfig; summary: UNetSummary; cuda_available: boolean; device_name: string }> {
    const res = await fetch(`${this.baseUrl}/api/unet/config`);
    return await res.json();
  }

  async updateUNetConfig(config: Partial<UNetConfig>) {
    const res = await fetch(`${this.baseUrl}/api/unet/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    return await res.json();
  }

  async runCudaBenchmark(params = { batch_size: 1, channels: 64, height: 256, width: 256, iterations: 30 }): Promise<CudaBenchmarkResult> {
    const res = await fetch(`${this.baseUrl}/api/unet/benchmark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return await res.json();
  }

  async getDemoAssets() {
    const res = await fetch(`${this.baseUrl}/api/vision/demo-assets`);
    return await res.json();
  }

  async getDemoAsset(key: string): Promise<{ key: string; title: string; image_base64: string }> {
    const res = await fetch(`${this.baseUrl}/api/vision/demo-assets/${key}`);
    return await res.json();
  }

  async fetchWithTimeout(url: string, options: RequestInit = {}, timeoutMs = 15000) {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      });
      clearTimeout(id);
      return response;
    } catch (error) {
      clearTimeout(id);
      throw error;
    }
  }

  async detectPose(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/pose-detection${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectOCR(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/ocr-detection${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectFaceMesh(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/face-landmark${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async classifyImage(formData: FormData) {
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/classify`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async recognizeGesture(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/gesture${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectHolistic(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/holistic${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectObjects(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/object-detection${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async analyzeImage(formData: FormData): Promise<{ has_face: boolean; has_pose: boolean; has_hands: boolean }> {
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/analyze`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectFingerFrame(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/finger-frame${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectFaceFilter(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/face-filter${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectArucoProjection(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/aruco-projection${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectSelfieSegmentation(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/segmentation${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectPoseTrainer(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/pose-trainer${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectAirDraw(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/air-draw${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectFaceTryon(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/face-tryon${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async setInvisibilityBackground(formData: FormData) {
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/invisibility-cloak/background`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async runInvisibilityCloak(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/vision/invisibility-cloak${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async detectYoloObjects(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/yolo/detect${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async segmentYoloObjects(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/yolo/segment${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async estimateYoloPose(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/yolo/pose${query}`, {
      method: 'POST',
      body: formData,
    });
    return await res.json();
  }

  async getKnowledgeCatalog() {
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/knowledge`);
    return await res.json();
  }

  async getKnowledgeArticle(key: string) {
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/knowledge/${key}`);
    return await res.json();
  }

  // ── 3D Depth Lab ──────────────────────────────────────────────

  async estimateDepth(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/depth/estimate${query}`, {
      method: 'POST',
      body: formData,
    }, 60000);
    return await res.json();
  }

  async generatePointCloud(formData: FormData, params?: Record<string, any>) {
    const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/depth/pointcloud${query}`, {
      method: 'POST',
      body: formData,
    }, 60000);
    return await res.json();
  }

  async getDepthVideos() {
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/depth/videos`);
    return await res.json();
  }

  async downloadDepthVideo(videoId: string) {
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/depth/videos/${videoId}/download`, {
      method: 'POST',
    });
    return await res.json();
  }

  // ── Model Repository ──────────────────────────────────────────

  async getModels() {
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/models`);
    return await res.json();
  }

  async downloadModel(modelId: string) {
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/models/${modelId}/download`, {
      method: 'POST',
    });
    return await res.json();
  }

  async deleteModel(modelId: string) {
    const res = await this.fetchWithTimeout(`${this.baseUrl}/api/models/${modelId}`, {
      method: 'DELETE',
    });
    return await res.json();
  }
}

