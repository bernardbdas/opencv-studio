export interface UNetConfig {
  in_channels: number;
  out_channels: number;
  depth: number;
  init_features: number;
  use_attention: boolean;
  use_cuda_kernel: boolean;
  custom_weights_loaded?: boolean;
  weights_filename?: string;
}

export interface UNetSummary {
  in_channels: number;
  out_channels: number;
  depth: number;
  init_features: number;
  use_attention: boolean;
  use_cuda_kernel: boolean;
  total_parameters: number;
  trainable_parameters: number;
}

export interface CudaStatus {
  available: boolean;
  device: string;
}

export interface CudaBenchmarkResult {
  cuda_available: boolean;
  device_name: string;
  tensor_shape: number[];
  total_elements: number;
  cpu_latency_ms: number;
  cuda_latency_ms: number;
  speedup: number;
}

export interface VisionFilterParams {
  filter_type: 'canny' | 'threshold' | 'harris' | 'blur';
  param1: number;
  param2: number;
}

export interface DemoAsset {
  key: string;
  filename: string;
  title: string;
  description: string;
  category: string;
  available: boolean;
}

