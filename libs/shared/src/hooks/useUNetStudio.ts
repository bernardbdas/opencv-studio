import { useState, useEffect, useCallback } from 'react';
import { OpenCVStudioClient } from '../api/client';
import { UNetConfig, UNetSummary, CudaStatus, CudaBenchmarkResult } from '../types';

export function useUNetStudio(baseUrl = 'http://localhost:8000') {
  const [client] = useState(() => new OpenCVStudioClient(baseUrl));
  const [config, setConfig] = useState<UNetConfig>({
    in_channels: 3,
    out_channels: 1,
    depth: 4,
    init_features: 32,
    use_attention: true,
    use_cuda_kernel: true,
  });

  const [summary, setSummary] = useState<UNetSummary | null>(null);
  const [cudaStatus, setCudaStatus] = useState<CudaStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [weightsStatus, setWeightsStatus] = useState<string>('Default initialized weights active');
  const [benchmarkResult, setBenchmarkResult] = useState<CudaBenchmarkResult | null>(null);
  const [benchmarking, setBenchmarking] = useState(false);

  const fetchConfig = useCallback(async () => {
    try {
      const data = await client.getUNetConfig();
      setConfig(data.config);
      setSummary(data.summary);
      setCudaStatus({
        available: data.cuda_available,
        device: data.device_name,
      });
      if (data.config.custom_weights_loaded) {
        setWeightsStatus(`Custom weights loaded: ${data.config.weights_filename}`);
      }
    } catch (err) {
      console.error('Failed to load U-Net config', err);
    }
  }, [client]);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  const updateConfig = async (newConfig: UNetConfig) => {
    setLoading(true);
    try {
      const data = await client.updateUNetConfig(newConfig);
      setConfig(data.config);
      setSummary(data.summary);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const runBenchmark = async () => {
    setBenchmarking(true);
    try {
      const data = await client.runCudaBenchmark();
      setBenchmarkResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setBenchmarking(false);
    }
  };

  return {
    config,
    summary,
    cudaStatus,
    loading,
    weightsStatus,
    benchmarkResult,
    benchmarking,
    fetchConfig,
    updateConfig,
    runBenchmark,
    client,
  };
}
