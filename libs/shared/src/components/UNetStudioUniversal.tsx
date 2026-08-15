import React, { useState } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TouchableOpacity, 
  ScrollView, 
  ActivityIndicator, 
  Image, 
  Platform 
} from 'react-native';
import { useUNetStudio } from '../hooks/useUNetStudio';

interface UNetStudioUniversalProps {
  baseUrl?: string;
}

export function UNetStudioUniversal({ baseUrl = 'http://localhost:8000' }: UNetStudioUniversalProps) {
  const {
    config,
    summary,
    cudaStatus,
    weightsStatus,
    benchmarkResult,
    benchmarking,
    updateConfig,
    runBenchmark,
    fetchConfig,
  } = useUNetStudio(baseUrl);

  const [activeTab, setActiveTab] = useState<'architecture' | 'benchmark' | 'inference'>('architecture');
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [inferenceMeta, setInferenceMeta] = useState<any>(null);

  const handleUploadWeb = async () => {
    if (Platform.OS !== 'web') {
      alert("Custom weights upload is currently only supported on the Web browser client.");
      return;
    }
    
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.pt,.pth,.bin';
    input.onchange = async (e: any) => {
      const file = e.target.files?.[0];
      if (!file) return;
      
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const res = await fetch(`${baseUrl}/api/unet/weights/upload`, {
          method: 'POST',
          body: formData
        });
        if (res.ok) {
          alert("Weights uploaded successfully!");
          fetchConfig();
        } else {
          const data = await res.json();
          alert(`Error uploading weights: ${data.detail}`);
        }
      } catch (err) {
        console.error(err);
      }
    };
    input.click();
  };

  const handleExportWeights = () => {
    if (Platform.OS === 'web') {
      window.open(`${baseUrl}/api/unet/weights/export`, '_blank');
    } else {
      alert(`Export URL: ${baseUrl}/api/unet/weights/export`);
    }
  };

  const handleSelectImageWeb = () => {
    if (Platform.OS !== 'web') {
      alert("Static file upload is supported on Web. For mobile, use a local camera capture/stream.");
      return;
    }

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = async (e: any) => {
      const file = e.target.files?.[0];
      if (!file) return;
      
      setImagePreview(URL.createObjectURL(file));
      setResultImage(null);
      setPredicting(true);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch(`${baseUrl}/api/unet/predict`, {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (res.ok) {
          setResultImage(data.overlay_base64);
          setInferenceMeta(data);
        } else {
          alert(data.detail || 'Segmentation error');
        }
      } catch (err) {
        console.error(err);
      } finally {
        setPredicting(false);
      }
    };
    input.click();
  };

  return (
    <View style={styles.container}>
      <View style={styles.headerCard}>
        <View style={styles.headerLeft}>
          <Text style={styles.headerTitle}>🧠 U-Net Universal Studio</Text>
          <Text style={styles.headerSubtitle}>
            Configure layers, benchmark execution speed, and run custom segmentation tasks.
          </Text>
        </View>
        <View style={styles.hardwareBadge}>
          <Text style={styles.badgeLabel}>⚡ Hardware Runtime</Text>
          <Text style={styles.badgeText}>{cudaStatus?.device || 'CPU Fallback'}</Text>
        </View>
      </View>

      <View style={styles.tabBar}>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'architecture' && styles.activeTab]}
          onPress={() => setActiveTab('architecture')}
        >
          <Text style={[styles.tabText, activeTab === 'architecture' && styles.activeTabText]}>⚙️ Architecture</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'benchmark' && styles.activeTab]}
          onPress={() => setActiveTab('benchmark')}
        >
          <Text style={[styles.tabText, activeTab === 'benchmark' && styles.activeTabText]}>📊 CUDA Bench</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tab, activeTab === 'inference' && styles.activeTab]}
          onPress={() => setActiveTab('inference')}
        >
          <Text style={[styles.tabText, activeTab === 'inference' && styles.activeTabText]}>🔍 Inference</Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.content}>
        {activeTab === 'architecture' && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Model Parameters</Text>
            
            <View style={styles.controlRow}>
              <Text style={styles.controlLabel}>Network Depth:</Text>
              <View style={styles.segmentedControl}>
                {[2, 3, 4, 5].map((d) => (
                  <TouchableOpacity
                    key={d}
                    style={[styles.segmentButton, config.depth === d && styles.activeSegment]}
                    onPress={() => updateConfig({ ...config, depth: d })}
                  >
                    <Text style={[styles.segmentText, config.depth === d && styles.activeSegmentText]}>
                      {d}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            <View style={styles.controlRow}>
              <Text style={styles.controlLabel}>Initial Features:</Text>
              <View style={styles.segmentedControl}>
                {[16, 32, 64].map((feats) => (
                  <TouchableOpacity
                    key={feats}
                    style={[styles.segmentButton, config.init_features === feats && styles.activeSegment]}
                    onPress={() => updateConfig({ ...config, init_features: feats })}
                  >
                    <Text style={[styles.segmentText, config.init_features === feats && styles.activeSegmentText]}>
                      {feats}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>

            <View style={styles.toggleContainer}>
              <TouchableOpacity 
                style={styles.toggleRow}
                onPress={() => updateConfig({ ...config, use_attention: !config.use_attention })}
              >
                <Text style={styles.toggleLabel}>Attention Gates</Text>
                <View style={[styles.switch, config.use_attention && styles.switchOn]}>
                  <View style={[styles.switchThumb, config.use_attention && styles.switchThumbOn]} />
                </View>
              </TouchableOpacity>

              <TouchableOpacity 
                style={styles.toggleRow}
                onPress={() => updateConfig({ ...config, use_cuda_kernel: !config.use_cuda_kernel })}
              >
                <Text style={styles.toggleLabel}>CUDA Fused Activation</Text>
                <View style={[styles.switch, config.use_cuda_kernel && styles.switchOn]}>
                  <View style={[styles.switchThumb, config.use_cuda_kernel && styles.switchThumbOn]} />
                </View>
              </TouchableOpacity>
            </View>

            <View style={styles.weightsSection}>
              <Text style={styles.weightsStatusLabel}>Weights Checkpoint:</Text>
              <Text style={styles.weightsStatusText}>✅ {weightsStatus}</Text>
              
              <View style={styles.buttonRow}>
                <TouchableOpacity style={styles.actionButton} onPress={handleUploadWeb}>
                  <Text style={styles.actionButtonText}>📤 Upload Weights</Text>
                </TouchableOpacity>
                <TouchableOpacity style={[styles.actionButton, styles.secondaryButton]} onPress={handleExportWeights}>
                  <Text style={styles.secondaryButtonText}>📥 Export Checkpoint</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        )}

        {activeTab === 'benchmark' && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>CUDA Kernel Acceleration Speedup</Text>
            <Text style={styles.cardSubtitle}>
              Measures execution speedup of custom CUDA kernels vs PyTorch CPU baselines.
            </Text>

            <TouchableOpacity 
              style={styles.primaryButton} 
              onPress={runBenchmark} 
              disabled={benchmarking}
            >
              {benchmarking ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.primaryButtonText}>📊 Run Execution Benchmark</Text>
              )}
            </TouchableOpacity>

            {benchmarkResult && (
              <View style={styles.statsContainer}>
                <View style={styles.statBox}>
                  <Text style={styles.statLabel}>CPU Latency</Text>
                  <Text style={styles.statValue}>{benchmarkResult.cpu_latency_ms} ms</Text>
                </View>
                <View style={styles.statBox}>
                  <Text style={styles.statLabel}>CUDA Latency</Text>
                  <Text style={styles.statValue}>{benchmarkResult.cuda_latency_ms} ms</Text>
                </View>
                <View style={[styles.statBox, styles.highlightStatBox]}>
                  <Text style={styles.highlightStatLabel}>Speedup</Text>
                  <Text style={styles.highlightStatValue}>{benchmarkResult.speedup}x</Text>
                </View>
              </View>
            )}
          </View>
        )}

        {activeTab === 'inference' && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Image Segmentation Inference</Text>
            
            <TouchableOpacity style={styles.primaryButton} onPress={handleSelectImageWeb} disabled={predicting}>
              <Text style={styles.primaryButtonText}>🖼️ Select Test Image</Text>
            </TouchableOpacity>

            {predicting && <ActivityIndicator style={styles.spinner} color="#38bdf8" />}

            {imagePreview && (
              <View style={styles.imagePreviewContainer}>
                <Text style={styles.sectionHeader}>Input Preview:</Text>
                <Image source={{ uri: imagePreview }} style={styles.previewImage} resizeMode="contain" />
              </View>
            )}

            {resultImage && (
              <View style={styles.imagePreviewContainer}>
                <Text style={styles.sectionHeader}>U-Net Output Mask Overlay:</Text>
                <Image source={{ uri: resultImage }} style={styles.previewImage} resizeMode="contain" />
                
                {inferenceMeta && (
                  <View style={styles.metaBox}>
                    <Text style={styles.metaText}>Device: {inferenceMeta.device}</Text>
                    <Text style={styles.metaText}>Latency: {inferenceMeta.latency_ms} ms</Text>
                  </View>
                )}
              </View>
            )}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#060911',
  },
  headerCard: {
    backgroundColor: '#0f172a',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#1e293b',
    marginVertical: 8,
    gap: 12,
  },
  headerLeft: {
    flex: 1,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#38bdf8',
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#94a3b8',
    marginTop: 4,
  },
  hardwareBadge: {
    backgroundColor: '#1e293b',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 10,
    alignSelf: 'flex-start',
    marginTop: 4,
  },
  badgeLabel: {
    fontSize: 10,
    color: '#94a3b8',
    textTransform: 'uppercase',
  },
  badgeText: {
    fontSize: 12,
    color: '#e2e8f0',
    fontWeight: 'bold',
    marginTop: 2,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#0f172a',
    borderRadius: 12,
    padding: 4,
    marginBottom: 16,
  },
  tab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
  },
  activeTab: {
    backgroundColor: '#38bdf820',
    borderWidth: 1,
    borderColor: '#38bdf850',
  },
  tabText: {
    fontSize: 12,
    color: '#94a3b8',
    fontWeight: '500',
  },
  activeTabText: {
    color: '#38bdf8',
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
  },
  card: {
    backgroundColor: '#0f172a',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: '#1e293b',
    marginBottom: 20,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#f8fafc',
    marginBottom: 8,
  },
  cardSubtitle: {
    fontSize: 12,
    color: '#94a3b8',
    marginBottom: 16,
  },
  controlRow: {
    marginBottom: 16,
  },
  controlLabel: {
    fontSize: 13,
    color: '#e2e8f0',
    marginBottom: 8,
  },
  segmentedControl: {
    flexDirection: 'row',
    backgroundColor: '#020617',
    borderRadius: 10,
    padding: 2,
  },
  segmentButton: {
    flex: 1,
    paddingVertical: 8,
    alignItems: 'center',
    borderRadius: 8,
  },
  activeSegment: {
    backgroundColor: '#1e293b',
  },
  segmentText: {
    fontSize: 12,
    color: '#94a3b8',
  },
  activeSegmentText: {
    color: '#38bdf8',
    fontWeight: 'bold',
  },
  toggleContainer: {
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: '#1e293b',
    paddingVertical: 8,
    marginVertical: 8,
    gap: 12,
  },
  toggleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 4,
  },
  toggleLabel: {
    fontSize: 13,
    color: '#e2e8f0',
  },
  switch: {
    width: 44,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#1e293b',
    padding: 2,
  },
  switchOn: {
    backgroundColor: '#0284c7',
  },
  switchThumb: {
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: '#fff',
  },
  switchThumbOn: {
    alignSelf: 'flex-end',
  },
  weightsSection: {
    marginTop: 16,
    gap: 8,
  },
  weightsStatusLabel: {
    fontSize: 12,
    color: '#94a3b8',
  },
  weightsStatusText: {
    fontSize: 13,
    color: '#38bdf8',
    fontWeight: 'bold',
    backgroundColor: '#020617',
    padding: 10,
    borderRadius: 10,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#0284c7',
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: 'center',
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  secondaryButton: {
    backgroundColor: '#1e293b',
  },
  secondaryButtonText: {
    color: '#e2e8f0',
    fontSize: 12,
    fontWeight: 'bold',
  },
  primaryButton: {
    backgroundColor: '#0284c7',
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    marginVertical: 12,
  },
  primaryButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 13,
  },
  statsContainer: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 12,
  },
  statBox: {
    flex: 1,
    backgroundColor: '#020617',
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#1e293b',
  },
  highlightStatBox: {
    borderColor: '#eab30850',
    backgroundColor: '#eab30810',
  },
  statLabel: {
    fontSize: 10,
    color: '#94a3b8',
    textTransform: 'uppercase',
  },
  statValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#f8fafc',
    marginTop: 4,
  },
  highlightStatLabel: {
    fontSize: 10,
    color: '#eab308',
    textTransform: 'uppercase',
  },
  highlightStatValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#eab308',
    marginTop: 4,
  },
  spinner: {
    marginVertical: 16,
  },
  imagePreviewContainer: {
    marginTop: 16,
    alignItems: 'center',
  },
  sectionHeader: {
    fontSize: 13,
    color: '#94a3b8',
    alignSelf: 'flex-start',
    marginBottom: 8,
  },
  previewImage: {
    width: '100%',
    height: 200,
    borderRadius: 12,
    backgroundColor: '#020617',
  },
  metaBox: {
    marginTop: 8,
    backgroundColor: '#020617',
    padding: 8,
    borderRadius: 8,
    alignSelf: 'stretch',
  },
  metaText: {
    fontSize: 11,
    color: '#94a3b8',
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
  },
});
