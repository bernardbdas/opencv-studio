import React, { useState, useEffect } from 'react';
import { OpenCVStudioClient } from '@opencv-studio/shared';
import {
  Cpu, HardDrive, Search, Download, Trash2, CheckCircle2, AlertCircle,
  RefreshCw, Info, Server
} from 'lucide-react';
import { ModelInspectorModal } from './ModelInspectorModal';

const api = new OpenCVStudioClient();

interface ModelItem {
  id: string;
  name: string;
  filename: string;
  category: string;
  description: string;
  cached: boolean;
  size_mb: number;
  download_url: string;
  status: 'ready' | 'missing' | 'downloading' | 'error';
}

export function ModelRepository() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Inspection modal state
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspectModelId, setInspectModelId] = useState<string | undefined>(undefined);

  // Load models from API
  const loadModels = async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const data = await api.getModels();
      setModels(data);
      setError(null);
    } catch (err: any) {
      setError('Failed to fetch models registry: ' + err.message);
    } finally {
      if (!silent) setLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  // Poll downloading models
  useEffect(() => {
    const hasDownloading = models.some(m => m.status === 'downloading');
    if (!hasDownloading) return;

    const interval = setInterval(() => {
      loadModels(true);
    }, 2000);

    return () => clearInterval(interval);
  }, [models]);

  // Handle Download Trigger
  const handleDownload = async (modelId: string) => {
    try {
      setModels(prev =>
        prev.map(m => (m.id === modelId ? { ...m, status: 'downloading' as const } : m))
      );
      await api.downloadModel(modelId);
      loadModels(true);
    } catch (err: any) {
      setError(`Failed to download model ${modelId}: ` + err.message);
      loadModels(true);
    }
  };

  // Handle Delete Trigger
  const handleDelete = async (modelId: string) => {
    if (!window.confirm('Are you sure you want to delete this local model cache?')) return;
    try {
      await api.deleteModel(modelId);
      loadModels(true);
    } catch (err: any) {
      setError(`Failed to delete model ${modelId}: ` + err.message);
    }
  };

  // Open inspector modal
  const handleInspect = (modelId: string) => {
    let specId = modelId;
    if (modelId.startsWith('mp_')) {
      specId = modelId.substring(3);
    } else if (modelId.startsWith('depth_')) {
      // Map depth keys to spec IDs
      const depthSpecMap: Record<string, string> = {
        depth_midas_small: 'selfie_segmenter',
        depth_dpt_hybrid: 'unet_custom',
        depth_dpt_large: 'unet_custom'
      };
      specId = depthSpecMap[modelId] || 'unet_custom';
    } else if (modelId.startsWith('yolo_')) {
      specId = 'object_detector';
    }
    
    setInspectModelId(specId);
    setInspectorOpen(true);
  };

  // Calculations
  const totalCached = models.filter(m => m.cached).length;
  const totalSpace = models.reduce((acc, m) => acc + (m.cached ? m.size_mb : 0), 0);
  const categories = ['All', 'MediaPipe', 'YOLO', 'Depth Lab'];

  const filteredModels = models.filter(model => {
    const matchesSearch =
      model.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
      model.description.toLowerCase().includes(searchQuery.toLowerCase());
      
    const matchesCategory =
      selectedCategory === 'All' || model.category === selectedCategory;

    return matchesSearch && matchesCategory;
  });

  return (
    <div className="space-y-6">
      {/* Header Panel */}
      <div className="glass-panel p-6 rounded-2xl flex items-center justify-between flex-wrap gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight flex items-center space-x-3">
            <Server className="w-7 h-7 text-cyan-400" />
            <span>Integrated Model Repository</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1 leading-relaxed">
            Manage local neural weights, check cached states, monitor background downloads, and inspect model parameters.
          </p>
        </div>
        <button
          onClick={() => loadModels()}
          disabled={loading}
          className="flex items-center space-x-1.5 px-3 py-1.5 bg-slate-900 border border-slate-800 text-xs text-slate-350 hover:text-white rounded-xl transition cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Sync State</span>
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="glass-panel p-4 rounded-xl flex items-center space-x-4">
          <div className="w-10 h-10 bg-cyan-500/10 border border-cyan-500/25 rounded-xl flex items-center justify-center">
            <CheckCircle2 className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <div className="text-[10px] font-mono uppercase text-slate-500">Cached Models</div>
            <div className="text-xl font-bold text-white font-mono">{totalCached} / {models.length}</div>
          </div>
        </div>

        <div className="glass-panel p-4 rounded-xl flex items-center space-x-4">
          <div className="w-10 h-10 bg-violet-500/10 border border-violet-500/25 rounded-xl flex items-center justify-center">
            <HardDrive className="w-5 h-5 text-violet-400" />
          </div>
          <div>
            <div className="text-[10px] font-mono uppercase text-slate-500">Local Disk Space</div>
            <div className="text-xl font-bold text-white font-mono">{totalSpace.toFixed(1)} MB</div>
          </div>
        </div>

        <div className="glass-panel p-4 rounded-xl flex items-center space-x-4">
          <div className="w-10 h-10 bg-amber-500/10 border border-amber-500/25 rounded-xl flex items-center justify-center">
            <Cpu className="w-5 h-5 text-amber-400" />
          </div>
          <div>
            <div className="text-[10px] font-mono uppercase text-slate-500">Target Backend</div>
            <div className="text-sm font-bold text-white leading-tight mt-1">CPU / CUDA Fused</div>
          </div>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="glass-panel p-4 rounded-2xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex space-x-1.5 bg-slate-950/80 p-1 rounded-xl border border-slate-900 self-start md:self-auto">
          {categories.map(cat => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`py-1.5 px-3 rounded-lg text-xs font-bold transition cursor-pointer ${
                selectedCategory === cat
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/35'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="relative max-w-md w-full">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search models by name or file..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-slate-950/80 border border-slate-900 focus:border-slate-800 text-xs text-white rounded-xl focus:outline-none placeholder-slate-500 font-mono transition"
          />
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-xl text-xs flex items-center space-x-3">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Models Grid */}
      {loading ? (
        <div className="glass-panel p-16 text-center rounded-2xl space-y-3">
          <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin mx-auto" />
          <div className="text-xs text-slate-500 font-mono">Loading models registry...</div>
        </div>
      ) : filteredModels.length === 0 ? (
        <div className="glass-panel p-16 text-center rounded-2xl space-y-2">
          <Server className="w-8 h-8 text-slate-700 mx-auto" />
          <div className="text-xs text-slate-500 font-mono">No matching models found.</div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {filteredModels.map(model => {
            const statusColors = {
              ready: { text: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', label: 'Cached' },
              missing: { text: 'text-slate-500', bg: 'bg-slate-900/60', border: 'border-slate-800', label: 'Missing' },
              downloading: { text: 'text-cyan-400 animate-pulse', bg: 'bg-cyan-500/15', border: 'border-cyan-500/25', label: 'Downloading...' },
              error: { text: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/20', label: 'Failed' }
            };
            const c = statusColors[model.status] || statusColors.missing;

            return (
              <div key={model.id} className="glass-panel rounded-2xl overflow-hidden flex flex-col justify-between hover:border-slate-800 transition-all duration-300">
                {/* Card Header */}
                <div className="p-5 space-y-3 flex-grow">
                  <div className="flex items-center justify-between">
                    <span className="text-[9px] font-mono font-bold uppercase tracking-wider bg-slate-950 px-2 py-0.5 border border-slate-900 rounded text-slate-500">
                      {model.category}
                    </span>
                    <span className={`text-[8px] font-mono font-bold px-2 py-0.5 rounded border ${c.bg} ${c.text} ${c.border}`}>
                      {c.label}
                    </span>
                  </div>

                  <div>
                    <h4 className="text-xs font-bold text-white font-mono leading-snug">{model.name}</h4>
                    <p className="text-[10px] text-slate-400 mt-1 font-mono leading-relaxed truncate" title={model.filename}>
                      {model.filename}
                    </p>
                  </div>

                  <p className="text-[10px] text-slate-400 leading-relaxed font-sans line-clamp-3">
                    {model.description}
                  </p>
                </div>

                {/* Card Footer */}
                <div className="px-5 py-4 bg-slate-950/40 border-t border-slate-850 flex items-center justify-between">
                  <span className="text-[10px] font-mono text-slate-400">
                    {model.cached ? `${model.size_mb.toFixed(1)} MB` : 'Not cached'}
                  </span>

                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleInspect(model.id)}
                      className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-white hover:border-slate-700 transition cursor-pointer"
                      title="Inspect Specs"
                    >
                      <Info className="w-3.5 h-3.5" />
                    </button>

                    {model.cached ? (
                      <button
                        onClick={() => handleDelete(model.id)}
                        className="p-1.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 hover:bg-rose-500/20 transition cursor-pointer"
                        title="Delete Cache"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    ) : (
                      <button
                        onClick={() => handleDownload(model.id)}
                        disabled={model.status === 'downloading'}
                        className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 hover:bg-cyan-500/20 disabled:opacity-40 disabled:cursor-not-allowed transition cursor-pointer"
                        title="Download Weights"
                      >
                        <Download className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Inspect Spec Dialog */}
      <ModelInspectorModal
        open={inspectorOpen}
        onOpenChange={setInspectorOpen}
        selectedModelId={inspectModelId}
      />
    </div>
  );
}
