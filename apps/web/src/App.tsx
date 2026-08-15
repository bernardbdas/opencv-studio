import React, { useState, useEffect, useCallback, useRef, Component, ErrorInfo, ReactNode } from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import { Layers, Eye, Activity, Sparkles, AlertCircle, RefreshCw, Info, Sun, Moon, Palette, Cpu, BookOpen, Box, Server } from 'lucide-react';
import { UNetStudio } from './components/UNetStudio';
import { VisionStudio } from './components/VisionStudio';
import { YoloStudio } from './components/YoloStudio';
import { ARLaunchpad } from './components/ARLaunchpad';
import { AudioTextStudio } from './components/AudioTextStudio';
import { KnowledgeHub } from './components/KnowledgeHub';
import { DepthLab } from './components/DepthLab';
import { ModelRepository } from './components/ModelRepository';
import { ModelInspectorModal } from './components/ModelInspectorModal';
import { OpenCVStudioClient } from '@opencv-studio/shared';

const client = new OpenCVStudioClient();

// ── Tab Registry ────────────────────────────────────────────────────────────

const DEFAULT_TAB_ORDER = ['vision', 'yolo', 'ar', '3dlab', 'nlp', 'unet', 'models', 'kb'];

const TAB_ICONS: Record<string, React.ReactNode> = {
  unet: <Layers className="w-3.5 h-3.5 shrink-0" />,
  vision: <Eye className="w-3.5 h-3.5 shrink-0" />,
  yolo: <Cpu className="w-3.5 h-3.5 shrink-0" />,
  ar: <Sparkles className="w-3.5 h-3.5 shrink-0" />,
  nlp: <Sparkles className="w-3.5 h-3.5 shrink-0" />,
  models: <Server className="w-3.5 h-3.5 shrink-0" />,
  kb: <BookOpen className="w-3.5 h-3.5 shrink-0" />,
  '3dlab': <Box className="w-3.5 h-3.5 shrink-0" />,
};

const TAB_LABELS: Record<string, string> = {
  unet: 'U-Net & CUDA',
  vision: 'Mediapipe Lab',
  yolo: 'YOLO Studio',
  ar: 'AR Launchpad',
  nlp: 'GenAI',
  models: 'Models',
  kb: 'Knowledge',
  '3dlab': '3D Depth Lab',
};

const TAB_CONTENT: Record<string, React.ReactNode> = {
  unet: <UNetStudio />,
  vision: <VisionStudio />,
  yolo: <YoloStudio />,
  ar: <ARLaunchpad />,
  nlp: <AudioTextStudio />,
  models: <ModelRepository />,
  kb: <KnowledgeHub />,
  '3dlab': <DepthLab />,
};

// ── Persistence ─────────────────────────────────────────────────────────────

const STORAGE_KEY = 'opencv-studio-tab-order';

function loadTabOrder(): string[] {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      const parsed = JSON.parse(saved) as string[];
      if (
        parsed.length === DEFAULT_TAB_ORDER.length &&
        DEFAULT_TAB_ORDER.every(id => parsed.includes(id))
      ) {
        return parsed;
      }
    }
  } catch { /* ignore */ }
  return [...DEFAULT_TAB_ORDER];
}

function saveTabOrder(order: string[]) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(order)); } catch { /* ignore */ }
}

// ── Error Boundary ──────────────────────────────────────────────────────────

interface ErrorBoundaryProps { children: ReactNode; }
interface ErrorBoundaryState { hasError: boolean; error: Error | null; }

class StudioErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  public state: ErrorBoundaryState = { hasError: false, error: null };

  public static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Studio Tab Render Error:', error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="glass-panel p-8 rounded-2xl border border-rose-500/30 bg-rose-950/20 text-center space-y-4">
          <AlertCircle className="w-10 h-10 text-rose-400 mx-auto" />
          <h3 className="text-lg font-bold text-slate-100">Tab Render Exception</h3>
          <p className="text-xs text-rose-300 font-mono">{this.state.error?.message}</p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="px-4 py-2 bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-200 text-xs rounded-xl transition inline-flex items-center space-x-2"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Reload Tab</span>
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── App Component ───────────────────────────────────────────────────────────

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('vision');
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [inspectorModelId, setInspectorModelId] = useState('unet_custom');
  const [monitorOpen, setMonitorOpen] = useState(false);

  // Themes & Accent configurations
  const [themeMode, setThemeMode] = useState<'dark' | 'light'>('dark');
  const [glassTheme, setGlassTheme] = useState<'glass' | 'solid'>('glass');
  const [accentColor, setAccentColor] = useState<'cyan' | 'violet' | 'emerald' | 'rose'>('cyan');

  // Draggable tab order
  const [tabOrder, setTabOrder] = useState<string[]>(loadTabOrder);
  const [draggedTab, setDraggedTab] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);

  const checkHealth = async () => {
    try {
      const res = await client.getHealth();
      setBackendOnline(res?.status === 'online');
    } catch {
      setBackendOnline(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const openInspector = (modelId = 'unet_custom') => {
    setInspectorModelId(modelId);
    setInspectorOpen(true);
  };

  // ── Drag & Drop Handlers ────────────────────────────────────────

  const handleDragStart = useCallback((e: React.DragEvent, tabId: string) => {
    setDraggedTab(tabId);
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', tabId);
    // Fade the dragged element
    if (e.currentTarget instanceof HTMLElement) {
      requestAnimationFrame(() => {
        (e.currentTarget as HTMLElement).style.opacity = '0.35';
      });
    }
  }, []);

  const handleDragEnd = useCallback((e: React.DragEvent) => {
    setDraggedTab(null);
    setDropTarget(null);
    if (e.currentTarget instanceof HTMLElement) {
      e.currentTarget.style.opacity = '1';
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent, tabId: string) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (tabId !== draggedTab) {
      setDropTarget(tabId);
    }
  }, [draggedTab]);

  const handleDragLeave = useCallback(() => {
    setDropTarget(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent, targetTabId: string) => {
    e.preventDefault();
    const sourceTabId = e.dataTransfer.getData('text/plain');
    if (!sourceTabId || sourceTabId === targetTabId) {
      setDraggedTab(null);
      setDropTarget(null);
      return;
    }

    setTabOrder(prev => {
      const newOrder = [...prev];
      const fromIdx = newOrder.indexOf(sourceTabId);
      const toIdx = newOrder.indexOf(targetTabId);
      if (fromIdx < 0 || toIdx < 0) return prev;

      newOrder.splice(fromIdx, 1);
      newOrder.splice(toIdx, 0, sourceTabId);

      saveTabOrder(newOrder);
      return newOrder;
    });

    setDraggedTab(null);
    setDropTarget(null);
  }, []);

  return (
    <div className={`min-h-screen flex flex-col bg-[#060911] transition-colors duration-300 selection:bg-cyan-500 selection:text-black ${themeMode === 'light' ? 'light-mode' : ''} ${glassTheme === 'solid' ? 'theme-solid' : ''} accent-${accentColor}`}>
      {/* Model Specs Modal */}
      <ModelInspectorModal
        open={inspectorOpen}
        onOpenChange={setInspectorOpen}
        selectedModelId={inspectorModelId}
      />

      {/* Top Header */}
      <header className="border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-xl sticky top-0 z-50 transition-all duration-300">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Eye className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="font-extrabold text-lg tracking-tight flex items-center space-x-2.5">
                <span className="text-gradient">OpenCV Studio</span>
                <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded-full bg-slate-900/80 border border-slate-800/60">
                  <span className="text-[10px] uppercase font-mono text-slate-400">v1.0</span>
                  <div
                    className={`w-1.5 h-1.5 rounded-full transition-all duration-500 ${backendOnline === true
                        ? 'bg-emerald-500 shadow-[0_0_8px_#10b981]'
                        : backendOnline === false
                          ? 'bg-rose-500 shadow-[0_0_8px_#f43f5e]'
                          : 'bg-amber-500 animate-pulse'
                      }`}
                    title={
                      backendOnline === true
                        ? 'Backend Connected'
                        : backendOnline === false
                          ? 'Backend Offline'
                          : 'Connecting to Backend...'
                    }
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            {/* Custom Theme & Accent Controller Card */}
            <div className="flex items-center space-x-2.5 px-2.5 py-1.5 rounded-xl bg-slate-900/60 border border-slate-805/80 text-xs shadow-sm">
              {/* Accents Selector */}
              <div className="flex items-center space-x-1.5 pr-2.5 border-r border-slate-800/85">
                {(['cyan', 'violet', 'emerald', 'rose'] as const).map((color) => {
                  const colorsHex = { cyan: '#06b6d4', violet: '#8b5cf6', emerald: '#10b981', rose: '#f43f5e' };
                  const isActive = accentColor === color;
                  return (
                    <button
                      key={color}
                      onClick={() => setAccentColor(color)}
                      className={`w-3.5 h-3.5 rounded-full border border-white/20 transition-all ${isActive
                          ? 'ring-2 ring-offset-2 ring-offset-slate-950 ring-cyan-500 scale-110'
                          : 'opacity-40 hover:opacity-100'
                        }`}
                      style={{ backgroundColor: colorsHex[color] }}
                      title={`Switch accent to ${color}`}
                    />
                  );
                })}
              </div>

              {/* Light / Dark Mode toggle */}
              <button
                onClick={() => setThemeMode(themeMode === 'dark' ? 'light' : 'dark')}
                className="p-1 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition flex items-center justify-center"
                title={themeMode === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
              >
                {themeMode === 'dark' ? (
                  <Sun className="w-3.5 h-3.5 text-amber-400" />
                ) : (
                  <Moon className="w-3.5 h-3.5 text-indigo-400" />
                )}
              </button>
            </div>

              {/* Realtime Performance Monitor Button */}
              <button
                onClick={() => setMonitorOpen(true)}
                className="flex items-center space-x-1.5 px-3 py-1.5 bg-indigo-500/10 hover:bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 rounded-xl text-xs font-semibold transition"
              >
                <Activity className="w-3.5 h-3.5 text-indigo-400" />
                <span>Monitor</span>
              </button>

              {/* Model Specs Button */}
              <button
                onClick={() => openInspector('unet_custom')}
                className="flex items-center space-x-1.5 px-3 py-1.5 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 text-cyan-300 rounded-xl text-xs font-semibold transition"
              >
                <Info className="w-3.5 h-3.5 text-cyan-400" />
                <span>Specs</span>
              </button>
            </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8">
        <Tabs.Root value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <Tabs.List className="custom-tab-list">
            {tabOrder.map((tabId) => (
              <Tabs.Trigger
                key={tabId}
                value={tabId}
                className={`custom-tab-trigger transition-all duration-150 ${draggedTab === tabId ? 'opacity-30 scale-[0.96]' : ''
                  } ${dropTarget === tabId && draggedTab !== tabId
                    ? 'ring-2 ring-cyan-400/50 bg-cyan-500/10 scale-[1.03]'
                    : ''
                  }`}
                draggable
                onDragStart={(e) => handleDragStart(e, tabId)}
                onDragEnd={handleDragEnd}
                onDragOver={(e) => handleDragOver(e, tabId)}
                onDragLeave={handleDragLeave}
                onDrop={(e) => handleDrop(e, tabId)}
              >
                {TAB_ICONS[tabId]}
                <span>{TAB_LABELS[tabId]}</span>
              </Tabs.Trigger>
            ))}
          </Tabs.List>

          <StudioErrorBoundary>
            {tabOrder.map((tabId) => (
              <Tabs.Content key={tabId} value={tabId}>
                {TAB_CONTENT[tabId]}
              </Tabs.Content>
            ))}
          </StudioErrorBoundary>
        </Tabs.Root>
      </main>

      {/* Real-time Performance Monitor Modal */}
      {monitorOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-in fade-in duration-200">
          <div className="bg-[#0b0f19] border border-slate-800 rounded-3xl w-full max-w-4xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
            {/* Header */}
            <div className="p-5 border-b border-slate-900 flex items-center justify-between bg-slate-950/30">
              <div className="flex items-center space-x-3">
                <Activity className="w-5 h-5 text-indigo-400" />
                <div>
                  <h3 className="text-base font-bold text-white">System Performance Monitor</h3>
                  <p className="text-[10px] text-slate-500 font-mono">Real-time XY Charting Engine</p>
                </div>
              </div>
              <button
                onClick={() => setMonitorOpen(false)}
                className="px-3 py-1.5 bg-slate-900 hover:bg-slate-850 text-slate-400 hover:text-white rounded-xl text-xs font-semibold border border-slate-800 transition"
              >
                Close
              </button>
            </div>
            
            {/* Chart Area */}
            <div className="flex-1 bg-[#070a13] p-4 flex flex-col items-center justify-center relative min-h-[450px]">
              {backendOnline === false ? (
                <div className="text-center space-y-2">
                  <AlertCircle className="w-8 h-8 text-rose-500 mx-auto animate-bounce" />
                  <p className="text-xs text-slate-400 font-medium">Performance monitor offline. Connect to FastAPI backend.</p>
                </div>
              ) : (
                <iframe
                  src="http://localhost:8000/api/performance/chart"
                  className="w-full h-full border-0 rounded-2xl min-h-[450px]"
                  title="Performance Monitor Chart"
                />
              )}
            </div>
            
            {/* Actions Footer */}
            <div className="p-4 border-t border-slate-900 bg-slate-950/20 flex justify-between items-center text-[10px] text-slate-500 font-mono">
              <span>Data updates on model execution</span>
              <div className="flex items-center space-x-2">
                <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-ping" />
                <span>Live telemetry active</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-500 font-mono">
        OpenCV Studio — Computer Vision & Deep Learning Workspace
      </footer>
    </div>
  );
};

export default App;
