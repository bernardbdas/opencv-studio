import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { OpenCVStudioClient } from '@opencv-studio/shared';
import {
  Box, Upload, Camera, CameraOff, Sliders, Download, AlertCircle,
  Maximize2, X, Image as ImageIcon, Cpu, RotateCcw, Layers, Eye, Folder,
  BarChart3, ToggleLeft, ToggleRight, Zap, Sparkles,
} from 'lucide-react';
import * as Slider from '@radix-ui/react-slider';
import * as THREE from 'three';
import { AnaglyphEffect } from 'three/addons/effects/AnaglyphEffect.js';

const client = new OpenCVStudioClient();

// ─── Types ──────────────────────────────────────────────────────────────────

interface Vertex {
  x: number; y: number; z: number; r: number; g: number; b: number;
}

interface EngineResult {
  depthHeatmap: string | null;
  sourceImage: string | null;
  speed: { preprocess: number; inference: number; postprocess: number } | null;
  e2eLatency: number;
  pointCloud: Vertex[];
  vertexCount: number;
  error: string | null;
  processing: boolean;
}

interface DemoAsset {
  key: string;
  title: string;
  category: string;
  description: string;
  image_base64: string;
}

const ENGINE_META: Record<string, { name: string; desc: string; color: string; border: string; bg: string; text: string; bar: string }> = {
  MiDaS_small: { name: 'MiDaS Small', desc: 'Fast CPU-friendly', color: '#06b6d4', border: 'border-cyan-500/40', bg: 'bg-cyan-500/15', text: 'text-cyan-300', bar: 'bg-cyan-500' },
  DPT_Hybrid:  { name: 'DPT Hybrid',  desc: 'Balanced accuracy/speed', color: '#f59e0b', border: 'border-amber-500/40', bg: 'bg-amber-500/15', text: 'text-amber-300', bar: 'bg-amber-500' },
  DPT_Large:   { name: 'DPT Large',   desc: 'Highest accuracy (GPU)', color: '#f43f5e', border: 'border-rose-500/40', bg: 'bg-rose-500/15', text: 'text-rose-300', bar: 'bg-rose-500' },
};

const ENGINE_IDS = Object.keys(ENGINE_META);

const emptyResult = (): EngineResult => ({
  depthHeatmap: null, sourceImage: null, speed: null, e2eLatency: 0,
  pointCloud: [], vertexCount: 0, error: null, processing: false,
});

// ─── Point Cloud Mesh ───────────────────────────────────────────────────────

function PointCloudMesh({ 
  vertices, 
  disableRotation = false,
  coloringMode = 'rgb'
}: { 
  vertices: Vertex[]; 
  disableRotation?: boolean;
  coloringMode?: 'rgb' | 'height' | 'depth';
}) {
  const meshRef = useRef<THREE.Points>(null);
  const prevGeoRef = useRef<THREE.BufferGeometry | null>(null);

  const { geometry } = useMemo(() => {
    if (prevGeoRef.current) prevGeoRef.current.dispose();
    const positions = new Float32Array(vertices.length * 3);
    const colors = new Float32Array(vertices.length * 3);
    
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    if (coloringMode !== 'rgb' && vertices.length > 0) {
      for (const v of vertices) {
        if (v.y < minY) minY = v.y;
        if (v.y > maxY) maxY = v.y;
        if (v.z < minZ) minZ = v.z;
        if (v.z > maxZ) maxZ = v.z;
      }
    }
    
    const dy = maxY - minY || 1;
    const dz = maxZ - minZ || 1;

    for (let i = 0; i < vertices.length; i++) {
      const v = vertices[i];
      positions[i * 3] = v.x; positions[i * 3 + 1] = v.y; positions[i * 3 + 2] = v.z;
      
      if (coloringMode === 'height') {
        const ny = (v.y - minY) / dy;
        let r = 0, g = 0, b = 0;
        if (ny < 0.4) {
          const t = ny / 0.4;
          r = 34 + (139 - 34) * t;
          g = 139 + (69 - 139) * t;
          b = 34 + (19 - 34) * t;
        } else if (ny < 0.8) {
          const t = (ny - 0.4) / 0.4;
          r = 139 + (160 - 139) * t;
          g = 69 + (82 - 69) * t;
          b = 19 + (45 - 19) * t;
        } else {
          const t = (ny - 0.8) / 0.2;
          r = 160 + (255 - 160) * t;
          g = 82 + (255 - 82) * t;
          b = 45 + (255 - 45) * t;
        }
        colors[i * 3] = r / 255;
        colors[i * 3 + 1] = g / 255;
        colors[i * 3 + 2] = b / 255;
      } else if (coloringMode === 'depth') {
        const nz = (v.z - minZ) / dz;
        let r = 0, g = 0, b = 0;
        if (nz < 0.5) {
          const t = nz / 0.5;
          r = 255;
          g = 255 * t;
          b = 0;
        } else {
          const t = (nz - 0.5) / 0.5;
          r = 255 * (1 - t);
          g = 255 * (1 - t);
          b = 255 * t;
        }
        colors[i * 3] = r / 255;
        colors[i * 3 + 1] = g / 255;
        colors[i * 3 + 2] = b / 255;
      } else {
        colors[i * 3] = v.r / 255;
        colors[i * 3 + 1] = v.g / 255;
        colors[i * 3 + 2] = v.b / 255;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    prevGeoRef.current = geo;
    return { geometry: geo };
  }, [vertices, coloringMode]);

  useFrame(() => { 
    if (meshRef.current && !disableRotation) {
      meshRef.current.rotation.y += 0.002; 
    }
  });

  return (
    <points ref={meshRef} geometry={geometry}>
      <pointsMaterial vertexColors size={0.04} sizeAttenuation />
    </points>
  );
}

// ─── ThreeJS Custom Controller (Fly-through & Anaglyph) ──────────────────────

function ThreeController({ 
  flyThrough, 
  anaglyph,
  zSpacing,
  historyLength
}: { 
  flyThrough: boolean; 
  anaglyph: boolean;
  zSpacing: number;
  historyLength: number;
}) {
  const { gl } = useThree();
  
  const effect = useMemo(() => {
    return new AnaglyphEffect(gl);
  }, [gl]);

  useEffect(() => {
    const handleResize = () => {
      effect.setSize(gl.domElement.clientWidth, gl.domElement.clientHeight);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [effect, gl]);

  useFrame(({ scene, camera }) => {
    if (flyThrough) {
      camera.position.z -= 0.08;
      const limit = -(historyLength * zSpacing) - 2;
      if (camera.position.z < limit) {
        camera.position.z = 6;
      }
    }

    if (anaglyph) {
      effect.render(scene, camera);
    }
  }, anaglyph ? 1 : 0);

  return null;
}

function AutoFitCamera({ vertices }: { vertices: Vertex[] }) {
  const { camera } = useThree();
  useEffect(() => {
    if (vertices.length === 0) return;
    let cx = 0, cy = 0, cz = 0;
    for (const v of vertices) { cx += v.x; cy += v.y; cz += v.z; }
    cx /= vertices.length; cy /= vertices.length; cz /= vertices.length;
    let maxR = 0;
    for (const v of vertices) {
      const d = Math.sqrt((v.x - cx) ** 2 + (v.y - cy) ** 2 + (v.z - cz) ** 2);
      if (d > maxR) maxR = d;
    }
    camera.position.set(cx, cy, cz + maxR * 1.8);
    camera.lookAt(cx, cy, cz);
  }, [vertices, camera]);
  return null;
}

// ─── Mini 3D Viewer ─────────────────────────────────────────────────────────

function Mini3DViewer({ 
  vertices, 
  vertexCount, 
  engineId,
  accumulate3D,
  history = [],
  zSpacing,
  coloringMode = 'rgb',
  flyThrough = false,
  anaglyph = false
}: { 
  vertices: Vertex[]; 
  vertexCount: number; 
  engineId: string;
  accumulate3D: boolean;
  history?: { id: number; vertices: Vertex[] }[];
  zSpacing: number;
  coloringMode?: 'rgb' | 'height' | 'depth';
  flyThrough?: boolean;
  anaglyph?: boolean;
}) {
  const meta = ENGINE_META[engineId];
  const hasData = accumulate3D ? history.length > 0 : vertices.length > 0;
  
  if (!hasData) {
    return (
      <div className="w-full h-full flex items-center justify-center text-slate-600">
        <div className="text-center space-y-1">
          <Box className="w-6 h-6 mx-auto text-slate-700" />
          <p className="text-[9px] font-mono">Awaiting 3D data</p>
        </div>
      </div>
    );
  }
  
  const displayPointsCount = accumulate3D 
    ? history.reduce((sum, f) => sum + f.vertices.length, 0)
    : vertexCount;

  return (
    <div className="relative w-full h-full">
      <Canvas camera={{ position: [0, 0, 8], fov: 60 }} dpr={[1, 1.5]}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        {accumulate3D ? (
          history.map((frame, index) => (
            <group key={frame.id} position={[0, 0, index * zSpacing]}>
              <PointCloudMesh vertices={frame.vertices} disableRotation={true} coloringMode={coloringMode} />
            </group>
          ))
        ) : (
          <PointCloudMesh vertices={vertices} coloringMode={coloringMode} />
        )}
        <AutoFitCamera vertices={accumulate3D ? (history[0]?.vertices || []) : vertices} />
        <OrbitControls enableDamping dampingFactor={0.15} rotateSpeed={0.8} zoomSpeed={1.2} panSpeed={0.6} />
        <ThreeController flyThrough={flyThrough && accumulate3D} anaglyph={anaglyph} zSpacing={zSpacing} historyLength={history.length} />
      </Canvas>
      <div className="absolute top-2 left-2 bg-slate-950/80 backdrop-blur-sm px-2 py-0.5 rounded-md text-[8px] font-mono border border-slate-800 pointer-events-none" style={{ color: meta.color }}>
        {displayPointsCount.toLocaleString()} pts {accumulate3D && `(${history.length} frames)`}
      </div>
      <div className="absolute bottom-1.5 left-1.5 bg-slate-900/70 px-1.5 py-0.5 rounded text-[7px] font-mono text-slate-500 pointer-events-none">
        Drag · Scroll · R-click
      </div>
    </div>
  );
}

// ─── Performance Comparison Chart ───────────────────────────────────────────

function PerformanceChart({ results }: { results: Record<string, EngineResult> }) {
  const entries = ENGINE_IDS
    .filter(id => results[id]?.speed && !results[id]?.error)
    .map(id => ({ id, ...results[id] }));

  if (entries.length === 0) return null;

  const maxLatency = Math.max(...entries.map(e => e.e2eLatency || 0), 1);

  return (
    <div className="p-4 bg-slate-955/20 rounded-2xl border border-slate-800/85 space-y-3 font-mono">
      <div className="text-xs font-bold text-slate-300 flex items-center space-x-2">
        <BarChart3 className="w-4 h-4 text-violet-400" />
        <span>Engine Performance Comparison</span>
      </div>
      <div className="space-y-2.5">
        {entries.map(e => {
          const meta = ENGINE_META[e.id];
          const total = e.e2eLatency || 0;
          const inf = e.speed?.inference || 0;
          const pre = e.speed?.preprocess || 0;
          const post = e.speed?.postprocess || 0;
          const pct = Math.max(5, (total / maxLatency) * 100);
          return (
            <div key={e.id} className="space-y-1">
              <div className="flex items-center justify-between text-[10px]">
                <div className="flex items-center space-x-2">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: meta.color }} />
                  <span className={meta.text + ' font-bold'}>{meta.name}</span>
                </div>
                <div className="flex items-center space-x-3 text-slate-500">
                  <span>Pre: <span className="text-slate-400">{pre.toFixed(0)}ms</span></span>
                  <span>Inf: <span className="font-bold" style={{ color: meta.color }}>{inf.toFixed(0)}ms</span></span>
                  <span>Post: <span className="text-slate-400">{post.toFixed(0)}ms</span></span>
                  <span className="text-slate-300 font-extrabold">{total}ms</span>
                </div>
              </div>
              <div className="h-2.5 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ease-out ${meta.bar}`}
                  style={{ width: `${pct}%`, opacity: 0.85 }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────

export const DepthLab: React.FC = () => {
  // Engine toggle switches
  const [enabledEngines, setEnabledEngines] = useState<Record<string, boolean>>({
    MiDaS_small: true, DPT_Hybrid: false, DPT_Large: false,
  });

  const activeEngines = ENGINE_IDS.filter(id => enabledEngines[id]);

  // Per-engine results
  const [engineResults, setEngineResults] = useState<Record<string, EngineResult>>({
    MiDaS_small: emptyResult(), DPT_Hybrid: emptyResult(), DPT_Large: emptyResult(),
  });

  // Input state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isVideo, setIsVideo] = useState(false);

  // Demo assets
  const [demoCatalog, setDemoCatalog] = useState<DemoAsset[]>([]);
  const [isDemoLibraryOpen, setIsDemoLibraryOpen] = useState(false);
  const [activeDemoCategory, setActiveDemoCategory] = useState('Objects & Scenes');
  const [activeDemoKey, setActiveDemoKey] = useState<string | null>(null);

  // 3D World Accumulation
  const [accumulate3D, setAccumulate3D] = useState(false);
  const [historyLimit, setHistoryLimit] = useState(15);
  const [zSpacing, setZSpacing] = useState(0.8);
  const [worldFrames, setWorldFrames] = useState<Record<string, { id: number; vertices: Vertex[] }[]>>({
    MiDaS_small: [], DPT_Hybrid: [], DPT_Large: [],
  });

  const frameCountRef = useRef(0);
  const accumulate3DRef = useRef(accumulate3D);
  const historyLimitRef = useRef(historyLimit);

  useEffect(() => { accumulate3DRef.current = accumulate3D; }, [accumulate3D]);
  useEffect(() => { historyLimitRef.current = historyLimit; }, [historyLimit]);

  // Advanced features
  const [coloringMode, setColoringMode] = useState<'rgb' | 'height' | 'depth'>('rgb');
  const [flyThrough, setFlyThrough] = useState(false);
  const [anaglyph, setAnaglyph] = useState(false);

  // Video assets
  interface DepthVideo {
    id: string;
    title: string;
    description: string;
    url: string;
    filename: string;
    cached: boolean;
    status: string;
  }
  const [videoCatalog, setVideoCatalog] = useState<DepthVideo[]>([]);
  const [selectedVideoKey, setSelectedVideoKey] = useState<string | null>(null);

  const loadVideos = useCallback(async () => {
    try {
      const videos = await client.getDepthVideos();
      setVideoCatalog(videos);
    } catch { /* offline */ }
  }, []);

  useEffect(() => {
    loadVideos();
  }, [loadVideos]);

  // Poll video download progress
  useEffect(() => {
    const hasDownloading = videoCatalog.some(v => v.status === 'downloading');
    if (!hasDownloading) return;
    const interval = setInterval(loadVideos, 3000);
    return () => clearInterval(interval);
  }, [videoCatalog, loadVideos]);

  const selectVideo = (video: DepthVideo) => {
    setSelectedVideoKey(video.id);
    setIsVideo(true);
    setUseWebcam(false);
    setSelectedFile(null);
    setWorldFrames({ MiDaS_small: [], DPT_Hybrid: [], DPT_Large: [] });
    frameCountRef.current = 0;
    const url = video.cached 
      ? `${client.baseUrl}/api/demo-files/${video.filename}`
      : video.url;
    setPreviewUrl(url);
  };

  const downloadVideo = async (videoId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await client.downloadDepthVideo(videoId);
      loadVideos();
    } catch { /* skip */ }
  };

  // Webcam
  const [useWebcam, setUseWebcam] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const sourceVideoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isProcessingRef = useRef(false);

  // 3D controls
  const [downsample, setDownsample] = useState(2);

  // UI
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Fullscreen
  const [fullscreenUrl, setFullscreenUrl] = useState<string | null>(null);
  const [fullscreenTitle, setFullscreenTitle] = useState('');
  const [expanded3dEngineId, setExpanded3dEngineId] = useState<string | null>(null);

  // ── Toggle engine ──────────────────────────────────────────────
  const toggleEngine = (id: string) => {
    setEnabledEngines(prev => ({ ...prev, [id]: !prev[id] }));
  };

  // ── Fetch demo assets ──────────────────────────────────────────
  useEffect(() => {
    const load = async () => {
      try {
        const assets = await client.getDemoAssets();
        setDemoCatalog(assets);
      } catch { /* offline */ }
    };
    load();
  }, []);

  // ── Select demo asset ──────────────────────────────────────────
  const selectDemoAsset = async (key: string) => {
    try {
      setActiveDemoKey(key);
      const asset = await client.getDemoAsset(key);
      const res = await fetch(asset.image_base64);
      const blob = await res.blob();
      const file = new File([blob], `${key}.jpg`, { type: 'image/jpeg' });
      setSelectedFile(file);
      setIsVideo(false);
      setPreviewUrl(asset.image_base64);
      setUseWebcam(false);
      setErrorMsg(null);
      if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    } catch { setErrorMsg('Failed to download demo asset.'); }
  };

  // ── File handling ──────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setIsVideo(file.type.startsWith('video/'));
    setPreviewUrl(URL.createObjectURL(file));
    setErrorMsg(null);
    setActiveDemoKey(null);
    setUseWebcam(false);
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
  };

  // ── Webcam ─────────────────────────────────────────────────────
  const toggleWebcam = async (val: boolean) => {
    setUseWebcam(val);
    setIsVideo(false);
    setActiveDemoKey(null);
    if (val) {
      setErrorMsg(null);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch { setErrorMsg('Could not access webcam.'); setUseWebcam(false); }
    } else {
      if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    }
  };

  useEffect(() => {
    if (!useWebcam && streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
  }, [useWebcam]);

  useEffect(() => {
    return () => { if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop()); };
  }, []);

  // ── Process single engine ──────────────────────────────────────
  const processEngine = useCallback(async (engineId: string, file: File) => {
    setEngineResults(prev => ({
      ...prev,
      [engineId]: { ...prev[engineId], processing: true, error: null },
    }));

    const formData = new FormData();
    formData.append('file', file);
    const pcForm = new FormData();
    pcForm.append('file', file);
    const startTime = performance.now();

    try {
      // Run depth estimation and point cloud generation concurrently
      const [data, pcData] = await Promise.all([
        client.estimateDepth(formData, { model_type: engineId }),
        client.generatePointCloud(pcForm, { model_type: engineId, downsample: 4 }).catch(() => null)
      ]);
      const e2e = Math.round(performance.now() - startTime);

      if (data && data.status === 'success') {
        let pc: Vertex[] = [];
        let vc = 0;
        if (pcData && pcData.vertices) {
          pc = pcData.vertices;
          vc = pcData.vertex_count;
        }

        if (accumulate3DRef.current && pc.length > 0) {
          setWorldFrames(prev => {
            const history = prev[engineId] || [];
            const newFrame = { id: frameCountRef.current++, vertices: pc };
            return {
              ...prev,
              [engineId]: [newFrame, ...history].slice(0, historyLimitRef.current),
            };
          });
        }

        setEngineResults(prev => ({
          ...prev,
          [engineId]: {
            depthHeatmap: data.depth_heatmap,
            sourceImage: data.source_image,
            speed: data.speed || null,
            e2eLatency: e2e,
            pointCloud: pc,
            vertexCount: vc,
            error: null,
            processing: false,
          },
        }));
      } else {
        setEngineResults(prev => ({
          ...prev,
          [engineId]: { ...emptyResult(), error: data?.detail || 'Depth estimation failed.', processing: false },
        }));
      }
    } catch (err) {
      setEngineResults(prev => ({
        ...prev,
        [engineId]: { ...emptyResult(), error: 'Failed to connect to depth API.', processing: false },
      }));
    }
  }, []);

  // ── Run all enabled engines on static image ────────────────────
  const runAllEngines = useCallback(async (file: File) => {
    const enabled = ENGINE_IDS.filter(id => enabledEngines[id]);
    await Promise.all(enabled.map(id => processEngine(id, file)));
  }, [enabledEngines, processEngine]);

  // Auto-run on file change or engine toggle
  useEffect(() => {
    if (selectedFile && !useWebcam && !isVideo) {
      runAllEngines(selectedFile);
    }
  }, [selectedFile, enabledEngines, runAllEngines, isVideo, useWebcam]);

  // ── Webcam depth loop (multi-engine) ───────────────────────────
  useEffect(() => {
    if (!useWebcam) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;

    const processFrame = async () => {
      if (!active || !videoRef.current || !canvasRef.current || isProcessingRef.current) {
        if (active) timer = setTimeout(processFrame, 400);
        return;
      }
      isProcessingRef.current = true;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      if (!ctx) { isProcessingRef.current = false; return; }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise<Blob | null>(r => canvas.toBlob(r, 'image/jpeg', 0.8));
      if (!blob) { isProcessingRef.current = false; return; }

      const file = new File([blob], 'webcam.jpg', { type: 'image/jpeg' });
      const enabled = ENGINE_IDS.filter(id => enabledEngines[id]);
      await Promise.all(enabled.map(id => processEngine(id, file)));

      isProcessingRef.current = false;
      if (active) timer = setTimeout(processFrame, 300);
    };

    processFrame();
    return () => { active = false; clearTimeout(timer); };
  }, [useWebcam, enabledEngines, processEngine]);

  // ── Video depth loop (multi-engine) ────────────────────────────
  useEffect(() => {
    if (!isVideo || !previewUrl) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;

    const processVideoFrame = async () => {
      const video = sourceVideoRef.current;
      if (!active || !video || video.paused || video.ended || isProcessingRef.current) {
        if (active) timer = setTimeout(processVideoFrame, 400);
        return;
      }
      isProcessingRef.current = true;

      const canvas = canvasRef.current;
      if (canvas && video.readyState >= 2) {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const blob = await new Promise<Blob | null>(r => canvas.toBlob(r, 'image/jpeg', 0.8));
          if (blob && active) {
            const file = new File([blob], 'video_frame.jpg', { type: 'image/jpeg' });
            const enabled = ENGINE_IDS.filter(id => enabledEngines[id]);
            await Promise.all(enabled.map(id => processEngine(id, file)));
          }
        }
      }

      isProcessingRef.current = false;
      if (active) timer = setTimeout(processVideoFrame, 300);
    };

    processVideoFrame();

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [isVideo, previewUrl, enabledEngines, processEngine]);

  // ── Manual high-res 3D generation ──────────────────────────────
  const generateHighRes3D = useCallback(async () => {
    let file = selectedFile;
    if (useWebcam && videoRef.current && canvasRef.current) {
      const c = canvasRef.current, ctx = c.getContext('2d');
      if (ctx) {
        c.width = videoRef.current.videoWidth || 640; c.height = videoRef.current.videoHeight || 480;
        ctx.drawImage(videoRef.current, 0, 0, c.width, c.height);
        const blob = await new Promise<Blob | null>(r => c.toBlob(r, 'image/jpeg', 0.85));
        if (blob) file = new File([blob], 'webcam.jpg', { type: 'image/jpeg' });
      }
    }
    if (!file) return;

    const enabled = ENGINE_IDS.filter(id => enabledEngines[id]);
    for (const engineId of enabled) {
      const pcForm = new FormData();
      pcForm.append('file', file);
      try {
        const pcData = await client.generatePointCloud(pcForm, { model_type: engineId, downsample });
        if (pcData && pcData.vertices) {
          setEngineResults(prev => ({
            ...prev,
            [engineId]: { ...prev[engineId], pointCloud: pcData.vertices, vertexCount: pcData.vertex_count },
          }));
        }
      } catch { /* skip */ }
    }
  }, [selectedFile, useWebcam, enabledEngines, downsample]);

  // ── Download ───────────────────────────────────────────────────
  const handleDownload = (engineId: string) => {
    const heatmap = engineResults[engineId]?.depthHeatmap;
    if (!heatmap) return;
    const a = document.createElement('a');
    a.href = heatmap;
    a.download = `depth_${engineId}.jpg`;
    a.click();
  };

  // Any engine processing?
  const anyProcessing = activeEngines.some(id => engineResults[id]?.processing);

  // ─── Render ────────────────────────────────────────────────────
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h2 className="text-2xl font-extrabold text-white flex items-center space-x-3 tracking-tight">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-violet-500 via-indigo-500 to-cyan-500 flex items-center justify-center shadow-lg shadow-violet-500/20">
              <Box className="w-5 h-5 text-white" />
            </div>
            <span className="text-gradient font-mono">3D Depth Lab</span>
          </h2>
          <p className="text-xs text-slate-400 mt-1 max-w-xl">
            Monocular depth estimation via Intel MiDaS — compare engines side-by-side with interactive 3D point cloud scans.
          </p>
        </div>
      </div>

      {/* Demo Catalog */}
      <div className="border border-slate-850 rounded-2xl overflow-hidden bg-slate-900/10 mb-6">
        <button
          onClick={() => setIsDemoLibraryOpen(!isDemoLibraryOpen)}
          className="w-full px-5 py-4 bg-slate-900/40 flex items-center justify-between text-left text-xs font-bold text-slate-205 border-b border-slate-850 hover:bg-slate-900/60 transition"
        >
          <div className="flex items-center space-x-2">
            <Folder className="w-4.5 h-4.5 text-violet-400" />
            <span>Studio Test Asset Catalog</span>
          </div>
          <div className="flex items-center space-x-3">
            <span className="text-[10px] bg-slate-900 border border-slate-800 text-slate-400 font-medium px-2.5 py-0.5 rounded-full font-mono">
              {Array.isArray(demoCatalog) && demoCatalog.length > 0 ? `${demoCatalog.length} samples` : 'Offline'}
            </span>
            <span className="text-[10px] text-slate-400 font-mono">{isDemoLibraryOpen ? 'Hide' : 'Show'}</span>
          </div>
        </button>
        {isDemoLibraryOpen && (
          <div className="p-4 bg-slate-955/20 space-y-4 animate-in fade-in slide-in-from-top-2 duration-200">
            <div className="flex space-x-1 p-1 bg-slate-950/80 rounded-xl border border-slate-900 max-w-3xl overflow-x-auto">
              {['Face & Hands', 'Body & Pose', 'Objects & Scenes', 'OCR & Translation', '3D Video Streams'].map(cat => (
                <button key={cat} onClick={() => setActiveDemoCategory(cat)}
                  className={`py-1.5 px-3.5 rounded-lg text-[10px] font-bold transition shrink-0 ${activeDemoCategory === cat ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30' : 'text-slate-400 hover:text-white'}`}
                >{cat}</button>
              ))}
            </div>
            {activeDemoCategory === '3D Video Streams' ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5">
                {videoCatalog.map(video => (
                  <div key={video.id} onClick={() => selectVideo(video)}
                    className={`p-3 bg-slate-900/65 border rounded-xl hover:bg-slate-900/90 transition flex flex-col justify-between group cursor-pointer ${selectedVideoKey === video.id ? 'border-violet-500 bg-violet-500/5 shadow-md shadow-violet-500/5' : 'border-slate-800/80'}`}
                  >
                    <div className="space-y-1">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[11px] font-bold text-slate-300 group-hover:text-violet-300 transition truncate">{video.title}</span>
                        <span className={`text-[7px] px-1.5 py-0.5 rounded font-mono font-bold uppercase shrink-0 ${
                          video.status === 'ready' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
                          video.status === 'downloading' ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20 animate-pulse' :
                          'bg-slate-800 text-slate-400 border border-slate-700'
                        }`}>
                          {video.status}
                        </span>
                      </div>
                      <p className="text-[9px] text-slate-400 leading-normal">{video.description}</p>
                    </div>
                    <div className="mt-3 flex items-center justify-between text-[8px] font-mono border-t border-slate-850 pt-2.5">
                      <span className="text-slate-500">File: {video.filename}</span>
                      {video.status !== 'ready' && (
                        <button 
                          onClick={(e) => downloadVideo(video.id, e)}
                          disabled={video.status === 'downloading'}
                          className="px-2 py-0.5 bg-violet-500/20 hover:bg-violet-500/35 text-violet-300 border border-violet-500/35 rounded transition disabled:opacity-50 font-bold"
                        >
                          {video.status === 'downloading' ? 'Downloading...' : 'Cache'}
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : Array.isArray(demoCatalog) && demoCatalog.length > 0 ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {demoCatalog.filter(a => a.category === activeDemoCategory).map(asset => (
                  <button key={asset.key} onClick={() => selectDemoAsset(asset.key)}
                    className={`p-2.5 bg-slate-900/60 border rounded-xl hover:bg-slate-900/90 text-center transition space-y-1 flex flex-col items-center justify-center group ${activeDemoKey === asset.key ? 'border-violet-400 bg-violet-500/5' : 'border-slate-800'}`}
                  >
                    <span className="text-[10px] font-bold text-slate-300 truncate w-full group-hover:text-violet-300 transition">{asset.title}</span>
                    <span className="text-[8px] font-mono text-slate-500 block uppercase">{asset.description || asset.category}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-xs text-slate-400 py-2">Preset demo catalog requires backend connection.</div>
            )}
          </div>
        )}
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-5">
        {/* Left — Controls */}
        <div className="glass-panel p-5 rounded-2xl space-y-5">
          {/* Engine Toggle Switches */}
          <div className="space-y-2">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block font-mono">Engine Toggles</label>
            <div className="space-y-2">
              {ENGINE_IDS.map(id => {
                const meta = ENGINE_META[id];
                const on = enabledEngines[id];
                return (
                  <button key={id} onClick={() => toggleEngine(id)}
                    className={`w-full p-2.5 rounded-xl border text-left transition flex items-center justify-between ${on ? `${meta.bg} ${meta.border} ${meta.text}` : 'border-slate-800 bg-slate-900/35 text-slate-500'}`}
                  >
                    <div>
                      <span className="text-[11px] font-bold font-mono block">{meta.name}</span>
                      <span className="text-[8px] text-slate-500 leading-tight">{meta.desc}</span>
                    </div>
                    {on ? <ToggleRight className="w-5 h-5" style={{ color: meta.color }} /> : <ToggleLeft className="w-5 h-5 text-slate-700" />}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Input Source */}
          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block font-mono">Input Source</label>
            <div className="flex flex-wrap gap-2">
              <button onClick={() => toggleWebcam(!useWebcam)}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold border transition ${useWebcam ? 'bg-violet-500/15 border-violet-500/40 text-violet-300' : 'bg-slate-900/60 border-slate-800 text-slate-500 hover:text-slate-300'}`}
              >
                <Eye className="w-3 h-3" />
                <span>Webcam</span>
                <span className={`w-1.5 h-1.5 rounded-full ${useWebcam ? 'bg-violet-400 animate-pulse' : 'bg-slate-700'}`} />
              </button>
            </div>
          </div>

          {/* 3D Resolution */}
          <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800/80 space-y-2">
            <div className="flex justify-between items-center">
              <div className="text-xs font-semibold text-slate-200 flex items-center space-x-1.5">
                <Layers className="w-3.5 h-3.5 text-violet-400" />
                <span>3D Resolution</span>
              </div>
              <span className="text-[10px] font-mono text-violet-400">Every {downsample}px</span>
            </div>
            <Slider.Root value={[downsample]} onValueChange={([v]) => setDownsample(v)} min={1} max={6} step={1} className="relative flex items-center select-none touch-none w-full h-5">
              <Slider.Track className="bg-slate-800 relative grow rounded-full h-1">
                <Slider.Range className="absolute bg-violet-500 rounded-full h-full" />
              </Slider.Track>
              <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-violet-400 focus:outline-none" />
            </Slider.Root>
            <div className="text-[9px] text-slate-500">1 = full (~300K pts), 4 = sparse (~18K pts)</div>
          </div>

          {/* 3D World Accumulator Panel */}
          <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800/80 space-y-3.5">
            <div className="flex items-center justify-between">
              <div className="text-xs font-semibold text-slate-200 flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5 text-violet-400" />
                <span>3D World Accumulator</span>
              </div>
              <button
                onClick={() => setAccumulate3D(!accumulate3D)}
                className={`px-2 py-0.5 rounded text-[8px] font-mono font-bold uppercase transition ${
                  accumulate3D 
                    ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' 
                    : 'bg-slate-800 text-slate-400 border border-slate-700'
                }`}
              >
                {accumulate3D ? 'Active' : 'Disabled'}
              </button>
            </div>

            {accumulate3D && (
              <div className="space-y-3 animate-in fade-in slide-in-from-top-1 duration-150">
                {/* History Buffer Size */}
                <div className="space-y-1">
                  <div className="flex justify-between items-center text-[9px]">
                    <span className="text-slate-400 font-mono">History Buffer</span>
                    <span className="font-mono text-violet-400">{historyLimit} frames</span>
                  </div>
                  <Slider.Root 
                    value={[historyLimit]} 
                    onValueChange={([v]) => setHistoryLimit(v)} 
                    min={5} 
                    max={30} 
                    step={1} 
                    className="relative flex items-center select-none touch-none w-full h-4"
                  >
                    <Slider.Track className="bg-slate-800 relative grow rounded-full h-0.5">
                      <Slider.Range className="absolute bg-violet-500 rounded-full h-full" />
                    </Slider.Track>
                    <Slider.Thumb className="block w-2.5 h-2.5 bg-white rounded-full border border-violet-400 focus:outline-none" />
                  </Slider.Root>
                </div>

                {/* Frame Spacing (Z-Spacing) */}
                <div className="space-y-1">
                  <div className="flex justify-between items-center text-[9px]">
                    <span className="text-slate-400 font-mono">Frame Spacing (Z)</span>
                    <span className="font-mono text-violet-400">{zSpacing.toFixed(1)} units</span>
                  </div>
                  <Slider.Root 
                    value={[zSpacing]} 
                    onValueChange={([v]) => setZSpacing(v)} 
                    min={0.1} 
                    max={3.0} 
                    step={0.1} 
                    className="relative flex items-center select-none touch-none w-full h-4"
                  >
                    <Slider.Track className="bg-slate-800 relative grow rounded-full h-0.5">
                      <Slider.Range className="absolute bg-violet-500 rounded-full h-full" />
                    </Slider.Track>
                    <Slider.Thumb className="block w-2.5 h-2.5 bg-white rounded-full border border-violet-400 focus:outline-none" />
                  </Slider.Root>
                </div>

                {/* 3D Coloring Mode */}
                <div className="space-y-1">
                  <label className="text-[9px] text-slate-400 font-mono">Coloring Mode</label>
                  <div className="flex space-x-1 p-0.5 bg-slate-950 rounded-lg border border-slate-800">
                    {(['rgb', 'height', 'depth'] as const).map(mode => (
                      <button 
                        key={mode} 
                        onClick={() => setColoringMode(mode)}
                        className={`flex-1 py-1 rounded text-[8px] font-bold uppercase transition ${
                          coloringMode === mode 
                            ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30 font-bold' 
                            : 'text-slate-500 hover:text-slate-300'
                        }`}
                      >
                        {mode === 'rgb' ? 'Original' : mode}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Auto Fly-Through Toggle */}
                <div className="flex items-center justify-between text-[9px] pt-1">
                  <span className="text-slate-400 font-mono">Auto Fly-Through</span>
                  <button
                    onClick={() => setFlyThrough(!flyThrough)}
                    className={`px-2 py-0.5 rounded text-[8px] font-mono font-bold uppercase transition ${
                      flyThrough 
                        ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30' 
                        : 'bg-slate-950 text-slate-500 border border-slate-850'
                    }`}
                  >
                    {flyThrough ? 'On' : 'Off'}
                  </button>
                </div>

                {/* Anaglyph 3D Toggle */}
                <div className="flex items-center justify-between text-[9px] pt-0.5">
                  <span className="text-slate-400 font-mono">Anaglyph 3D (Stereo)</span>
                  <button
                    onClick={() => setAnaglyph(!anaglyph)}
                    className={`px-2 py-0.5 rounded text-[8px] font-mono font-bold uppercase transition ${
                      anaglyph 
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' 
                        : 'bg-slate-950 text-slate-500 border border-slate-850'
                    }`}
                  >
                    {anaglyph ? 'Glasses On' : 'Off'}
                  </button>
                </div>

                {/* Clear accumulated frames */}
                <button
                  onClick={() => setWorldFrames({ MiDaS_small: [], DPT_Hybrid: [], DPT_Large: [] })}
                  className="w-full py-1.5 bg-slate-950 hover:bg-slate-900 border border-slate-800 rounded-lg text-[9px] font-mono text-rose-400 hover:text-rose-300 font-bold transition flex items-center justify-center space-x-1"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>Clear Reconstructed 3D</span>
                </button>
              </div>
            )}
          </div>

          {/* High-Res 3D Button */}
          <button
            onClick={generateHighRes3D}
            disabled={activeEngines.length === 0 || (!selectedFile && !useWebcam)}
            className="w-full py-3 rounded-xl font-bold text-sm flex items-center justify-center space-x-2 transition bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white shadow-lg shadow-violet-500/20 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Box className="w-4 h-4" />
            <span>Generate Hi-Res 3D</span>
          </button>

          {/* Active engine count badge */}
          <div className="text-center text-[9px] font-mono text-slate-600">
            {activeEngines.length} engine{activeEngines.length !== 1 ? 's' : ''} active
          </div>
        </div>

        {/* Right — Display (3 columns) */}
        <div className="lg:col-span-3 space-y-5">
          {/* Header Row */}
          <div className="glass-panel p-4 rounded-2xl">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <h3 className="text-lg font-bold text-white flex items-center space-x-2 font-mono">
                  <Cpu className="w-5 h-5 text-violet-400" />
                  <span>Multi-Engine Depth Viewport</span>
                  {anyProcessing && <span className="w-2 h-2 rounded-full bg-violet-400 animate-ping" />}
                </h3>
                <div className="text-[10px] text-slate-500 font-mono mt-0.5">
                  {activeEngines.map(id => ENGINE_META[id].name).join(' · ') || 'No engines enabled'}
                </div>
              </div>
              <div className="flex items-center space-x-3">
                {activeEngines.length > 0 && engineResults[activeEngines[0]]?.depthHeatmap && (
                  <button onClick={() => handleDownload(activeEngines[0])}
                    className="flex items-center space-x-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs text-violet-300 rounded-xl transition font-medium"
                  >
                    <Download className="w-3.5 h-3.5" /><span>Download</span>
                  </button>
                )}
                <input type="file" accept="image/*,video/*" onChange={handleFileChange}
                  className="text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-300 hover:file:bg-slate-700 cursor-pointer"
                />
              </div>
            </div>
          </div>

          {/* Error */}
          {errorMsg && (
            <div className="p-3.5 bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-xl text-xs flex items-center space-x-2">
              <AlertCircle className="w-4.5 h-4.5 text-rose-400 shrink-0" /><span>{errorMsg}</span>
            </div>
          )}

          {/* Source Input Row */}
          <div className="glass-panel p-4 rounded-2xl">
            <div className="bg-slate-950/85 rounded-xl border border-slate-800 p-4 flex items-center justify-center relative group min-h-[180px]">
              <div className="absolute top-3 left-3 bg-slate-900/90 px-2.5 py-1 rounded-md text-[10px] font-mono text-slate-400 border border-slate-800 z-10">Source Input</div>
              {useWebcam ? (
                <div className="relative w-full max-h-52 rounded-lg overflow-hidden flex items-center justify-center bg-black">
                  <video ref={videoRef} autoPlay playsInline muted className="max-h-52 w-full object-contain rounded-lg transform -scale-x-100" />
                  <canvas ref={canvasRef} className="hidden" />
                </div>
              ) : isVideo && previewUrl ? (
                <div className="relative w-full max-h-52 rounded-lg overflow-hidden flex items-center justify-center bg-black">
                  <video ref={sourceVideoRef} src={previewUrl} crossOrigin="anonymous" controls autoPlay loop muted className="max-h-52 w-full object-contain rounded-lg" />
                  <canvas ref={canvasRef} className="hidden" />
                </div>
              ) : previewUrl ? (
                <>
                  <img src={previewUrl} alt="Source" className="max-h-52 rounded-lg object-contain" />
                  <button onClick={() => { setFullscreenUrl(previewUrl); setFullscreenTitle('Source Input'); }}
                    className="absolute top-3 right-3 p-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-700 transition opacity-0 group-hover:opacity-100"
                  ><Maximize2 className="w-4 h-4" /></button>
                </>
              ) : (
                <div className="text-center text-slate-500 text-sm space-y-2">
                  <ImageIcon className="w-8 h-8 mx-auto text-slate-600" />
                  <div>Upload image/video or enable webcam</div>
                </div>
              )}
            </div>
          </div>

          {/* Depth Heatmap Outputs Row */}
          {activeEngines.length > 0 && (
            <div className={`grid gap-4 ${activeEngines.length === 1 ? 'grid-cols-1' : activeEngines.length === 2 ? 'grid-cols-2' : 'grid-cols-3'}`}>
              {activeEngines.map(engineId => {
                const meta = ENGINE_META[engineId];
                const result = engineResults[engineId] || emptyResult();
                return (
                  <div key={engineId} className="glass-panel rounded-2xl overflow-hidden">
                    {/* Engine badge header */}
                    <div className={`px-3 py-2 flex items-center justify-between border-b border-slate-800/60 ${meta.bg}`}>
                      <div className="flex items-center space-x-2">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: meta.color }} />
                        <span className={`text-[10px] font-bold font-mono ${meta.text}`}>{meta.name}</span>
                        {result.processing && <span className="w-1.5 h-1.5 rounded-full animate-ping" style={{ backgroundColor: meta.color }} />}
                      </div>
                      {result.speed && (
                        <span className="text-[9px] font-mono text-slate-500">
                          <Zap className="w-2.5 h-2.5 inline mr-0.5" style={{ color: meta.color }} />
                          {result.speed.inference.toFixed(0)}ms
                        </span>
                      )}
                    </div>
                    {/* Depth heatmap */}
                    <div 
                      onClick={() => { if (result.depthHeatmap) { setFullscreenUrl(result.depthHeatmap); setFullscreenTitle(`Depth — ${meta.name}`); } }}
                      className={`p-3 min-h-[160px] flex items-center justify-center bg-slate-950/85 relative group transition-colors ${result.depthHeatmap ? 'cursor-pointer hover:bg-slate-900/45' : ''}`}
                    >
                      {result.error ? (
                        <div className="text-center space-y-1">
                          <AlertCircle className="w-5 h-5 text-rose-500 mx-auto" />
                          <p className="text-[9px] text-rose-400 font-mono">{result.error}</p>
                        </div>
                      ) : result.depthHeatmap ? (
                        <>
                          <img src={result.depthHeatmap} alt={`Depth ${meta.name}`} className="max-h-48 rounded-lg object-contain" />
                          <button onClick={(e) => { e.stopPropagation(); setFullscreenUrl(result.depthHeatmap); setFullscreenTitle(`Depth — ${meta.name}`); }}
                            className="absolute top-2 right-2 p-1 rounded-md bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-700 transition opacity-0 group-hover:opacity-100 cursor-pointer"
                          ><Maximize2 className="w-3 h-3" /></button>
                        </>
                      ) : (
                        <div className="text-center text-slate-600 text-[10px] font-mono">
                          {result.processing ? 'Processing...' : 'Awaiting input'}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* 3D Point Cloud Viewers Row */}
          {activeEngines.length > 0 && activeEngines.some(id => 
            accumulate3D 
              ? (worldFrames[id]?.length || 0) > 0 
              : (engineResults[id]?.pointCloud?.length || 0) > 0
          ) && (
            <div className={`grid gap-4 ${activeEngines.length === 1 ? 'grid-cols-1' : activeEngines.length === 2 ? 'grid-cols-2' : 'grid-cols-3'}`}>
              {activeEngines.map(engineId => {
                const meta = ENGINE_META[engineId];
                const result = engineResults[engineId] || emptyResult();
                return (
                  <div key={engineId} className={`rounded-2xl border overflow-hidden bg-black/90 relative group ${meta.border}`} style={{ height: activeEngines.length === 1 ? 380 : 260 }}>
                    <div className={`px-3 py-1.5 flex items-center justify-between border-b border-slate-800/40 ${meta.bg}`}>
                      <div className="flex items-center space-x-2">
                        <Box className="w-3 h-3" style={{ color: meta.color }} />
                        <span className={`text-[9px] font-bold font-mono ${meta.text}`}>3D — {meta.name}</span>
                      </div>
                      {(accumulate3D ? (worldFrames[engineId]?.length || 0) > 0 : (result.pointCloud?.length || 0) > 0) && (
                        <button
                          onClick={() => setExpanded3dEngineId(engineId)}
                          className="p-1 rounded bg-slate-900/80 hover:bg-slate-850 text-slate-400 hover:text-white border border-slate-800 hover:border-slate-700 transition cursor-pointer"
                          title="Expand 3D view"
                        >
                          <Maximize2 className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                    <div className="w-full" style={{ height: activeEngines.length === 1 ? 340 : 225 }}>
                      <Mini3DViewer 
                        vertices={result.pointCloud} 
                        vertexCount={result.vertexCount} 
                        engineId={engineId}
                        accumulate3D={accumulate3D}
                        history={worldFrames[engineId] || []}
                        zSpacing={zSpacing}
                        coloringMode={coloringMode}
                        flyThrough={flyThrough}
                        anaglyph={anaglyph}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Performance Comparison Chart */}
          <PerformanceChart results={engineResults} />
        </div>
      </div>

      {/* Lightbox */}
      {fullscreenUrl && (
        <div className="fixed inset-0 bg-black/90 backdrop-blur-lg z-[200] flex items-center justify-center p-8 animate-in fade-in" onClick={() => setFullscreenUrl(null)}>
          <div className="relative max-w-5xl max-h-full">
            <div className="absolute -top-10 left-0 text-xs font-mono text-slate-400">{fullscreenTitle}</div>
            <button onClick={(e) => { e.stopPropagation(); setFullscreenUrl(null); }}
              className="absolute -top-10 right-0 p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-700 transition"
            ><X className="w-4 h-4" /></button>
            {fullscreenUrl === 'webcam' ? (
              <div className="max-h-[85vh] aspect-video w-[640px] bg-black rounded-xl overflow-hidden flex items-center justify-center">
                <video ref={(el) => { if (el && videoRef.current && el.srcObject !== videoRef.current.srcObject) el.srcObject = videoRef.current.srcObject; }}
                  autoPlay playsInline muted className="w-full h-full object-contain transform -scale-x-100" />
              </div>
            ) : isVideo && fullscreenUrl === previewUrl ? (
              <video src={previewUrl} crossOrigin="anonymous" controls autoPlay loop muted className="max-h-[85vh] rounded-xl object-contain shadow-2xl" />
            ) : (
              <img src={fullscreenUrl} alt="Fullscreen" className="max-h-[85vh] rounded-xl object-contain shadow-2xl" />
            )}
          </div>
        </div>
      )}

      {/* Expanded 3D Viewport Lightbox */}
      {expanded3dEngineId && (
        <div className="fixed inset-0 bg-black/90 backdrop-blur-lg z-[200] flex items-center justify-center p-8 animate-in fade-in" onClick={() => setExpanded3dEngineId(null)}>
          <div className="relative max-w-5xl w-full h-[80vh] flex flex-col items-center justify-center" onClick={(e) => e.stopPropagation()}>
            <div className="absolute -top-10 left-0 text-xs font-mono text-slate-400">
              3D Point Cloud — {ENGINE_META[expanded3dEngineId].name} ({
                accumulate3D 
                  ? `${(worldFrames[expanded3dEngineId] || []).reduce((sum, f) => sum + f.vertices.length, 0).toLocaleString()} pts (${(worldFrames[expanded3dEngineId] || []).length} frames)`
                  : `${(engineResults[expanded3dEngineId]?.vertexCount || 0).toLocaleString()} pts`
              })
            </div>
            <button onClick={() => setExpanded3dEngineId(null)}
              className="absolute -top-10 right-0 p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-700 transition cursor-pointer"
            ><X className="w-4 h-4" /></button>
            <div className="w-full h-full bg-slate-950/90 rounded-2xl border border-slate-800/80 overflow-hidden relative">
              <Canvas camera={{ position: [0, 0, 8], fov: 60 }} dpr={[1, 1.5]}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1} />
                {accumulate3D ? (
                  (worldFrames[expanded3dEngineId] || []).map((frame, index) => (
                    <group key={frame.id} position={[0, 0, index * zSpacing]}>
                      <PointCloudMesh vertices={frame.vertices} disableRotation={true} coloringMode={coloringMode} />
                    </group>
                  ))
                ) : (
                  <PointCloudMesh vertices={engineResults[expanded3dEngineId]?.pointCloud || []} coloringMode={coloringMode} />
                )}
                <AutoFitCamera vertices={accumulate3D ? ((worldFrames[expanded3dEngineId] || [])[0]?.vertices || []) : (engineResults[expanded3dEngineId]?.pointCloud || [])} />
                <OrbitControls enableDamping dampingFactor={0.15} rotateSpeed={0.8} zoomSpeed={1.2} panSpeed={0.6} />
                <ThreeController 
                  flyThrough={flyThrough && accumulate3D} 
                  anaglyph={anaglyph} 
                  zSpacing={zSpacing} 
                  historyLength={(worldFrames[expanded3dEngineId] || []).length} 
                />
              </Canvas>
              <div className="absolute bottom-3 left-3 bg-slate-900/70 px-2.5 py-1 rounded text-[9px] font-mono text-slate-400 pointer-events-none border border-slate-800/60">
                Drag to rotate · Scroll to zoom · Right-click to pan
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DepthLab;
