import React, { useState, useEffect, useCallback } from 'react';
import { Sliders, Eye, Image as ImageIcon, Play, Sparkles, Folder, Download, Zap, User, Tag, Maximize2, X, RefreshCw, AlertCircle, Cpu } from 'lucide-react';
import * as Slider from '@radix-ui/react-slider';
import * as Switch from '@radix-ui/react-switch';
import { OpenCVStudioClient, DemoAsset } from '@opencv-studio/shared';

const client = new OpenCVStudioClient();

export const YoloStudio: React.FC = () => {
  const [activeTask, setActiveTask] = useState('detect'); // 'detect', 'segment', 'pose'
  const [yoloVersion, setYoloVersion] = useState<string>('v8'); // 'v8', 'v9', 'v10', 'v11'
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);
  const [showLabels, setShowLabels] = useState(true);

  // Fullscreen Lightbox states
  const [fullscreenUrl, setFullscreenUrl] = useState<string | null>(null);
  const [fullscreenTitle, setFullscreenTitle] = useState<string>('');
  const [fullscreenType, setFullscreenType] = useState<'source' | 'output' | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setFullscreenUrl(null);
        setFullscreenType(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const isProcessingRef = React.useRef(false);
  const [livePreview, setLivePreview] = useState(true);
  const [activeTaskName, setActiveTaskName] = useState('YOLO Object Detection');
  const [detectionsList, setDetectionsList] = useState<{ class_name: string; confidence: number; bbox?: [number, number, number, number] }[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Performance diagnostics states
  const [speedMetrics, setSpeedMetrics] = useState<{ preprocess: number; inference: number; postprocess: number } | null>(null);
  const [fps, setFps] = useState<number>(0);
  const [e2eLatency, setE2eLatency] = useState<number>(0);
  const lastFrameTimeRef = React.useRef<number>(0);

  // Demo Assets
  const [demoCatalog, setDemoCatalog] = useState<DemoAsset[]>([]);
  const [activeDemoKey, setActiveDemoKey] = useState<string | null>(null);
  const [isDemoLibraryOpen, setIsDemoLibraryOpen] = useState(true);
  const [activeDemoCategory, setActiveDemoCategory] = useState<string>('Face & Hands');

  // YOLO Intelligence states
  const [geofenceActive, setGeofenceActive] = useState(false);
  const [geofencePoints, setGeofencePoints] = useState<{ x: number; y: number }[]>([]);
  const [isDrawingGeofence, setIsDrawingGeofence] = useState(false);
  const [intrusionCount, setIntrusionCount] = useState(0);
  const [hasIntrusion, setHasIntrusion] = useState(false);

  const [speedActive, setSpeedActive] = useState(false);
  const [speedCalibration, setSpeedCalibration] = useState(1.2);

  const [heatmapActive, setHeatmapActive] = useState(false);
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.6);

  // Refs for tracking and drawing overlays
  const overlayCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const heatmapCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const tracksRef = React.useRef<Record<string, { centroid: { x: number; y: number }; lastSeen: number; speed: number; consecutiveFrames: number; class_name: string }>>({});
  const lastTrackTimeRef = React.useRef<number>(0);
  const nextTrackIdRef = React.useRef<number>(1);

  // Webcam states/refs
  const [useWebcam, setUseWebcam] = useState(false);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);

  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);

  const toggleWebcam = (val: boolean) => {
    setUseWebcam(val);
    if (val) {
      setResultUrl(null);
      setErrorMsg(null);
      setDetectionsList([]);
      setSpeedMetrics(null);
      setFps(0);
      setE2eLatency(0);
      lastFrameTimeRef.current = 0;
    }
  };

  // Fetch demo assets on mount
  useEffect(() => {
    const loadDemoAssets = async () => {
      try {
        const assets = await client.getDemoAssets();
        setDemoCatalog(assets);
        // Load default asset
        if (assets.length > 0) {
          const defaultAsset = assets.find((a: any) => a.key === 'object_detection') || assets[0];
          selectDemoAsset(defaultAsset.key);
        }
      } catch (err) {
        console.error("Failed to load demo assets catalog:", err);
      }
    };
    loadDemoAssets();
  }, []);

  const selectDemoAsset = async (key: string) => {
    try {
      setProcessing(true);
      setActiveDemoKey(key);
      const asset = await client.getDemoAsset(key);
      
      // Convert base64 data url directly to File
      const res = await fetch(asset.image_base64);
      const blob = await res.blob();
      const file = new File([blob], `${key}.jpg`, { type: 'image/jpeg' });
      
      setSelectedFile(file);
      setPreviewUrl(asset.image_base64);
      setUseWebcam(false); // turn off webcam
      setErrorMsg(null);
    } catch (err) {
      console.error("Failed to select demo asset:", err);
      setErrorMsg("Failed to download selected demo asset.");
    } finally {
      setProcessing(false);
    }
  };

  const selectTask = (task: string) => {
    setActiveTask(task);
    if (task === 'detect') {
      setActiveTaskName('YOLO Object Detection');
    } else if (task === 'segment') {
      setActiveTaskName('YOLO Instance Segmentation');
    } else if (task === 'pose') {
      setActiveTaskName('YOLO Human Pose Estimation');
    }
    setResultUrl(null);
    setDetectionsList([]);
    setErrorMsg(null);
  };

  const handleVersionChange = (version: string) => {
    setYoloVersion(version);
    if (['v10', 'v6', 'world'].includes(version) && activeTask !== 'detect') {
      selectTask('detect');
    }
    setResultUrl(null);
    setDetectionsList([]);
    setErrorMsg(null);
    setSpeedMetrics(null);
    setE2eLatency(0);
  };

  // ── YOLO Intelligence Helpers & Analytics ───────────────────────

  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (heatmapActive && heatmapCanvasRef.current) {
      ctx.globalAlpha = heatmapOpacity;
      ctx.drawImage(heatmapCanvasRef.current, 0, 0, w, h);
      ctx.globalAlpha = 1.0;
    }

    if (geofencePoints.length > 0) {
      ctx.lineWidth = Math.max(3, w * 0.004);
      ctx.strokeStyle = hasIntrusion ? '#f43f5e' : '#10b981';
      ctx.fillStyle = hasIntrusion ? 'rgba(244, 63, 94, 0.15)' : 'rgba(16, 185, 129, 0.08)';

      ctx.beginPath();
      ctx.moveTo(geofencePoints[0].x * w, geofencePoints[0].y * h);
      for (let i = 1; i < geofencePoints.length; i++) {
        ctx.lineTo(geofencePoints[i].x * w, geofencePoints[i].y * h);
      }
      if (geofencePoints.length > 2) {
        ctx.closePath();
        ctx.fill();
      }
      ctx.stroke();

      if (isDrawingGeofence) {
        ctx.fillStyle = '#67e8f9';
        for (const pt of geofencePoints) {
          ctx.beginPath();
          ctx.arc(pt.x * w, pt.y * h, Math.max(6, w * 0.007), 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    if (speedActive && detectionsList.length > 0) {
      ctx.fillStyle = '#ffffff';
      ctx.font = `bold ${Math.max(10, Math.round(w * 0.012))}px monospace`;
      
      for (const det of detectionsList) {
        if (!det.bbox) continue;
        const [x1, y1, x2, y2] = det.bbox;
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        
        const matchedTrack = Object.values(tracksRef.current).find(t => {
          const d = Math.sqrt((t.centroid.x - cx) ** 2 + (t.centroid.y - cy) ** 2);
          return d < 80 && t.class_name === det.class_name;
        });

        if (matchedTrack && matchedTrack.speed > 5 && matchedTrack.consecutiveFrames > 3) {
          const speedText = `${Math.round(matchedTrack.speed)} MPH`;
          const textY = Math.max(y1 - 10, 20);
          ctx.shadowColor = 'black';
          ctx.shadowBlur = 4;
          ctx.fillText(speedText, x1 + 5, textY);
          ctx.shadowBlur = 0;
        }
      }
    }
  }, [geofencePoints, hasIntrusion, isDrawingGeofence, heatmapActive, heatmapOpacity, speedActive, detectionsList]);

  const isPointInPolygon = (point: { x: number; y: number }, vs: { x: number; y: number }[]) => {
    const x = point.x, y = point.y;
    let inside = false;
    for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
      const xi = vs[i].x, yi = vs[i].y;
      const xj = vs[j].x, yj = vs[j].y;
      const intersect = ((yi > y) !== (yj > y))
          && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  };

  const processYoloAnalytics = useCallback((detections: any[]) => {
    const now = performance.now();
    const dt = lastTrackTimeRef.current > 0 ? (now - lastTrackTimeRef.current) / 1000 : 0.1;
    lastTrackTimeRef.current = now;

    if (speedActive && dt < 1.5) {
      const activeIds = new Set<string>();
      
      for (const det of detections) {
        if (!det.bbox) continue;
        const [x1, y1, x2, y2] = det.bbox;
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        
        let bestTrackId: string | null = null;
        let minDist = 80;
        
        for (const [tid, t] of Object.entries(tracksRef.current)) {
          if (t.class_name !== det.class_name) continue;
          const d = Math.sqrt((t.centroid.x - cx) ** 2 + (t.centroid.y - cy) ** 2);
          if (d < minDist) {
            minDist = d;
            bestTrackId = tid;
          }
        }
        
        if (bestTrackId) {
          const t = tracksRef.current[bestTrackId];
          const dist = Math.sqrt((cx - t.centroid.x) ** 2 + (cy - t.centroid.y) ** 2);
          const rawSpeed = (dist / dt) * 0.15 * speedCalibration;
          
          t.centroid = { x: cx, y: cy };
          t.speed = t.speed * 0.8 + rawSpeed * 0.2;
          t.lastSeen = now;
          t.consecutiveFrames += 1;
          activeIds.add(bestTrackId);
        } else {
          const newId = String(nextTrackIdRef.current++);
          tracksRef.current[newId] = {
            centroid: { x: cx, y: cy },
            lastSeen: now,
            speed: 0,
            consecutiveFrames: 1,
            class_name: det.class_name
          };
          activeIds.add(newId);
        }
      }
      
      for (const [tid, t] of Object.entries(tracksRef.current)) {
        if (!activeIds.has(tid) && (now - t.lastSeen > 1000)) {
          delete tracksRef.current[tid];
        }
      }
    } else {
      tracksRef.current = {};
    }

    let detectedIntrusions = 0;
    if (geofenceActive && geofencePoints.length > 2 && overlayCanvasRef.current) {
      const w = overlayCanvasRef.current.width;
      const h = overlayCanvasRef.current.height;
      
      for (const det of detections) {
        if (!det.bbox) continue;
        const [x1, y1, x2, y2] = det.bbox;
        const bx = (x1 + x2) / 2;
        const by = y2;
        
        const npx = bx / w;
        const npy = by / h;
        
        if (isPointInPolygon({ x: npx, y: npy }, geofencePoints)) {
          detectedIntrusions++;
        }
      }
    }
    setIntrusionCount(detectedIntrusions);
    setHasIntrusion(detectedIntrusions > 0);

    if (heatmapActive && heatmapCanvasRef.current) {
      const hCanvas = heatmapCanvasRef.current;
      const hCtx = hCanvas.getContext('2d');
      if (hCtx) {
        hCtx.globalCompositeOperation = 'destination-out';
        hCtx.fillStyle = 'rgba(0, 0, 0, 0.025)';
        hCtx.fillRect(0, 0, hCanvas.width, hCanvas.height);
        
        hCtx.globalCompositeOperation = 'screen';
        for (const det of detections) {
          if (!det.bbox) continue;
          const [x1, y1, x2, y2] = det.bbox;
          const cx = (x1 + x2) / 2;
          const cy = (y1 + y2) / 2;
          
          const grad = hCtx.createRadialGradient(cx, cy, 2, cx, cy, 25);
          grad.addColorStop(0, 'rgba(0, 255, 255, 0.2)');
          grad.addColorStop(0.4, 'rgba(0, 128, 255, 0.1)');
          grad.addColorStop(1, 'rgba(0, 0, 255, 0)');
          hCtx.fillStyle = grad;
          hCtx.beginPath();
          hCtx.arc(cx, cy, 25, 0, Math.PI * 2);
          hCtx.fill();
        }
      }
    }

    drawOverlay();
  }, [geofenceActive, geofencePoints, heatmapActive, speedActive, speedCalibration, drawOverlay]);

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    
    if (overlayCanvasRef.current) {
      overlayCanvasRef.current.width = w;
      overlayCanvasRef.current.height = h;
    }
    if (heatmapCanvasRef.current) {
      if (heatmapCanvasRef.current.width !== w || heatmapCanvasRef.current.height !== h) {
        heatmapCanvasRef.current.width = w;
        heatmapCanvasRef.current.height = h;
        const ctx = heatmapCanvasRef.current.getContext('2d');
        if (ctx) {
          ctx.fillStyle = '#000000';
          ctx.fillRect(0, 0, w, h);
        }
      }
    }
    drawOverlay();
  };

  const handleOutputContainerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDrawingGeofence || !overlayCanvasRef.current) return;
    
    const container = e.currentTarget;
    const img = container.querySelector('img');
    if (!img) return;

    const rect = img.getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const screenY = e.clientY - rect.top;

    const npx = screenX / rect.width;
    const npy = screenY / rect.height;

    if (npx >= 0 && npx <= 1 && npy >= 0 && npy <= 1) {
      setGeofencePoints(prev => [...prev, { x: npx, y: npy }]);
    }
  };

  useEffect(() => {
    drawOverlay();
  }, [geofencePoints, hasIntrusion, isDrawingGeofence, heatmapActive, heatmapOpacity, speedActive, drawOverlay]);

  // Core vision request execution
  const runVisionProcessing = useCallback(async () => {
    if (!selectedFile || useWebcam) return;
    setProcessing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    const startTime = performance.now();
    try {
      const params = {
        version: yoloVersion,
        conf: confThreshold,
        iou: iouThreshold,
        show_labels: showLabels,
      };

      let data: any = null;
      if (activeTask === 'detect') {
        data = await client.detectYoloObjects(formData, params);
      } else if (activeTask === 'segment') {
        data = await client.segmentYoloObjects(formData, params);
      } else if (activeTask === 'pose') {
        data = await client.estimateYoloPose(formData, params);
      }

      if (data && data.status === 'success') {
        const latency = performance.now() - startTime;
        setE2eLatency(Math.round(latency));
        if (data.speed) {
          setSpeedMetrics(data.speed);
        }
        setResultUrl(data.image_base64);
        if (data.detections) {
          setDetectionsList(data.detections);
          processYoloAnalytics(data.detections);
        } else {
          setDetectionsList([]);
          processYoloAnalytics([]);
        }
        setErrorMsg(null);
      } else {
        setErrorMsg(data?.detail || "YOLO processing returned error status.");
      }
    } catch (err) {
      console.error("YOLO processing exception:", err);
      setErrorMsg("Failed to connect to YOLO API. Make sure backend is running.");
    } finally {
      setProcessing(false);
    }
  }, [selectedFile, useWebcam, activeTask, confThreshold, iouThreshold, showLabels, yoloVersion, processYoloAnalytics]);

  // Turn webcam on / off
  useEffect(() => {
    if (useWebcam) {
      navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
        .then((stream) => {
          setWebcamStream(stream);
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.play();
          }
        })
        .catch((err) => {
          console.error("Failed to acquire webcam video track:", err);
          setErrorMsg("Failed to open webcam. Please verify camera permissions.");
          setUseWebcam(false);
        });
    } else {
      if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
        setWebcamStream(null);
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
    }

    return () => {
      if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [useWebcam]);

  // Webcam processing loop
  useEffect(() => {
    let active = true;
    let timer: any = null;

    const processFrame = async () => {
      if (!active) return;
      if (isProcessingRef.current) {
        if (active) timer = setTimeout(processFrame, 100);
        return;
      }

      let startedProcessing = false;

      if (useWebcam && webcamStream && videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        if (ctx && video.readyState === video.HAVE_ENOUGH_DATA) {
          // Process at a lightweight 480x360 or 640x480 resolution for YOLO performance
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          startedProcessing = true;
          isProcessingRef.current = true;
          setProcessing(true);

          canvas.toBlob(async (blob) => {
            if (!blob) {
              isProcessingRef.current = false;
              if (active) {
                setProcessing(false);
                timer = setTimeout(processFrame, 180);
              }
              return;
            }
            if (!active) return;
            
            const file = new File([blob], 'webcam.jpg', { type: 'image/jpeg' });
            const formData = new FormData();
            formData.append('file', file);

            const startTime = performance.now();
            try {
              const params = {
                version: yoloVersion,
                conf: confThreshold,
                iou: iouThreshold,
                show_labels: showLabels,
              };

              let data: any = null;
              if (activeTask === 'detect') {
                data = await client.detectYoloObjects(formData, params);
              } else if (activeTask === 'segment') {
                data = await client.segmentYoloObjects(formData, params);
              } else if (activeTask === 'pose') {
                data = await client.estimateYoloPose(formData, params);
              }

              if (data && data.status === 'success' && active) {
                const latency = performance.now() - startTime;
                setE2eLatency(Math.round(latency));
                if (data.speed) {
                  setSpeedMetrics(data.speed);
                }
                const now = performance.now();
                if (lastFrameTimeRef.current > 0) {
                  const currentFps = 1000 / (now - lastFrameTimeRef.current);
                  setFps(prev => Math.round(prev * 0.85 + currentFps * 0.15));
                }
                lastFrameTimeRef.current = now;

                setResultUrl(data.image_base64);
                if (data.detections) {
                  setDetectionsList(data.detections);
                  processYoloAnalytics(data.detections);
                } else if (activeTask === 'pose' && data.people_detected !== undefined) {
                  const fakeDets = Array.from({ length: data.people_detected }).map(() => ({ class_name: 'person', confidence: 1.0 }));
                  setDetectionsList(fakeDets);
                  processYoloAnalytics(fakeDets);
                } else {
                  setDetectionsList([]);
                  processYoloAnalytics([]);
                }
                setErrorMsg(null);
              }
            } catch (err) {
              console.error("YOLO live loop exception:", err);
            } finally {
              isProcessingRef.current = false;
              if (active) {
                setProcessing(false);
                timer = setTimeout(processFrame, 150); // YOLO runs slightly heavier, wait 150ms before next frame
              }
            }
          }, 'image/jpeg', 0.55);
        }
      }

      if (!startedProcessing && active) {
        timer = setTimeout(processFrame, 180);
      }
    };

    if (useWebcam && webcamStream) {
      timer = setTimeout(processFrame, 500);
    }

    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [useWebcam, webcamStream, activeTask, confThreshold, iouThreshold, showLabels, yoloVersion]);

  // Live real-time preview trigger for static images
  useEffect(() => {
    if (livePreview && selectedFile && !useWebcam) {
      const timer = setTimeout(() => {
        runVisionProcessing();
      }, 150);
      return () => clearTimeout(timer);
    }
    return () => {};
  }, [livePreview, selectedFile, useWebcam, runVisionProcessing]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setUseWebcam(false); // disable webcam
    }
  };

  const handleDownloadFrame = () => {
    if (!resultUrl) return;
    const a = document.createElement('a');
    a.href = resultUrl;
    a.download = `yolo_${activeTask}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="space-y-6">
      {/* Studio Test Asset Catalog */}
      <div className="border border-slate-850 rounded-2xl overflow-hidden bg-slate-900/10 mb-6">
        <button
          onClick={() => setIsDemoLibraryOpen(!isDemoLibraryOpen)}
          className="w-full px-5 py-4 bg-slate-900/40 flex items-center justify-between text-left text-xs font-bold text-slate-205 border-b border-slate-850 hover:bg-slate-900/60 transition"
        >
          <div className="flex items-center space-x-2">
            <Folder className="w-4.5 h-4.5 text-cyan-400" />
            <span>Studio Test Asset Catalog</span>
          </div>
          <div className="flex items-center space-x-3">
            <span className="text-[10px] bg-slate-900 border border-slate-800 text-slate-400 font-medium px-2.5 py-0.5 rounded-full font-mono">
              {Array.isArray(demoCatalog) && demoCatalog.length > 0 ? `${demoCatalog.length} samples available` : 'Demo library offline'}
            </span>
            <span className="text-[10px] text-slate-400 font-mono">
              {isDemoLibraryOpen ? 'Hide' : 'Show'} Library
            </span>
          </div>
        </button>

        {isDemoLibraryOpen && (
          <div className="p-4 bg-slate-955/20 space-y-4 animate-in fade-in slide-in-from-top-2 duration-200">
            {/* Category Tabs */}
            <div className="flex space-x-1 p-1 bg-slate-950/80 rounded-xl border border-slate-900 max-w-xl">
              {['Face & Hands', 'Body & Pose', 'Objects & Scenes', 'OCR & Translation'].map((cat) => (
                <button
                  key={cat}
                  onClick={() => setActiveDemoCategory(cat)}
                  className={`flex-1 py-1.5 px-3 rounded-lg text-[10px] font-bold transition ${
                    activeDemoCategory === cat
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {Array.isArray(demoCatalog) && demoCatalog.length > 0 ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {demoCatalog
                  .filter((asset) => asset.category === activeDemoCategory)
                  .map((asset) => (
                    <button
                      key={asset.key}
                      onClick={() => selectDemoAsset(asset.key)}
                      className={`p-2.5 bg-slate-900/60 border rounded-xl hover:bg-slate-900/90 text-center transition space-y-1 flex flex-col items-center justify-center group ${
                        activeDemoKey === asset.key ? 'border-cyan-400 bg-cyan-500/5' : 'border-slate-800'
                      }`}
                    >
                      <span className="text-[10px] font-bold text-slate-350 truncate w-full group-hover:text-cyan-300 transition">
                        {asset.title}
                      </span>
                      <span className="text-[8px] font-mono text-slate-500 block uppercase">
                        {asset.description || asset.category}
                      </span>
                    </button>
                  ))}
              </div>
            ) : (
              <div className="text-xs text-slate-400 py-2">
                Preset demo catalog requires backend connection.
              </div>
            )}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Column */}
        <div className="glass-panel p-6 rounded-2xl space-y-6">
          <div className="flex items-center space-x-3">
            <Cpu className="w-6 h-6 text-cyan-400" />
            <h3 className="text-xl font-bold text-white font-mono">YOLO Model Selector</h3>
          </div>

          {/* Input Source & Mode Controls */}
          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Input Source & Mode</label>
            <div className="flex flex-wrap gap-2">
              {/* Webcam toggle */}
              <button
                onClick={() => toggleWebcam(!useWebcam)}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold border transition ${
                  useWebcam
                    ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
                    : 'bg-slate-900/60 border-slate-800 text-slate-500 hover:text-slate-300'
                }`}
              >
                <Eye className="w-3 h-3" />
                <span>Webcam</span>
                <span className={`w-1.5 h-1.5 rounded-full ${useWebcam ? 'bg-cyan-400 animate-pulse' : 'bg-slate-700'}`} />
              </button>
            </div>
          </div>



          {/* YOLO Version Selector */}
          <div className="space-y-2 pt-2 border-t border-slate-800/60">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block font-mono">YOLO Engine Version</label>
            <div className="grid grid-cols-2 gap-2">
              {[
                { id: 'v11', name: 'YOLOv11', desc: 'SOTA High-Efficiency' },
                { id: 'v10', name: 'YOLOv10', desc: 'End-to-End (NMS-free)' },
                { id: 'v9', name: 'YOLOv9', desc: 'PGI Feature Conserv' },
                { id: 'v8', name: 'YOLOv8', desc: 'Legacy Generalist' },
                { id: 'world', name: 'YOLO-World', desc: 'Open-Vocabulary Zero-Shot' },
                { id: 'v6', name: 'YOLOv6', desc: 'Meituan Industrial Edge' },
                { id: 'v5', name: 'YOLOv5', desc: 'Legendary PyTorch baseline' },
              ].map((v) => (
                <button
                  key={v.id}
                  onClick={() => handleVersionChange(v.id)}
                  className={`p-2 rounded-xl border text-left transition flex flex-col justify-between ${
                    yoloVersion === v.id
                      ? 'bg-cyan-500/25 border-cyan-400 text-cyan-300'
                      : 'border-slate-850 bg-slate-900/35 text-slate-450 hover:bg-slate-900'
                  }`}
                >
                  <span className="text-[11px] font-bold font-mono">{v.name}</span>
                  <span className="text-[8px] text-slate-500 mt-0.5 leading-tight">{v.desc}</span>
                </button>
              ))}
            </div>
          </div>

          {/* YOLO Tasks */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-450 block font-mono">Select YOLO Model Task:</label>
            <div className="grid grid-cols-1 gap-2">
              {[
                { id: 'detect', name: `Object Detection (${yoloVersion.toUpperCase()})`, desc: 'Detect 80 classes of objects with clean vector bounding boxes' },
                { id: 'segment', name: `Instance Segmentation (${yoloVersion.toUpperCase()}-seg)`, desc: 'Generate pixel-precise overlay masks for individual objects', disabled: ['v10', 'v6', 'world'].includes(yoloVersion) },
                { id: 'pose', name: `Human Pose Estimation (${yoloVersion.toUpperCase()}-pose)`, desc: 'Extract 17 keypoint skeletal tracks mapped on detected bodies', disabled: ['v10', 'v6', 'world'].includes(yoloVersion) },
              ].map((item) => (
                <button
                  key={item.id}
                  onClick={() => selectTask(item.id)}
                  disabled={item.disabled}
                  className={`p-3 rounded-xl border text-left flex items-start space-x-3 transition ${
                    item.disabled
                      ? 'opacity-40 cursor-not-allowed border-slate-900 bg-slate-950/20 text-slate-600'
                      : activeTask === item.id
                      ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                      : 'border-slate-850 bg-slate-900/35 text-slate-400 hover:bg-slate-900'
                  }`}
                >
                  <Cpu className={`w-5 h-5 mt-0.5 ${item.disabled ? 'text-slate-700' : 'text-cyan-455'}`} />
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-bold text-slate-200 flex items-center justify-between">
                      <span>{item.name}</span>
                      {item.disabled && <span className="text-[8px] px-1.5 py-0.5 bg-slate-900 text-slate-500 rounded border border-slate-800">Unsupported</span>}
                    </div>
                    <div className="text-[10px] text-slate-450 mt-0.5 leading-relaxed">{item.desc}</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Parameters sliders */}
          <div className="space-y-4 pt-4 border-t border-slate-800">
            <span className="text-xs font-bold text-slate-450 uppercase tracking-widest block font-mono">Inference Hyperparameters</span>

            {/* Confidence Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-[11px] text-slate-400">
                <span>Confidence Score Cutoff:</span>
                <span className="font-mono text-cyan-350">{Math.round(confThreshold * 100)}%</span>
              </div>
              <Slider.Root
                className="relative flex items-center select-none touch-none w-full h-3"
                value={[confThreshold]}
                min={0.05}
                max={0.95}
                step={0.05}
                onValueChange={(val) => setConfThreshold(val[0])}
              >
                <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                  <Slider.Range className="absolute bg-cyan-500 rounded-full h-full" />
                </Slider.Track>
                <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-cyan-400 focus:outline-none" />
              </Slider.Root>
            </div>

            {/* Overlap / IoU Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-[11px] text-slate-400">
                <span>NMS Overlap (IoU) Threshold:</span>
                <span className="font-mono text-cyan-350">{Math.round(iouThreshold * 100)}%</span>
              </div>
              <Slider.Root
                className="relative flex items-center select-none touch-none w-full h-3"
                value={[iouThreshold]}
                min={0.05}
                max={0.95}
                step={0.05}
                onValueChange={(val) => setIouThreshold(val[0])}
              >
                <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                  <Slider.Range className="absolute bg-cyan-500 rounded-full h-full" />
                </Slider.Track>
                <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-cyan-400 focus:outline-none" />
              </Slider.Root>
            </div>

            {/* Show Labels switch */}
            {activeTask === 'detect' && (
              <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800/80 flex items-center justify-between">
                <div>
                  <div className="text-xs font-semibold text-slate-200 flex items-center space-x-1.5">
                    <Tag className="w-3.5 h-3.5 text-cyan-400" />
                    <span>Show Class Labels</span>
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">Render class text tag</div>
                </div>
                <Switch.Root
                  checked={showLabels}
                  onCheckedChange={setShowLabels}
                  className="w-11 h-6 bg-slate-800 rounded-full relative data-[state=checked]:bg-cyan-500 transition cursor-pointer flex items-center px-0.5"
                >
                  <Switch.Thumb className="block w-5 h-5 bg-white rounded-full transition transform translate-x-0 data-[state=checked]:translate-x-5" />
                </Switch.Root>
              </div>
            )}
          </div>

          {/* YOLO Intelligence Add-ons */}
          <div className="space-y-4 pt-4 border-t border-slate-800 font-mono">
            <span className="text-xs font-bold text-slate-450 uppercase tracking-widest block">YOLO Intelligence</span>

            {/* Geofencing Card */}
            <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800/80 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-200">Intrusion Zone (Geofence)</span>
                <Switch.Root
                  checked={geofenceActive}
                  onCheckedChange={(val) => {
                    setGeofenceActive(val);
                    if (!val) {
                      setHasIntrusion(false);
                      setIntrusionCount(0);
                    }
                  }}
                  className="w-9 h-5 bg-slate-800 rounded-full relative data-[state=checked]:bg-cyan-500 transition cursor-pointer flex items-center px-0.5"
                >
                  <Switch.Thumb className="block w-4 h-4 bg-white rounded-full transition transform translate-x-0 data-[state=checked]:translate-x-4" />
                </Switch.Root>
              </div>

              {geofenceActive && (
                <div className="space-y-2 animate-in fade-in slide-in-from-top-1 duration-150">
                  <div className="flex space-x-2">
                    <button
                      onClick={() => setIsDrawingGeofence(!isDrawingGeofence)}
                      className={`flex-1 py-1.5 rounded-lg text-[10px] font-bold border transition ${
                        isDrawingGeofence 
                          ? 'bg-cyan-500/20 text-cyan-300 border-cyan-400' 
                          : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200'
                      }`}
                    >
                      {isDrawingGeofence ? 'Click Output to Place Points' : 'Draw Polygon'}
                    </button>
                    <button
                      onClick={() => {
                        setGeofencePoints([]);
                        setHasIntrusion(false);
                        setIntrusionCount(0);
                        setIsDrawingGeofence(false);
                      }}
                      className="px-2 py-1.5 bg-slate-955 text-rose-400 border border-slate-800 rounded-lg text-[10px] font-bold hover:text-rose-350 transition"
                    >
                      Clear
                    </button>
                  </div>
                  {geofencePoints.length > 0 && (
                    <div className="flex items-center justify-between text-[9px] text-slate-400 border-t border-slate-850 pt-1.5">
                      <span>{geofencePoints.length} vertices</span>
                      <span className={`font-bold ${hasIntrusion ? 'text-rose-400 animate-pulse' : 'text-emerald-400'}`}>
                        {hasIntrusion ? `${intrusionCount} INTRUSIONS!` : 'Zone Secure'}
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Speed Estimation Card */}
            <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800/80 space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-200">Vehicle Speed Tracker</span>
                <Switch.Root
                  checked={speedActive}
                  onCheckedChange={setSpeedActive}
                  className="w-9 h-5 bg-slate-800 rounded-full relative data-[state=checked]:bg-cyan-500 transition cursor-pointer flex items-center px-0.5"
                >
                  <Switch.Thumb className="block w-4 h-4 bg-white rounded-full transition transform translate-x-0 data-[state=checked]:translate-x-4" />
                </Switch.Root>
              </div>
              {speedActive && (
                <div className="space-y-1.5 animate-in fade-in slide-in-from-top-1 duration-150">
                  <div className="flex justify-between text-[9px] text-slate-400">
                    <span>Calibration Scale:</span>
                    <span className="text-cyan-300">{speedCalibration.toFixed(1)}x</span>
                  </div>
                  <Slider.Root
                    className="relative flex items-center select-none touch-none w-full h-3"
                    value={[speedCalibration]}
                    min={0.5}
                    max={3.0}
                    step={0.1}
                    onValueChange={(val) => setSpeedCalibration(val[0])}
                  >
                    <Slider.Track className="bg-slate-900 relative grow rounded-full h-0.5">
                      <Slider.Range className="absolute bg-cyan-500 rounded-full h-full" />
                    </Slider.Track>
                    <Slider.Thumb className="block w-2.5 h-2.5 bg-white rounded-full border border-cyan-400 focus:outline-none" />
                  </Slider.Root>
                </div>
              )}
            </div>

            {/* Heatmap Card */}
            <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800/80 space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-200">Traffic Heatmap</span>
                <Switch.Root
                  checked={heatmapActive}
                  onCheckedChange={(val) => {
                    setHeatmapActive(val);
                    if (val && heatmapCanvasRef.current) {
                      const ctx = heatmapCanvasRef.current.getContext('2d');
                      if (ctx) {
                        ctx.fillStyle = '#000000';
                        ctx.fillRect(0, 0, heatmapCanvasRef.current.width, heatmapCanvasRef.current.height);
                      }
                    }
                  }}
                  className="w-9 h-5 bg-slate-800 rounded-full relative data-[state=checked]:bg-cyan-500 transition cursor-pointer flex items-center px-0.5"
                >
                  <Switch.Thumb className="block w-4 h-4 bg-white rounded-full transition transform translate-x-0 data-[state=checked]:translate-x-4" />
                </Switch.Root>
              </div>
              {heatmapActive && (
                <div className="space-y-2 animate-in fade-in slide-in-from-top-1 duration-150">
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[9px] text-slate-400">
                      <span>Heatmap Opacity:</span>
                      <span className="text-cyan-300">{Math.round(heatmapOpacity * 100)}%</span>
                    </div>
                    <Slider.Root
                      className="relative flex items-center select-none touch-none w-full h-3"
                      value={[heatmapOpacity]}
                      min={0.1}
                      max={0.9}
                      step={0.05}
                      onValueChange={(val) => setHeatmapOpacity(val[0])}
                    >
                      <Slider.Track className="bg-slate-900 relative grow rounded-full h-0.5">
                        <Slider.Range className="absolute bg-cyan-500 rounded-full h-full" />
                      </Slider.Track>
                      <Slider.Thumb className="block w-2.5 h-2.5 bg-white rounded-full border border-cyan-400 focus:outline-none" />
                    </Slider.Root>
                  </div>
                  <button
                    onClick={() => {
                      if (heatmapCanvasRef.current) {
                        const ctx = heatmapCanvasRef.current.getContext('2d');
                        if (ctx) {
                          ctx.fillStyle = '#000000';
                          ctx.fillRect(0, 0, heatmapCanvasRef.current.width, heatmapCanvasRef.current.height);
                          drawOverlay();
                        }
                      }
                    }}
                    className="w-full py-1 bg-slate-950 text-cyan-400 border border-slate-800 rounded-lg text-[9px] font-bold hover:text-cyan-300 transition"
                  >
                    Reset Heatmap
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Display Column */}
        <div className="lg:col-span-2 glass-panel p-6 rounded-2xl space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div>
              <h3 className="text-xl font-bold text-white flex items-center space-x-2 font-mono">
                <Cpu className="w-5 h-5 text-cyan-400" />
                <span>YOLO Inference Playground</span>
              </h3>
              {activeTaskName && (
                <div className="text-xs text-cyan-400 font-mono mt-0.5">Pipeline: {activeTaskName}</div>
              )}
            </div>

            <div className="flex items-center space-x-3">
              {resultUrl && (
                <button
                  onClick={handleDownloadFrame}
                  className="flex items-center space-x-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs text-cyan-300 rounded-xl transition font-medium"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Download Frame</span>
                </button>
              )}

              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="text-xs text-slate-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-xl file:border-0 file:text-xs file:font-semibold file:bg-slate-800 file:text-slate-300 hover:file:bg-slate-700 cursor-pointer"
              />
            </div>
          </div>

          {/* Error / Warning Alert Box */}
          {errorMsg && (
            <div className="p-3.5 bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-xl text-xs flex items-center space-x-2 shadow-sm animate-in fade-in slide-in-from-top-2 duration-200">
              <AlertCircle className="w-4.5 h-4.5 text-rose-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Source and output side-by-side */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 min-h-[320px]">
            {/* Source */}
            <div className="bg-slate-955/85 rounded-xl border border-slate-800 p-4 flex items-center justify-center relative group">
              <div className="absolute top-3 left-3 bg-slate-900/90 px-2.5 py-1 rounded-md text-[10px] font-mono text-slate-400 border border-slate-800">
                Source Input
              </div>
              {useWebcam ? (
                <div className="relative w-full h-full max-h-72 rounded-lg overflow-hidden flex items-center justify-center bg-black">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="max-h-72 w-full object-contain rounded-lg transform -scale-x-100"
                  />
                  <canvas ref={canvasRef} className="hidden" />
                  <button
                    onClick={() => {
                      setFullscreenUrl('webcam');
                      setFullscreenTitle('Source Input (Webcam)');
                      setFullscreenType('source');
                    }}
                    className="absolute top-3 right-3 p-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-700 transition opacity-0 group-hover:opacity-100 focus:opacity-100 animate-in fade-in"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </button>
                </div>
              ) : previewUrl ? (
                <>
                  <img src={previewUrl} alt="Source Input" className="max-h-72 rounded-lg object-contain" />
                  <button
                    onClick={() => {
                      setFullscreenUrl(previewUrl);
                      setFullscreenTitle('Source Input Frame');
                      setFullscreenType('source');
                    }}
                    className="absolute top-3 right-3 p-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-700 transition opacity-0 group-hover:opacity-100 focus:opacity-100"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </button>
                </>
              ) : (
                <div className="text-center text-slate-500 text-sm space-y-2">
                  <ImageIcon className="w-8 h-8 mx-auto text-slate-600" />
                  <div>Select demo asset or upload image</div>
                </div>
              )}
            </div>

            {/* Output */}
            <div 
              onClick={handleOutputContainerClick}
              className={`bg-slate-955/85 rounded-xl border p-4 flex items-center justify-center relative group transition ${
                isDrawingGeofence ? 'border-cyan-500/50 cursor-crosshair' : 'border-cyan-500/20'
              }`}
            >
              <div className="absolute top-3 left-3 bg-slate-900/90 px-2.5 py-1 rounded-md text-[10px] font-mono text-cyan-400 border border-slate-800 flex items-center space-x-1 z-20">
                {processing && <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />}
                <span>Processed YOLO Output</span>
              </div>
              {resultUrl ? (
                <div className="relative flex items-center justify-center w-full h-full max-h-72">
                  <img src={resultUrl} alt="Processed YOLO Output" className="max-h-72 rounded-lg object-contain" onLoad={handleImageLoad} />
                  <canvas ref={overlayCanvasRef} className="absolute max-h-72 object-contain pointer-events-none rounded-lg z-10" />
                  <canvas ref={heatmapCanvasRef} className="hidden" />
                  <button
                    onClick={() => {
                      setFullscreenUrl(resultUrl);
                      setFullscreenTitle(`YOLO Output: ${activeTask.toUpperCase()}`);
                      setFullscreenType('output');
                    }}
                    className="absolute top-3 right-3 p-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-cyan-400 hover:text-cyan-300 border border-slate-700 transition opacity-0 group-hover:opacity-100 focus:opacity-100 z-20"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </button>
                </div>
              ) : (
                <div className="text-center text-slate-500 text-sm">
                  YOLO overlay will appear here
                </div>
              )}
            </div>
          </div>

          {/* Diagnostics Panel */}
          {(speedMetrics || e2eLatency > 0 || (useWebcam && fps > 0)) && (
            <div className="p-4 bg-slate-950/70 rounded-2xl border border-slate-800/80 space-y-3 font-mono">
              <div className="text-xs font-bold text-slate-300 flex items-center space-x-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                <span>Real-Time Performance Diagnostics</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-[10px]">
                {/* Inference time */}
                {speedMetrics && (
                  <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800 space-y-1">
                    <span className="text-slate-500 uppercase tracking-wider block text-[8px] font-bold">Inference Latency</span>
                    <span className="text-sm font-extrabold text-cyan-400">
                      {speedMetrics.inference.toFixed(1)} <span className="text-[10px] font-medium text-slate-500">ms</span>
                    </span>
                  </div>
                )}
                
                {/* FPS (Webcam only) */}
                {useWebcam && (
                  <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800 space-y-1">
                    <span className="text-slate-500 uppercase tracking-wider block text-[8px] font-bold">Streaming Performance</span>
                    <span className="text-sm font-extrabold text-emerald-400 flex items-center space-x-1.5">
                      <span>{fps || 30}</span> 
                      <span className="text-[10px] font-medium text-slate-500">FPS</span>
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping inline-block" />
                    </span>
                  </div>
                )}

                {/* Pre / Post process */}
                {speedMetrics && (
                  <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800 space-y-1">
                    <span className="text-slate-500 uppercase tracking-wider block text-[8px] font-bold">Pre / Post Process</span>
                    <span className="text-[10px] font-semibold text-slate-300">
                      {speedMetrics.preprocess.toFixed(1)} ms / {speedMetrics.postprocess.toFixed(1)} ms
                    </span>
                  </div>
                )}

                {/* Roundtrip Latency */}
                {e2eLatency > 0 && (
                  <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800 space-y-1">
                    <span className="text-slate-500 uppercase tracking-wider block text-[8px] font-bold">End-to-End Latency</span>
                    <span className="text-sm font-extrabold text-pink-400">
                      {e2eLatency} <span className="text-[10px] font-medium text-slate-500">ms</span>
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Detections Summary List */}
          {detectionsList.length > 0 && (
            <div className="p-4 bg-slate-950/70 rounded-2xl border border-slate-800/80 space-y-3">
              <div className="text-xs font-bold text-slate-300 flex items-center space-x-2">
                <Tag className="w-4 h-4 text-cyan-400" />
                <span>Detected Classes & Metrics ({detectionsList.length})</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {detectionsList.map((item, idx) => (
                  <span
                    key={`${item.class_name}_${idx}`}
                    className="text-[10px] px-2.5 py-1 bg-cyan-500/10 border border-cyan-500/20 text-cyan-300 rounded-full font-mono font-medium"
                  >
                    {item.class_name} {item.confidence < 1.0 ? `(${Math.round(item.confidence * 100)}%)` : ''}
                  </span>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>

      {/* Lightbox */}
      {fullscreenUrl && (
        <div 
          className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/95 backdrop-blur-md p-4 animate-in fade-in duration-200"
          onClick={() => {
            setFullscreenUrl(null);
            setFullscreenType(null);
          }}
        >
          <div className="absolute top-4 left-6 flex items-center space-x-3 text-slate-350">
            <span className="text-xs font-mono uppercase bg-slate-900/85 border border-slate-800 px-3 py-1 rounded-full text-cyan-400">
              {fullscreenTitle}
            </span>
          </div>

          <div className="absolute top-4 right-4 flex items-center space-x-3">
            {!(fullscreenType === 'source' && useWebcam) && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  const a = document.createElement('a');
                  a.href = (fullscreenType === 'output' && useWebcam) ? (resultUrl || '') : (fullscreenUrl || '');
                  a.download = `yolo_fullscreen.jpg`;
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                }}
                className="p-2 bg-slate-900/90 hover:bg-slate-800 text-cyan-400 hover:text-cyan-300 rounded-full border border-slate-800 transition shadow-lg"
                title="Download Current Frame"
              >
                <Download className="w-5 h-5" />
              </button>
            )}
            <button
              onClick={(e) => {
                e.stopPropagation();
                setFullscreenUrl(null);
                setFullscreenType(null);
              }}
              className="p-2 bg-slate-900/90 hover:bg-rose-950/50 text-slate-400 hover:text-rose-400 rounded-full border border-slate-800 hover:border-rose-900/50 transition shadow-lg"
              title="Close Fullscreen"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div 
            className="w-full max-w-5xl h-[80vh] flex items-center justify-center p-2"
            onClick={(e) => e.stopPropagation()}
          >
            {fullscreenType === 'source' && useWebcam ? (
              <video
                ref={(el) => {
                  if (el && videoRef.current && el.srcObject !== videoRef.current.srcObject) {
                    el.srcObject = videoRef.current.srcObject;
                  }
                }}
                autoPlay
                playsInline
                muted
                className="w-full h-full rounded-2xl object-contain transform -scale-x-100 border border-slate-800 shadow-2xl select-none"
              />
            ) : fullscreenType === 'output' && useWebcam ? (
              <img 
                src={resultUrl || ''} 
                alt="Fullscreen Processed YOLO Output" 
                className="w-full h-full rounded-2xl object-contain border border-slate-800 shadow-2xl select-none"
              />
            ) : (
              <img 
                src={fullscreenUrl || ''} 
                alt="Fullscreen Preview" 
                className="w-full h-full rounded-2xl object-contain border border-slate-800 shadow-2xl select-none animate-in zoom-in-95 duration-200"
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};
