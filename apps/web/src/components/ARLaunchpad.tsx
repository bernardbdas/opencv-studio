import React, { useState, useEffect, useCallback } from 'react';
import { Sliders, Eye, EyeOff, Image as ImageIcon, Download, Zap, Sparkles, User, Hand, Compass, Maximize2, X, Play, RefreshCw, AlertCircle } from 'lucide-react';
import * as Slider from '@radix-ui/react-slider';
import * as Switch from '@radix-ui/react-switch';
import { OpenCVStudioClient, DemoAsset } from '@opencv-studio/shared';

const client = new OpenCVStudioClient();

export const ARLaunchpad: React.FC = () => {
  const [activeTask, setActiveTask] = useState('finger_frame');
  const [param1, setParam1] = useState(100);
  const [param2, setParam2] = useState(200);

  // Pre-processing
  const [scale, setScale] = useState(0.5); // Default 0.5 for webcam speed
  const [preBlur, setPreBlur] = useState(0);
  const [grayscale, setGrayscale] = useState(false);

  // Visualization styling
  const [overlayColor, setOverlayColor] = useState('cyan');
  const [lineThickness, setLineThickness] = useState(3);
  const [pointRadius, setPointRadius] = useState(4);
  const [showLabels, setShowLabels] = useState(true);

  // Lightbox
  const [fullscreenUrl, setFullscreenUrl] = useState<string | null>(null);
  const [fullscreenTitle, setFullscreenTitle] = useState<string>('');
  const [fullscreenType, setFullscreenType] = useState<'source' | 'output' | null>(null);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [resultUrl, setResultUrl] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const isProcessingRef = React.useRef(false);
  const [livePreview, setLivePreview] = useState(true);
  const [activeTaskName, setActiveTaskName] = useState('AR Finger Portal Frame');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Webcam States
  const [useWebcam, setUseWebcam] = useState(true); // Default to webcam for AR Launchpad!
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);

  // Invisibility Cloak States
  const [invisibilityMode, setInvisibilityMode] = useState<'ai' | 'color'>('ai');
  const [cloakColor, setCloakColor] = useState<'green' | 'red' | 'blue'>('green');
  const [hasCapturedBackground, setHasCapturedBackground] = useState(false);
  const [capturingBg, setCapturingBg] = useState(false);

  // AR Finger Portal Filter States
  const portalFiltersList = ['sketch', 'thermal', 'neon', 'pixel', 'cartoon'];
  const [portalFilter, setPortalFilter] = useState<'sketch' | 'thermal' | 'neon' | 'pixel' | 'cartoon'>('sketch');

  // Phase 3 States
  const [shouldClearAirDraw, setShouldClearAirDraw] = useState(false);
  const [shouldResetPoseTrainer, setShouldResetPoseTrainer] = useState(false);

  const captureBackground = useCallback(() => {
    console.log("[AR Debug] captureBackground clicked. Refs:", { 
      video: videoRef.current, 
      canvas: canvasRef.current 
    });
    if (!videoRef.current || !canvasRef.current) {
      console.warn("[AR Debug] videoRef or canvasRef is null. Exiting.");
      return;
    }
    setCapturingBg(true);
    setErrorMsg(null);
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.warn("[AR Debug] Failed to get 2D context from canvas. Exiting.");
      setCapturingBg(false);
      return;
    }
    
    // Set canvas dimensions matching video feed
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    console.log("[AR Debug] Canvas dimensions set to:", canvas.width, "x", canvas.height);
    
    // Draw current webcam frame onto hidden canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    console.log("[AR Debug] Frame drawn to canvas. Converting to blob...");
    
    canvas.toBlob(async (blob) => {
      console.log("[AR Debug] toBlob completed. Blob:", blob);
      if (!blob) {
        setErrorMsg("Failed to capture background frame from camera feed.");
        setCapturingBg(false);
        return;
      }
      
      const file = new File([blob], 'background.jpg', { type: 'image/jpeg' });
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        console.log("[AR Debug] Uploading background frame to backend...");
        const res = await (client as any).setInvisibilityBackground(formData);
        console.log("[AR Debug] Background upload response:", res);
        if (res.status === 'success') {
          setHasCapturedBackground(true);
          setErrorMsg(null);
          console.log("[AR Debug] Background frame cached successfully.");
        } else {
          setErrorMsg(res.message || "Failed to set background frame.");
          console.warn("[AR Debug] Backend returned error status:", res);
        }
      } catch (err) {
        console.error("[AR Debug] Failed to upload background frame:", err);
        setErrorMsg("Network error when uploading background frame.");
      } finally {
        setCapturingBg(false);
      }
    }, 'image/jpeg', 0.85);
  }, []);

  const [imageCapabilities, setImageCapabilities] = useState<{ hasFace: boolean; hasPose: boolean; hasHands: boolean } | null>(null);

  const analyzeImageContents = useCallback(async (file: File) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      const data = await client.analyzeImage(formData);
      setImageCapabilities({
        hasFace: data.has_face,
        hasPose: data.has_pose,
        hasHands: data.has_hands,
      });
    } catch (err) {
      console.error("Capability analyze error:", err);
      setImageCapabilities(null);
    }
  }, []);

  useEffect(() => {
    if (selectedFile && !useWebcam) {
      analyzeImageContents(selectedFile);
    } else {
      setImageCapabilities(null);
    }
  }, [selectedFile, useWebcam, analyzeImageContents]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setFullscreenUrl(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const selectTask = (task: string) => {
    setActiveTask(task);
    setErrorMsg(null);
    setResultUrl(null);
    if (task === 'finger_frame') {
      setLineThickness(3);
      setPointRadius(4);
      setActiveTaskName('AR Finger Portal Frame');
    } else if (task === 'face_filter') {
      setLineThickness(3);
      setPointRadius(2);
      setActiveTaskName('Cyberpunk Face Visor');
    } else if (task === 'face_tryon') {
      setActiveTaskName('Virtual Try-On Glasses');
    } else if (task === 'pose_trainer') {
      setActiveTaskName('Squat Rep Counter');
    } else if (task === 'air_draw') {
      setActiveTaskName('Index-Pinch Air Drawing');
    } else if (task === 'aruco') {
      setLineThickness(3);
      setActiveTaskName('3D ArUco Marker Projection');
    } else if (task === 'segmentation') {
      setActiveTaskName('AI Background Segmenter');
    } else if (task === 'invisibility_cloak') {
      setActiveTaskName('Invisibility Cloak');
    }
  };

  const runVisionProcessing = useCallback(async () => {
    if (!selectedFile || useWebcam) return;
    setProcessing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const params: Record<string, any> = {
        scale,
        pre_blur: preBlur,
        grayscale,
        overlay_color: overlayColor,
        line_thickness: lineThickness,
        show_labels: showLabels,
      };

      let data: any = null;
      if (activeTask === 'finger_frame') {
        params.portal_filter = portalFilter;
        data = await client.detectFingerFrame(formData, params);
      } else if (activeTask === 'face_filter') {
        params.point_radius = pointRadius;
        data = await client.detectFaceFilter(formData, params);
      } else if (activeTask === 'face_tryon') {
        params.overlay_color = overlayColor;
        data = await (client as any).detectFaceTryon(formData, params);
      } else if (activeTask === 'pose_trainer') {
        data = await (client as any).detectPoseTrainer(formData, params);
      } else if (activeTask === 'air_draw') {
        data = await (client as any).detectAirDraw(formData, { ...params, clear: shouldClearAirDraw });
        if (shouldClearAirDraw) {
          setShouldClearAirDraw(false);
        }
      } else if (activeTask === 'aruco') {
        data = await client.detectArucoProjection(formData, params);
      } else if (activeTask === 'segmentation') {
        data = await client.detectSelfieSegmentation(formData, params);
      } else if (activeTask === 'invisibility_cloak') {
        params.mode = invisibilityMode;
        params.color = cloakColor;
        data = await (client as any).runInvisibilityCloak(formData, params);
      }

      if (data && data.image_base64) {
        setResultUrl(data.image_base64);

        // Detect warnings
        if (activeTask === 'finger_frame' && data.hands_detected === 0) {
          setErrorMsg("AR Finger Frame Warning: No hands detected to construct portal borders. Defaulting to centered frame.");
        } else if (activeTask === 'face_filter' && !data.face_detected) {
          setErrorMsg("Face Filter Warning: No face detected in the image.");
        } else if (activeTask === 'aruco' && !data.marker_detected) {
          setErrorMsg("ArUco Warning: Place a DICT_4X4 ArUco tag in front of the camera.");
        } else if (activeTask === 'segmentation' && !data.person_segmented) {
          setErrorMsg("Segmentation Warning: Selfie background replacement fell back to overlay.");
        } else if (activeTask === 'invisibility_cloak' && data.status === 'error') {
          setErrorMsg(data.message || "Invisibility Cloak Error: No background frame captured.");
        } else {
          setErrorMsg(null);
        }
      }
    } catch (err) {
      console.error("AR processing exception:", err);
    } finally {
      setProcessing(false);
    }
  }, [selectedFile, useWebcam, activeTask, scale, preBlur, grayscale, overlayColor, lineThickness, pointRadius, showLabels, invisibilityMode, cloakColor, portalFilter]);

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
          console.error("Error accessing webcam:", err);
          alert("Could not access webcam. Please check permissions.");
          setUseWebcam(false);
        });
    } else {
      if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
        setWebcamStream(null);
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
      
      // If we are currently processing a frame, wait and check again shortly
      if (isProcessingRef.current) {
        if (active) {
          timer = setTimeout(processFrame, 100);
        }
        return;
      }

      let startedProcessing = false;

      if (useWebcam && webcamStream && videoRef.current && canvasRef.current) {
        if (activeTask === 'invisibility_cloak' && !hasCapturedBackground) {
          setErrorMsg("Please step out of the frame and click 'Capture Background' to initialize the cloak.");
          if (active) {
            timer = setTimeout(processFrame, 180);
          }
          return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (ctx && video.readyState === video.HAVE_ENOUGH_DATA) {
          canvas.width = video.videoWidth || 640;
          canvas.height = video.videoHeight || 480;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          startedProcessing = true;
          isProcessingRef.current = true;
          setProcessing(true);

          canvas.toBlob(async (blob) => {
            if (!blob) {
              console.error("[AR Debug] Failed to convert canvas to blob");
              isProcessingRef.current = false;
              if (active) {
                setProcessing(false);
                timer = setTimeout(processFrame, 180);
              }
              return;
            }
            if (!active) return;
            console.log("[AR Debug] Captured blob successfully. Sending frame to backend for task:", activeTask);
            const file = new File([blob], 'webcam.jpg', { type: 'image/jpeg' });
            const formData = new FormData();
            formData.append('file', file);
            try {
              const params: Record<string, any> = {
                scale,
                pre_blur: preBlur,
                grayscale,
                overlay_color: overlayColor,
                line_thickness: lineThickness,
                show_labels: showLabels,
              };

              let data: any = null;
              if (activeTask === 'finger_frame') {
                params.portal_filter = portalFilter;
                data = await client.detectFingerFrame(formData, params);
                if (data && data.head_gesture && data.head_gesture !== 'none' && active) {
                  console.log("[AR Debug] Head gesture transition triggered! Direction:", data.head_gesture);
                  setPortalFilter((prev) => {
                    const currentIndex = portalFiltersList.indexOf(prev);
                    let nextIndex = currentIndex;
                    if (data.head_gesture === 'right') {
                      nextIndex = (currentIndex + 1) % portalFiltersList.length;
                    } else if (data.head_gesture === 'left') {
                      nextIndex = (currentIndex - 1 + portalFiltersList.length) % portalFiltersList.length;
                    }
                    const nextFilter = portalFiltersList[nextIndex] as any;
                    console.log("[AR Debug] Switching portal filter to:", nextFilter);
                    return nextFilter;
                  });
                }
              } else if (activeTask === 'face_filter') {
                params.point_radius = pointRadius;
                data = await client.detectFaceFilter(formData, params);
              } else if (activeTask === 'face_tryon') {
                data = await (client as any).detectFaceTryon(formData, params);
              } else if (activeTask === 'pose_trainer') {
                data = await (client as any).detectPoseTrainer(formData, { ...params, reset: shouldResetPoseTrainer });
                if (shouldResetPoseTrainer) {
                  setShouldResetPoseTrainer(false);
                }
              } else if (activeTask === 'air_draw') {
                data = await (client as any).detectAirDraw(formData, { ...params, clear: shouldClearAirDraw });
                if (shouldClearAirDraw) {
                  setShouldClearAirDraw(false);
                }
              } else if (activeTask === 'aruco') {
                data = await client.detectArucoProjection(formData, params);
              } else if (activeTask === 'segmentation') {
                data = await client.detectSelfieSegmentation(formData, params);
              } else if (activeTask === 'invisibility_cloak') {
                params.mode = invisibilityMode;
                params.color = cloakColor;
                data = await (client as any).runInvisibilityCloak(formData, params);
              }

              if (data && data.image_base64 && active) {
                console.log("[AR Debug] Received response, status:", data.status, "has image:", !!data.image_base64);
                setResultUrl(data.image_base64);

                // Detect warnings
                if (activeTask === 'finger_frame' && data.hands_detected === 0) {
                  setErrorMsg("AR Finger Frame Warning: No hands detected to construct portal borders.");
                } else if (activeTask === 'face_filter' && !data.face_detected) {
                  setErrorMsg("Face Filter Warning: No face detected in camera frame.");
                } else if (activeTask === 'face_tryon' && !data.face_detected) {
                  setErrorMsg("Face Try-On Warning: Fit your face in front of the camera.");
                } else if (activeTask === 'pose_trainer' && !data.pose_detected) {
                  setErrorMsg("Pose Trainer Warning: Step back to fit your hip, knee, and ankle in the frame.");
                } else if (activeTask === 'air_draw' && !data.hands_detected) {
                  setErrorMsg("Air Drawing Warning: Hold your hand up and pinch index + thumb to paint.");
                } else if (activeTask === 'aruco' && !data.marker_detected) {
                  setErrorMsg("ArUco Warning: Place a DICT_4X4 ArUco tag in front of the camera.");
                } else if (activeTask === 'segmentation' && !data.person_segmented) {
                  setErrorMsg("Segmentation Warning: Selfie background replacement fell back to overlay.");
                } else if (activeTask === 'invisibility_cloak' && data.status === 'error') {
                  setErrorMsg(data.message || "Invisibility Cloak Error: No background frame captured.");
                } else {
                  setErrorMsg(null);
                }
              } else {
                console.warn("[AR Debug] No data returned or task inactive");
              }
            } catch (err) {
              console.error("[AR Debug] Webcam AR frame process error:", err);
            } finally {
              isProcessingRef.current = false;
              if (active) {
                setProcessing(false);
                timer = setTimeout(processFrame, 80); // Quick turnaround once finished!
              }
            }
          }, 'image/jpeg', 0.40);
        }
      }
      
      // If we didn't start the async processing, schedule next frame check in 180ms
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
  }, [useWebcam, webcamStream, activeTask, scale, preBlur, grayscale, overlayColor, lineThickness, pointRadius, showLabels, invisibilityMode, cloakColor, hasCapturedBackground, portalFilter]);

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
      setUseWebcam(false); // Disable webcam if custom file uploaded
    }
  };

  const handleDownloadFrame = () => {
    if (!resultUrl) return;
    const a = document.createElement('a');
    a.href = resultUrl;
    a.download = `ar_launchpad_${activeTask}_frame.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const fingerFrameAllowed = true;
  const faceFilterAllowed = true;
  const arucoAllowed = true;
  const segmentationAllowed = true;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Column */}
        <div className="glass-panel p-6 rounded-2xl space-y-6">
          <div className="flex items-center space-x-3">
            <Sparkles className="w-6 h-6 text-cyan-400" />
            <h3 className="text-xl font-bold text-white">AR Pipeline Selector</h3>
          </div>

          {/* Input Source & Mode Controls */}
          <div className="space-y-1.5">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-wider block">Input Source & Mode</label>
            <div className="flex flex-wrap gap-2">
              {/* Webcam toggle */}
              <button
                onClick={() => setUseWebcam(!useWebcam)}
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



          {/* Active Image Capabilities Summary */}
          {imageCapabilities && !useWebcam && (
            <div className="p-3 bg-slate-955/60 rounded-xl border border-slate-900 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-bold">Image Pre-Scan Detection:</span>
              <div className="flex flex-wrap gap-1.5">
                <span className={`text-[9px] px-2 py-0.5 rounded-full border font-mono ${imageCapabilities.hasFace ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-slate-900 border-slate-800 text-slate-500'}`}>
                  Face: {imageCapabilities.hasFace ? 'Detected' : 'None'}
                </span>
                <span className={`text-[9px] px-2 py-0.5 rounded-full border font-mono ${imageCapabilities.hasPose ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-slate-900 border-slate-800 text-slate-500'}`}>
                  Pose: {imageCapabilities.hasPose ? 'Detected' : 'None'}
                </span>
                <span className={`text-[9px] px-2 py-0.5 rounded-full border font-mono ${imageCapabilities.hasHands ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-slate-900 border-slate-800 text-slate-505'}`}>
                  Hands: {imageCapabilities.hasHands ? 'Detected' : 'None'}
                </span>
              </div>
            </div>
          )}

          {/* AR Tasks */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-400 block">Select AR Filter Effect:</label>
            <div className="grid grid-cols-1 gap-2">
              {[
                { id: 'finger_frame', name: 'AR Finger Portal Frame', icon: Sparkles, desc: 'Pencil-sketch portal mapped between thumbs and index fingers', allowed: fingerFrameAllowed, reason: '' },
                { id: 'face_filter', name: 'Cyberpunk Face Visor', icon: User, desc: 'Dynamic sci-fi glowing neon visor mapped to face structure', allowed: faceFilterAllowed, reason: '' },
                { id: 'face_tryon', name: 'Virtual Try-On Glasses', icon: User, desc: 'Project glowing cyberpunk neon sunglasses onto face landmarks', allowed: true, reason: '' },
                { id: 'pose_trainer', name: 'Squat Rep Counter', icon: Play, desc: 'Workout pose tracker evaluating hip-knee-ankle angles to count squats', allowed: true, reason: '' },
                { id: 'air_draw', name: 'Index-Pinch Air Drawing', icon: Sparkles, desc: 'Pinch index and thumb together to paint glowing lines in the air', allowed: true, reason: '' },
                { id: 'aruco', name: '3D ArUco Marker Projection', icon: Compass, desc: 'Project a 3D wireframe cube in perspective onto DICT_4X4 tag', allowed: arucoAllowed, reason: '' },
                { id: 'segmentation', name: 'AI Background Segmenter', icon: Hand, desc: 'Replaces background with a vector synthwave sunset space grid', allowed: segmentationAllowed, reason: '' },
                { id: 'invisibility_cloak', name: 'Invisibility Cloak', icon: EyeOff, desc: 'Steal the background or key a color to render you completely invisible', allowed: true, reason: '' },
              ].map((item) => {
                const Icon = item.icon;
                const isTaskAllowed = item.allowed;
                return (
                  <button
                    key={item.id}
                    disabled={!isTaskAllowed}
                    onClick={() => selectTask(item.id)}
                    className={`p-3 rounded-xl border text-left flex items-start space-x-3 transition disabled:opacity-30 disabled:cursor-not-allowed ${
                      activeTask === item.id
                        ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                        : 'border-slate-850 bg-slate-900/40 text-slate-400 hover:bg-slate-900'
                    }`}
                    title={isTaskAllowed ? `Select ${item.name}` : item.reason}
                  >
                    <Icon className="w-5 h-5 mt-0.5 text-cyan-400" />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-bold text-slate-200 flex items-center justify-between">
                        <span>{item.name}</span>
                      </div>
                      <div className="text-[10px] text-slate-450 mt-0.5 leading-relaxed">{item.desc}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Pipeline Stage Controls */}
          <div className="space-y-4 pt-4 border-t border-slate-850">
            <span className="text-xs font-bold text-slate-450 uppercase tracking-widest block">Styling & Pre-Processing</span>


            {/* Pose Trainer Settings */}
            {activeTask === 'pose_trainer' && (
              <div className="p-3 bg-slate-955/40 rounded-xl border border-cyan-500/30 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
                <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-1.5">
                  <Play className="w-3.5 h-3.5" />
                  <span>Squat Trainer Info</span>
                </span>
                <p className="text-[10px] text-slate-400 leading-relaxed">
                  Fit your full body (hip, knee, and ankle) in the frame. Perform squats to count repetitions.
                </p>
                <button
                  type="button"
                  onClick={() => setShouldResetPoseTrainer(true)}
                  className="w-full py-2 px-3 rounded-lg border border-red-500/30 bg-red-500/10 hover:bg-red-500/20 text-red-400 text-xs font-bold transition flex items-center justify-center space-x-2"
                >
                  <span>Reset Rep Counter</span>
                </button>
              </div>
            )}

            {/* Air Draw Settings */}
            {activeTask === 'air_draw' && (
              <div className="p-3 bg-slate-955/40 rounded-xl border border-cyan-500/30 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
                <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-1.5">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Air Drawing Info</span>
                </span>
                <p className="text-[10px] text-slate-400 leading-relaxed">
                  Pinch your index finger and thumb together to paint. Release pinch to lift the brush.
                </p>
                <button
                  type="button"
                  onClick={() => setShouldClearAirDraw(true)}
                  className="w-full py-2 px-3 rounded-lg border border-red-500/30 bg-red-500/10 hover:bg-red-500/20 text-red-400 text-xs font-bold transition flex items-center justify-center space-x-2"
                >
                  <span>Clear Drawing Path</span>
                </button>
              </div>
            )}

            {/* Face Try-On Settings */}
            {activeTask === 'face_tryon' && (
              <div className="p-3 bg-slate-955/40 rounded-xl border border-cyan-500/30 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
                <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-1.5">
                  <User className="w-3.5 h-3.5" />
                  <span>Sunglasses Color</span>
                </span>
                <div className="grid grid-cols-3 gap-1.5">
                  {['cyan', 'pink', 'amber'].map((col) => (
                    <button
                      key={col}
                      type="button"
                      onClick={() => setOverlayColor(col)}
                      className={`py-1 rounded-md border text-[9px] font-bold uppercase transition ${
                        overlayColor === col
                          ? 'border-cyan-400 bg-cyan-500/15 text-cyan-300'
                          : 'border-slate-800 bg-slate-900/40 text-slate-400 hover:bg-slate-900'
                      }`}
                    >
                      {col}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Invisibility Cloak Settings */}
            {activeTask === 'invisibility_cloak' && (
              <div className="p-3 bg-slate-955/40 rounded-xl border border-cyan-500/30 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
                <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-1.5">
                  <EyeOff className="w-3.5 h-3.5" />
                  <span>Cloak Settings</span>
                </span>

                {/* Capture Button */}
                <div className="space-y-1.5">
                  <label className="text-[10px] text-slate-400 block">Invisibility Background:</label>
                  <button
                    type="button"
                    onClick={captureBackground}
                    disabled={capturingBg || !useWebcam}
                    className={`w-full py-2 px-3 rounded-lg border text-xs font-bold transition flex items-center justify-center space-x-2 ${
                      !useWebcam
                        ? 'border-slate-800 bg-slate-900/20 text-slate-500 cursor-not-allowed'
                        : hasCapturedBackground
                        ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20'
                        : 'border-cyan-500/30 bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20'
                    }`}
                  >
                    {capturingBg ? (
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Play className="w-3.5 h-3.5" />
                    )}
                    <span>{hasCapturedBackground ? 'Recapture Background' : 'Capture Background'}</span>
                  </button>
                  {!useWebcam && (
                    <p className="text-[9px] text-amber-500 mt-1">Webcam feed must be active to capture background.</p>
                  )}
                  {hasCapturedBackground && (
                    <p className="text-[9px] text-emerald-400 mt-1">✓ Background frame cached successfully.</p>
                  )}
                </div>

                {/* Cloak Mode Selector */}
                <div className="space-y-1.5">
                  <label className="text-[10px] text-slate-400 block">Cloak Detection Method:</label>
                  <div className="grid grid-cols-2 gap-1.5">
                    <button
                      type="button"
                      onClick={() => setInvisibilityMode('ai')}
                      className={`py-1.5 rounded-md border text-[10px] font-bold transition ${
                        invisibilityMode === 'ai'
                          ? 'border-cyan-400 bg-cyan-500/15 text-cyan-300'
                          : 'border-slate-800 bg-slate-900/40 text-slate-400 hover:bg-slate-900'
                      }`}
                    >
                      AI Segmentation
                    </button>
                    <button
                      type="button"
                      onClick={() => setInvisibilityMode('color')}
                      className={`py-1.5 rounded-md border text-[10px] font-bold transition ${
                        invisibilityMode === 'color'
                          ? 'border-cyan-400 bg-cyan-500/15 text-cyan-300'
                          : 'border-slate-800 bg-slate-900/40 text-slate-400 hover:bg-slate-900'
                      }`}
                    >
                      HSV Color Keying
                    </button>
                  </div>
                </div>

                {/* HSV Color Selector */}
                {invisibilityMode === 'color' && (
                  <div className="space-y-1.5 animate-in fade-in slide-in-from-top-1 duration-150">
                    <label className="text-[10px] text-slate-400 block">Target Cloak Color:</label>
                    <div className="grid grid-cols-3 gap-1.5">
                      {['green', 'red', 'blue'].map((col) => (
                        <button
                          key={col}
                          type="button"
                          onClick={() => setCloakColor(col as any)}
                          className={`py-1 rounded-md border text-[9px] font-bold uppercase transition ${
                            cloakColor === col
                              ? 'border-pink-500 bg-pink-500/15 text-pink-300'
                              : 'border-slate-800 bg-slate-900/40 text-slate-400 hover:bg-slate-900'
                          }`}
                        >
                          {col}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Stage 1: Pre-processing */}
            <div className="p-3 bg-slate-955/40 rounded-xl border border-slate-900 space-y-3">
              <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-1">
                <span>Stage 1: Pre-Processing</span>
              </span>

              {/* Scale Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Webcam Scale (Performance Boost):</span>
                  <span className="font-mono text-cyan-350">{Math.round(scale * 100)}%</span>
                </div>
                <Slider.Root
                  className="relative flex items-center select-none touch-none w-full h-3"
                  value={[scale]}
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  onValueChange={(val) => setScale(val[0])}
                >
                  <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                    <Slider.Range className="absolute bg-cyan-500 rounded-full h-full" />
                  </Slider.Track>
                  <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-cyan-400 focus:outline-none" />
                </Slider.Root>
              </div>

              {/* Denoise Blur */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Denoise Blur Kernel:</span>
                  <span className="font-mono text-cyan-355">{preBlur === 0 ? 'Disabled' : `${preBlur}x${preBlur}`}</span>
                </div>
                <Slider.Root
                  className="relative flex items-center select-none touch-none w-full h-3"
                  value={[preBlur]}
                  min={0}
                  max={15}
                  step={2}
                  onValueChange={(val) => setPreBlur(val[0])}
                >
                  <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                    <Slider.Range className="absolute bg-cyan-500 rounded-full h-full" />
                  </Slider.Track>
                  <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-cyan-400 focus:outline-none" />
                </Slider.Root>
              </div>

              {/* Grayscale */}
              <div className="flex items-center justify-between pt-1">
                <span className="text-[11px] text-slate-400">Force Grayscale Conversion:</span>
                <Switch.Root
                  checked={grayscale}
                  onCheckedChange={setGrayscale}
                  className="w-9 h-5 bg-slate-900 rounded-full relative data-[state=checked]:bg-cyan-500 transition cursor-pointer flex items-center px-0.5"
                >
                  <Switch.Thumb className="block w-4 h-4 bg-white rounded-full transition transform translate-x-0 data-[state=checked]:translate-x-4" />
                </Switch.Root>
              </div>
            </div>

            {/* Stage 3: Rendering Styling */}
            <div className="p-3 bg-slate-955/40 rounded-xl border border-slate-900 space-y-3">
              <span className="text-[11px] font-bold text-pink-400 uppercase tracking-wider flex items-center space-x-1">
                <span>Stage 3: Visualization Styling</span>
              </span>

              {/* Overlay Color */}
              <div className="space-y-1.5">
                <label className="text-[11px] text-slate-400 block">Overlay Color Theme:</label>
                <div className="flex flex-wrap gap-1.5">
                  {[
                    { id: 'cyan', hex: '#06b6d4', name: 'Cyan' },
                    { id: 'amber', hex: '#f59e0b', name: 'Amber' },
                    { id: 'green', hex: '#10b981', name: 'Green' },
                    { id: 'pink', hex: '#ec4899', name: 'Pink' },
                    { id: 'blue', hex: '#3b82f6', name: 'Blue' },
                    { id: 'magenta', hex: '#d946ef', name: 'Magenta' },
                    { id: 'white', hex: '#ffffff', name: 'White' },
                  ].map((col) => (
                    <button
                      key={col.id}
                      type="button"
                      onClick={() => setOverlayColor(col.id)}
                      title={col.name}
                      className={`w-6 h-6 rounded-full border transition flex items-center justify-center ${
                        overlayColor === col.id ? 'ring-2 ring-cyan-500/55 border-white' : 'border-slate-800'
                      }`}
                      style={{ backgroundColor: col.hex }}
                    >
                      {overlayColor === col.id && (
                        <span className="w-1.5 h-1.5 rounded-full bg-black" style={{ backgroundColor: col.id === 'white' ? '#000' : '#fff' }} />
                      )}
                    </button>
                  ))}
                </div>
              </div>

              {/* Line Thickness */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Overlay Line Thickness:</span>
                  <span className="font-mono text-pink-300">{lineThickness} px</span>
                </div>
                <Slider.Root
                  className="relative flex items-center select-none touch-none w-full h-3"
                  value={[lineThickness]}
                  min={1}
                  max={8}
                  step={1}
                  onValueChange={(val) => setLineThickness(val[0])}
                >
                  <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                    <Slider.Range className="absolute bg-pink-500 rounded-full h-full" />
                  </Slider.Track>
                  <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-pink-400 focus:outline-none" />
                </Slider.Root>
              </div>

              {/* Point Radius */}
              {activeTask === 'holistic' && (
                <div className="space-y-1.5">
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>Landmark Point Radius:</span>
                    <span className="font-mono text-pink-300">{pointRadius} px</span>
                  </div>
                  <Slider.Root
                    className="relative flex items-center select-none touch-none w-full h-3"
                    value={[pointRadius]}
                    min={1}
                    max={8}
                    step={1}
                    onValueChange={(val) => setPointRadius(val[0])}
                  >
                    <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                      <Slider.Range className="absolute bg-pink-500 rounded-full h-full" />
                    </Slider.Track>
                    <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-pink-400 focus:outline-none" />
                  </Slider.Root>
                </div>
              )}

              {/* Show Labels */}
              <div className="flex items-center justify-between pt-1">
                <span className="text-[11px] text-slate-400">Show Legend / Labels:</span>
                <Switch.Root
                  checked={showLabels}
                  onCheckedChange={setShowLabels}
                  className="w-9 h-5 bg-slate-900 rounded-full relative data-[state=checked]:bg-pink-500 transition cursor-pointer flex items-center px-0.5"
                >
                  <Switch.Thumb className="block w-4 h-4 bg-white rounded-full transition transform translate-x-0 data-[state=checked]:translate-x-4" />
                </Switch.Root>
              </div>
            </div>
          </div>
        </div>

        {/* Media & Output Canvas Column */}
        <div className="lg:col-span-2 glass-panel p-6 rounded-2xl space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div>
              <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                <Eye className="w-5 h-5 text-cyan-400" />
                <span>AR Render Workspace</span>
              </h3>
              {activeTaskName && (
                <div className="text-xs text-cyan-400 font-mono mt-0.5">Active Portal: {activeTaskName}</div>
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

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 min-h-[320px]">
            {/* Source */}
            <div className="bg-slate-950/85 rounded-xl border border-slate-800 p-4 flex items-center justify-center relative group">
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
            <div className="bg-slate-950/85 rounded-xl border border-cyan-500/20 p-4 flex items-center justify-center relative group">
              <div className="absolute top-3 left-3 bg-slate-900/90 px-2.5 py-1 rounded-md text-[10px] font-mono text-cyan-400 border border-slate-800 flex items-center space-x-1">
                {processing && <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />}
                <span>Processed Output</span>
              </div>
              {resultUrl ? (
                <>
                  <img src={resultUrl} alt="Processed AR Output" className="max-h-72 rounded-lg object-contain" />
                  <button
                    onClick={() => {
                      setFullscreenUrl(resultUrl);
                      setFullscreenTitle(`AR Frame: ${activeTask.toUpperCase()}`);
                      setFullscreenType('output');
                    }}
                    className="absolute top-3 right-3 p-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-cyan-400 hover:text-cyan-300 border border-slate-700 transition opacity-0 group-hover:opacity-100 focus:opacity-100"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </button>
                </>
              ) : (
                <div className="text-center text-slate-500 text-sm">
                  AR overlay will appear here
                </div>
              )}
            </div>
          </div>

          {/* AR Finger Portal Filter Dock */}
          {activeTask === 'finger_frame' && (
            <div className="mt-4 p-4 bg-slate-950/80 rounded-xl border border-slate-800/80 space-y-3 animate-in fade-in slide-in-from-bottom-2 duration-200">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center space-x-2">
                  <Sparkles className="w-4 h-4 text-cyan-400" />
                  <span className="text-xs font-bold text-slate-205">Finger Portal Filter Dock</span>
                </div>
                <div className="text-[10px] text-slate-400 flex items-center space-x-1.5 bg-slate-900 px-2.5 py-1 rounded-full border border-slate-850">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  <span>Gesture Control: Extend Left Pinky ⟵ / Right Pinky ⟶ to cycle</span>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                {[
                  { id: 'sketch', name: 'Pencil Sketch', desc: 'Monochrome sketch' },
                  { id: 'thermal', name: 'Thermal Cam', desc: 'Heat-map coloring' },
                  { id: 'neon', name: 'Neon Outline', desc: 'Glowing boundaries' },
                  { id: 'pixel', name: '8-Bit Pixel', desc: 'Retro pixelation' },
                  { id: 'cartoon', name: 'Cartoon', desc: 'Posterized styling' },
                ].map((filt) => {
                  const isActive = portalFilter === filt.id;
                  return (
                    <button
                      key={filt.id}
                      type="button"
                      onClick={() => setPortalFilter(filt.id as any)}
                      className={`p-2.5 rounded-xl border text-center transition-all duration-200 ${
                        isActive
                          ? 'bg-cyan-500/15 border-cyan-400 text-cyan-300 shadow-lg shadow-cyan-500/10 scale-[1.02]'
                          : 'border-slate-850 bg-slate-900/35 text-slate-450 hover:text-slate-200 hover:bg-slate-900 hover:border-slate-800'
                      }`}
                    >
                      <div className="text-[11px] font-bold block">{filt.name}</div>
                      <div className="text-[9px] text-slate-500 mt-0.5 truncate">{filt.desc}</div>
                    </button>
                  );
                })}
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
                  a.download = `ar_launchpad_fullscreen.jpg`;
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
                alt="Fullscreen Processed AR Output" 
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
