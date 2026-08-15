import React, { useState, useEffect, useCallback } from 'react';
import { Sliders, Eye, Image as ImageIcon, Play, Sparkles, Folder, Download, Zap, UserCheck, FileText, User, Tag, Hand, Compass, Maximize2, X, RefreshCw, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import * as Slider from '@radix-ui/react-slider';
import * as Switch from '@radix-ui/react-switch';
import { OpenCVStudioClient, DemoAsset } from '@opencv-studio/shared';

const client = new OpenCVStudioClient();

export const VisionStudio: React.FC = () => {
  const [activeGroup, setActiveGroup] = useState<'classic' | 'mediapipe'>('classic');
  const [activeTask, setActiveTask] = useState('canny');
  const [param1, setParam1] = useState(100);
  const [param2, setParam2] = useState(200);
  
  // Stage 1: Pre-processing controls
  const [scale, setScale] = useState(1.0);
  const [preBlur, setPreBlur] = useState(0);
  const [grayscale, setGrayscale] = useState(true);

  // Stage 3: Styling & Visualization controls
  const [overlayColor, setOverlayColor] = useState('cyan');
  const [lineThickness, setLineThickness] = useState(3);
  const [pointRadius, setPointRadius] = useState(6);
  const [showLabels, setShowLabels] = useState(true);

  // Fullscreen Lightbox states
  const [fullscreenUrl, setFullscreenUrl] = useState<string | null>(null);
  const [fullscreenTitle, setFullscreenTitle] = useState<string>('');
  const [fullscreenType, setFullscreenType] = useState<'source' | 'output' | 'anpr' | null>(null);

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
  const [activeTaskName, setActiveTaskName] = useState('Canny Edge Filter');
  const [categoriesList, setCategoriesList] = useState<{ category_name: string; score: number }[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [ocrResult, setOcrResult] = useState<{ extractedText: string; detectedLanguage: string; translation: string | null } | null>(null);

  // Demo Assets
  const [demoCatalog, setDemoCatalog] = useState<DemoAsset[]>([]);
  const [activeDemoKey, setActiveDemoKey] = useState<string | null>(null);
  const [isDemoLibraryOpen, setIsDemoLibraryOpen] = useState(true);

  const [activeDemoCategory, setActiveDemoCategory] = useState<string>('Face & Hands');
  const [useTrafficAnpr, setUseTrafficAnpr] = useState(false);

  // Webcam & Capability states/refs
  const [imageCapabilities, setImageCapabilities] = useState<{ hasFace: boolean; hasPose: boolean; hasHands: boolean } | null>(null);
  const [useWebcam, setUseWebcam] = useState(false);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);

  const toggleWebcam = (val: boolean) => {
    setUseWebcam(val);
    if (val) {
      setUseTrafficAnpr(false);
      setResultUrl(null);
      setErrorMsg(null);
    }
  };

  const toggleTrafficAnpr = (val: boolean) => {
    setUseTrafficAnpr(val);
    if (val) {
      setUseWebcam(false);
      setResultUrl(null);
      setErrorMsg(null);
      setActiveTaskName('Real-Time License Plate ANPR Feed');
    }
  };
  const videoRef = React.useRef<HTMLVideoElement | null>(null);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);

  // Map demo asset keys to the best-matching pipeline task
  const demoKeyToTask: Record<string, { group: 'classic' | 'mediapipe'; task: string }> = {
    face_landmark:              { group: 'mediapipe', task: 'facemesh' },
    face_detection:             { group: 'mediapipe', task: 'facemesh' },
    hand_landmark:              { group: 'mediapipe', task: 'gesture' },
    gesture_recognition:        { group: 'mediapipe', task: 'gesture' },
    holistic_landmark:          { group: 'mediapipe', task: 'holistic' },
    pose_landmark:              { group: 'mediapipe', task: 'pose' },
    object_detection:           { group: 'mediapipe', task: 'objects' },
    image_segmentation:         { group: 'mediapipe', task: 'classify' },
    interactive_segmentation:   { group: 'mediapipe', task: 'objects' },
    classic_cv:                 { group: 'classic',   task: 'canny' },
    image_embedding_1:          { group: 'mediapipe', task: 'classify' },
    image_embedding_2:          { group: 'mediapipe', task: 'classify' },
    ocr_spanish:                { group: 'mediapipe', task: 'ocr' },
    ocr_french:                 { group: 'mediapipe', task: 'ocr' },
    ocr_japanese:               { group: 'mediapipe', task: 'ocr' },
  };

  // Analyze image contents dynamically (Haar cascades, skin-tone HSV heuristics)
  const analyzeImageContents = useCallback(async (file: File) => {
    if (activeDemoKey) {
      const overrides: Record<string, any> = {
        // Face & Hands
        face_landmark:            { has_face: true,  has_pose: false, has_hands: false },
        face_detection:           { has_face: true,  has_pose: false, has_hands: false },
        hand_landmark:            { has_face: false, has_pose: false, has_hands: true },
        gesture_recognition:      { has_face: false, has_pose: false, has_hands: true },
        holistic_landmark:        { has_face: true,  has_pose: true,  has_hands: true },
        // Body & Pose
        pose_landmark:            { has_face: true,  has_pose: true,  has_hands: true },
        // Objects & Scenes
        object_detection:         { has_face: false, has_pose: false, has_hands: false },
        image_segmentation:       { has_face: true,  has_pose: true,  has_hands: true },
        interactive_segmentation: { has_face: false, has_pose: false, has_hands: false },
        classic_cv:               { has_face: false, has_pose: false, has_hands: false },
        image_embedding_1:        { has_face: false, has_pose: false, has_hands: false },
        image_embedding_2:        { has_face: false, has_pose: false, has_hands: false },
        // OCR & Translation
        ocr_spanish:              { has_face: false, has_pose: false, has_hands: false },
        ocr_french:               { has_face: false, has_pose: false, has_hands: false },
        ocr_japanese:             { has_face: false, has_pose: false, has_hands: false },
      };
      const override = overrides[activeDemoKey];
      if (override) {
        setImageCapabilities({
          hasFace: override.has_face,
          hasPose: override.has_pose,
          hasHands: override.has_hands,
        });
        return;
      }
    }

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
  }, [activeDemoKey]);

  // Run capability analysis when selectedFile changes
  useEffect(() => {
    if (selectedFile && !useWebcam) {
      analyzeImageContents(selectedFile);
    } else {
      setImageCapabilities(null);
    }
  }, [selectedFile, useWebcam, analyzeImageContents]);

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

  // Webcam frame processing loop
  useEffect(() => {
    let active = true;
    let timer: any = null;

    const processFrame = async () => {
      if (!active) return;

      // Synchronously check if we are already processing a frame
      if (isProcessingRef.current) {
        if (active) {
          timer = setTimeout(processFrame, 100);
        }
        return;
      }

      let startedProcessing = false;

      if (useWebcam && webcamStream && videoRef.current && canvasRef.current) {
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
              console.error("[Webcam Debug] Failed to convert canvas to blob");
              isProcessingRef.current = false;
              if (active) {
                setProcessing(false);
                timer = setTimeout(processFrame, 180);
              }
              return;
            }
            if (!active) return;
            console.log("[Webcam Debug] Captured blob successfully. Sending frame to backend for task:", activeTask);
            const file = new File([blob], 'webcam.jpg', { type: 'image/jpeg' });
            const formData = new FormData();
            formData.append('file', file);
            try {
              let data: any = null;
              if (activeGroup === 'classic') {
                const queryParams = new URLSearchParams({
                  filter_type: activeTask,
                  param1: param1.toString(),
                  param2: param2.toString(),
                  scale: scale.toString(),
                  pre_blur: preBlur.toString(),
                  grayscale: grayscale.toString(),
                });
                const res = await fetch(`/api/vision/classic?${queryParams.toString()}`, {
                  method: 'POST',
                  body: formData
                });
                data = await res.json();
                if (res.ok && data.image_base64 && active) {
                  setResultUrl(data.image_base64);
                }
              } else {
                const params: Record<string, any> = {
                  scale,
                  pre_blur: preBlur,
                  grayscale,
                  overlay_color: overlayColor,
                  line_thickness: lineThickness,
                  show_labels: showLabels,
                };
                if (activeTask === 'pose') {
                  params.joint_radius = pointRadius;
                  data = await client.detectPose(formData, params);
                } else if (activeTask === 'ocr') {
                  data = await client.detectOCR(formData, params);
                } else if (activeTask === 'facemesh') {
                  params.point_radius = pointRadius;
                  data = await client.detectFaceMesh(formData, params);
                } else if (activeTask === 'classify') {
                  data = await client.classifyImage(formData);
                } else if (activeTask === 'gesture') {
                  data = await client.recognizeGesture(formData, params);
                } else if (activeTask === 'holistic') {
                  params.point_radius = pointRadius;
                  data = await client.detectHolistic(formData, params);
                } else if (activeTask === 'objects') {
                  params.min_area = param1;
                  data = await (client as any).detectObjects(formData, params);
                } else if (activeTask === 'finger_frame') {
                  data = await client.detectFingerFrame(formData, params);
                }

                if (data && active) {
                  console.log("[Webcam Debug] Received response, status:", data.status, "has image:", !!data.image_base64);
                  if (data.image_base64) {
                    setResultUrl(data.image_base64);
                  }
                  if (data.categories) {
                    setCategoriesList(data.categories);
                  } else {
                    setCategoriesList([]);
                  }

                  // Check for warnings or empty detections
                  if (activeTask === 'pose' && data.landmarks_detected === 0) {
                    setErrorMsg("Pose Tracking Warning: No skeletal structure detected in camera frame.");
                  } else if (activeTask === 'facemesh' && data.mesh_points_count === 0) {
                    setErrorMsg("Face Mesh Warning: No human face structure detected in camera frame.");
                  } else if (activeTask === 'holistic' && data.face_mesh_points === 0 && data.pose_landmarks === 0 && data.hand_landmarks === 0) {
                    setErrorMsg("Holistic Warning: No face, pose, or hand landmarks detected in camera frame.");
                  } else if (activeTask === 'gesture' && data.gesture_count === 0 && data.hand_count === 0) {
                    setErrorMsg("Hand Gesture Warning: No hands detected in camera frame.");
                  } else if (activeTask === 'ocr' && data.text_regions === 0) {
                    setErrorMsg("OCR Warning: No text regions detected in camera frame.");
                  } else {
                    setErrorMsg(null);
                  }
                } else {
                  console.warn("[Webcam Debug] No data returned or task inactive");
                }
              }
            } catch (err) {
              console.error("[Webcam Debug] Webcam frame process error:", err);
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
  }, [useWebcam, webcamStream, activeGroup, activeTask, param1, param2, scale, preBlur, grayscale, overlayColor, lineThickness, pointRadius, showLabels]);

  useEffect(() => {
    if (typeof client?.getDemoAssets === 'function') {
      client.getDemoAssets()
        .then((data) => {
          if (Array.isArray(data)) {
            setDemoCatalog(data);
          } else {
            setDemoCatalog([]);
          }
        })
        .catch((err) => {
          console.error("Failed to load demo assets catalog:", err);
          setDemoCatalog([]);
        });
    } else {
      setDemoCatalog([]);
    }
  }, []);

  const selectTask = (group: 'classic' | 'mediapipe', task: string) => {
    setActiveGroup(group);
    setActiveTask(task);
    
    // Set reasonable defaults based on the chosen task
    if (group === 'classic') {
      setGrayscale(true);
      if (task === 'canny') {
        setParam1(100);
        setParam2(200);
        setActiveTaskName('Canny Edge Filter');
      } else if (task === 'threshold') {
        setParam1(127);
        setParam2(255);
        setActiveTaskName('Binary Threshold');
      } else if (task === 'harris') {
        setParam1(2);
        setParam2(2);
        setActiveTaskName('Harris Corners');
      } else if (task === 'blur') {
        setParam1(5);
        setParam2(1);
        setActiveTaskName('Gaussian Blur');
      }
    } else {
      setGrayscale(false);
      if (task === 'pose') {
        setPointRadius(6);
        setLineThickness(3);
        setActiveTaskName('Pose Skeleton Tracking');
      } else if (task === 'ocr') {
        setActiveTaskName('OCR Text Region extraction');
      } else if (task === 'facemesh') {
        setPointRadius(2);
        setActiveTaskName('Face Mesh 3D');
      } else if (task === 'classify') {
        setActiveTaskName('MediaPipe Image Classification');
      } else if (task === 'gesture') {
        setActiveTaskName('Hand Gesture Recognition');
      } else if (task === 'holistic') {
        setActiveTaskName('Holistic Pipeline (Face + Hand + Pose)');
      } else if (task === 'objects') {
        setParam1(500); // min_area default
        setActiveTaskName('MediaPipe Object Detection');
      } else if (task === 'finger_frame') {
        setLineThickness(3);
        setPointRadius(4);
        setActiveTaskName('AR Finger Portal Frame');
      }
    }
  };

  const runVisionProcessing = useCallback(async () => {
    if (!selectedFile) return;
    setProcessing(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      let data: any = null;
      if (activeGroup === 'classic') {
        const queryParams = new URLSearchParams({
          filter_type: activeTask,
          param1: param1.toString(),
          param2: param2.toString(),
          scale: scale.toString(),
          pre_blur: preBlur.toString(),
          grayscale: grayscale.toString(),
        });
        const res = await fetch(`/api/vision/classic?${queryParams.toString()}`, {
          method: 'POST',
          body: formData
        });
        data = await res.json();
        if (res.ok && data.image_base64) {
          setResultUrl(data.image_base64);
          setCategoriesList([]);
        }
      } else {
        // MediaPipe / ML Tasks
        const params: Record<string, any> = {
          scale,
          pre_blur: preBlur,
          grayscale,
          overlay_color: overlayColor,
          line_thickness: lineThickness,
          show_labels: showLabels,
        };

        if (activeTask === 'pose') {
          params.joint_radius = pointRadius;
          data = await client.detectPose(formData, params);
        } else if (activeTask === 'ocr') {
          data = await client.detectOCR(formData, params);
        } else if (activeTask === 'facemesh') {
          params.point_radius = pointRadius;
          data = await client.detectFaceMesh(formData, params);
        } else if (activeTask === 'classify') {
          data = await client.classifyImage(formData);
        } else if (activeTask === 'gesture') {
          data = await client.recognizeGesture(formData, params);
        } else if (activeTask === 'holistic') {
          params.point_radius = pointRadius;
          data = await client.detectHolistic(formData, params);
        } else if (activeTask === 'objects') {
          params.min_area = param1;
          data = await (client as any).detectObjects(formData, params);
        } else if (activeTask === 'finger_frame') {
          data = await client.detectFingerFrame(formData, params);
        }

        if (data) {
          if (data.image_base64) {
            setResultUrl(data.image_base64);
          }
          if (data.categories) {
            setCategoriesList(data.categories);
          } else {
            setCategoriesList([]);
          }

          // Capture OCR results for display panel
          if (activeTask === 'ocr' && data.extracted_text !== undefined) {
            setOcrResult({
              extractedText: data.extracted_text || '',
              detectedLanguage: data.detected_language || 'Unknown',
              translation: data.translation || null,
            });
          } else {
            setOcrResult(null);
          }

          // Check for warnings or empty detections
          if (activeTask === 'pose' && data.landmarks_detected === 0) {
            setErrorMsg("Pose Tracking Warning: No skeletal structure detected in the image. Try another picture with a person standing.");
          } else if (activeTask === 'facemesh' && data.mesh_points_count === 0) {
            setErrorMsg("Face Mesh Warning: No human face structure detected in the image. Try another picture with a face visible.");
          } else if (activeTask === 'holistic' && data.face_mesh_points === 0 && data.pose_landmarks === 0 && data.hand_landmarks === 0) {
            setErrorMsg("Holistic Warning: No face, pose, or hand landmarks detected in the image.");
          } else if (activeTask === 'gesture' && data.gesture_count === 0 && data.hand_count === 0) {
            setErrorMsg("Hand Gesture Warning: No hands detected in the image. Try an image with visible hands.");
          } else if (activeTask === 'ocr' && data.text_regions === 0) {
            setErrorMsg("OCR Warning: No text regions detected in the image. Try an image with readable text.");
          } else if (data.error) {
            setErrorMsg(`Pipeline Error: ${data.error}`);
          } else {
            setErrorMsg(null);
          }
        }
      }
    } catch (err) {
      console.error("Vision processing exception:", err);
    } finally {
      setProcessing(false);
    }
  }, [selectedFile, activeGroup, activeTask, param1, param2, scale, preBlur, grayscale, overlayColor, lineThickness, pointRadius, showLabels]);

  useEffect(() => {
    if (livePreview && selectedFile) {
      const timer = setTimeout(() => {
        runVisionProcessing();
      }, 150);
      return () => clearTimeout(timer);
    }
    return () => {};
  }, [livePreview, selectedFile, runVisionProcessing]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setActiveDemoKey(null);
      setResultUrl(null);
      setCategoriesList([]);
      setOcrResult(null);
    }
  };

  const handleSelectDemoAsset = async (asset: DemoAsset) => {
    setActiveDemoKey(asset.key);
    setErrorMsg(null);
    setUseTrafficAnpr(false);

    // Auto-switch to the best-matching pipeline task for this demo asset
    const mapping = demoKeyToTask[asset.key];
    if (mapping) {
      selectTask(mapping.group, mapping.task);
    }

    try {
      if (typeof client?.getDemoAsset === 'function') {
        const data = await client.getDemoAsset(asset.key);
        if (data?.image_base64) {
          setPreviewUrl(data.image_base64);
          setResultUrl(null);
          setCategoriesList([]);
          setOcrResult(null);

          // Convert base64 payload to File object
          const res = await fetch(data.image_base64);
          const blob = await res.blob();
          const file = new File([blob], asset.filename, { type: 'image/jpeg' });
          setSelectedFile(file);
        }
      }
    } catch (err) {
      console.error('Error fetching demo asset:', err);
    }
  };

  const handleDownloadFrame = () => {
    if (!resultUrl) return;
    const a = document.createElement('a');
    a.href = resultUrl;
    a.download = `opencv_studio_${activeTask}_frame.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const poseAllowed = true;
  const faceAllowed = true;
  const handsAllowed = true;
  const holisticAllowed = true;

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
          <div className="p-4 bg-slate-950/20 space-y-4 animate-in fade-in slide-in-from-top-2 duration-200">
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
                      onClick={() => handleSelectDemoAsset(asset)}
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
                Preset demo catalog requires backend connection. Start FastAPI backend with <code className="text-cyan-400 font-mono">just start-backend</code> or upload a custom image below.
              </div>
            )}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls Column */}
        <div className="glass-panel p-6 rounded-2xl space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Sliders className="w-6 h-6 text-cyan-400" />
              <h3 className="text-xl font-bold text-white">OpenCV & MediaPipe Controls</h3>
            </div>
          </div>

          {/* Input Source & Mode Controls — Compact toolbar */}
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

              {/* ANPR Stream toggle */}
              <button
                onClick={() => toggleTrafficAnpr(!useTrafficAnpr)}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-[11px] font-semibold border transition ${
                  useTrafficAnpr
                    ? 'bg-purple-500/15 border-purple-500/40 text-purple-300'
                    : 'bg-slate-900/60 border-slate-800 text-slate-500 hover:text-slate-300'
                }`}
              >
                <Play className="w-3 h-3" />
                <span>ANPR Stream</span>
                <span className={`w-1.5 h-1.5 rounded-full ${useTrafficAnpr ? 'bg-purple-400 animate-pulse' : 'bg-slate-700'}`} />
              </button>
            </div>
          </div>



          {/* Active Image Capabilities Summary */}
          {imageCapabilities && !useWebcam && !useTrafficAnpr && (
            <div className="p-3 bg-slate-955/60 rounded-xl border border-slate-900 space-y-1">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider block font-bold">Image Pre-Scan Detection:</span>
              <div className="flex flex-wrap gap-1.5">
                <span className={`text-[9px] px-2 py-0.5 rounded-full border font-mono ${imageCapabilities.hasFace ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-slate-900 border-slate-800 text-slate-500'}`}>
                  Face: {imageCapabilities.hasFace ? 'Detected' : 'None'}
                </span>
                <span className={`text-[9px] px-2 py-0.5 rounded-full border font-mono ${imageCapabilities.hasPose ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-slate-900 border-slate-800 text-slate-500'}`}>
                  Pose: {imageCapabilities.hasPose ? 'Detected' : 'None'}
                </span>
                <span className={`text-[9px] px-2 py-0.5 rounded-full border font-mono ${imageCapabilities.hasHands ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-slate-900 border-slate-800 text-slate-500'}`}>
                  Hands: {imageCapabilities.hasHands ? 'Detected' : 'None'}
                </span>
              </div>
            </div>
          )}

          {/* Level 1: Pipeline Group Selection */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-400 block">Select Pipeline Type:</label>
            <div className="flex bg-slate-950/80 p-1.5 rounded-xl border border-slate-800">
              <button
                onClick={() => selectTask('classic', 'canny')}
                className={`flex-1 py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center space-x-1.5 transition ${activeGroup === 'classic' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400 hover:text-white'}`}
              >
                <Sliders className="w-3.5 h-3.5" />
                <span>Classic OpenCV</span>
              </button>
              <button
                onClick={() => selectTask('mediapipe', 'pose')}
                className={`flex-1 py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center space-x-1.5 transition ${activeGroup === 'mediapipe' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40' : 'text-slate-400 hover:text-white'}`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>MediaPipe AI</span>
              </button>
            </div>
          </div>

          {/* Level 2: Dynamic Task Selection Grid */}
          {activeGroup === 'classic' ? (
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-400 block">Select Classic CV Operation:</label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { id: 'canny', name: 'Canny Edge' },
                  { id: 'threshold', name: 'Binary Threshold' },
                  { id: 'harris', name: 'Harris Corners' },
                  { id: 'blur', name: 'Gaussian Blur' },
                ].map((mode) => (
                  <button
                    key={mode.id}
                    onClick={() => selectTask('classic', mode.id)}
                    className={`py-2 px-2.5 rounded-xl text-xs font-medium border transition ${
                      activeTask === mode.id
                        ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-md shadow-cyan-500/10'
                        : 'border-slate-850 bg-slate-900/40 text-slate-400 hover:bg-slate-900'
                    }`}
                  >
                    {mode.name}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="space-y-1.5">
                <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Body & Pose Mesh</span>
                <div className="grid grid-cols-2 gap-1.5">
                  {[
                    { id: 'pose', name: 'Pose Tracking', icon: UserCheck, allowed: true, reason: '' },
                    { id: 'facemesh', name: 'Face Tracking', icon: User, allowed: true, reason: '' },
                    { id: 'gesture', name: 'Hand Tracking', icon: Hand, allowed: true, reason: '' },
                    { id: 'holistic', name: 'Holistic Mesh', icon: Compass, allowed: true, reason: '' },
                  ].map((item) => {
                    const Icon = item.icon;
                    const isTaskAllowed = item.allowed;
                    return (
                      <button
                        key={item.id}
                        disabled={!isTaskAllowed}
                        onClick={() => selectTask('mediapipe', item.id)}
                        className={`py-2 px-1.5 rounded-xl text-[11px] font-medium border flex flex-col items-center justify-center space-y-1 transition disabled:opacity-30 disabled:cursor-not-allowed ${
                          activeTask === item.id
                            ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                            : 'border-slate-850 bg-slate-900/40 text-slate-400 hover:bg-slate-900'
                        }`}
                        title={isTaskAllowed ? `Select ${item.name}` : item.reason}
                      >
                        <div className="flex items-center space-x-1">
                          <Icon className="w-3.5 h-3.5" />
                          {!isTaskAllowed && <span className="text-[9px] text-rose-455 font-bold">🔒</span>}
                        </div>
                        <span className="text-center truncate w-full">{item.name}</span>
                        {!isTaskAllowed && (
                          <span className="text-[7px] text-rose-400 font-bold uppercase tracking-tighter block">Locked</span>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="space-y-1.5">
                <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Features & Text</span>
                <div className="grid grid-cols-2 gap-1.5">
                  {[
                    { id: 'objects', name: 'Object Detection', icon: Sparkles },
                    { id: 'ocr', name: 'OCR Extract', icon: FileText },
                  ].map((item) => {
                    const Icon = item.icon;
                    return (
                      <button
                        key={item.id}
                        onClick={() => selectTask('mediapipe', item.id)}
                        className={`py-2 px-2.5 rounded-xl text-xs font-medium border flex items-center justify-center space-x-2 transition ${
                          activeTask === item.id
                            ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                            : 'border-slate-850 bg-slate-900/40 text-slate-400 hover:bg-slate-900'
                        }`}
                      >
                        <Icon className="w-3.5 h-3.5" />
                        <span>{item.name}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="space-y-1.5">
                <span className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Classification</span>
                <div className="grid grid-cols-1">
                  {[
                    { id: 'classify', name: 'Image Classification', icon: Tag, allowed: true },
                  ].map((item) => {
                    const Icon = item.icon;
                    return (
                      <button
                        key={item.id}
                        onClick={() => selectTask('mediapipe', item.id)}
                        className={`py-2 px-2 rounded-xl text-[11px] font-medium border flex items-center justify-center space-x-1.5 transition ${
                          activeTask === item.id
                            ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                            : 'border-slate-850 bg-slate-900/40 text-slate-400 hover:bg-slate-900'
                        }`}
                      >
                        <Icon className="w-3.5 h-3.5 text-center" />
                        <span className="truncate">{item.name}</span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* Level 3: Dynamic Hierarchical Stage Controls */}
          <div className="space-y-4 pt-4 border-t border-slate-850">
            <span className="text-xs font-bold text-slate-450 uppercase tracking-widest block">Pipeline Stage Controls</span>
            
            {/* Stage 1: Pre-processing controls */}
            <div className="p-3 bg-slate-955/40 rounded-xl border border-slate-900 space-y-3">
              <span className="text-[11px] font-bold text-cyan-400 uppercase tracking-wider flex items-center space-x-1">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                <span>Stage 1: Pre-Processing</span>
              </span>

              {/* Resize Scale Slider */}
              <div className="space-y-1.5">
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span>Image Scale Factor:</span>
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

              {/* Noise Pre-Blur Slider */}
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

              {/* Grayscale pre-process */}
              <div className="flex items-center justify-between pt-1">
                <span className="text-[11px] text-slate-400">Force Grayscale Conversion:</span>
                <Switch.Root
                  checked={grayscale}
                  onCheckedChange={setGrayscale}
                  className="w-8 h-4.5 bg-slate-900 rounded-full relative data-[state=checked]:bg-cyan-500 transition"
                >
                  <Switch.Thumb className="block w-3.5 h-3.5 bg-white rounded-full transition transform translate-x-0.5 data-[state=checked]:translate-x-4" />
                </Switch.Root>
              </div>
            </div>

            {/* Stage 2: Core task controls */}
            <div className="p-3 bg-slate-955/40 rounded-xl border border-slate-900 space-y-3">
              <span className="text-[11px] font-bold text-amber-400 uppercase tracking-wider flex items-center space-x-1">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                <span>Stage 2: Core Algorithm</span>
              </span>

              {/* Task Specific sliders */}
              {activeTask === 'canny' && (
                <>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Canny Low Threshold:</span>
                      <span className="font-mono text-amber-300">{param1}</span>
                    </div>
                    <Slider.Root
                      className="relative flex items-center select-none touch-none w-full h-3"
                      value={[param1]}
                      min={1}
                      max={300}
                      step={1}
                      onValueChange={(val) => setParam1(val[0])}
                    >
                      <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                        <Slider.Range className="absolute bg-amber-500 rounded-full h-full" />
                      </Slider.Track>
                      <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-amber-400 focus:outline-none" />
                    </Slider.Root>
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Canny High Threshold:</span>
                      <span className="font-mono text-amber-300">{param2}</span>
                    </div>
                    <Slider.Root
                      className="relative flex items-center select-none touch-none w-full h-3"
                      value={[param2]}
                      min={1}
                      max={300}
                      step={1}
                      onValueChange={(val) => setParam2(val[0])}
                    >
                      <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                        <Slider.Range className="absolute bg-amber-500 rounded-full h-full" />
                      </Slider.Track>
                      <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-amber-400 focus:outline-none" />
                    </Slider.Root>
                  </div>
                </>
              )}

              {activeTask === 'threshold' && (
                <>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Threshold Cutoff Value:</span>
                      <span className="font-mono text-amber-300">{param1}</span>
                    </div>
                    <Slider.Root
                      className="relative flex items-center select-none touch-none w-full h-3"
                      value={[param1]}
                      min={0}
                      max={255}
                      step={1}
                      onValueChange={(val) => setParam1(val[0])}
                    >
                      <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                        <Slider.Range className="absolute bg-amber-500 rounded-full h-full" />
                      </Slider.Track>
                      <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-amber-400 focus:outline-none" />
                    </Slider.Root>
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Binary Max Value:</span>
                      <span className="font-mono text-amber-300">{param2}</span>
                    </div>
                    <Slider.Root
                      className="relative flex items-center select-none touch-none w-full h-3"
                      value={[param2]}
                      min={0}
                      max={255}
                      step={1}
                      onValueChange={(val) => setParam2(val[0])}
                    >
                      <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                        <Slider.Range className="absolute bg-amber-500 rounded-full h-full" />
                      </Slider.Track>
                      <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-amber-400 focus:outline-none" />
                    </Slider.Root>
                  </div>
                </>
              )}

              {activeTask === 'harris' && (
                <>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Harris Window (Block):</span>
                      <span className="font-mono text-amber-300">{param1} px</span>
                    </div>
                    <Slider.Root
                      className="relative flex items-center select-none touch-none w-full h-3"
                      value={[param1]}
                      min={2}
                      max={10}
                      step={1}
                      onValueChange={(val) => setParam1(val[0])}
                    >
                      <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                        <Slider.Range className="absolute bg-amber-500 rounded-full h-full" />
                      </Slider.Track>
                      <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-amber-400 focus:outline-none" />
                    </Slider.Root>
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Post-Dilate Kernel:</span>
                      <span className="font-mono text-amber-300">{param2}x{param2}</span>
                    </div>
                    <Slider.Root
                      className="relative flex items-center select-none touch-none w-full h-3"
                      value={[param2]}
                      min={1}
                      max={5}
                      step={1}
                      onValueChange={(val) => setParam2(val[0])}
                    >
                      <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                        <Slider.Range className="absolute bg-amber-500 rounded-full h-full" />
                      </Slider.Track>
                      <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-amber-400 focus:outline-none" />
                    </Slider.Root>
                  </div>
                </>
              )}

              {activeTask === 'blur' && (
                <>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Gaussian Kernel:</span>
                      <span className="font-mono text-amber-300">{param1}x{param1}</span>
                    </div>
                    <Slider.Root
                      className="relative flex items-center select-none touch-none w-full h-3"
                      value={[param1]}
                      min={1}
                      max={31}
                      step={2}
                      onValueChange={(val) => setParam1(val[0])}
                    >
                      <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                        <Slider.Range className="absolute bg-amber-500 rounded-full h-full" />
                      </Slider.Track>
                      <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-amber-400 focus:outline-none" />
                    </Slider.Root>
                  </div>
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Sigma Dev:</span>
                      <span className="font-mono text-amber-300">{param2}</span>
                    </div>
                    <Slider.Root
                      className="relative flex items-center select-none touch-none w-full h-3"
                      value={[param2]}
                      min={0}
                      max={10}
                      step={0.5}
                      onValueChange={(val) => setParam2(val[0])}
                    >
                      <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                        <Slider.Range className="absolute bg-amber-500 rounded-full h-full" />
                      </Slider.Track>
                      <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-amber-400 focus:outline-none" />
                    </Slider.Root>
                  </div>
                </>
              )}

              {activeTask === 'objects' && (
                <div className="space-y-1.5">
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>Min Contour Area:</span>
                    <span className="font-mono text-amber-300">{param1} px²</span>
                  </div>
                  <Slider.Root
                    className="relative flex items-center select-none touch-none w-full h-3"
                    value={[param1]}
                    min={50}
                    max={5000}
                    step={50}
                    onValueChange={(val) => setParam1(val[0])}
                  >
                    <Slider.Track className="bg-slate-900 relative grow rounded-full h-1">
                      <Slider.Range className="absolute bg-amber-500 rounded-full h-full" />
                    </Slider.Track>
                    <Slider.Thumb className="block w-3.5 h-3.5 bg-white rounded-full border border-amber-400 focus:outline-none" />
                  </Slider.Root>
                </div>
              )}

              {activeGroup === 'mediapipe' && activeTask !== 'objects' && (
                <div className="text-[11px] text-slate-500 leading-relaxed py-1 italic">
                  Pretrained weights active. Dynamic overlays rendering is processed with local config.
                </div>
              )}
            </div>

            {/* Stage 3: Rendering & Visualization controls */}
            {activeTask !== 'classify' && (
              <div className="p-3 bg-slate-955/40 rounded-xl border border-slate-900 space-y-3">
                <span className="text-[11px] font-bold text-pink-400 uppercase tracking-wider flex items-center space-x-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-pink-400 animate-pulse" />
                  <span>Stage 3: Visualization Styling</span>
                </span>

                {/* Overlay Color Palette Selector */}
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
                {activeTask !== 'facemesh' && (
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
                )}

                {/* Point/Joint Radius */}
                {(activeTask === 'pose' || activeTask === 'facemesh' || activeTask === 'holistic') && (
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-[11px] text-slate-400">
                      <span>Landmark Point Radius:</span>
                      <span className="font-mono text-pink-300">{pointRadius} px</span>
                    </div>
                    <Slider.Root
                      className="relative flex items-center select-none touch-none w-full h-3"
                      value={[pointRadius]}
                      min={1}
                      max={12}
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

                {/* Labels switch */}
                {(activeTask === 'gesture' || activeTask === 'holistic' || activeTask === 'objects' || activeTask === 'ocr') && (
                  <div className="flex items-center justify-between pt-1">
                    <span className="text-[11px] text-slate-400">Show Legend / Labels:</span>
                    <Switch.Root
                      checked={showLabels}
                      onCheckedChange={setShowLabels}
                      className="w-8 h-4.5 bg-slate-900 rounded-full relative data-[state=checked]:bg-pink-500 transition"
                    >
                      <Switch.Thumb className="block w-3.5 h-3.5 bg-white rounded-full transition transform translate-x-0.5 data-[state=checked]:translate-x-4" />
                    </Switch.Root>
                  </div>
                )}
              </div>
            )}
          </div>


        </div>

        {/* Media & Output Canvas Column */}
        <div className="lg:col-span-2 glass-panel p-6 rounded-2xl space-y-6">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <div>
              <h3 className="text-xl font-bold text-white flex items-center space-x-2">
                <Eye className="w-5 h-5 text-cyan-400" />
                <span>Vision Processing Canvas</span>
              </h3>
              {activeTaskName && (
                <div className="text-xs text-cyan-400 font-mono mt-0.5">Active Task: {activeTaskName}</div>
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

          {/* Classification Results Box if active */}
          {categoriesList.length > 0 && (
            <div className="p-4 bg-slate-900/90 rounded-xl border border-blue-500/30 space-y-2">
              <div className="text-xs font-bold text-blue-300 flex items-center space-x-2">
                <Tag className="w-4 h-4 text-blue-400" />
                <span>MediaPipe Image Classification Top Probabilities</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                {categoriesList.map((cat, idx) => (
                  <div key={idx} className="p-2.5 bg-slate-950/80 rounded-lg border border-slate-800 flex justify-between items-center text-xs">
                    <span className="font-medium text-slate-200">{cat.category_name}</span>
                    <span className="font-mono text-cyan-400 font-bold">{(cat.score * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* OCR Translation Results Box */}
          {ocrResult && (
            <div className="p-4 bg-slate-900/90 rounded-xl border border-emerald-500/30 space-y-3">
              <div className="text-xs font-bold text-emerald-300 flex items-center space-x-2">
                <FileText className="w-4 h-4 text-emerald-400" />
                <span>OCR Text Extraction & Translation</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                <div className="p-2.5 bg-slate-950/80 rounded-lg border border-slate-800 text-xs">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Detected Language</div>
                  <div className="font-semibold text-cyan-400">{ocrResult.detectedLanguage}</div>
                </div>
                <div className="sm:col-span-2 p-2.5 bg-slate-950/80 rounded-lg border border-slate-800 text-xs">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Extracted Text</div>
                  <div className="font-medium text-slate-200 leading-relaxed">{ocrResult.extractedText || '[No text detected]'}</div>
                </div>
              </div>
              {ocrResult.translation && (
                <div className="p-3 bg-emerald-500/5 rounded-lg border border-emerald-500/20">
                  <div className="text-[10px] text-emerald-500 uppercase tracking-wider mb-1 font-bold">English Translation</div>
                  <div className="text-sm font-medium text-emerald-200 leading-relaxed">{ocrResult.translation}</div>
                </div>
              )}
            </div>
          )}

          {/* Error / Warning Alert Box */}
          {errorMsg && (
            <div className="p-3.5 bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-xl text-xs flex items-center space-x-2 shadow-sm animate-in fade-in slide-in-from-top-2 duration-200">
              <AlertCircle className="w-4.5 h-4.5 text-rose-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 min-h-[320px]">
            <div className="bg-slate-950/80 rounded-xl border border-slate-800 p-4 flex items-center justify-center relative group">
              <div className="absolute top-3 left-3 bg-slate-900/90 px-2.5 py-1 rounded-md text-[10px] font-mono text-slate-400 border border-slate-800">
                Source Frame
              </div>
              {useTrafficAnpr ? (
                <div className="text-center text-slate-400 text-xs space-y-3 p-6 bg-slate-900/40 border border-slate-800 rounded-xl max-w-xs animate-pulse">
                  <Play className="w-8 h-8 mx-auto text-cyan-400" />
                  <div className="font-bold text-slate-200">CCTV_HIGHWAY_SEC_30 FEED</div>
                  <div className="text-[10px] text-slate-500 font-mono">Stream: Mapped via FastAPI StreamingResponse</div>
                </div>
              ) : useWebcam ? (
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
                      setFullscreenTitle('Source Frame (Webcam)');
                      setFullscreenType('source');
                    }}
                    className="absolute top-3 right-3 p-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-700 transition opacity-0 group-hover:opacity-100 focus:opacity-100 animate-in fade-in"
                    title="View Larger Preview"
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
                    title="View Larger Preview"
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

            <div className="bg-slate-950/80 rounded-xl border border-cyan-500/20 p-4 flex items-center justify-center relative group">
              <div className="absolute top-3 left-3 bg-slate-900/90 px-2.5 py-1 rounded-md text-[10px] font-mono text-cyan-400 border border-slate-800 flex items-center space-x-1">
                {(processing || useTrafficAnpr) && <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />}
                <span>Processed Output</span>
              </div>
              {useTrafficAnpr ? (
                <div className="relative w-full h-full max-h-72 rounded-lg overflow-hidden flex items-center justify-center bg-black">
                  <img
                    src={`${client.baseUrl}/api/vision/traffic-anpr-stream`}
                    alt="Real-Time Traffic ANPR Stream"
                    className="max-h-72 w-full object-contain rounded-lg border border-cyan-500/20"
                  />
                  <button
                    onClick={() => {
                      setFullscreenUrl('anpr');
                      setFullscreenTitle('Real-Time Traffic ANPR Stream');
                      setFullscreenType('anpr');
                    }}
                    className="absolute top-3 right-3 p-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-cyan-400 hover:text-cyan-300 border border-slate-700 transition opacity-0 group-hover:opacity-100 focus:opacity-100 animate-in fade-in"
                    title="View Larger Preview"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </button>
                </div>
              ) : resultUrl ? (
                <>
                  <img src={resultUrl} alt="Processed Vision Output" className="max-h-72 rounded-lg object-contain" />
                  <button
                    onClick={() => {
                      setFullscreenUrl(resultUrl);
                      setFullscreenTitle(`Processed Output: ${activeTask.toUpperCase()}`);
                      setFullscreenType('output');
                    }}
                    className="absolute top-3 right-3 p-1.5 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-cyan-400 hover:text-cyan-300 border border-slate-700 transition opacity-0 group-hover:opacity-100 focus:opacity-100"
                    title="View Larger Preview"
                  >
                    <Maximize2 className="w-4 h-4" />
                  </button>
                </>
              ) : (
                <div className="text-center text-slate-500 text-sm">
                  Output preview will appear here
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* On-Demand Large Preview Lightbox Overlay Modal */}
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
            {!(fullscreenType === 'source' && useWebcam) && fullscreenType !== 'anpr' && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  const a = document.createElement('a');
                  a.href = (fullscreenType === 'output' && useWebcam) ? (resultUrl || '') : (fullscreenUrl || '');
                  a.download = `opencv_studio_large_preview.jpg`;
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
                alt="Fullscreen Processed Output" 
                className="w-full h-full rounded-2xl object-contain border border-slate-800 shadow-2xl select-none"
              />
            ) : fullscreenType === 'anpr' ? (
              <img 
                src={`${client.baseUrl}/api/vision/traffic-anpr-stream`} 
                alt="Fullscreen Real-Time Traffic ANPR Stream" 
                className="w-full h-full rounded-2xl object-contain border border-cyan-500/20 shadow-2xl select-none"
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
