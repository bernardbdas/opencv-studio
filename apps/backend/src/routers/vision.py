import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from apps.backend.src.services.vision_service import vision_service
from apps.backend.src.services.mediapipe_service import mediapipe_service
from apps.backend.src.services.metrics_service import metrics_service

router = APIRouter(prefix="/api/vision", tags=["Vision Tasks"])

@router.post("/classic")
async def process_classic_cv(filter_type: str = "canny", param1: float = 100.0, param2: float = 200.0, scale: float = 1.0, pre_blur: int = 0, grayscale: bool = True, file: UploadFile = File(...)):
    """Process image using classic OpenCV filters (Canny Edge, Thresholding, Harris Corner, Blur)."""
    try:
        contents = await file.read()
        return vision_service.process_classic_filter(contents, filter_type, param1, param2, scale=scale, pre_blur=pre_blur, grayscale=grayscale)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenCV processing failed: {str(e)}")

@router.post("/object-detection")
async def process_object_detection(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, min_area: float = 500.0, overlay_color: str = "green", line_thickness: int = 2, show_labels: bool = True, file: UploadFile = File(...)):
    """Perform Object Detection & Bounding Box extraction."""
    try:
        contents = await file.read()
        start = time.time()
        res = vision_service.detect_objects(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, min_area=min_area, overlay_color=overlay_color, line_thickness=line_thickness, show_labels=show_labels)
        latency = (time.time() - start) * 1000
        metrics_service.record("Mediapipe Lab", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")

@router.post("/pose-detection")
async def process_pose_detection(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, overlay_color: str = "cyan", line_thickness: int = 3, joint_radius: int = 6, file: UploadFile = File(...)):
    """Perform Pose Landmark Skeleton Detection."""
    try:
        contents = await file.read()
        start = time.time()
        res = vision_service.detect_pose(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, overlay_color=overlay_color, line_thickness=line_thickness, joint_radius=joint_radius)
        latency = (time.time() - start) * 1000
        metrics_service.record("Mediapipe Lab", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pose detection failed: {str(e)}")

@router.post("/ocr-detection")
async def process_ocr_detection(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, overlay_color: str = "magenta", line_thickness: int = 2, show_labels: bool = True, file: UploadFile = File(...)):
    """Perform OCR Text Region & Bounding Box extraction."""
    try:
        contents = await file.read()
        return vision_service.detect_ocr(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, overlay_color=overlay_color, line_thickness=line_thickness, show_labels=show_labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR text extraction failed: {str(e)}")

@router.post("/face-landmark")
async def process_face_landmark(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, overlay_color: str = "green", point_radius: int = 2, file: UploadFile = File(...)):
    """Perform Face Mesh Landmark mesh regressor."""
    try:
        contents = await file.read()
        start = time.time()
        res = vision_service.detect_face_landmarks(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, overlay_color=overlay_color, point_radius=point_radius)
        latency = (time.time() - start) * 1000
        metrics_service.record("Mediapipe Lab", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face landmark detection failed: {str(e)}")

@router.post("/classify")
async def process_image_classification(file: UploadFile = File(...)):
    """MediaPipe ImageClassifier task."""
    try:
        contents = await file.read()
        start = time.time()
        res = mediapipe_service.classify_image(contents)
        latency = (time.time() - start) * 1000
        metrics_service.record("Mediapipe Lab", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image classification failed: {str(e)}")

@router.post("/gesture")
async def process_gesture_recognition(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, overlay_color: str = "pink", line_thickness: int = 2, show_labels: bool = True, file: UploadFile = File(...)):
    """MediaPipe GestureRecognizer task."""
    try:
        contents = await file.read()
        start = time.time()
        res = mediapipe_service.recognize_gesture(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, overlay_color=overlay_color, line_thickness=line_thickness, show_labels=show_labels)
        latency = (time.time() - start) * 1000
        metrics_service.record("Mediapipe Lab", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gesture recognition failed: {str(e)}")

@router.post("/holistic")
async def process_holistic_detection(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, overlay_color: str = "cyan", line_thickness: int = 3, point_radius: int = 2, show_labels: bool = True, file: UploadFile = File(...)):
    """MediaPipe HolisticLandmarker task (Face + Hand + Pose)."""
    try:
        contents = await file.read()
        start = time.time()
        res = mediapipe_service.detect_holistic(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, overlay_color=overlay_color, line_thickness=line_thickness, point_radius=point_radius, show_labels=show_labels)
        latency = (time.time() - start) * 1000
        metrics_service.record("Mediapipe Lab", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Holistic detection failed: {str(e)}")

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze image components to return flags indicating presence of face, body, or hands."""
    try:
        contents = await file.read()
        return vision_service.analyze_image_contents(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@router.post("/finger-frame")
async def process_finger_frame(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, overlay_color: str = "cyan", line_thickness: int = 3, show_labels: bool = True, portal_filter: str = "sketch", file: UploadFile = File(...)):
    """Perform AR Finger Frame crop & interactive filter overlay inside frame."""
    try:
        contents = await file.read()
        start = time.time()
        res = mediapipe_service.detect_finger_frame(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, overlay_color=overlay_color, line_thickness=line_thickness, show_labels=show_labels, portal_filter=portal_filter)
        latency = (time.time() - start) * 1000
        metrics_service.record("Mediapipe Lab", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AR Finger frame processing failed: {str(e)}")

@router.post("/similarity")
async def process_image_similarity(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """MediaPipe ImageEmbedder task for calculating cosine similarity between two images."""
    try:
        c1 = await file1.read()
        c2 = await file2.read()
        start = time.time()
        res = mediapipe_service.compute_image_similarity(c1, c2)
        latency = (time.time() - start) * 1000
        metrics_service.record("Mediapipe Lab", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")

@router.post("/face-filter")
async def process_face_filter(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, overlay_color: str = "cyan", line_thickness: int = 3, point_radius: int = 2, file: UploadFile = File(...)):
    """Perform Face Mesh Visor Overlay AR filter."""
    try:
        contents = await file.read()
        start = time.time()
        res = mediapipe_service.detect_face_filter(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, overlay_color=overlay_color, line_thickness=line_thickness, point_radius=point_radius)
        latency = (time.time() - start) * 1000
        metrics_service.record("Mediapipe Lab", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AR Face filter failed: {str(e)}")

@router.post("/aruco-projection")
async def process_aruco_projection(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, overlay_color: str = "cyan", line_thickness: int = 3, file: UploadFile = File(...)):
    """Perform ArUco Marker detection and 3D wireframe cube projection."""
    try:
        contents = await file.read()
        start = time.time()
        res = mediapipe_service.detect_aruco_projection(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, overlay_color=overlay_color, line_thickness=line_thickness)
        latency = (time.time() - start) * 1000
        metrics_service.record("Mediapipe Lab", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ArUco 3D projection failed: {str(e)}")

@router.post("/segmentation")
async def process_selfie_segmentation(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, overlay_color: str = "cyan", file: UploadFile = File(...)):
    """Perform AI selfie segmentation and background replacement."""
    try:
        contents = await file.read()
        start = time.time()
        res = mediapipe_service.detect_selfie_segmentation(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, overlay_color=overlay_color)
        latency = (time.time() - start) * 1000
        metrics_service.record("Mediapipe Lab", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Selfie background segmentation failed: {str(e)}")

@router.post("/pose-trainer")
async def process_pose_trainer(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, file: UploadFile = File(...)):
    """Perform pose trainer squat rep counting."""
    try:
        contents = await file.read()
        start = time.time()
        res = mediapipe_service.detect_pose_trainer(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale)
        latency = (time.time() - start) * 1000
        metrics_service.record("AR Launchpad", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pose trainer processing failed: {str(e)}")

@router.post("/air-draw")
async def process_air_draw(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, clear: bool = False, file: UploadFile = File(...)):
    """Perform real-time pinch-to-draw air writing."""
    try:
        contents = await file.read()
        start = time.time()
        res = mediapipe_service.detect_air_draw(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, clear=clear)
        latency = (time.time() - start) * 1000
        metrics_service.record("AR Launchpad", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Air draw processing failed: {str(e)}")

@router.post("/face-tryon")
async def process_face_tryon(scale: float = 1.0, pre_blur: int = 0, grayscale: bool = False, overlay_color: str = "cyan", file: UploadFile = File(...)):
    """Perform face try-on neon cyberpunk glasses projection."""
    try:
        contents = await file.read()
        start = time.time()
        res = mediapipe_service.detect_face_tryon(contents, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale, overlay_color=overlay_color)
        latency = (time.time() - start) * 1000
        metrics_service.record("AR Launchpad", latency)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face try-on processing failed: {str(e)}")

@router.get("/demo-assets")
def get_demo_assets():
    """Get catalog of bundled demo images available for instant testing."""
    return vision_service.get_demo_assets_catalog()

@router.get("/demo-assets/{key}")
def get_demo_asset_by_key(key: str):
    """Retrieve demo image asset base64 payload by asset key."""
    try:
        return vision_service.get_demo_asset_image(key)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/traffic-anpr-stream")
def get_traffic_anpr_stream():
    """Stream live processed traffic video frames with ANPR overlays."""
    try:
        return StreamingResponse(
            vision_service.stream_traffic_anpr(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ANPR streaming failed: {str(e)}")

@router.post("/invisibility-cloak/background")
async def set_invisibility_cloak_background(file: UploadFile = File(...)):
    """Set the background frame for the invisibility cloak."""
    try:
        contents = await file.read()
        return mediapipe_service.set_invisibility_background(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set invisibility background: {str(e)}")

@router.post("/invisibility-cloak")
async def process_invisibility_cloak(
    mode: str = "ai",
    color: str = "green",
    scale: float = 1.0,
    pre_blur: int = 0,
    grayscale: bool = False,
    show_labels: bool = True,
    file: UploadFile = File(...)
):
    """Run invisibility cloak frame processing using AI segmentation or color thresholding."""
    try:
        contents = await file.read()
        return mediapipe_service.detect_invisibility_cloak(
            contents,
            mode=mode,
            color=color,
            scale=scale,
            pre_blur=pre_blur,
            force_grayscale=grayscale,
            show_labels=show_labels
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invisibility cloak processing failed: {str(e)}")
