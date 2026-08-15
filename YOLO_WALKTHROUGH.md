# Walkthrough - YOLO AI Studio

Successfully implemented the new **YOLO AI Studio** tab in OpenCV Studio, providing premium real-time demonstrations of state-of-the-art YOLOv8 models.

---

## Architecture and Components

### 1. Backend Services (`yolo_service.py`)
* Implemented in [yolo_service.py](file:///Users/bernard/Developer/DATA-SCIENCE/COMPUTER-VISION/opencv-studio/apps/backend/src/services/yolo_service.py).
* Caches YOLOv8 singleton models in-memory on demand to maximize speed and prevent double-loading:
  * **YOLOv8n** (`yolov8n.pt`) for Object Detection.
  * **YOLOv8n-seg** (`yolov8n-seg.pt`) for Instance Segmentation.
  * **YOLOv8n-pose** (`yolov8n-pose.pt`) for Keypoint Pose Estimation.
* Uses **`cv2.LINE_AA` (anti-aliased lines)** for all visual decorations (bounding boxes, polygons, keypoints, text labels, and skeleton links) to guarantee smooth vector rendering without pixelation.

### 2. API Routers (`yolo.py`)
* Implemented in [yolo.py](file:///Users/bernard/Developer/DATA-SCIENCE/COMPUTER-VISION/opencv-studio/apps/backend/src/routers/yolo.py).
* Exposes three endpoints:
  * `POST /api/yolo/detect`: Runs bounding-box object detection, returning labels and scores.
  * `POST /api/yolo/segment`: Runs polygon instance segmentation, blending translucent color fills.
  * `POST /api/yolo/pose`: Runs skeleton keypoint mapping.
* Registered in [main.py](file:///Users/bernard/Developer/DATA-SCIENCE/COMPUTER-VISION/opencv-studio/apps/backend/src/main.py#L30-L34).

### 3. API Client Bindings (`client.ts`)
* Added `detectYoloObjects`, `segmentYoloObjects`, and `estimateYoloPose` methods to the shared client wrapper in [client.ts](file:///Users/bernard/Developer/DATA-SCIENCE/COMPUTER-VISION/opencv-studio/libs/shared/src/api/client.ts#L190-L215).

### 4. Interactive Frontend (`YoloStudio.tsx` & `App.tsx`)
* Implemented in [YoloStudio.tsx](file:///Users/bernard/Developer/DATA-SCIENCE/COMPUTER-VISION/opencv-studio/apps/web/src/components/YoloStudio.tsx).
* Added the **YOLO AI Studio** tab trigger and content inside [App.tsx](file:///Users/bernard/Developer/DATA-SCIENCE/COMPUTER-VISION/opencv-studio/apps/web/src/App.tsx#L195-L225).
* Features:
  * **Interactive Task Sidebar:** Toggle between object detection, segmentation, and pose skeleton tracking.
  * **Hyperparameter Tuning:** Live sliders to modify **Confidence threshold** and **NMS IoU/Overlap threshold** on the fly.
  * **Real-time Metrics:** Displays a glowing flex-grid badge summary of all detected classes and their confidence percentages.
  * **Test Catalog Integration:** Accompanying demo photo selector block to try models instantly without a camera.
  * **Webcam & Fullscreen Lightbox:** Mirror webcam live streams and view upscaled predictions in a fullscreen overlay with aspect-ratio preservation.

---

## Verification Instructions

1. Run the project locally (`just start`).
2. Open http://localhost:3000 in your web browser.
3. Click the new **YOLO AI Studio** tab at the top.
4. Try each of the demonstrations:
   * **Object Detection:** Select it, choose a demo image (like "Object Detection"), and view bounding boxes. Toggle "Webcam Live Video Feed" to run it live!
   * **Instance Segmentation:** Turn it on to see colored masks contouring objects (e.g. people, chairs, laptops).
   * **Pose Estimation:** Stand in front of the camera and watch the 17-point skeletal track map onto your body in real-time.
