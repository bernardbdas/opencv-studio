"""Model registry and configuration mapping module for opencv-studio MediaPipe tasks."""

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    task_name: str
    filename: str
    search_pattern: str
    fallback_url: str

# 1. Object Detector
OBJECT_DETECTION = ModelConfig(
    task_name="object_detector",
    filename="efficientdet_lite0.tflite",
    search_pattern="object_detector/efficientdet_lite0/float16/latest",
    fallback_url="https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite"
)

# 2. Image Classifier
IMAGE_CLASSIFICATION = ModelConfig(
    task_name="image_classifier",
    filename="efficientnet_lite0.tflite",
    search_pattern="image_classifier/efficientnet_lite0/float32/latest",
    fallback_url="https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/latest/efficientnet_lite0.tflite"
)

# 3. Image Segmenter
IMAGE_SEGMENTATION = ModelConfig(
    task_name="image_segmenter",
    filename="selfie_segmenter.tflite",
    search_pattern="image_segmenter/selfie_segmenter/float16/latest",
    fallback_url="https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)

# 4. Gesture Recognizer
GESTURE_RECOGNIZER = ModelConfig(
    task_name="gesture_recognizer",
    filename="gesture_recognizer.task",
    search_pattern="gesture_recognizer/gesture_recognizer/float16/latest",
    fallback_url="https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
)

# 5. Hand Landmarker
HAND_LANDMARKER = ModelConfig(
    task_name="hand_landmarker",
    filename="hand_landmarker.task",
    search_pattern="hand_landmarker/hand_landmarker/float16/latest",
    fallback_url="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# 6. Face Detector
FACE_DETECTION = ModelConfig(
    task_name="face_detector",
    filename="blaze_face_short_range.tflite",
    search_pattern="face_detector/blaze_face_short_range/float16/latest",
    fallback_url="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

# 7. Face Landmarker
FACE_LANDMARKER = ModelConfig(
    task_name="face_landmarker",
    filename="face_landmarker.task",
    search_pattern="face_landmarker/face_landmarker/float16/latest",
    fallback_url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

# 8. Pose Landmarker
POSE_LANDMARKER = ModelConfig(
    task_name="pose_landmarker",
    filename="pose_landmarker_full.task",
    search_pattern="pose_landmarker/pose_landmarker_full/float16/latest",
    fallback_url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

# 9. Holistic Landmarker
HOLISTIC_LANDMARKER = ModelConfig(
    task_name="holistic_landmarker",
    filename="holistic_landmarker.task",
    search_pattern="holistic_landmarker/holistic_landmarker/float16/latest",
    fallback_url="https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)

# 10. Image Embedder
IMAGE_EMBEDDING = ModelConfig(
    task_name="image_embedder",
    filename="mobilenet_v3_small.tflite",
    search_pattern="image_embedder/mobilenet_v3_small/float32/latest",
    fallback_url="https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite"
)

# 11. Language Detector
LANGUAGE_DETECTION = ModelConfig(
    task_name="language_detector",
    filename="language_detector.tflite",
    search_pattern="language_detector/language_detector/float32/latest",
    fallback_url="https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/latest/language_detector.tflite"
)
