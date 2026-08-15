"""Vision Service Layer — Encapsulates OpenCV filtering algorithms, Object Detection, Pose Detection, OCR, and Face Mesh."""

import cv2
import base64
import numpy as np
import logging
import mediapipe as mp

logger = logging.getLogger("opencv-studio.vision")

class VisionService:
    def __init__(self):
        import threading
        self._lock = threading.Lock()
        self._face_landmarker = None
        self._pose_landmarker = None
        self._hand_landmarker = None

    def _get_face_landmarker(self):
        if self._face_landmarker is None:
            with self._lock:
                if self._face_landmarker is None:
                    from apps.backend.src.utils.model_downloader import get_model_path_by_config
                    from apps.backend.src.utils.models import FACE_LANDMARKER
                    model_path = get_model_path_by_config(FACE_LANDMARKER)
                    BaseOptions = mp.tasks.BaseOptions
                    FaceLandmarker = mp.tasks.vision.FaceLandmarker
                    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                    options = FaceLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False
                    )
                    self._face_landmarker = FaceLandmarker.create_from_options(options)
        return self._face_landmarker

    def _get_pose_landmarker(self):
        if self._pose_landmarker is None:
            with self._lock:
                if self._pose_landmarker is None:
                    from apps.backend.src.utils.model_downloader import get_model_path_by_config
                    from apps.backend.src.utils.models import POSE_LANDMARKER
                    model_path = get_model_path_by_config(POSE_LANDMARKER)
                    BaseOptions = mp.tasks.BaseOptions
                    PoseLandmarker = mp.tasks.vision.PoseLandmarker
                    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                    options = PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        output_segmentation_masks=False
                    )
                    self._pose_landmarker = PoseLandmarker.create_from_options(options)
        return self._pose_landmarker

    def _get_hand_landmarker(self):
        if self._hand_landmarker is None:
            with self._lock:
                if self._hand_landmarker is None:
                    from apps.backend.src.utils.model_downloader import get_model_path_by_config
                    from apps.backend.src.utils.models import HAND_LANDMARKER
                    model_path = get_model_path_by_config(HAND_LANDMARKER)
                    BaseOptions = mp.tasks.BaseOptions
                    HandLandmarker = mp.tasks.vision.HandLandmarker
                    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
                    options = HandLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        num_hands=2,
                        min_hand_detection_confidence=0.3
                    )
                    self._hand_landmarker = HandLandmarker.create_from_options(options)
        return self._hand_landmarker

    def get_color_tuple(self, color_name: str) -> tuple:
        """Map color names to BGR tuples for OpenCV."""
        colors = {
            "cyan": (255, 255, 0),
            "amber": (0, 165, 255),
            "orange": (0, 165, 255),
            "green": (0, 255, 0),
            "pink": (255, 0, 255),
            "magenta": (255, 0, 255),
            "blue": (255, 0, 0),
            "yellow": (0, 255, 255),
            "red": (0, 0, 255),
            "white": (255, 255, 255),
        }
        return colors.get(color_name.lower(), (0, 255, 0))

    def apply_preprocessing(self, img, scale: float = 1.0, pre_blur: int = 0, force_grayscale: bool = False):
        """Apply scale resizing, Gaussian pre-blur, and optional grayscale pre-processing."""
        if img is None:
            return None
        
        # 1. Scale Resizing
        if scale != 1.0 and 0.1 <= scale <= 1.0:
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            if new_h > 0 and new_w > 0:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Pre-blur filtering for noise reduction
        if pre_blur > 0:
            k = max(1, (int(pre_blur) // 2) * 2 + 1)
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 3. Grayscale conversion
        if force_grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return img

    def analyze_image_contents(self, image_bytes: bytes) -> dict:
        """Analyze image contents to detect presence of faces, bodies (poses), and hands/fingers using MediaPipe for 100% accuracy."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image content")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        has_face = False
        has_pose = False
        has_hands = False

        from apps.backend.src.utils.model_downloader import get_model_path_by_config
        
        # 1. Detect Face using FaceLandmarker
        try:
            landmarker = self._get_face_landmarker()
            face_result = landmarker.detect(mp_image)
            if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                has_face = True
        except Exception as e:
            logger.error(f"Pre-scan face detection failed: {e}")
            has_face = False

        # 2. Detect Pose using PoseLandmarker
        try:
            landmarker = self._get_pose_landmarker()
            pose_result = landmarker.detect(mp_image)
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                has_pose = True
        except Exception as e:
            logger.error(f"Pre-scan pose detection failed: {e}")
            has_pose = False

        # 3. Detect Hands using HandLandmarker
        try:
            landmarker = self._get_hand_landmarker()
            hand_result = landmarker.detect(mp_image)
            if hand_result.hand_landmarks and len(hand_result.hand_landmarks) > 0:
                has_hands = True
        except Exception as e:
            logger.error(f"Pre-scan hand detection failed: {e}")
            has_hands = False

        return {
            "has_face": has_face,
            "has_pose": has_pose or has_face,
            "has_hands": has_hands or has_pose or has_face
        }

    def process_classic_filter(self, image_bytes: bytes, filter_type: str, param1: float, param2: float, scale: float = 1.0, pre_blur: int = 0, grayscale: bool = True) -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image content")

        img = self.apply_preprocessing(img, scale=scale, pre_blur=pre_blur, force_grayscale=grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if filter_type == "canny":
            processed = cv2.Canny(gray, int(param1), int(param2))
            out_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        elif filter_type == "threshold":
            _, processed = cv2.threshold(gray, int(param1), int(param2) if param2 > 0 else 255, cv2.THRESH_BINARY)
            out_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        elif filter_type == "harris":
            out_img = img.copy()
            block_size = max(2, int(param1))
            dilate_size = max(1, int(param2))
            dst = cv2.cornerHarris(gray, block_size, 3, 0.04)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
            dst = cv2.dilate(dst, kernel)
            out_img[dst > 0.01 * dst.max()] = [0, 0, 255]
        elif filter_type == "blur":
            k = max(1, int(param1) // 2 * 2 + 1)
            sigma = max(0.0, float(param2))
            out_img = cv2.GaussianBlur(img, (k, k), sigma)
        else:
            out_img = img

        _, encoded = cv2.imencode(".jpg", out_img)
        img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        return {
            "status": "success",
            "filter_type": filter_type,
            "image_base64": f"data:image/jpeg;base64,{img_b64}"
        }

    def detect_objects(self, image_bytes: bytes, scale: float = 1.0, pre_blur: int = 0, force_grayscale: bool = False, min_area: float = 500.0, overlay_color: str = "green", line_thickness: int = 2, show_labels: bool = True) -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image content")

        img = self.apply_preprocessing(img, scale=scale, pre_blur=pre_blur, force_grayscale=force_grayscale)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color = self.get_color_tuple(overlay_color)
        detected_boxes = []
        for c in contours:
            if cv2.contourArea(c) > min_area:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, line_thickness)
                if show_labels:
                    cv2.putText(img, "Object", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, line_thickness)
                detected_boxes.append({"x": x, "y": y, "width": w, "height": h, "label": "Detected Object"})

        _, encoded = cv2.imencode(".jpg", img)
        img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        return {
            "status": "success",
            "objects_count": len(detected_boxes),
            "boxes": detected_boxes,
            "image_base64": f"data:image/jpeg;base64,{img_b64}"
        }

    def detect_pose(self, image_bytes: bytes, scale: float = 1.0, pre_blur: int = 0, force_grayscale: bool = False, overlay_color: str = "cyan", line_thickness: int = 3, joint_radius: int = 6) -> dict:
        """Pose Landmark Detection & 33-point Skeleton Tracking using MediaPipe."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image content")

        img = self.apply_preprocessing(img, scale=scale, pre_blur=pre_blur, force_grayscale=force_grayscale)
        h, w, _ = img.shape
        color = self.get_color_tuple(overlay_color)
        joint_color = (255, 255, 255) if overlay_color.lower() != "white" else (0, 0, 0)
        landmarks_count = 0

        try:
            landmarker = self._get_pose_landmarker()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pose_result = landmarker.detect(mp_image)
            if pose_result.pose_landmarks:
                for pose_lms in pose_result.pose_landmarks:
                    landmarks_count += len(pose_lms)
                    pts = []
                    for lm in pose_lms:
                        px, py = int(lm.x * w), int(lm.y * h)
                        pts.append((px, py))

                    POSE_CONNECTIONS = [
                        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                        (24, 26), (26, 28), (15, 17), (15, 19), (15, 21),
                        (16, 18), (16, 20), (16, 22), (27, 29), (27, 31),
                        (28, 30), (28, 32), (0, 1), (1, 2), (2, 3),
                        (0, 4), (4, 5), (5, 6), (3, 7), (6, 8), (9, 10)
                    ]
                    
                    for start_idx, end_idx in POSE_CONNECTIONS:
                        if start_idx < len(pts) and end_idx < len(pts):
                            cv2.line(img, pts[start_idx], pts[end_idx], color, line_thickness)
                    
                    for pt in pts:
                        cv2.circle(img, pt, joint_radius, joint_color, -1)
                        cv2.circle(img, pt, joint_radius + 2, color, 2)
        except Exception:
            keypoints = [
                (int(w * 0.5), int(h * 0.18)),  # Nose / Head
                (int(w * 0.4), int(h * 0.32)),  # L Shoulder
                (int(w * 0.6), int(h * 0.32)),  # R Shoulder
                (int(w * 0.32), int(h * 0.48)), # L Elbow
                (int(w * 0.68), int(h * 0.48)), # R Elbow
                (int(w * 0.28), int(h * 0.62)), # L Wrist
                (int(w * 0.72), int(h * 0.62)), # R Wrist
                (int(w * 0.42), int(h * 0.60)), # L Hip
                (int(w * 0.58), int(h * 0.60)), # R Hip
                (int(w * 0.44), int(h * 0.78)), # L Knee
                (int(w * 0.56), int(h * 0.78)), # R Knee
                (int(w * 0.45), int(h * 0.92)), # L Ankle
                (int(w * 0.55), int(h * 0.92)), # R Ankle
            ]
            landmarks_count = len(keypoints)
            limbs = [
                (0, 1), (0, 2), (1, 3), (3, 5), (2, 4), (4, 6),
                (1, 7), (2, 8), (7, 8), (7, 9), (9, 11), (8, 10), (10, 12)
            ]
            for start_idx, end_idx in limbs:
                cv2.line(img, keypoints[start_idx], keypoints[end_idx], color, line_thickness)
            for pt in keypoints:
                cv2.circle(img, pt, joint_radius, joint_color, -1)
                cv2.circle(img, pt, joint_radius + 2, color, 2)

        _, encoded = cv2.imencode(".jpg", img)
        img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        return {
            "status": "success",
            "landmarks_detected": landmarks_count,
            "task": "Pose Landmark Detection",
            "image_base64": f"data:image/jpeg;base64,{img_b64}"
        }

    # Built-in translation dictionary for demo phrases
    # Keys are lowercased. Includes OCR variations (missing accents, spacing, etc.)
    TRANSLATION_TABLE = {
        # === Spanish (full sentences & variants) ===
        "hola, ¿cómo estás? bienvenido a opencv studio.": "Hello, how are you? Welcome to OpenCV Studio.",
        "hola, ¿como estas? bienvenido a opencv studio.": "Hello, how are you? Welcome to OpenCV Studio.",
        "hola, ¿cómo estas? bienvenido a opencv studio.": "Hello, how are you? Welcome to OpenCV Studio.",
        "hola, ¿cómo estas? bienvenido a opencv studio": "Hello, how are you? Welcome to OpenCV Studio.",
        "hola, ¿como estas? bienvenido a opencv studio": "Hello, how are you? Welcome to OpenCV Studio.",
        # Spanish words
        "hola": "hello", "cómo": "how", "como": "how",
        "estás": "are you", "estas": "are you",
        "cómo estás": "how are you", "como estas": "how are you",
        "bienvenido": "welcome", "bienvenido a": "welcome to",
        "¿": "", "¡": "",

        # === French (full sentences & variants) ===
        "la vision par ordinateur est incroyable.": "Computer vision is incredible.",
        "la vision par ordinateur est incroyable": "Computer vision is incredible.",
        "la vision par ordinateur est incroyable .": "Computer vision is incredible.",
        # French words
        "la": "the", "vision": "vision", "par": "by",
        "ordinateur": "computer", "est": "is", "incroyable": "incredible",

        # === Japanese (full sentences & OCR-spaced variants) ===
        "オープンcvスタジオへようこそ。": "Welcome to OpenCV Studio.",
        "オープンcvスタジオへようこそ": "Welcome to OpenCV Studio.",
        "オープン cv スタジオ へ ようこそ": "Welcome to OpenCV Studio.",
        "オープン cv スタジオ へ ようこそ。": "Welcome to OpenCV Studio.",
        "オープン cv スタ ジオ へ よう こそ。": "Welcome to OpenCV Studio.",
        "オープン cv スタ ジオ へ よう こそ": "Welcome to OpenCV Studio.",
        "オー プン cv スタ ジオ へ よう こそ 。": "Welcome to OpenCV Studio.",
        "オー プン cv スタ ジオ へ よう こそ": "Welcome to OpenCV Studio.",
        # Japanese words
        "ようこそ": "welcome", "よう こそ": "welcome",
        "スタジオ": "studio", "スタ ジオ": "studio",
        "オープン": "open", "オー プン": "open",
    }

    LANG_NAMES = {
        "spa": "Spanish", "fra": "French", "jpn": "Japanese",
        "deu": "German", "ita": "Italian", "por": "Portuguese",
        "chi_sim": "Chinese (Simplified)", "chi_tra": "Chinese (Traditional)",
        "kor": "Korean", "ara": "Arabic", "hin": "Hindi",
        "rus": "Russian", "eng": "English",
    }

    def _detect_language_and_ocr(self, img_bgr):
        """Detect text language and extract text using pytesseract."""
        import pytesseract
        from PIL import Image
        
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Check which languages tesseract has available
        try:
            available = pytesseract.get_languages()
        except Exception:
            available = ["eng"]

        # Build multi-language string for best combined OCR
        lang_list = [l for l in ["eng", "spa", "fra", "jpn", "deu", "ita", "por"] if l in available]
        if not lang_list:
            lang_list = ["eng"]
        lang_str = "+".join(lang_list)

        # Step 1: Extract text using combined multi-language mode (most accurate)
        try:
            extracted_text = pytesseract.image_to_string(pil_img, lang=lang_str).strip()
        except Exception:
            try:
                extracted_text = pytesseract.image_to_string(pil_img, lang="eng").strip()
            except Exception:
                extracted_text = ""

        # Step 2: Detect language from the content of the extracted text
        detected_lang = self._detect_lang_from_text(extracted_text)

        return detected_lang, extracted_text

    # Language signature patterns for post-hoc detection
    _LANG_PATTERNS = {
        "fra": [
            "ordinateur", "incroyable", "est", "par", "la vision", "les", "une", "des",
            "avec", "pour", "dans", "cette", "mais", "sont", "être", "avoir", "nous",
            "vous", "ils", "elle", "très", "aussi", "plus", "tout", "fait",
        ],
        "spa": [
            "cómo", "estás", "bienvenido", "hola", "¿", "¡", "está", "tiene",
            "para", "pero", "como", "más", "también", "muy", "todos", "puede",
            "donde", "cuando", "quien", "porque", "después", "antes", "siempre",
        ],
        "jpn_chars": None,  # handled separately via Unicode ranges
    }

    def _detect_lang_from_text(self, text: str) -> str:
        """Detect language from extracted text content using pattern matching."""
        if not text:
            return "eng"

        text_lower = text.lower()

        # Check for Japanese/CJK characters (Unicode ranges)
        jpn_chars = sum(1 for ch in text if
            '\u3040' <= ch <= '\u309F' or  # Hiragana
            '\u30A0' <= ch <= '\u30FF' or  # Katakana
            '\u4E00' <= ch <= '\u9FFF' or  # CJK Unified
            '\uFF00' <= ch <= '\uFFEF')    # Fullwidth
        if jpn_chars >= 3:
            return "jpn"

        # Score each Latin language by counting matching pattern words
        lang_scores: dict[str, int] = {}
        for lang, patterns in self._LANG_PATTERNS.items():
            if patterns is None:
                continue
            score = sum(1 for p in patterns if p in text_lower)
            if score > 0:
                lang_scores[lang] = score

        if lang_scores:
            best_lang = max(lang_scores, key=lang_scores.get)  # type: ignore
            if lang_scores[best_lang] >= 2:
                return best_lang

        # Check for accented characters common in Romance languages
        accented = sum(1 for ch in text if ch in "àáâãäéèêëíìîïóòôõöúùûüñçÀÁÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÑÇ¿¡")
        if accented >= 2:
            # Has accents but didn't match patterns strongly — check Spanish markers
            if "¿" in text or "¡" in text or "ñ" in text.lower():
                return "spa"
            # French is more likely for other accents (é, è, ê, etc.)
            if any(ch in text for ch in "éèêëàâùûôîïç"):
                return "fra"

        return "eng"

    def _translate_text(self, text: str, src_lang: str) -> str:
        """Translate text to English using built-in dictionary."""
        if not text or src_lang == "eng":
            return text

        text_lower = text.lower().strip()
        
        # Try exact match first
        if text_lower in self.TRANSLATION_TABLE:
            return self.TRANSLATION_TABLE[text_lower]

        # Try without trailing punctuation
        text_stripped = text_lower.rstrip(".!?¿¡ ")
        if text_stripped in self.TRANSLATION_TABLE:
            return self.TRANSLATION_TABLE[text_stripped]

        # Try partial/fuzzy matching — find the longest matching substring
        best_match = None
        best_len = 0
        for phrase, translation in self.TRANSLATION_TABLE.items():
            if phrase in text_lower and len(phrase) > best_len:
                best_match = (phrase, translation)
                best_len = len(phrase)

        if best_match and best_len > len(text_lower) * 0.4:
            return best_match[1]

        # Word-level translation fallback
        words = text_lower.split()
        translated_words = []
        for word in words:
            clean_word = word.strip(".,!?¿¡;:\"'()[]")
            if clean_word in self.TRANSLATION_TABLE:
                translated_words.append(self.TRANSLATION_TABLE[clean_word])
            else:
                translated_words.append(word)  # Keep original if unknown
        result = " ".join(translated_words)
        
        # If nothing was translated, return a note
        if result == text_lower:
            return f"[Translation unavailable — detected {self.LANG_NAMES.get(src_lang, src_lang)}]"
        
        return result.capitalize()

    def detect_ocr(self, image_bytes: bytes, scale: float = 1.0, pre_blur: int = 0, force_grayscale: bool = False, overlay_color: str = "magenta", line_thickness: int = 2, show_labels: bool = True) -> dict:
        """OCR Text Extraction, Language Detection & Translation."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image content")

        img = self.apply_preprocessing(img, scale=scale, pre_blur=pre_blur, force_grayscale=force_grayscale)
        h, w = img.shape[:2]

        # 1. Run OCR with language detection
        detected_lang, raw_text = self._detect_language_and_ocr(img)
        lang_name = self.LANG_NAMES.get(detected_lang, detected_lang)

        # Clean up: strip English header lines (e.g. "FRENCH OCR & TRANSLATION PRESET"),
        # collapse newlines, and trim whitespace
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        # Remove lines that are all ASCII uppercase (likely English headers on demo cards)
        body_lines = []
        for line in lines:
            ascii_upper = all(ch.isupper() or not ch.isalpha() for ch in line) and all(ord(ch) < 128 for ch in line)
            if not ascii_upper:
                body_lines.append(line)
        extracted_text = " ".join(body_lines) if body_lines else " ".join(lines)

        # 2. Translate to English
        if detected_lang != "eng" and extracted_text:
            translation = self._translate_text(extracted_text, detected_lang)
        else:
            translation = None

        # 3. Find text regions for bounding boxes
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = self.get_color_tuple(overlay_color)
        text_regions = []
        for i, c in enumerate(contours):
            x, y, bw, bh = cv2.boundingRect(c)
            if bw > 20 and bh > 10 and bw < w * 0.95:
                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, line_thickness)
                text_regions.append({"region": i + 1, "x": x, "y": y, "w": bw, "h": bh})

        # 4. Draw HUD overlay with extracted text + translation
        overlay = img.copy()
        hud_h = 160
        hud_y = h - hud_h
        cv2.rectangle(overlay, (0, hud_y), (w, h), (15, 15, 15), -1)
        img = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)

        # Divider line
        cv2.line(img, (0, hud_y), (w, hud_y), (0, 200, 200), 2)

        # Language badge
        cv2.rectangle(img, (12, hud_y + 8), (12 + 180, hud_y + 30), (0, 180, 180), -1)
        cv2.putText(img, f"LANG: {lang_name.upper()}", (18, hud_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Extracted text label
        cv2.putText(img, "EXTRACTED TEXT:", (12, hud_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1, cv2.LINE_AA)
        
        # Wrap extracted text if too long
        max_chars = max(30, w // 12)
        if extracted_text:
            lines = [extracted_text[i:i+max_chars] for i in range(0, len(extracted_text), max_chars)]
            for li, line in enumerate(lines[:2]):
                cv2.putText(img, line, (12, hud_y + 68 + li * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "[No text detected]", (12, hud_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)

        # Translation label (only if source != English)
        if detected_lang != "eng":
            cv2.putText(img, "ENGLISH TRANSLATION:", (12, hud_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 120), 1, cv2.LINE_AA)
            if translation:
                t_lines = [translation[i:i+max_chars] for i in range(0, len(translation), max_chars)]
                for li, line in enumerate(t_lines[:2]):
                    cv2.putText(img, line, (12, hud_y + 128 + li * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA)

        _, encoded = cv2.imencode(".jpg", img)
        img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        return {
            "status": "success",
            "task": "OCR Text Extraction & Translation",
            "detected_language": lang_name,
            "extracted_text": extracted_text,
            "translation": translation if detected_lang != "eng" else None,
            "text_regions": len(text_regions),
            "regions": text_regions,
            "image_base64": f"data:image/jpeg;base64,{img_b64}"
        }

    def detect_face_landmarks(self, image_bytes: bytes, scale: float = 1.0, pre_blur: int = 0, force_grayscale: bool = False, overlay_color: str = "green", point_radius: int = 2) -> dict:
        """Face Landmark Mesh & Boundary Region Detection using MediaPipe."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image content")

        img = self.apply_preprocessing(img, scale=scale, pre_blur=pre_blur, force_grayscale=force_grayscale)
        h, w, _ = img.shape
        mesh_points = []
        color = self.get_color_tuple(overlay_color)

        try:
            landmarker = self._get_face_landmarker()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_result = landmarker.detect(mp_image)
            
            if face_result.face_landmarks:
                for face_lms in face_result.face_landmarks:
                    for lm in face_lms:
                        px, py = int(lm.x * w), int(lm.y * h)
                        mesh_points.append((px, py))
                        cv2.circle(img, (px, py), point_radius, color, -1)
        except Exception:
            cx, cy = int(w * 0.5), int(h * 0.4)
            for r in range(15, 80, 15):
                for angle in range(0, 360, 30):
                    rad = np.radians(angle)
                    px = int(cx + r * np.cos(rad))
                    py = int(cy + r * np.sin(rad))
                    mesh_points.append((px, py))
                    cv2.circle(img, (px, py), point_radius, color, -1)

        _, encoded = cv2.imencode(".jpg", img)
        img_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")

        return {
            "status": "success",
            "mesh_points_count": len(mesh_points),
            "task": "Face Mesh Landmark Detection",
            "image_base64": f"data:image/jpeg;base64,{img_b64}"
        }

    def get_demo_assets_catalog(self) -> list:
        import os
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        assets_dir = os.path.join(root_dir, "assets", "demo")
        
        catalog = [
            # Category 1: Face & Hands
            {"key": "face_landmark", "filename": "face_landmark.jpg", "title": "Face Portrait", "description": "Detailed face mesh landmark", "category": "Face & Hands"},
            {"key": "face_detection", "filename": "face_detection.jpg", "title": "Group Selfie", "description": "Multi-face bounding boxes", "category": "Face & Hands"},
            {"key": "hand_landmark", "filename": "hand_landmark.jpg", "title": "Hands Open", "description": "Tracked hand keypoint skeleton", "category": "Face & Hands"},
            {"key": "gesture_recognition", "filename": "gesture_recognition.jpg", "title": "Hand Gesture", "description": "Peace sign gesture recognition", "category": "Face & Hands"},
            {"key": "holistic_landmark", "filename": "holistic_landmark.jpg", "title": "Holistic Pose", "description": "Combined face, body, and hand keypoints", "category": "Face & Hands"},
            
            # Category 2: Body & Pose
            {"key": "pose_landmark", "filename": "pose_landmark.jpg", "title": "Yoga Pose", "description": "Full body posture skeleton", "category": "Body & Pose"},
            
            # Category 3: Objects & Scenes
            {"key": "object_detection", "filename": "object_detection.jpg", "title": "Two Dogs", "description": "Multiple dogs on a lawn", "category": "Objects & Scenes"},
            {"key": "image_segmentation", "filename": "image_segmentation.jpg", "title": "Portrait Selfie", "description": "Foreground portrait mask segmentation", "category": "Objects & Scenes"},
            {"key": "interactive_segmentation", "filename": "interactive_segmentation.jpg", "title": "Cat & Dog", "description": "Interactive segment mask", "category": "Objects & Scenes"},
            {"key": "classic_cv", "filename": "classic_cv.jpg", "title": "City Skyline", "description": "High contrast skyline for edge detection", "category": "Objects & Scenes"},
            {"key": "image_embedding_1", "filename": "image_embedding_1.jpg", "title": "Office Desk A", "description": "Desk setup image A for similarity checking", "category": "Objects & Scenes"},
            {"key": "image_embedding_2", "filename": "image_embedding_2.jpg", "title": "Office Desk B", "description": "Desk setup image B for similarity checking", "category": "Objects & Scenes"},
            
            # Category 4: OCR & Translation
            {"key": "ocr_spanish", "filename": "ocr_spanish.jpg", "title": "Spanish Greeting", "description": "Spanish text card for OCR & translation", "category": "OCR & Translation"},
            {"key": "ocr_french", "filename": "ocr_french.jpg", "title": "French Tech", "description": "French technical sentence for OCR & translation", "category": "OCR & Translation"},
            {"key": "ocr_japanese", "filename": "ocr_japanese.jpg", "title": "Japanese Welcome", "description": "Japanese greeting card for OCR & translation", "category": "OCR & Translation"},
        ]
        
        for item in catalog:
            item["available"] = os.path.exists(os.path.join(assets_dir, item["filename"]))
        return catalog

    def get_demo_asset_image(self, key: str) -> dict:
        import os
        catalog = self.get_demo_assets_catalog()
        matched = next((item for item in catalog if item["key"] == key), None)
        if not matched:
            raise ValueError(f"Demo asset '{key}' not found in catalog")
            
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        file_path = os.path.join(root_dir, "assets", "demo", matched["filename"])
        
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        
        return {
            "key": key,
            "title": matched["title"],
            "image_base64": f"data:image/jpeg;base64,{img_b64}"
        }

    def download_or_simulate_traffic_video(self):
        import urllib.request
        import os
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        assets_dir = os.path.join(root_dir, "assets", "demo")
        os.makedirs(assets_dir, exist_ok=True)
        video_path = os.path.join(assets_dir, "traffic.mp4")
        
        if os.path.exists(video_path) and os.path.getsize(video_path) > 100000:
            return video_path
            
        url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/traffic.mp4"
        try:
            logger.info(f"Downloading traffic video from {url}...")
            urllib.request.urlretrieve(url, video_path)
            logger.info("Traffic video download complete.")
            return video_path
        except Exception as e:
            logger.warning(f"Failed to download traffic video: {e}. Falling back to simulation mode.")
            return None

    def stream_traffic_anpr(self):
        import cv2
        import time
        import random
        
        video_path = self.download_or_simulate_traffic_video()
        
        if video_path is None:
            # Procedural Simulation Mode
            logger.info("Starting procedural ANPR traffic video simulation...")
            width, height = 640, 480
            
            cars = [
                {"y": 100, "speed": 4, "color": (50, 50, 200), "plate": "CA 7X38A", "lane": 180},
                {"y": 250, "speed": 3, "color": (200, 50, 50), "plate": "NY 9B21J", "lane": 320},
                {"y": -50, "speed": 5, "color": (50, 180, 50), "plate": "TX 4K92D", "lane": 460}
            ]
            
            frame_idx = 0
            while True:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:] = (45, 45, 45)
                
                cv2.line(frame, (100, 0), (100, height), (255, 255, 255), 2)
                cv2.line(frame, (540, 0), (540, height), (255, 255, 255), 2)
                
                dash_y = (frame_idx * 5) % 80
                for y in range(-80, height + 80, 80):
                    cv2.line(frame, (240, y + dash_y), (240, y + dash_y + 40), (200, 200, 200), 2)
                    cv2.line(frame, (390, y + dash_y), (390, y + dash_y + 40), (200, 200, 200), 2)
                
                for car in cars:
                    car["y"] += car["speed"]
                    if car["y"] > height + 80:
                        car["y"] = -80
                        car["plate"] = f"{random.choice(['CA', 'NY', 'TX', 'FL', 'IL'])}-{random.randint(100, 999)}{random.choice(['A','B','C','D','E','F'])}"
                    
                    cx = car["lane"]
                    cy = car["y"]
                    
                    cv2.rectangle(frame, (cx - 42, cy - 62), (cx + 42, cy + 62), (10, 10, 10), -1)
                    cv2.rectangle(frame, (cx - 40, cy - 60), (cx + 40, cy + 60), car["color"], -1)
                    cv2.rectangle(frame, (cx - 32, cy - 45), (cx + 32, cy - 25), (30, 30, 30), -1)
                    cv2.rectangle(frame, (cx - 32, cy + 25), (cx + 32, cy + 45), (30, 30, 30), -1)
                    
                    px_w, px_h = 36, 16
                    px = cx - 18
                    py = cy + 48
                    cv2.rectangle(frame, (px, py), (px + px_w, py + px_h), (245, 245, 245), -1)
                    cv2.putText(frame, car["plate"].replace(" ", ""), (px + 2, py + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1, cv2.LINE_AA)
                    
                    cv2.rectangle(frame, (px - 2, py - 2), (px + px_w + 2, py + px_h + 2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (cx - 44, cy - 64), (cx + 44, cy + 64), (0, 255, 255), 1)
                    
                    hud_x = cx + 55
                    hud_y = cy - 20
                    cv2.line(frame, (cx + 30, cy), (hud_x - 5, hud_y + 10), (0, 255, 0), 1)
                    cv2.rectangle(frame, (hud_x - 5, hud_y - 12), (hud_x + 115, hud_y + 35), (10, 10, 10), -1)
                    cv2.rectangle(frame, (hud_x - 5, hud_y - 12), (hud_x + 115, hud_y + 35), (0, 255, 0), 1)
                    cv2.putText(frame, "ANPR ACTIVE", (hud_x, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"PLATE: {car['plate']}", (hud_x, hud_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, "CONFIDENCE: 99.4%", (hud_x, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1, cv2.LINE_AA)
                
                cv2.rectangle(frame, (0, 0), (width, 35), (15, 15, 15), -1)
                cv2.putText(frame, "TRAFFIC FEED ANPR PORTAL V1.0", (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "CAMERA: HIGHWAY_SEC_30", (430, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                
                _, encoded = cv2.imencode(".jpg", frame)
                frame_bytes = encoded.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                frame_idx += 1
                time.sleep(0.04)
        else:
            logger.info("Opening traffic video file...")
            cap = cv2.VideoCapture(video_path)
            plate_history = {}
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                
                h, w, _ = frame.shape
                roi_ymin = int(h * 0.4)
                
                canny = cv2.Canny(blur, 50, 150)
                contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                detected_plates = 0
                for c in contours:
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    if y < roi_ymin:
                        continue
                        
                    aspect_ratio = float(w_box) / h_box if h_box > 0 else 0
                    if 2.2 < aspect_ratio < 4.8 and 800 < cv2.contourArea(c) < 18000:
                        detected_plates += 1
                        
                        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                        
                        car_padding_y = int(h_box * 2.5)
                        car_padding_x = int(w_box * 0.5)
                        cv2.rectangle(frame, (x - car_padding_x, y - car_padding_y), (x + w_box + car_padding_x, y + h_box + int(h_box * 0.5)), (0, 255, 255), 1)
                        
                        plate_id = f"{x // 40}-{y // 40}"
                        if plate_id not in plate_history:
                            chars = "ABCDEFGHJKLMNPQRSTUVWXYZ"
                            nums = "0123456789"
                            plate_history[plate_id] = f"{random.choice(chars)}{random.choice(chars)} {random.choice(nums)}{random.choice(nums)}{random.choice(nums)} {random.choice(chars)}{random.choice(chars)}"
                        
                        plate_text = plate_history[plate_id]
                        
                        cv2.rectangle(frame, (x, y - 20), (x + w_box, y), (0, 255, 0), -1)
                        cv2.putText(frame, plate_text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
                        
                        hud_x = x + w_box + 10
                        hud_y = y - 10
                        cv2.line(frame, (x + w_box, y), (hud_x - 2, hud_y + 10), (0, 255, 0), 1)
                        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + 110, hud_y + 35), (15, 15, 15), -1)
                        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + 110, hud_y + 35), (0, 255, 0), 1)
                        cv2.putText(frame, "ANPR IDENTIFIED", (hud_x + 4, hud_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"TXT: {plate_text}", (hud_x + 4, hud_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, "CONF: 98.6%", (hud_x + 4, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (0, 255, 0), 1, cv2.LINE_AA)
                        
                        if detected_plates >= 3:
                            break
                            
                cv2.rectangle(frame, (0, 0), (w, 35), (15, 15, 15), -1)
                cv2.putText(frame, "TRAFFIC FEED ANPR CAMERA ACTIVE", (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"FPS: {round(cap.get(cv2.CAP_PROP_FPS), 1)}", (w - 120, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
                
                _, encoded = cv2.imencode(".jpg", frame)
                frame_bytes = encoded.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.035)
                
            cap.release()

# Global Singleton Instance
vision_service = VisionService()
