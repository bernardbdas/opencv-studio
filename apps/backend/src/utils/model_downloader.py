"""Model downloader utility module for opencv-studio FastAPI backend."""

import os
import logging
import requests
import xml.etree.ElementTree as ET
from apps.backend.src.utils.models import ModelConfig

logger = logging.getLogger("opencv-studio.downloader")

def get_workspace_root() -> str:
    """Returns absolute path to repository root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

def resolve_url_from_xml(config: ModelConfig) -> str:
    root_dir = get_workspace_root()
    xml_path = os.path.join(root_dir, ".mediapipe", "mediapipe_models.xml")
    
    xml_content = None
    if os.path.exists(xml_path):
        try:
            with open(xml_path, "r", encoding="utf-8") as f:
                xml_content = f.read()
        except Exception:
            pass
            
    if not xml_content:
        try:
            r = requests.get("https://storage.googleapis.com/mediapipe-models", timeout=5)
            if r.status_code == 200:
                xml_content = r.text
        except Exception:
            pass
            
    if not xml_content:
        return config.fallback_url
        
    try:
        root = ET.fromstring(xml_content)
        matched_keys = []
        for elem in root.iter():
            if elem.tag.endswith("Key") and elem.text:
                key_text = elem.text.strip()
                if config.search_pattern in key_text:
                    matched_keys.append(key_text)
                    
        if matched_keys:
            for k in matched_keys:
                if k.endswith(config.filename):
                    return f"https://storage.googleapis.com/mediapipe-models/{k}"
            return f"https://storage.googleapis.com/mediapipe-models/{matched_keys[0]}"
    except Exception as e:
        logger.warning(f"Error parsing XML bucket directory: {e}")
        
    return config.fallback_url

def get_model_path_by_config(config: ModelConfig) -> str:
    resolved_url = resolve_url_from_xml(config)
    return get_model_path(config.task_name, config.filename, resolved_url)

def get_model_path(task_name: str, model_filename: str, download_url: str) -> str:
    # Resolve directory for MediaPipe models in apps/backend/src/models/mediapipe/<task_name>/
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(src_dir, "models", "mediapipe", task_name)
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, model_filename)
    
    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        logger.info(f"Downloading model: {model_filename} from {download_url}...")
        try:
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            if os.path.exists(model_path):
                os.remove(model_path)
            logger.error(f"Failed to download model: {e}")
            raise e
            
    return model_path
