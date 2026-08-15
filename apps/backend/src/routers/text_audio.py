"""FastAPI Router for Text, Audio, and LLM Inference tasks in opencv-studio."""

from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter(prefix="/api/nlp", tags=["Text & Audio"])

class TextPromptRequest(BaseModel):
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256

class TextEmbedRequest(BaseModel):
    text: str

@router.post("/llm")
def llm_inference(req: TextPromptRequest):
    """Generative AI LLM Inference response lab."""
    response_text = f"🤖 [OpenCV Studio LLM]: Response generated for prompt: '{req.prompt}'. Modern vision models integrated with transformer language heads enable multimodal image reasoning."
    return {
        "status": "success",
        "prompt": req.prompt,
        "response": response_text,
        "tokens_generated": len(response_text.split())
    }

@router.post("/text-embedding")
def generate_text_embedding(req: TextEmbedRequest):
    """Generate dense vector embedding for text."""
    # Compute deterministic mock embedding vector
    import hashlib
    hash_obj = hashlib.sha256(req.text.encode()).digest()
    vec = [round(float(b) / 255.0 - 0.5, 4) for b in hash_obj[:16]]
    return {
        "status": "success",
        "text": req.text,
        "embedding": vec,
        "dimensions": len(vec)
    }
