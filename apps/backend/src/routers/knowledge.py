"""FastAPI Router for Google Open Knowledge Format (OKF) Wiki articles."""

import os
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge Base"])

KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../knowledge-base"))

class ArticleMeta(BaseModel):
    key: str
    type: str
    title: str
    description: str
    tags: List[str]
    timestamp: str

class ArticleDetail(BaseModel):
    key: str
    type: str
    title: str
    description: str
    tags: List[str]
    timestamp: str
    content: str

def parse_frontmatter(file_path: str) -> tuple[Dict[str, Any], str]:
    """Helper to parse YAML frontmatter and body from a markdown file."""
    if not os.path.exists(file_path):
        return {}, ""
        
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    meta = {}
    body = text
    
    # Check if file has frontmatter
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter_text = parts[1]
            body = parts[2]
            
            # Simple line-by-line parser for standard OKF frontmatter fields
            for line in frontmatter_text.strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    
                    # Parse lists like: tags: ["yolo", "cv"]
                    if v.startswith("[") and v.endswith("]"):
                        # strip braces and split by commas
                        items = v[1:-1].split(",")
                        v = [i.strip().strip('"').strip("'") for i in items if i.strip()]
                        
                    meta[k] = v
                    
    return meta, body.strip()

@router.get("", response_model=List[ArticleMeta])
async def get_knowledge_catalog():
    """Lists all available articles in the OKF Knowledge Base."""
    if not os.path.exists(KB_DIR):
        # Gracefully handle missing directory
        return []
        
    catalog = []
    for filename in sorted(os.listdir(KB_DIR)):
        if filename.endswith(".md"):
            key = os.path.splitext(filename)[0]
            # Skip readme/index files in list view if desired, or include them.
            # We include all but let App filter or display them.
            file_path = os.path.join(KB_DIR, filename)
            try:
                meta, _ = parse_frontmatter(file_path)
                catalog.append({
                    "key": key,
                    "type": meta.get("type", "article"),
                    "title": meta.get("title", key.replace("_", " ")),
                    "description": meta.get("description", ""),
                    "tags": meta.get("tags", []),
                    "timestamp": meta.get("timestamp", "")
                })
            except Exception as e:
                # Log error and skip
                continue
    return catalog

@router.get("/{key}", response_model=ArticleDetail)
async def get_knowledge_article(key: str):
    """Retrieves metadata and full body text for a specific article key."""
    # Prevent directory traversal attacks
    clean_key = re.sub(r"[^a-zA-Z0-9_\-]", "", key)
    filename = f"{clean_key}.md"
    file_path = os.path.join(KB_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Knowledge article '{key}' not found.")
        
    try:
        meta, body = parse_frontmatter(file_path)
        return {
            "key": clean_key,
            "type": meta.get("type", "article"),
            "title": meta.get("title", clean_key.replace("_", " ")),
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
            "timestamp": meta.get("timestamp", ""),
            "content": body
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading knowledge article: {str(e)}")
