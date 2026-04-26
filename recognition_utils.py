
# recognition_utils.py
# Purpose: Face embeddings (FaceNet) & duplicate detection utilities.

from typing import Optional, Dict, Tuple
from pathlib import Path
import numpy as np
import csv

import torch
from facenet_pytorch import InceptionResnetV1
import cv2

_MODEL = None

def init_facenet_model() -> InceptionResnetV1:
    global _MODEL
    if _MODEL is None:
        _MODEL = InceptionResnetV1(pretrained='vggface2').eval()
    return _MODEL

def compute_embedding(bgr_img: np.ndarray) -> Optional[np.ndarray]:
    model = init_facenet_model()
    img = cv2.resize(bgr_img, (160, 160), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        emb = model(t).cpu().numpy().flatten()
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb.astype(np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def load_registry(path: Path) -> Dict[str, np.ndarray]:
    if path.exists():
        try:
            return np.load(path, allow_pickle=True).item()
        except Exception:
            return {}
    return {}

def save_registry(path: Path, reg: Dict[str, np.ndarray]):
    np.save(path, reg)

def detect_duplicates(embedding: np.ndarray, registry: Dict[str, np.ndarray], threshold: float = 0.85) -> Tuple[Optional[str], Optional[float]]:
    best_id, best_sim = None, None
    for pid, vec in registry.items():
        sim = cosine_sim(embedding, np.asarray(vec, dtype=np.float32))
        if best_sim is None or sim > best_sim:
            best_sim, best_id = sim, pid
    if best_sim is not None and best_sim >= threshold:
        return best_id, best_sim
    return None, None

def append_duplicate_report(path: Path, original_id: str, duplicate_id: str, similarity: float, video_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["original_id","duplicate_id","similarity","video"])
        w.writerow([original_id, duplicate_id, f"{similarity:.4f}", video_name])
