
# fusion_utils.py
# Purpose: FaceMesh extraction, robust fusion, metrics, and OBJ exports.

from typing import List, Dict, Optional, Tuple
from contextlib import closing
from collections import defaultdict
import numpy as np
import math, json, csv, time
from pathlib import Path

# ---- MediaPipe ----
import mediapipe as mp

# ---- OpenCV (for frames & drawing) ----
import cv2

# Landmark indices map
IDX = dict(
    L_EAR=234, R_EAR=454,
    L_ZYG=93, R_ZYG=323,
    NASION=168, PRONASALE=1,
    SUBNASALE=2, STOMION=13, MENTON=152
)

LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153, 154, 155]
RIGHT_EYE = [263, 362, 387, 386, 385, 373, 374, 380, 381, 382]

# -------------------------------
# Video IO
# -------------------------------
def read_video_frames(video_path: Path, max_dim: int = 1280, frame_stride: int = 1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    idx = 0
    last_bgr = None
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        last_bgr = frame_bgr.copy()
        if (idx % max(1, frame_stride)) != 0:
            idx += 1; continue
        idx += 1
        h, w = frame_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            frame_bgr = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        yield frame_rgb, frame_bgr, idx
    cap.release()

# -------------------------------
# FaceMesh helpers
# -------------------------------
def get_facemesh(max_faces: int = 1):
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

def facemesh_process_rgb(face_mesh, rgb_img: np.ndarray) -> Optional[np.ndarray]:
    """Run MediaPipe FaceMesh on an RGB frame, return Nx3 array in pixel units (x,y,z)."""
    H, W = rgb_img.shape[:2]
    res = face_mesh.process(rgb_img)
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark  # single-person assumption
    pts = np.zeros((len(lms), 3), dtype=np.float32)
    for i, p in enumerate(lms):
        pts[i, 0] = float(p.x) * W
        pts[i, 1] = float(p.y) * H
        pts[i, 2] = float(p.z) * W  # scale z to width
    return pts

# -------------------------------
# Confidence scoring & interpolation
# -------------------------------
def interocular_distance_px(pts: np.ndarray) -> float:
    le = np.mean(pts[LEFT_EYE, :2], axis=0)
    re = np.mean(pts[RIGHT_EYE, :2], axis=0)
    return float(np.linalg.norm(re - le)) + 1e-9

def get_mp_edges() -> List[Tuple[int, int]]:
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION, FACEMESH_CONTOURS
    E = set()
    for s in (FACEMESH_TESSELATION, FACEMESH_CONTOURS):
        for a, b in s:
            if a == b: continue
            E.add(tuple(sorted((int(a), int(b)))))
    return sorted(list(E))

def confidence_filter(pts: np.ndarray, edges: List[Tuple[int,int]], thresh: float = 0.5) -> np.ndarray:
    out = pts.copy().astype(np.float32)
    iod = interocular_distance_px(pts)
    if iod <= 1e-6:
        return out
    n = len(pts)
    adj = [[] for _ in range(n)]
    for i, j in edges:
        if i < n and j < n:
            adj[i].append(j); adj[j].append(i)
    for i in range(n):
        nbrs = adj[i]
        if not nbrs: 
            continue
        nbr_mean = out[nbrs].mean(axis=0)
        err = float(np.linalg.norm((out[i, :2] - nbr_mean[:2])))
        c = 1.0 / (1.0 + (err / (0.03 * iod)))  # confidence ~ smoothness
        if c < thresh:
            out[i] = nbr_mean
    return out

# -------------------------------
# Fusion
# -------------------------------
def procrustes_align(base_xy: np.ndarray, target_xy: np.ndarray) -> Tuple[np.ndarray, float]:
    b = base_xy - base_xy.mean(axis=0, keepdims=True)
    t = target_xy - target_xy.mean(axis=0, keepdims=True)
    nb = np.linalg.norm(b) + 1e-9
    nt = np.linalg.norm(t) + 1e-9
    b /= nb; t /= nt
    U, _, Vt = np.linalg.svd(t.T @ b)
    R = U @ Vt
    t_aligned = (t @ R) * (nb / nt)
    diff = t_aligned - b
    rms = float(np.sqrt((diff**2).sum() / diff.size))
    return t_aligned, rms

def huber_weights(errors: np.ndarray, delta: float = 1.345) -> np.ndarray:
    w = np.ones_like(errors, dtype=np.float32)
    mask = np.abs(errors) > delta
    w[mask] = delta / (np.abs(errors[mask]) + 1e-9)
    return w

def fuse_landmarks(frames: List[np.ndarray]) -> np.ndarray:
    if not frames:
        return np.zeros((0, 3), dtype=np.float32)
    shapes = [f[:, :2] for f in frames]
    K = len(shapes)
    if K == 1:
        return frames[0].copy()
    pair_rms = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(i+1, K):
            _, rms = procrustes_align(shapes[i], shapes[j])
            pair_rms[i, j] = pair_rms[j, i] = rms
    ref_idx = int(np.argmin(pair_rms.sum(axis=1)))
    ref_xy = shapes[ref_idx]
    aligned_full = []
    errs = []
    for f in frames:
        a2d, rms = procrustes_align(ref_xy, f[:, :2])
        z = f[:, 2] - f[:, 2].mean()
        aligned_full.append(np.hstack([a2d, z[:, None]]))
        errs.append(rms)
    w = huber_weights(np.array(errs, dtype=np.float32)).reshape((-1, 1, 1))
    aligned_full = np.stack(aligned_full, axis=0)
    fused = np.sum(aligned_full * w, axis=0) / np.sum(w, axis=0)
    return fused.astype(np.float32)

# -------------------------------
# Geodesic path (Dijkstra)
# -------------------------------
def dijkstra_path(points: np.ndarray, edges: List[Tuple[int, int]], src: int, dst: int):
    import heapq
    n = len(points)
    if src >= n or dst >= n: return None, None
    adj = [[] for _ in range(n)]
    for i, j in edges:
        if i < n and j < n:
            xi, yi, zi = points[i]
            xj, yj, zj = points[j]
            w = float(np.sqrt((xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2))
            adj[i].append((j, w)); adj[j].append((i, w))
    dist = [float("inf")] * n
    prev = [-1] * n
    pq = []
    dist[src] = 0.0
    heapq.heappush(pq, (0.0, src))
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]: continue
        if u == dst: break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dist[dst] == float("inf"): return None, None
    path = []
    cur = dst
    while cur != -1:
        path.append(cur); cur = prev[cur]
    path.reverse()
    return float(dist[dst]), path

# -------------------------------
# Metrics
# -------------------------------
def get_mp_triangles() -> np.ndarray:
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
    edges = np.array(list(FACEMESH_TESSELATION), dtype=np.int32)
    adj = defaultdict(set)
    for a, b in edges:
        a, b = int(a), int(b)
        adj[a].add(b); adj[b].add(a)
    tris = set()
    for i, nbrs in adj.items():
        nbrs = list(nbrs)
        for a in range(len(nbrs)):
            for b in range(a+1, len(nbrs)):
                j, k = int(nbrs[a]), int(nbrs[b])
                if j in adj[k]:
                    tri = tuple(sorted((i, j, k)))
                    tris.add(tri)
    return np.array(sorted(list(tris)), dtype=np.int32)

def _P(M: np.ndarray, i: int) -> Tuple[float, float, float]:
    p = M[i]
    return float(p[0]), float(p[1]), float(p[2])

def _euclid2(a, b):
    if a is None or b is None: return None
    (x1,y1,_),(x2,y2,_) = a,b
    return float(np.hypot(x1-x2, y1-y2))

def _euclid3(a, b):
    if a is None or b is None: return None
    (x1,y1,z1),(x2,y2,z2) = a,b
    return float(np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2))

def get_mp_edges_cached():
    # convenience wrapper for in-module cache
    if not hasattr(get_mp_edges_cached, "_cache"):
        get_mp_edges_cached._cache = get_mp_edges()
    return get_mp_edges_cached._cache

def compute_metrics(mesh: np.ndarray, edges: Optional[List[Tuple[int,int]]] = None) -> Dict:
    M = mesh
    pL_e, pR_e = _P(M, IDX["L_EAR"]), _P(M, IDX["R_EAR"])
    pL_z, pR_z = _P(M, IDX["L_ZYG"]), _P(M, IDX["R_ZYG"])
    pNa, pPr = _P(M, IDX["NASION"]), _P(M, IDX["PRONASALE"])
    pSn, pSt, pMe = _P(M, IDX["SUBNASALE"]), _P(M, IDX["STOMION"]), _P(M, IDX["MENTON"])

    ear_to_ear_straight = _euclid2(pL_e, pR_e)
    ear_to_ear_over_bridge = _euclid2(pL_e, pNa) + _euclid2(pNa, pR_e)

    if edges is None:
        edges = get_mp_edges_cached()

    ear_to_ear_geodesic_over_bridge = None
    path_over_bridge = None
    d1, path1 = dijkstra_path(M, edges, IDX["L_EAR"], IDX["NASION"])
    d2, path2 = dijkstra_path(M, edges, IDX["NASION"], IDX["R_EAR"])
    if d1 is not None and d2 is not None:
        ear_to_ear_geodesic_over_bridge = float(d1 + d2)
        path_over_bridge = (path1 or []) + ((path2[1:] if path2 else []))

    bizygomatic_width = _euclid2(pL_z, pR_z)
    menton_sellion_length = _euclid2(pMe, pNa)
    nose_bridge_height_2d = _euclid2(pNa, pPr)
    nose_bridge_height_3d = _euclid3(pNa, pPr)
    lip_to_chin_length = _euclid2(pSt, pMe)
    face_length_to_width_ratio = (menton_sellion_length / bizygomatic_width) if bizygomatic_width else None
    subnasale_to_menton = _euclid2(pSn, pMe)

    return {
        "units": "pixels",
        "ear_to_ear_straight": ear_to_ear_straight,
        "ear_to_ear_over_bridge": ear_to_ear_over_bridge,
        "ear_to_ear_geodesic_over_bridge": ear_to_ear_geodesic_over_bridge,
        "bizygomatic_width": bizygomatic_width,
        "menton_sellion_length": menton_sellion_length,
        "nose_bridge_height_2d": nose_bridge_height_2d,
        "nose_bridge_height_3d": nose_bridge_height_3d,
        "lip_to_chin_length": lip_to_chin_length,
        "face_length_to_width_ratio": face_length_to_width_ratio,
        "subnasale_to_menton": subnasale_to_menton,
        "points": IDX.copy(),
        "paths": {"path_indices_over_bridge": path_over_bridge}
    }

# -------------------------------
# OBJ & rendering
# -------------------------------
def vertices_from_landmarks(landmarks: np.ndarray, z_mult: float = 1.0, flip_y: bool = False) -> np.ndarray:
    V = landmarks.copy().astype(np.float32)
    V[:, 0] -= V[:, 0].mean()
    V[:, 1] -= V[:, 1].mean()
    if flip_y:
        V[:, 1] *= -1.0
    xy_range = max(1e-6, float(np.ptp(V[:,0]) + np.ptp(V[:,1])) / 2.0)
    z_range  = max(1e-6, float(np.ptp(V[:,2])))
    z_factor = (xy_range / z_range) * float(z_mult)
    V[:, 2] = (V[:, 2] - V[:, 2].mean()) * z_factor
    return V

def write_full_obj(path: Path, vertices: np.ndarray, faces: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

def write_feature_obj_subset(path: Path, landmarks: np.ndarray, seq: List[int], group_name: str):
    unique_idx = []
    idx_map = {}
    for i in seq:
        if i not in idx_map:
            idx_map[i] = len(unique_idx)
            unique_idx.append(i)
    verts = landmarks[unique_idx, :]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write(f"g {group_name}\n")
        remapped = [idx_map[i] + 1 for i in seq]
        f.write("l " + " ".join(str(x) for x in remapped) + "\n")

def render_fused_image_on_frame(fused_xy: np.ndarray, frame_bgr: Optional[np.ndarray], out_path: Path):
    if frame_bgr is None:
        W = H = 800
        img = np.ones((H, W, 3), dtype=np.uint8) * 255
        x = fused_xy[:,0]; y = fused_xy[:,1]
        x = (x - x.min()) / (x.ptp() + 1e-6) * (W * 0.8) + W*0.1
        y = (y - y.min()) / (y.ptp() + 1e-6) * (H * 0.8) + H*0.1
        pts = np.stack([x, y], axis=1).astype(int)
    else:
        img = frame_bgr.copy()
        pts = fused_xy.astype(int)
    for p in pts:
        cv2.circle(img, (int(p[0]), int(p[1])), 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

# -------------------------------
# Feature sequences builder
# -------------------------------
def build_feature_sequences(mesh: np.ndarray, metrics: Dict) -> Dict[str, List[int]]:
    feats = {
        "ear2ear":       [IDX["L_EAR"], IDX["R_EAR"]],
        "bizygomatic":   [IDX["L_ZYG"], IDX["R_ZYG"]],
        "menton_sellion":[IDX["MENTON"], IDX["NASION"]],
        "nosebridge":    [IDX["NASION"], IDX["PRONASALE"]],
        "lip2chin":      [IDX["STOMION"], IDX["MENTON"]],
    }
    path_over_bridge = metrics.get("paths", {}).get("path_indices_over_bridge")
    if path_over_bridge:
        feats["ear_over_bridge"] = path_over_bridge
    return feats
