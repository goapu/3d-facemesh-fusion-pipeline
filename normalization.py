
# normalization.py
# Purpose: Provide two normalization modes for facial landmarks.
# - Generic normalization: centroid translation + interocular scaling + eye-line rotation
# - Mask normalization: nasion-centered + bizygomatic scaling + ear-line rotation
#
# Landmark indices are based on MediaPipe FaceMesh (468 points).

from typing import Tuple, Dict
import numpy as np

# Landmark indices
LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153, 154, 155]
RIGHT_EYE = [263, 362, 387, 386, 385, 373, 374, 380, 381, 382]

IDX = dict(
    L_EAR=234, R_EAR=454,
    L_ZYG=93, R_ZYG=323,
    NASION=168, PRONASALE=1,
    SUBNASALE=2, STOMION=13, MENTON=152
)

def _eye_centers_xy(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    le = np.mean(pts[LEFT_EYE, :2], axis=0)
    re = np.mean(pts[RIGHT_EYE, :2], axis=0)
    return le.astype(np.float32), re.astype(np.float32)

def normalize_generic(pts: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Generic normalization for invariance:
      - translate to centroid
      - rotate to make eye line horizontal
      - scale by interocular distance
    Returns normalized XY and metadata.
    """
    M = pts.astype(np.float32).copy()
    xy = M[:, :2]
    centroid = xy.mean(axis=0)
    xy_c = xy - centroid

    le, re = _eye_centers_xy(M)
    le -= centroid; re -= centroid
    v = re - le
    theta = float(np.arctan2(v[1], v[0] + 1e-9))
    c, s = float(np.cos(-theta)), float(np.sin(-theta))
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    xy_r = xy_c @ R.T

    iod = float(np.linalg.norm(re - le)) + 1e-9
    xy_n = xy_r / iod

    meta = {"mode": "generic", "centroid": centroid.tolist(), "theta": theta, "scale": iod}
    return xy_n.astype(np.float32), meta

def normalize_for_mask(pts: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Mask-fitting normalization:
      - translate to nasion (#168)
      - rotate to make ear line (#234-#454) horizontal
      - scale by bizygomatic width (#93-#323)
    Returns normalized XY and metadata.
    """
    M = pts.astype(np.float32).copy()
    xy = M[:, :2]
    nasion = xy[IDX["NASION"]].copy()

    # translate
    xy_c = xy - nasion

    left_ear, right_ear = xy[IDX["L_EAR"]], xy[IDX["R_EAR"]]
    v = right_ear - left_ear
    theta = float(np.arctan2(v[1], v[0] + 1e-9))

    c, s = float(np.cos(-theta)), float(np.sin(-theta))
    R = np.array([[c, -s],[s, c]], dtype=np.float32)
    xy_r = xy_c @ R.T

    left_z, right_z = xy[IDX["L_ZYG"]], xy[IDX["R_ZYG"]]
    bizyg = float(np.linalg.norm(right_z - left_z)) + 1e-9
    xy_n = xy_r / bizyg

    meta = {"mode": "mask", "center": nasion.tolist(), "theta": theta, "scale": bizyg}
    return xy_n.astype(np.float32), meta

def nasion_menton_ratio(pts: np.ndarray) -> float:
    """Return ratio (nasion->menton) / bizygomatic width (in pixel units)."""
    M = pts.astype(np.float32)
    nasion = M[IDX["NASION"], :2]
    menton = M[IDX["MENTON"], :2]
    left_z, right_z = M[IDX["L_ZYG"], :2], M[IDX["R_ZYG"], :2]
    num = float(np.linalg.norm(menton - nasion))
    den = float(np.linalg.norm(right_z - left_z)) + 1e-9
    return num / den
