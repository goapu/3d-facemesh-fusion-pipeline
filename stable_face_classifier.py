import cv2
import numpy as np
import argparse
import sys
import os
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Import the "brain" (Fusion) from your professional pipeline
try:
    from fusion_utils import (
        get_facemesh, 
        facemesh_process_rgb, 
        confidence_filter, 
        get_mp_edges, 
        fuse_landmarks
    )
except ImportError:
    print("Error: Could not import 'fusion_utils.py'. Please ensure it is in the same directory.")
    sys.exit(1)

# ---------------------------------------------------------
# LANDMARK INDICES (Standardized MP FaceMesh)
# ---------------------------------------------------------
# Eyes for IPD scale
LEFT_EYE = [33, 133, 160, 159, 158, 144, 145, 153, 154, 155]
RIGHT_EYE = [263, 362, 387, 386, 385, 373, 374, 380, 381, 382]

# Widths & Lengths
# Bizygomatic width (Cheekbones) - The standard for "Face Width"
IDX_ZYGOMA_L = 234 
IDX_ZYGOMA_R = 454 

# Bigonial width (Jaw) - The corner of the jaw
IDX_JAW_L = 58
IDX_JAW_R = 288

# Forehead (Temples/Frontotemporal)
IDX_TEMPLE_L = 21
IDX_TEMPLE_R = 251

# Face Height Landmarks
# 1. Physiognomic Height (Hairline to Chin) - Used for aesthetic "Shape" (Oval/Square)
IDX_HAIRLINE = 10 
IDX_MENTON = 152

# 2. Morphological Height (Nasion to Chin) - Used for MEDICAL Facial Index
IDX_NASION = 168 # Approximate Nasion (root of nose between eyes)

# ---------------------------------------------------------
# Literature Research (Medical Norms: Banister / Martin-Saller)
# ---------------------------------------------------------

def perform_literature_research():
    """
    Returns established medical anthropometric data for face measurements.
    Based on the Banister Classification for the Morphological Facial Index.
    Formula: (Nasion-Gnathion Height / Bizygomatic Width) * 100
    """
    print("Using standard medical anthropometry data (Banister Scale)...")
    
    return {
        "average_face_width_mm": 140,  # Approximate adult mean
        # The Banister Classification (Percentages)
        "facial_index_ranges": {
            "Hypereuryprosopic": (0, 79.9),     # Very Broad
            "Euryprosopic":      (80.0, 84.9),  # Broad
            "Mesoprosopic":      (85.0, 89.9),  # Medium (The Medical "Normal")
            "Leptoprosopic":     (90.0, 94.9),  # Narrow/Long
            "Hyperleptoprosopic":(95.0, 200.0)  # Very Narrow/Long
        }
    }

# ---------------------------------------------------------
# 1. The Classifier Logic
# ---------------------------------------------------------

def reproject_to_frame(fused_norm: np.ndarray, reference_pts: np.ndarray) -> np.ndarray:
    """
    The fusion process returns [Normalized X, Normalized Y, Pixel Z].
    This function scales X/Y back to pixels, but keeps Z as-is.
    """
    # 1. Compute centroids
    fused_center = fused_norm.mean(axis=0)
    ref_center = reference_pts.mean(axis=0)
    
    # 2. Center both sets
    fused_c = fused_norm - fused_center
    ref_c = reference_pts - ref_center
    
    # 3. Compute Scale Ratio based ONLY on X and Y
    scale_fused_xy = np.linalg.norm(fused_c[:, :2])
    scale_ref_xy = np.linalg.norm(ref_c[:, :2])
    
    if scale_fused_xy < 1e-6: 
        return reference_pts # Fallback
        
    s = scale_ref_xy / scale_fused_xy
    
    # 4. Apply transform:
    aligned_xy = fused_c[:, :2] * s
    aligned_z = fused_c[:, 2:] 
    
    # Recombine
    aligned_c = np.hstack([aligned_xy, aligned_z])
    
    # Translate to reference position
    return aligned_c + ref_center

def align_face_upright(points: np.ndarray):
    """
    Rotates the landmarks so the eyes are perfectly horizontal.
    """
    pts = points.copy()
    
    # Get Eye Centers
    le_center = np.mean(pts[LEFT_EYE, :2], axis=0)
    re_center = np.mean(pts[RIGHT_EYE, :2], axis=0)
    
    # Calculate angle
    dY = re_center[1] - le_center[1]
    dX = re_center[0] - le_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Center of rotation (between eyes)
    center = (le_center + re_center) / 2
    
    # Rotation Matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply to all points (Z remains unchanged)
    ones = np.ones(shape=(len(pts), 1))
    points_ones = np.hstack([pts[:, :2], ones])
    
    rotated_xy = M.dot(points_ones.T).T
    
    # Update points
    pts[:, 0] = rotated_xy[:, 0]
    pts[:, 1] = rotated_xy[:, 1]
    
    return pts

def get_dist(pts, idx1, idx2):
    return np.linalg.norm(pts[idx1] - pts[idx2])

def compute_face_dimensions(points: np.ndarray):
    """
    Calculates widths using SPECIFIC LANDMARKS.
    """
    if points is None or len(points) == 0:
        return None, None, None
        
    # 1. Align face first (remove tilt)
    points = align_face_upright(points)

    # 2. Measure Distances (in pixels)
    physio_height = get_dist(points, IDX_HAIRLINE, IDX_MENTON) # For Shape (Oval/Square)
    morph_height = get_dist(points, IDX_NASION, IDX_MENTON)    # For Medical Index
    
    forehead_w = get_dist(points, IDX_TEMPLE_L, IDX_TEMPLE_R)
    cheek_w = get_dist(points, IDX_ZYGOMA_L, IDX_ZYGOMA_R)
    jaw_w = get_dist(points, IDX_JAW_L, IDX_JAW_R)

    # --- SCALE ESTIMATION (IPD) ---
    le_center = np.mean(points[LEFT_EYE, :2], axis=0)
    re_center = np.mean(points[RIGHT_EYE, :2], axis=0)
    ipd_px = np.linalg.norm(re_center - le_center)
    
    # Average human IPD is ~63mm.
    if ipd_px < 1.0:
        mm_per_px = 0.0
    else:
        mm_per_px = 63.0 / (ipd_px + 1e-6)

    dims = {
        "physio_height": physio_height,
        "morph_height": morph_height,
        "forehead_width": forehead_w,
        "cheek_width": cheek_w,
        "jaw_width": jaw_w
    }

    scale_info = {
        "ipd_px": ipd_px,
        "mm_per_px": mm_per_px,
        "cheek_width_mm": cheek_w * mm_per_px
    }

    debug_indices = {
        "morph_height": (IDX_NASION, IDX_MENTON),
        "forehead": (IDX_TEMPLE_L, IDX_TEMPLE_R),
        "cheek": (IDX_ZYGOMA_L, IDX_ZYGOMA_R),
        "jaw": (IDX_JAW_L, IDX_JAW_R)
    }

    return dims, debug_indices, scale_info

def classify_face_shape(dims: Dict[str, float], medical_data: Dict = None):
    """
    Classifies face using the Medical Banister Scale and Aesthetic sub-types.
    """
    HM = dims["morph_height"] # Morphological Height (Nasion-Menton)
    HP = dims["physio_height"] # Physiognomic Height (Hairline-Menton)
    F = dims["forehead_width"]
    C = dims["cheek_width"]
    J = dims["jaw_width"]

    if any(v is None or v == 0 for v in [HM, HP, F, C, J]):
        return "Unknown", "Unknown", {}

    # --- MEDICAL INDICES ---
    # Formula: (Morphological Height / Bizygomatic Width) * 100
    facial_index = (HM / C) * 100
    
    # Ratios for Aesthetic Shape Logic
    jaw_cheek_ratio = J / C
    fore_cheek_ratio = F / C
    
    ratios = {
        "facial_index": facial_index,
        "jaw_cheek_ratio": jaw_cheek_ratio,
        "fore_cheek_ratio": fore_cheek_ratio,
    }

    # --- 1. MEDICAL CLASSIFICATION (Banister) ---
    medical_cat = "Unknown"
    ranges = medical_data.get("facial_index_ranges") if medical_data else {}
    
    if ranges:
        if facial_index < ranges["Hypereuryprosopic"][1]:
            medical_cat = "Hypereuryprosopic" # Very Broad
        elif facial_index < ranges["Euryprosopic"][1]:
            medical_cat = "Euryprosopic"      # Broad
        elif facial_index < ranges["Mesoprosopic"][1]:
            medical_cat = "Mesoprosopic"      # Medium/Average
        elif facial_index < ranges["Leptoprosopic"][1]:
            medical_cat = "Leptoprosopic"     # Narrow
        else:
            medical_cat = "Hyperleptoprosopic"# Very Narrow

    # --- 2. AESTHETIC SHAPE SUB-CLASSIFICATION ---
    # We filter primarily by the Medical Category, then refine by Jaw/Forehead
    
    aesthetic_shape = "Unknown"
    
    # BROAD FACES (Hypereuryprosopic / Euryprosopic)
    if "euryprosopic" in medical_cat.lower():
        # Broad face. Is it Square (angular) or Round (soft)?
        # High jaw width suggests Square.
        if jaw_cheek_ratio > 0.88:
            aesthetic_shape = "Square"
        else:
            aesthetic_shape = "Round"

    # MEDIUM FACES (Mesoprosopic)
    elif "mesoprosopic" in medical_cat.lower():
        # The Mathematical Average.
        if jaw_cheek_ratio > 0.9:
            aesthetic_shape = "Rectangle" # Strong jaw on average face
        elif jaw_cheek_ratio < 0.76:
            # Narrow jaw. Check Forehead.
            if fore_cheek_ratio > 0.95:
                aesthetic_shape = "Heart" # Wide forehead
            else:
                aesthetic_shape = "Diamond" # Narrow forehead & chin
        else:
            aesthetic_shape = "Oval" # Balanced

    # LONG FACES (Leptoprosopic / Hyperleptoprosopic)
    elif "leptoprosopic" in medical_cat.lower(): 
        if jaw_cheek_ratio > 0.88:
            aesthetic_shape = "Rectangle" # Long face + Broad jaw
        else:
            aesthetic_shape = "Oblong" # Long face + Narrow/Average jaw

    # Special Case: Triangle (Pear)
    if (J / F) > 1.15:
        aesthetic_shape = "Triangle"

    return medical_cat, aesthetic_shape, ratios

def classify_face_size(width_mm: float, medical_data: Dict = None):
    if width_mm <= 0: return "Unknown"
    if width_mm > 500: return "Error" 
    
    avg_width = 140
    if medical_data:
        avg_width = medical_data.get("average_face_width_mm", 140)
        
    if width_mm < (avg_width - 8):
        return "Small"
    elif width_mm <= (avg_width + 8):
        return "Medium" 
    else:
        return "Big"

# ---------------------------------------------------------
# 2. The Integrated Pipeline
# ---------------------------------------------------------

def process_source(source, frames_to_capture=30, interactive=True):
    if interactive:
        print(f"--- Starting Medical Face Classifier ---")
        medical_data = perform_literature_research() 
        print(f"Target: Collect {frames_to_capture} stable frames.")
    else:
        medical_data = perform_literature_research()
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source {source}")
        return None, None, None, None, None, None, None

    face_mesh = get_facemesh(max_faces=1)
    edges = get_mp_edges()
    
    frames_lms = [] 
    last_frame_bgr = None
    last_valid_pts = None 
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        last_frame_bgr = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pts = facemesh_process_rgb(face_mesh, frame_rgb)
        
        if interactive:
            status_text = f"Scanning... {len(frames_lms)}/{frames_to_capture}"
            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 255), 2, cv2.LINE_AA)
        
        if pts is not None:
            pts = confidence_filter(pts, edges, thresh=0.5)
            frames_lms.append(pts)
            last_valid_pts = pts 
            
            if interactive:
                # Visualize Morphological Height (Nasion to Chin)
                n = pts[IDX_NASION]
                g = pts[IDX_MENTON]
                cv2.line(frame, (int(n[0]), int(n[1])), (int(g[0]), int(g[1])), (0,255,255), 2)
                for x, y, _ in pts[::5]:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        if interactive:
            cv2.imshow("Stable Face Classifier (Scanning)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release(); cv2.destroyAllWindows(); return None, None, None, None, None, None, None
        
        if len(frames_lms) >= frames_to_capture:
            break

    cap.release()
    if interactive: cv2.destroyWindow("Stable Face Classifier (Scanning)")
    
    if len(frames_lms) < 5 or last_valid_pts is None:
        if interactive: print("Not enough frames.")
        return None, None, None, None, None, None, None

    if interactive: print("Fusing frames...")
    
    fused_mesh = fuse_landmarks(frames_lms)
    
    if fused_mesh.size == 0:
        if interactive: print("Fusion failed.")
        return None, None, None, None, None, None, None

    fused_mesh = reproject_to_frame(fused_mesh, last_valid_pts)

    # 3. Compute
    dims, debug_indices, scale_info = compute_face_dimensions(fused_mesh)
    med_label, aes_label, ratios = classify_face_shape(dims, medical_data)
    size_label = classify_face_size(scale_info["cheek_width_mm"], medical_data)
    
    if interactive:
        print(f"\n>>> FINAL DIAGNOSIS: {med_label} ({aes_label}) <<<")
        print(f"Facial Index: {ratios['facial_index']:.1f} (Banister Scale)")
        print(f"Est. Bizygomatic Width: {scale_info['cheek_width_mm']:.1f}mm")

    # Visualize
    result_img = last_frame_bgr.copy()
    overlay = result_img.copy()
    cv2.rectangle(overlay, (0, 0), (result_img.shape[1], 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, result_img, 0.3, 0, result_img)

    for x, y, _ in fused_mesh:
        cv2.circle(result_img, (int(x), int(y)), 1, (0, 255, 0), -1)

    # Draw Morphological Height (Nasion to Menton) - YELLOW
    idxA, idxB = debug_indices["morph_height"]
    ptA = (int(fused_mesh[idxA][0]), int(fused_mesh[idxA][1]))
    ptB = (int(fused_mesh[idxB][0]), int(fused_mesh[idxB][1]))
    cv2.line(result_img, ptA, ptB, (0, 255, 255), 2)
    cv2.putText(result_img, "Medical Height", (ptA[0]+10, ptA[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    # Widths (Red)
    colors = [(0,0,255), (0,0,255), (0,0,255)]
    keys = ["forehead", "cheek", "jaw"]
    for i, k in enumerate(keys):
        idx1, idx2 = debug_indices[k]
        p1 = (int(fused_mesh[idx1][0]), int(fused_mesh[idx1][1]))
        p2 = (int(fused_mesh[idx2][0]), int(fused_mesh[idx2][1]))
        cv2.line(result_img, p1, p2, colors[i], 2)
        cv2.putText(result_img, k.capitalize(), (p2[0]+5, p2[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200),1)

    # Text Info
    # Medical Term
    cv2.putText(result_img, f"Medical: {med_label}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Aesthetic Term
    cv2.putText(result_img, f"Shape:   {aes_label}", (30, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 2, cv2.LINE_AA)
    
    # Size
    size_text = f"Size:    {size_label} ({scale_info['cheek_width_mm']:.0f}mm)"
    cv2.putText(result_img, size_text, (30, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Index Score
    ratio_text = f"Facial Index: {ratios['facial_index']:.1f}"
    cv2.putText(result_img, ratio_text, (30, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    
    return med_label, aes_label, size_label, dims, scale_info, result_img, ratios

def process_batch_folder(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file_path = output_path / "face_shape_summary.csv"
    csv_exists = csv_file_path.exists()
    
    csv_file = open(csv_file_path, 'a', newline='')
    fieldnames = [
        'filename', 'medical_class', 'aesthetic_shape', 'face_size', 
        'bizygomatic_width_mm', 'morph_height_mm', 'facial_index', 
        'jaw_cheek_ratio', 'fore_cheek_ratio'
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    if not csv_exists:
        writer.writeheader()

    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    files = []
    for ext in video_extensions:
        files.extend(input_path.glob(ext))
    
    files = sorted(files)
    print(f"Found {len(files)} videos in {input_dir}")

    for idx, video_file in enumerate(files):
        print(f"[{idx+1}/{len(files)}] Processing: {video_file.name} ...")
        
        # Unpack result including new medical label
        med_label, aes_label, size_label, metrics, scale_info, result_img, ratios = process_source(str(video_file), frames_to_capture=30, interactive=False)
        
        if med_label:
            out_img_name = f"{video_file.stem}_shape.jpg"
            cv2.imwrite(str(output_path / out_img_name), result_img)
            
            mm = scale_info['mm_per_px']
            row = {
                'filename': video_file.name,
                'medical_class': med_label,
                'aesthetic_shape': aes_label,
                'face_size': size_label,
                'bizygomatic_width_mm': f"{scale_info['cheek_width_mm']:.1f}", 
                'morph_height_mm': f"{metrics['morph_height'] * mm:.1f}",
                'facial_index': f"{ratios['facial_index']:.3f}",
                'jaw_cheek_ratio': f"{ratios['jaw_cheek_ratio']:.3f}",
                'fore_cheek_ratio': f"{ratios['fore_cheek_ratio']:.3f}",
            }
            writer.writerow(row)
            print(f"   -> {med_label} ({aes_label}) Index: {ratios['facial_index']:.1f}")
            
            cv2.imshow("Batch Processing", result_img)
            cv2.waitKey(1)
        else:
            print("   -> Failed: No face detected.")

    csv_file.close()
    print(f"\nBatch processing complete. Results saved to: {output_dir}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Medical-Standard Face Shape Classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--webcam", action="store_true", help="Use webcam (index 0)")
    group.add_argument("--video", type=str, help="Path to single video file")
    group.add_argument("--input_dir", type=str, help="Directory containing video files for batch processing")
    
    parser.add_argument("--output_dir", type=str, help="Directory to save results (required for input_dir)")
    
    args = parser.parse_args()
    
    if args.input_dir:
        if not args.output_dir:
            print("Error: --output_dir is required when using --input_dir")
            return
        process_batch_folder(args.input_dir, args.output_dir)
    else:
        source = 0 if args.webcam else args.video
        # Unpack 7 values
        result = process_source(source, interactive=True)
        if result[5] is not None:
            img = result[5]
            cv2.imshow("Final Result", img)
            print("Press any key to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if args.output_dir:
                out_path = Path(args.output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                fname = "webcam_result.jpg" if args.webcam else Path(args.video).stem + "_result.jpg"
                cv2.imwrite(str(out_path / fname), img)
                print(f"Saved result to {out_path / fname}")

if __name__ == "__main__":
    main()