
import os, sys, csv, json, time, math, argparse, re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2

# ---- Local modules ----
from normalization import normalize_generic, normalize_for_mask, nasion_menton_ratio
from fusion_utils import (
    read_video_frames, get_facemesh, facemesh_process_rgb,
    get_mp_triangles, get_mp_edges, confidence_filter,
    fuse_landmarks, compute_metrics, vertices_from_landmarks,
    write_full_obj, write_feature_obj_subset, render_fused_image_on_frame,
    build_feature_sequences
)
from recognition_utils import (
    compute_embedding, load_registry, save_registry,
    detect_duplicates, append_duplicate_report
)

# -------------------------------
# Helpers
# -------------------------------
def extract_id_from_stem(stem: str) -> str:
    m = re.search(r'(\d+)', stem)
    if m:
        return m.group(1)
    return ''.join([c for c in stem if c.isalnum()]) or "1"

# -------------------------------
# Main per-video processing
# -------------------------------
def process_single_video(video_path: Path, out_dir: Path, out_root: Path,
                         export_type: str = "all",
                         frame_stride: int = 1,
                         min_frames: int = 12,
                         z_mult: float = 1.0,
                         flip_y: bool = False,
                         mask_normalization: bool = False,
                         ref_bizyg_mm: Optional[float] = None,
                         mm_per_px: Optional[float] = None,
                         overwrite: bool = False,
                         verbose: bool = False,
                         dup_threshold: float = 0.85,
                         registry_path: Optional[Path] = None,
                         dup_report_path: Optional[Path] = None) -> Dict:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    video_id = extract_id_from_stem(video_path.stem)

    fused_json_path = out_dir / f"{video_id}_fused.json"
    if fused_json_path.exists() and (export_type in ("all","fused","json_only","mesh_only")) and (not overwrite):
        if verbose: print(f"[SKIP] fused already exists for {video_id}: {fused_json_path.name}")
        return {"video": video_path.name, "id": video_id, "status": "skipped"}

    # Load FaceMesh
    face_mesh = get_facemesh(max_faces=1)
    edges = get_mp_edges()
    frames_lms: List[np.ndarray] = []
    last_bgr = None

    # Progress window
    win_name = f"Processing {video_path.name} (press 'q' to stop preview)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Read frames
    for frgb, fbgr, idx in read_video_frames(video_path, frame_stride=frame_stride):
        last_bgr = fbgr
        pts = facemesh_process_rgb(face_mesh, frgb)
        # draw progress
        preview = fbgr.copy()
        if pts is not None and len(pts) >= 468:
            pts = confidence_filter(pts, edges, thresh=0.5)
            frames_lms.append(pts)
            for (x,y,_) in pts[::8]:  # subsample for speed
                cv2.circle(preview, (int(x), int(y)), 1, (0,255,0), -1)
        cv2.putText(preview, f"Frames collected: {len(frames_lms)}", (12,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,200,255), 2, cv2.LINE_AA)
        cv2.imshow(win_name, preview)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cv2.destroyWindow(win_name)

    if len(frames_lms) < max(1, min_frames):
        print(f"[WARN] Not enough landmark frames in {video_path.name} (got {len(frames_lms)}, need >= {min_frames})")
        return {"video": video_path.name, "id": video_id, "status": "insufficient_frames"}

    fused = fuse_landmarks(frames_lms)
    if fused.size == 0:
        print(f"[WARN] Fusion failed for {video_path.name}")
        return {"video": video_path.name, "id": video_id, "status": "fuse_failed"}

    # Metrics (px)
    metrics_px = compute_metrics(fused, edges=edges)

    # Optional mask-specific metric
    extra_metrics = {}
    if mask_normalization:
        try:
            extra_metrics["nasion_menton_over_bizyg_ratio"] = float(nasion_menton_ratio(fused))
        except Exception:
            pass

    # mm scaling
    mm_scale = None
    if mm_per_px and mm_per_px > 0:
        mm_scale = float(mm_per_px)
    elif ref_bizyg_mm and metrics_px.get("bizygomatic_width"):
        bw_px = float(metrics_px["bizygomatic_width"])
        if bw_px > 0:
            mm_scale = float(ref_bizyg_mm) / bw_px
    metrics_mm = None
    if mm_scale:
        metrics_mm = {}
        for k, v in metrics_px.items():
            if isinstance(v, (int, float)) and (k != "face_length_to_width_ratio"):
                metrics_mm[k] = float(v) * mm_scale
        if "nasion_menton_over_bizyg_ratio" in extra_metrics:
            # ratio is unitless; keep as-is
            pass
        metrics_mm["units"] = "mm"
        metrics_mm["mm_per_pixel"] = mm_scale

    # Fused visualization and embedding
    fused_img_path = out_dir / f"{video_id}_fused.jpg"
    render_fused_image_on_frame(fused[:, :2], last_bgr, fused_img_path)

    # Duplicate detection before saving outputs
    is_duplicate = False
    matched_id = None
    sim_val = None
    if registry_path is not None:
        registry = load_registry(registry_path)
        bgr = cv2.imread(str(fused_img_path))
        emb_vec = compute_embedding(bgr)
        match_id, sim = detect_duplicates(emb_vec, registry, threshold=dup_threshold)
        if match_id is not None:
            is_duplicate = True
            matched_id, sim_val = match_id, sim
            append_duplicate_report(dup_report_path, matched_id, video_id, sim_val, video_path.name)
            # remove fused image; skip other saves
            try: os.remove(fused_img_path)
            except Exception: pass
            if verbose:
                print(f"[DUPLICATE] {video_id} matches {matched_id} (sim={sim_val:.3f}). Skipping save.")
            return {"video": video_path.name, "id": video_id, "status": "duplicate", "match": matched_id, "sim": sim_val}
        else:
            # Save embedding and registry
            registry[str(video_id)] = emb_vec.astype(np.float32)
            save_registry(registry_path, registry)
            emb_path = out_dir / f"{video_id}_embedding.npy"
            np.save(emb_path, emb_vec.astype(np.float32))

    # Save outputs according to export_type
    faces = get_mp_triangles()
    V = vertices_from_landmarks(fused, z_mult=z_mult, flip_y=flip_y)

    # Save fused json
    if export_type in ("all","fused","json_only"):
        with open(fused_json_path, "w") as f:
            json.dump({
                "id": video_id,
                "landmarks": fused.astype(float).tolist(),
                "metrics": {**metrics_px, **extra_metrics},
                "metrics_mm": metrics_mm
            }, f, indent=2)
        if verbose: print(f"[OK] {fused_json_path.name}")

    # Save fused obj
    fused_obj_path = out_dir / f"{video_id}_fused.obj"
    if export_type in ("all","fused","mesh_only"):
        write_full_obj(fused_obj_path, V, faces)
        if verbose: print(f"[OK] {fused_obj_path.name}")

    # Feature subset OBJs
    if export_type in ("all","features_only"):
        feats = build_feature_sequences(fused, metrics_px)
        for name, seq in feats.items():
            out_obj = out_dir / f"{video_id}_{name}.obj"
            write_feature_obj_subset(out_obj, V, seq, name)
            if verbose: print(f"[OK] {out_obj.name}")

    # Per-video summary.csv
    if export_type in ("all","fused","json_only"):
        csv_path = out_dir / f"{video_id}_summary.csv"
        row_px = {f"{k}_px": (None if (not isinstance(v, (int, float))) else float(v))
                  for k, v in metrics_px.items() if k not in ("points","paths","units")}
        # add extra metrics
        for k, v in extra_metrics.items():
            row_px[k] = float(v) if isinstance(v, (int,float)) else v

        row = {"id": video_id, **row_px}
        if metrics_mm:
            for k, v in metrics_mm.items():
                if isinstance(v, (int, float)):
                    row[f"{k}_mm"] = float(v)
            row["mm_per_pixel"] = metrics_mm.get("mm_per_pixel", None)
        headers = ["id"] + sorted([k for k in row.keys() if k != "id"])
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader(); w.writerow(row)
        if verbose: print(f"[OK] {csv_path.name}")

    dt = time.time() - t0
    return {"video": video_path.name, "id": video_id, "status": "ok", "frames": len(frames_lms), "seconds": round(dt, 2)}

# -------------------------------
# Batch processing
# -------------------------------
def process_videos_folder(input_dir: Path, out_root: Path, pattern: str = "*.mp4",
                          export_type: str = "all",
                          frame_stride: int = 1,
                          min_frames: int = 12,
                          z_mult: float = 1.0,
                          flip_y: bool = False,
                          mask_normalization: bool = False,
                          ref_bizyg_mm: Optional[float] = None,
                          mm_per_px: Optional[float] = None,
                          overwrite: bool = False,
                          verbose: bool = False,
                          dup_threshold: float = 0.85):
    out_root.mkdir(parents=True, exist_ok=True)
    vids = sorted(list(input_dir.rglob(pattern)))
    if not vids:
        print(f"[ERROR] No videos matching {pattern} in {input_dir}")
        return

    registry_path = out_root / "embeddings.npy"
    dup_report_path = out_root / "duplicate_report.csv"
    master_rows = []

    for vp in vids:
        video_id = extract_id_from_stem(vp.stem)
        sub_out = out_root / video_id
        res = process_single_video(vp, sub_out, out_root,
                                   export_type=export_type,
                                   frame_stride=frame_stride,
                                   min_frames=min_frames,
                                   z_mult=z_mult, flip_y=flip_y,
                                   mask_normalization=mask_normalization,
                                   ref_bizyg_mm=ref_bizyg_mm, mm_per_px=mm_per_px,
                                   overwrite=overwrite, verbose=verbose,
                                   dup_threshold=dup_threshold,
                                   registry_path=registry_path,
                                   dup_report_path=dup_report_path)
        if res.get("status") == "ok":
            csv_path = sub_out / f"{video_id}_summary.csv"
            if csv_path.exists():
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    master_rows.append(df.iloc[0].to_dict())
                except Exception:
                    with open(csv_path) as f:
                        reader = csv.DictReader(f)
                        row = next(reader, None)
                        if row: master_rows.append(row)

    if master_rows and export_type in ("all","fused","json_only"):
        master_csv = out_root / "summary.csv"
        keys = set()
        for r in master_rows: keys.update(r.keys())
        keys = ["id"] + sorted([k for k in keys if k != "id"])
        with open(master_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in master_rows:
                for k in keys: r.setdefault(k, None)
                w.writerow(r)
        print(f"[OK] Master summary -> {master_csv} (videos: {len(master_rows)})")

# -------------------------------
# CLI
# -------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="FaceMesh v6: mask-normalization + fusion + metrics + selective features + embeddings.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    export_choices = ["all","fused","json_only","mesh_only","features_only"]

    # single video
    sp = sub.add_parser("video", help="Process a single video (one person).")
    sp.add_argument("--source", required=True, help="Path to a video file.")
    sp.add_argument("--out-dir", required=True, help="Output directory for this video (should include the ID).")
    sp.add_argument("--export-type", default="all", choices=export_choices, help="What to export (default: all).")
    sp.add_argument("--frame-stride", type=int, default=1, help="Use every N-th frame (speed/robustness).")
    sp.add_argument("--min-frames", type=int, default=12, help="Minimum frames required for fusion.")
    sp.add_argument("--z-mult", type=float, default=1.0, help="Extra Z scaling for OBJ depth.")
    sp.add_argument("--flip-y", action="store_true", help="Flip Y axis in OBJs (viewer compatibility).")
    sp.add_argument("--mask-normalization", action="store_true", help="Use mask-specific normalization (nasion/cheekbone/ear).")
    sp.add_argument("--ref-bizyg-mm", type=float, default=None, help="Known bizygomatic width in mm for scaling.")
    sp.add_argument("--mm-per-px", type=float, default=None, help="Direct scale (mm per pixel).")
    sp.add_argument("--overwrite", action="store_true", help="Overwrite existing fused outputs instead of skipping.")
    sp.add_argument("--verbose", action="store_true")
    sp.add_argument("--dup-threshold", type=float, default=0.85, help="Cosine similarity threshold for duplicates (default 0.85).")

    # folder
    sp = sub.add_parser("videos", help="Process all videos in a folder (one person per video).")
    sp.add_argument("--input", required=True, help="Input folder with videos.")
    sp.add_argument("--out-root", required=True, help="Root output folder (subfolder per video id).")
    sp.add_argument("--pattern", default="*.mp4", help="Glob for video files (default: *.mp4).")
    sp.add_argument("--export-type", default="all", choices=export_choices, help="What to export (default: all).")
    sp.add_argument("--frame-stride", type=int, default=1, help="Use every N-th frame.")
    sp.add_argument("--min-frames", type=int, default=12, help="Minimum frames required for fusion.")
    sp.add_argument("--z-mult", type=float, default=1.0, help="Extra Z scaling for OBJ depth.")
    sp.add_argument("--flip-y", action="store_true", help="Flip Y axis in OBJs.")
    sp.add_argument("--mask-normalization", action="store_true", help="Use mask-specific normalization (nasion/cheekbone/ear).")
    sp.add_argument("--ref-bizyg-mm", type=float, default=None, help="Known bizygomatic width in mm.")
    sp.add_argument("--mm-per-px", type=float, default=None, help="Direct scale (mm per pixel).")
    sp.add_argument("--overwrite", action="store_true", help="Overwrite existing fused outputs instead of skipping.")
    sp.add_argument("--verbose", action="store_true")
    sp.add_argument("--dup-threshold", type=float, default=0.85, help="Cosine similarity threshold for duplicates (default 0.85).")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.cmd == "video":
        out_dir = Path(args.out_dir)
        out_root = out_dir.parent
        process_single_video(Path(args.source), out_dir, out_root,
                             export_type=args.export_type,
                             frame_stride=args.frame_stride,
                             min_frames=args.min_frames,
                             z_mult=args.z_mult,
                             flip_y=args.flip_y,
                             mask_normalization=args.mask_normalization,
                             ref_bizyg_mm=args.ref_bizyg_mm,
                             mm_per_px=args.mm_per_px,
                             overwrite=args.overwrite,
                             verbose=args.verbose,
                             dup_threshold=args.dup_threshold,
                             registry_path=out_root / "embeddings.npy",
                             dup_report_path=out_root / "duplicate_report.csv")
    elif args.cmd == "videos":
        process_videos_folder(Path(args.input), Path(args.out_root),
                              pattern=args.pattern,
                              export_type=args.export_type,
                              frame_stride=args.frame_stride,
                              min_frames=args.min_frames,
                              z_mult=args.z_mult,
                              flip_y=args.flip_y,
                              mask_normalization=args.mask_normalization,
                              ref_bizyg_mm=args.ref_bizyg_mm,
                              mm_per_px=args.mm_per_px,
                              overwrite=args.overwrite,
                              verbose=args.verbose,
                              dup_threshold=args.dup_threshold)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
