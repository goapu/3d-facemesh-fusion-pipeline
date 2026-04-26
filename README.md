
# 3D FaceMesh Fusion Pipeline

**Author:** Dilip Goswami, MSc TU Berlin

A Python-based facial analysis pipeline for extracting stable facial landmarks from video, generating fused 3D FaceMesh outputs, measuring key facial distances, detecting duplicate identities across videos, and classifying face shape using anthropometric ratios.

The project uses **MediaPipe FaceMesh**, **OpenCV**, **NumPy**, **FaceNet**, and **PyTorch** to process video files and generate measurements, meshes, visualizations, embeddings, and summary CSV files locally.

## Features

- Extracts 468-point facial landmarks from video frames using MediaPipe FaceMesh.
- Fuses landmarks across multiple frames for a more stable face representation.
- Exports full 3D face meshes as `.obj` files.
- Exports selected facial feature measurements as separate `.obj` files.
- Generates per-video JSON, image, CSV, and embedding outputs locally.
- Supports duplicate detection using FaceNet embeddings and cosine similarity.
- Provides two landmark normalization modes:
  - Generic normalization for general face recognition and comparison.
  - Mask-specific normalization for surgical mask sizing and facial-fit measurements.
- Uses an ear-over-bridge path measurement for mask-specific facial fitting analysis.
- Classifies face shape using medical facial-index categories and aesthetic labels.
- Produces batch summary reports when run locally.

## Project Structure

```text
.
├── FaceAndLandmark_autoMesh_v6.py   # Main FaceMesh fusion, metric, OBJ, and duplicate-detection pipeline
├── stable_face_classifier.py        # Stable medical/aesthetic face-shape classifier
├── fusion_utils.py                  # FaceMesh extraction, landmark fusion, metrics, rendering, and OBJ utilities
├── recognition_utils.py             # FaceNet embedding and duplicate-detection utilities
├── normalization.py                 # Generic and mask-specific landmark normalization
├── requirements.txt                 # Python dependencies
├── LICENSE                          # Project license
└── README.md                        # Project documentation
```

Input videos and generated output folders are intentionally not included in this repository for privacy reasons.

Recommended local working folders:

```text
.
├── input/
│   └── videos/                      # Local input videos, not committed to Git
├── output/                          # Local pipeline outputs, not committed to Git
└── faceShape/                       # Local face-shape outputs, not committed to Git
```

> Note: Files such as `__pycache__`, `.DS_Store`, `__MACOSX`, input videos, and generated outputs are not required for source-code-only version control and should be ignored in Git.

## Requirements

Recommended Python version: **Python 3.10**

Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install mediapipe facenet-pytorch torch torchvision opencv-python numpy pandas scikit-learn
```

The `requirements.txt` file should contain:

```text
mediapipe
facenet-pytorch
torch
torchvision
opencv-python
numpy
pandas
scikit-learn
```

> Note: FaceNet uses a pretrained model with `pretrained='vggface2'`. On the first run, an internet connection may be required so the model can be downloaded and cached locally.

On some systems, OpenCV preview windows require a desktop environment. If running on a headless server, use a GUI-enabled environment or adjust the scripts to disable `cv2.imshow` calls.

## Main Pipeline Usage

The main script is:

```bash
python FaceAndLandmark_autoMesh_v6.py
```

It supports two modes:

1. Process a single video.
2. Process a folder of videos.

### Process a Folder of Videos

```bash
python FaceAndLandmark_autoMesh_v6.py videos \
  --input "input/videos" \
  --out-root "output" \
  --pattern "*.mp4" \
  --export-type all \
  --verbose
```

### Process a Single Video

```bash
python FaceAndLandmark_autoMesh_v6.py video \
  --source "input/videos/01.mp4" \
  --out-dir "output/01" \
  --export-type all \
  --verbose
```

### Mask-Specific Normalization

Use this option when the goal is mask fitting or mask-related facial measurements:

```bash
python FaceAndLandmark_autoMesh_v6.py videos \
  --input "input/videos" \
  --out-root "output" \
  --pattern "*.mp4" \
  --export-type all \
  --mask-normalization \
  --verbose
```

Mask-specific normalization is useful for facial-fit analysis because it focuses on measurements such as face width, ear-to-ear distance, nose bridge height, and ear-over-bridge distance.

### Optional Scaling to Millimeters

If a real-world scale is known, measurements can be converted from pixels to millimeters.

Using known bizygomatic width:

```bash
python FaceAndLandmark_autoMesh_v6.py video \
  --source "input/videos/01.mp4" \
  --out-dir "output/01" \
  --ref-bizyg-mm 140 \
  --export-type all
```

Using a direct millimeter-per-pixel value:

```bash
python FaceAndLandmark_autoMesh_v6.py video \
  --source "input/videos/01.mp4" \
  --out-dir "output/01" \
  --mm-per-px 0.5 \
  --export-type all
```

## Export Types

The main pipeline supports the following export options:

| Export Type     | Description                                                                |
| --------------- | -------------------------------------------------------------------------- |
| `all`           | Export JSON, full mesh, feature meshes, image preview, CSV, and embedding. |
| `fused`         | Export fused JSON, full mesh, preview image, and summary.                  |
| `json_only`     | Export only fused landmark/metric JSON and CSV summary.                    |
| `mesh_only`     | Export only the full fused OBJ mesh.                                       |
| `features_only` | Export only selected facial feature OBJ files.                             |

## Main Pipeline Outputs

When the main pipeline is run locally, each processed video may generate an output folder like this:

```text
output/01/
├── 01_fused.json
├── 01_fused.obj
├── 01_fused.jpg
├── 01_summary.csv
├── 01_embedding.npy
├── 01_ear2ear.obj
├── 01_ear_over_bridge.obj
├── 01_bizygomatic.obj
├── 01_menton_sellion.obj
├── 01_nosebridge.obj
└── 01_lip2chin.obj
```

Global output files may include:

```text
output/summary.csv             # Combined metric summary for processed videos
output/embeddings.npy          # Stored embeddings used for duplicate detection
output/duplicate_report.csv    # Duplicate matches found across videos
```

These generated files are not included in this repository because they may contain biometric or identity-related information.

Example metrics include:

* Bizygomatic width
* Ear-to-ear distance
* Ear-over-bridge path
* Menton-sellion length
* Nose bridge height
* Lip-to-chin length
* Face length-to-width ratio
* Nasion-to-menton over bizygomatic ratio when mask normalization is enabled

## Duplicate Detection

Duplicate detection is performed with FaceNet embeddings.

The default cosine similarity threshold is `0.85`:

```bash
python FaceAndLandmark_autoMesh_v6.py videos \
  --input "input/videos" \
  --out-root "output" \
  --dup-threshold 0.85
```

If a video is detected as a duplicate, the result is written locally to:

```text
output/duplicate_report.csv
```

Example report format:

```text
original_id,duplicate_id,similarity,video
03,04,1.0000,04.mp4
```

## Face Shape Classifier Usage

The second script classifies face shape using stable fused landmarks and anthropometric ratios.

### Batch Mode

```bash
python stable_face_classifier.py \
  --input_dir "input/videos" \
  --output_dir "faceShape"
```

### Single Video Mode

```bash
python stable_face_classifier.py \
  --video "input/videos/01.mp4" \
  --output_dir "faceShape"
```

### Webcam Mode

```bash
python stable_face_classifier.py --webcam
```

## Face Shape Outputs

When the face-shape classifier is run locally, it can generate annotated images and a CSV summary:

```text
faceShape/
├── 01_shape.jpg
├── 02_shape.jpg
├── 03_shape.jpg
└── face_shape_summary.csv
```

The generated `faceShape/` folder is not included in this repository for privacy reasons.

The summary CSV may include:

| Column                 | Description                                                           |
| ---------------------- | --------------------------------------------------------------------- |
| `filename`             | Source video filename.                                                |
| `medical_class`        | Medical facial-index class.                                           |
| `aesthetic_shape`      | Aesthetic face-shape label.                                           |
| `face_size`            | Estimated size category.                                              |
| `bizygomatic_width_mm` | Estimated cheekbone width in millimeters.                             |
| `morph_height_mm`      | Estimated morphological face height in millimeters.                   |
| `facial_index`         | Morphological height divided by bizygomatic width, multiplied by 100. |
| `jaw_cheek_ratio`      | Jaw width relative to cheek width.                                    |
| `fore_cheek_ratio`     | Forehead width relative to cheek width.                               |

Medical categories used by the classifier include:

* Hypereuryprosopic
* Euryprosopic
* Mesoprosopic
* Leptoprosopic
* Hyperleptoprosopic

Aesthetic labels include shapes such as:

* Oval
* Round
* Square
* Rectangle
* Oblong
* Triangle
* Heart

## Example Results

Example result files are not included in this repository because the pipeline processes facial videos and generates biometric-style outputs.

After running the scripts locally, users can inspect the generated CSV files, OBJ meshes, preview images, embeddings, and duplicate-detection reports in their local output folders.

## GitHub Recommendations

Before pushing to GitHub, add a `.gitignore` file to avoid committing videos, generated outputs, cache files, and biometric data.

Recommended `.gitignore`:

```gitignore
# Python cache files
__pycache__/
*.pyc
*.pyo

# System files
.DS_Store
__MACOSX/

# Virtual environments
.venv/
venv/
env/

# Model/cache folders
.cache/
models/

# Input videos and private media
input/videos/
*.mp4
*.mov
*.avi
*.mkv

# Generated outputs
output/
outputs/
results/
faceShape/

# Generated embeddings and arrays
*.npy
*.npz

# Generated images
*.jpg
*.jpeg
*.png
```

If videos or generated biometric outputs are important for reproducibility, store them separately with appropriate consent, anonymization, and access control. For large non-sensitive files, consider using Git LFS instead of committing them directly.

## Privacy Notice

This project processes face videos and generates biometric-style outputs such as facial landmarks, embeddings, measurements, meshes, and visualizations.

Treat all input videos and generated outputs as sensitive data. Do not publish personal face videos, embeddings, or identifiable facial data without consent.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Dilip Goswami, MSc TU Berlin

## Troubleshooting

### `ModuleNotFoundError`

Install the missing dependency:

```bash
pip install -r requirements.txt
```

Or install dependencies manually:

```bash
pip install mediapipe facenet-pytorch torch torchvision opencv-python numpy pandas scikit-learn
```

### FaceNet model download fails

The first initialization of the FaceNet model may require internet access to download the pretrained `vggface2` weights. Check your internet connection or firewall settings, then run the script again.

### OpenCV window does not appear

The scripts use `cv2.imshow`. Make sure you are running in an environment with GUI support.

If using a headless server, disable display calls such as `cv2.imshow` or run the script in an environment with graphical display support.

### Not enough landmark frames

The main pipeline requires enough stable frames for fusion. Try:

```bash
--min-frames 5
```

or use a clearer video with better lighting and a more frontal face angle.

### Duplicate detection seems too strict or too loose

Adjust the duplicate threshold:

```bash
--dup-threshold 0.90
```

Higher values are stricter. Lower values are more permissive.


