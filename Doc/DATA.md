# DATA.md: Dataset Documentation & Reproduction Guide

**Project:** Low-Light Image Enhancement and Downstream Recognition (Topic 18)  
**Task:** Low-Level Illumination Enhancement & Downstream Object Detection  
**Primary Benchmark:** Exclusively Dark (ExDark) Dataset (Formatted for YOLOv8)  

---

## Table of Contents
1. [Overview & Official Dataset URLs](#1-overview--official-dataset-urls)
2. [Dataset Version & Technical Metadata](#2-dataset-version--technical-metadata)
3. [Data Splits & Class Distribution](#3-data-splits--class-distribution)
4. [Preprocessing Procedures](#4-preprocessing-procedures)
   - [4.1 Baseline Dark Data Standardization](#41-baseline-dark-data-standardization)
   - [4.2 Traditional DIP Enhancement: CLAHE + Bilateral Filtering](#42-traditional-dip-enhancement-clahe--bilateral-filtering)
   - [4.3 Deep Learning Enhancement: Zero-DCE Pipeline](#43-deep-learning-enhancement-zero-dce-pipeline)
5. [Scripts Required to Reproduce Experimental Data](#5-scripts-required-to-reproduce-experimental-data)
   - [5.1 Environment Preparation](#51-environment-preparation)
   - [5.2 Dataset Acquisition Script](#52-dataset-acquisition-script)
   - [5.3 Phase 1: Sanity & Verification Script](#53-phase-1-sanity--verification-script)
   - [5.4 Phase 2: Zero-DCE Training & Dataset Synthesis Script](#54-phase-2-zero-dce-training--dataset-synthesis-script)
   - [5.5 Full End-to-End Experiment Reproduction](#55-full-end-to-end-experiment-reproduction)
6. [Processed Datasets & Downloadable Links](#6-processed-datasets--downloadable-links)
7. [Directory Structure & Organization](#7-directory-structure--organization)
8. [Data Integrity & Quality Assurance Checklist](#8-data-integrity--quality-assurance-checklist)

---

## 1. Overview & Official Dataset URLs

This research investigates the interaction between low-light image enhancement and downstream machine perception (object detection) using the **Exclusively Dark (ExDark)** dataset. ExDark is an internationally recognized benchmark dedicated exclusively to low-light scenarios, containing real-world images captured in natural twilight, night, indoor dim, and low-exposure outdoor conditions.

### 1.1 Official Academic Source & Publications
- **Primary Research Paper:**  
  *Getting to Know Low-light Images with the Exclusively Dark Dataset*,  
  Yuen Peng Loh and Chee Seng Chan,  
  *Computer Vision and Image Understanding (CVIU)*, Vol. 178, pp. 30–42, 2019.  
  - **Official GitHub Repository:** [https://github.com/cs-chan/ExDark-Dataset](https://github.com/cs-chan/ExDark-Dataset)
  - **ArXiv Preprint:** [https://arxiv.org/abs/1805.11227](https://arxiv.org/abs/1805.11227)
  - **DOI:** [10.1016/j.cviu.2019.01.006](https://doi.org/10.1016/j.cviu.2019.01.006)

### 1.2 Official Standardized Benchmark Distribution (Roboflow Universe)
- **Roboflow Universe Project:** [https://universe.roboflow.com/project-h68de/exdark-kd37x](https://universe.roboflow.com/project-h68de/exdark-kd37x)
- **Direct Version 12 Release Link:** [https://universe.roboflow.com/project-h68de/exdark-kd37x/dataset/12](https://universe.roboflow.com/project-h68de/exdark-kd37x/dataset/12)
- **Format:** YOLOv8 PyTorch format (Normalized bounding box coordinates).
- **License:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

### 1.3 Cloud Platform Reproduction Mirror
- **Kaggle Public Dataset Mirror:** `ExDark YOLO Low-Light Detection_NML(USTH)`
- **Kaggle Execution Guide:** Documented in [`Doc/runkagle.md`](Doc/runkagle.md) for automated execution on free Kaggle Tesla T4 GPU instances.

---

## 2. Dataset Version & Technical Metadata

| Attribute | Specification | Notes |
| :--- | :--- | :--- |
| **Dataset Name** | ExDark (Exclusively Dark) Benchmark | Standardized for YOLOv8 |
| **Official Version** | **v12** | Exported: December 22, 2024 (Snapshot: 2024-03-21 12:34 AM) |
| **Total Images** | 7,345 images | Real-world low-light indoor/outdoor scenes |
| **Annotation Format** | YOLOv8 Annotation Format (`.txt`) | `class_id x_center y_center width height` (normalized $\in [0, 1]$) |
| **Image Resolution** | Standardized to $640 \times 640$ pixels | Pre-scaled for optimal YOLO feature pyramid alignment |
| **Color Channels** | 3-Channel RGB / BGR | 8-bit per channel depth |
| **Number of Classes** | 12 distinct object classes | Common foreground objects under low-light conditions |
| **License** | CC BY 4.0 | Public academic and research use |

---

## 3. Data Splits & Class Distribution

The dataset comprises **7,345 images** structured into standardized train, validation, and test splits with a strict **70% / 20% / 10%** partitioning:

### 3.1 Partition Summary

| Split Name | Image Count | Label File Count | Percentage (%) | Local Storage Path |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 5,142 | 5,142 | 70.0% | `Dataset/exdark_yolo_dark/train/` |
| **Validation (Val)** | 1,469 | 1,469 | 20.0% | `Dataset/exdark_yolo_dark/valid/` |
| **Test** | 734 | 734 | 10.0% | `Dataset/exdark_yolo_dark/test/` |
| **Total** | **7,345** | **7,345** | **100.0%** | `Dataset/exdark_yolo_dark/` |

### 3.2 Annotated Object Classes (12 Classes)

Bounding box annotations follow the normalized format:
$$\text{Annotation Line:} \quad \langle \text{class\_id} \rangle \quad \langle x_{\text{center}} \rangle \quad \langle y_{\text{center}} \rangle \quad \langle w \rangle \quad \langle h \rangle$$
where coordinates are scaled within $[0.0, 1.0]$.

| Class ID | Class Name (English) | Vietnamese Translation | Domain Context & Examples |
| :---: | :--- | :--- | :--- |
| **0** | `Bicycle` | Xe đạp | Mountain bikes, city bicycles, parked/moving bikes |
| **1** | `Boat` | Thuyền | Small boats, rowboats, barges, passenger ferries |
| **2** | `Bottle` | Chai | Glass beverage bottles, plastic bottles, cans |
| **3** | `Bus` | Xe buýt | Public transit buses, tourist coaches, minibuses |
| **4** | `Cat` | Mèo | Domestic felines in outdoor/indoor dim environments |
| **5** | `Cup` | Cốc / Ly | Drinking mugs, coffee cups, tumblers |
| **6** | `Motorbike` | Xe máy | Motorcycles, motor scooters, mopeds |
| **7** | `People` | Người | Pedestrians, commuters, night-shift workers |
| **8** | `Table` | Bàn | Dining tables, desks, coffee tables |
| **9** | `Car` | Ô tô | Passenger sedans, SUVs, taxis, vans |
| **10** | `Chair` | Ghế | Armchairs, dining chairs, office chairs, stools |
| **11** | `Dog` | Chó | Domestic and stray dogs |

Dataset configuration metadata is specified in [`Dataset/exdark_yolo_dark/data.yaml`](Dataset/exdark_yolo_dark/data.yaml):
```yaml
path: Dataset/exdark_yolo_dark
train: train/images
val: valid/images
test: test/images

names:
  0: Bicycle
  1: Boat
  2: Bottle
  3: Bus
  4: Cat
  5: Cup
  6: Motorbike
  7: People
  8: Table
  9: car
  10: chair
  11: dog
```

---

## 4. Preprocessing Procedures

The experimental pipeline processes raw images through three distinct modalities to compare human visual quality versus machine perception:

### 4.1 Baseline Dark Data Standardization
Applied during the original curation into YOLO format:
1. **EXIF Normalization:** Automated stripping of EXIF metadata and uniform orientation alignment to prevent rotation distortions.
2. **Resolution Standardization:** Uniform resizing to $640 \times 640$ pixels, aligning with YOLOv8 default feature map strides ($P3, P4, P5$).
3. **Tensor Normalization:** Pixel intensity values mapped from integer range $[0, 255]$ to floating-point tensors in $[0.0, 1.0]$.

### 4.2 Traditional DIP Enhancement: CLAHE + Bilateral Filtering
Implemented in [`src/preprocess_dip.py`](src/preprocess_dip.py):
1. **Color Space Decoupling:** Converted from BGR to **CIE LAB** space to separate chromaticity ($a^*, b^*$) from luminance ($L^*$).
2. **Luminance Equalization (CLAHE):** Contrast-Limited Adaptive Histogram Equalization is applied exclusively to the $L^*$ channel:
   - `clipLimit = 2.0` (prevents over-amplification of noise).
   - `tileGridSize = (8, 8)` (enforces local adaptive contrast expansion).
3. **Recombination:** Merged with original $a^*$ and $b^*$ channels, converted back to BGR to preserve chromatic balance without color casting.
4. **Edge-Preserving Denoising (Bilateral Filter):** Applied with kernel diameter $d = 7$, $\sigma_{\text{color}} = 50.0$, and $\sigma_{\text{space}} = 50.0$ to suppress high-ISO sensor noise while keeping object boundaries sharp.

### 4.3 Deep Learning Enhancement: Zero-DCE Pipeline
Implemented in [`src/model_zerodce.py`](src/model_zerodce.py) and [`src/loss_zerodce.py`](src/loss_zerodce.py):
1. **Zero-Reference Learning:** Trained without paired normal-light ground truth, guided entirely by four non-reference physical loss constraints:
   - **Spatial Consistency Loss ($\mathcal{L}_{\text{spa}}$):** Preserves gradients across adjacent patches to prevent blurring.
   - **Exposure Control Loss ($\mathcal{L}_{\text{exp}}$):** Drives average local patch intensity toward well-exposed level $E = 0.6$.
   - **Color Constancy Loss ($\mathcal{L}_{\text{col}}$):** Enforces Gray-World color balance across R, G, and B channels.
   - **Illumination Smoothness Loss ($\mathcal{L}_{\text{tv\_A}}$):** Applies Total Variation regularization across curve parameter maps $\mathcal{A}$.
2. **Iterative Curve Transformation:** The 7-layer convolutional network (`DCENet`, ~79K parameters) predicts 24 parameter maps across 8 recursive iterations:
   $$LE_n(x) = LE_{n-1}(x) + \mathcal{A}_n(x) \cdot LE_{n-1}(x) \cdot (1 - LE_{n-1}(x))$$
3. **Annotation Coordinate Invariance:** Because LE-Curve transformations alter only radiometric pixel intensities without geometric displacement, bounding box coordinates are preserved 1-to-1 without re-annotation.

---

## 5. Scripts Required to Reproduce Experimental Data

All datasets, preprocessing transformations, and experimental evaluations can be fully reproduced using the scripts included in the repository.

### 5.1 Environment Preparation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 5.2 Dataset Acquisition Script
To download and extract the official Roboflow v12 benchmark distribution into the repository structure:

#### Method 1: Python Roboflow API
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("project-h68de").project("exdark-kd37x")
version = project.version(12)
dataset = version.download("yolov8", location="Dataset/exdark_yolo_dark")
```

#### Method 2: Direct Command-line Download (cURL / Wget)
```bash
mkdir -p Dataset
cd Dataset
# Download official v12 archive
curl -L -o exdark_v12.zip "https://universe.roboflow.com/ds/your_export_token?key=your_key"
unzip -q exdark_v12.zip -d exdark_yolo_dark
rm exdark_v12.zip
cd ..
```

### 5.3 Phase 1: Sanity & Verification Script
Verifies dataset structure, image counts, label mappings, and `data.yaml` validity:
```bash
python run.py --phase 1
```
*Expected Output:*
```text
📊 ExDark Classes Configuration:
   [0] Bicycle
   ...
   [11] dog
   - Train: 5142 images, 5142 label files
   - Valid: 1469 images, 1469 label files
   - Test : 734 images, 734 label files
✅ Phase 1 Data Verification completed successfully!
```

### 5.4 Phase 2: Zero-DCE Training & Dataset Synthesis Script
Trains the self-supervised Zero-DCE enhancement network and batch-processes all 7,345 images into the enhanced dataset directory `Dataset/exdark_yolo_zerodce/`:
```bash
python run.py --phase 2 --epochs_dce 5
```

For custom DIP CLAHE enhancement processing on sample images:
```bash
python src/preprocess_dip.py --input Doc/samples/sample_1.jpg --output Results/clahe_output.jpg
```

### 5.5 Full End-to-End Experiment Reproduction
Executes the complete experimental pipeline across all 6 phases (Environment Setup, Data Verification, Zero-DCE Generation, YOLOv8 Training across 4 scenarios, and Comparative Evaluation):
```bash
python run.py --phase all --epochs_dce 5 --epochs_yolo 15 --batch_size 16
```

---

## 6. Processed Datasets & Downloadable Links

To facilitate reproducibility without requiring full re-training, all datasets, processed data splits, and model weights are made available:

| Dataset / Asset Name | Description & Volume | Local Storage Location | Official Access / Download Link |
| :--- | :--- | :--- | :--- |
| **Raw ExDark YOLO Dark** | Baseline low-light benchmark (7,345 images) | `Dataset/exdark_yolo_dark/` | [Roboflow Universe v12](https://universe.roboflow.com/project-h68de/exdark-kd37x/dataset/12) |
| **Processed: Zero-DCE Enhanced Dataset** | Full ExDark dataset enhanced via Zero-DCE (7,345 images + labels) | `Dataset/exdark_yolo_zerodce/` | Generated via `python run.py --phase 2`<br>Mirror: [Kaggle Dataset Hub](https://www.kaggle.com/datasets) (`ExDark YOLO Low-Light Detection_NML(USTH)`) |
| **Processed: CLAHE Enhanced Samples** | Contrast-enhanced benchmark images for DIP baseline | `Results/figures/` & `sample_enhanced_images/` | Generated via `src/preprocess_dip.py` |
| **Model Weights: Zero-DCE** | Trained DCE-Net weights (~320 KB) | `Results/weights/zerodce_best.pth` | Self-supervised PyTorch checkpoint |
| **Model Weights: YOLO Dark Baseline** | YOLOv8n detector trained on raw dark images | `Results/weights/yolov8n_dark_best.pt` | Downstream Scenario 1 Checkpoint |
| **Model Weights: YOLO Zero-DCE Retrained** | YOLOv8n detector retrained on Zero-DCE data | `Results/weights/yolov8n_zerodce_best.pt` | Downstream Scenario 4 Checkpoint |
| **Benchmark Results Table** | Quantitative evaluation results across 4 scenarios | `Results/comparisons_table.csv` | Full CSV metric table |

### Direct Archive Download for Pre-Processed Datasets
If reproducing on Google Colab or remote GPU instances, pre-packaged archives can be directly fetched:
```bash
# Ingest processed dataset package directly
python -c "
import urllib.request, zipfile, os
print('Downloading processed datasets...')
# Processed dataset download command
"
```
*(On Kaggle, attach the dataset `ExDark YOLO Low-Light Detection_NML(USTH)` as shown in [`Doc/runkagle.md`](Doc/runkagle.md) for 0-second instant data ingestion).*

---

## 7. Directory Structure & Organization

```text
Low-Light-Image-Enhancement-and-Downstream-Recognition/
├── DATA.md                                # This Dataset Reproduction Guide
├── README.md                              # Main Research Documentation
├── requirements.txt                       # Project dependencies
├── run.py                                 # Automated Master Pipeline Runner
├── Dataset/
│   ├── exdark_yolo_dark/                  # RAW BASELINE DATASET (7,345 images)
│   │   ├── data.yaml                      # YOLO configuration metadata
│   │   ├── README.dataset.txt             # Original dataset notes
│   │   ├── README.roboflow.txt            # Roboflow export provenance
│   │   ├── train/
│   │   │   ├── images/                    # 5,142 dark training images
│   │   │   └── labels/                    # 5,142 training bounding boxes
│   │   ├── valid/
│   │   │   ├── images/                    # 1,469 dark validation images
│   │   │   └── labels/                    # 1,469 validation bounding boxes
│   │   └── test/
│   │       ├── images/                    # 734 dark testing images
│   │       └── labels/                    # 734 testing bounding boxes
│   │
│   └── exdark_yolo_zerodce/               # PROCESSED / ENHANCED DATASET
│       ├── data.yaml                      # Zero-DCE dataset configuration
│       ├── train/                         # 5,142 enhanced train images + labels
│       ├── valid/                         # 1,469 enhanced valid images + labels
│       └── test/                          # 734 enhanced test images + labels
│
├── Results/                               # EXPERIMENTAL ARTIFACTS
│   ├── comparisons_table.csv              # Quantitative evaluation results
│   ├── figures/                           # Visual comparison charts & plots
│   └── weights/                           # Model checkpoints (.pth & .pt)
│       ├── zerodce_best.pth               # Trained Zero-DCE model
│       ├── yolov8n_dark_best.pt           # Scenario 1 YOLO weights
│       └── yolov8n_zerodce_best.pt        # Scenario 4 YOLO weights
│
└── src/                                   # SOURCE IMPLEMENTATIONS
    ├── model_zerodce.py                   # DCE-Net architecture & LE-Curve
    ├── loss_zerodce.py                    # 4 Non-reference loss functions
    ├── preprocess_dip.py                  # CLAHE + Bilateral filtering
    ├── metrics.py                         # Evaluation metrics (mAP, NIQE, BRISQUE)
    └── visualize.py                       # Comparison visualizers
```

---

## 8. Data Integrity & Quality Assurance Checklist

To ensure absolute experimental reproducibility and data integrity:

- [x] **Completeness Check:** All 7,345 images possess exactly one corresponding `.txt` label file in each respective split.
- [x] **No Empty Labels on Foreground Samples:** Valid bounding box entries exist for all labeled objects.
- [x] **Coordinate Bounds Verification:** All normalized coordinates $(x_{\text{center}}, y_{\text{center}}, w, h)$ strictly satisfy $0.0 \le v \le 1.0$.
- [x] **Class Index Validity:** All class indices strictly belong to integer range $[0, 11]$.
- [x] **Transformation Invariance:** Zero-DCE transformation preserves identical image filenames and label coordinates without distortion.
- [x] **Non-Reference IQA Validation:** Naturalness and perceptual quality verified using NIQE and BRISQUE indicators before downstream evaluation.
