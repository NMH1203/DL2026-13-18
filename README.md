# Low-Light Image Enhancement and Downstream Object Detection

Group 13 — Project 18 — Deep Learning (2026–2027)

This repository studies whether low-light enhancement improves downstream object detection on ExDark. The comparison uses the same test split and the same YOLOv8n evaluation settings for all reported image variants so that differences are attributable to the input preprocessing pipeline.

## Research question

> Across raw, classical, learned, and hybrid low-light image variants, which enhancement strategies improve downstream YOLOv8n detection on ExDark, and does better visual appearance consistently correspond to better recognition performance?

The controlled comparison currently contains five input variants:

1. Raw low-light images.
2. CLAHE.
3. CLAHE followed by bilateral filtering.
4. Pretrained Zero-DCE++ (recorded as `Zero-DCE` in the result CSV).
5. Pretrained Zero-DCE++ followed by bilateral filtering.

The repository also contains a Zero-DCE implementation trained from scratch. Its result must be reported as a separate experiment and must not be treated as an architectural ablation against Zero-DCE++ until a controlled test row is added to the canonical CSV.

## Current controlled test results

The following values come from `Results/comparisons_table.csv` and use the same 734-image ExDark test split.

| Input variant | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Raw dark | 0.6671 | 0.5883 | 0.6146 | 0.2862 |
| CLAHE | 0.6543 | **0.5935** | 0.6188 | 0.2872 |
| CLAHE + bilateral | 0.6572 | 0.5800 | 0.6161 | 0.2856 |
| Zero-DCE++ | 0.6771 | 0.5281 | 0.5881 | 0.2705 |
| Zero-DCE++ + bilateral | **0.7040** | 0.5677 | **0.6244** | **0.2921** |

Within this recorded comparison, Zero-DCE++ followed by bilateral filtering has the highest precision, mAP@0.5, and mAP@0.5:0.95. CLAHE has the highest recall. These results support a limited conclusion: enhancement performance depends on the complete preprocessing pipeline, and improved visual appearance alone does not guarantee improved detection. Causal statements about sensor noise or domain shift require additional ablation evidence.

## Methods

### Zero-DCE and Zero-DCE++

The learned enhancement branch uses a recurrent light-enhancement curve:

```text
LE_n(x) = LE_(n-1)(x) + A_n(x) * LE_(n-1)(x) * (1 - LE_(n-1)(x))
```

The local from-scratch implementation is in `src/luong/model_zerodce.py`. Its loss implementation uses:

```text
L_total = L_spa + 10 L_exp + 5 L_col + 200 L_tv
```

The loss terms encourage spatial consistency, target exposure, color constancy, and smooth illumination-curve maps. A bounded curve formulation keeps values in the expected numerical range, but this property alone does not prove the absence of saturated highlights.

### Classical enhancement

`src/luong/preprocess_dip.py` implements CLAHE on the luminance channel in CIE LAB space. Bilateral filtering is evaluated separately so that its additional effect can be measured rather than assumed.

### Downstream detector

YOLOv8n is evaluated with a common ExDark split and matched detector settings. Precision, recall, mAP@0.5, and mAP@0.5:0.95 are the primary metrics.

The functions currently named `calculate_niqe` and `calculate_brisque` in `src/luong/metrics.py` are lightweight project-specific proxies, not validated standard NIQE or BRISQUE implementations. They must be labelled as proxy scores unless they are replaced by standard implementations.

## Repository structure

```text
.
├── README.md
├── DATA.md
├── document.md
├── review.md
├── requirements.txt
├── run.py
├── demo.py
├── app.py
├── Dataset/
│   └── exdark_yolo_dark/
├── Notebooks/
│   ├── NML/
│   │   ├── 01_data_preparation.ipynb
│   │   ├── 02_zerodce_enhancement.ipynb
│   │   └── 03_yolov8_experiments.ipynb
│   └── 04_final_demo.ipynb
├── src/
│   ├── luong/
│   │   ├── model_zerodce.py
│   │   ├── loss_zerodce.py
│   │   ├── preprocess_dip.py
│   │   ├── metrics.py
│   │   ├── visualize.py
│   │   └── pipeline.py
│   └── hienanh/
├── templates/
│   └── index.html
├── sample_enhanced_images/
│   └── 1_raw_dark/
└── Results/
    ├── comparisons_table.csv
    ├── figures/
    └── weights/
```

The LaTeX report is maintained at `report/Group13_Project18_Report.tex` in the working project. Add it to Git before relying on it as a repository deliverable.

## Installation

```bash
git clone https://github.com/NMH1203/Low-Light-Image-Enhancement-and-Downstream-Recognition.git
cd Low-Light-Image-Enhancement-and-Downstream-Recognition
pip install -r requirements.txt
```

Python 3.10 or newer is recommended. Training and inference require PyTorch and Ultralytics; CUDA is optional but recommended for training.

## Pipeline commands

```bash
# Verify the ExDark dataset
python run.py --phase 1

# Train the local Zero-DCE model and generate its enhanced dataset
python run.py --phase 2 --epochs_dce 10

# Train and evaluate detector pipelines
python run.py --phase 3,4 --epochs_yolo 40 --batch_size 16
```

Run `python run.py --help` before a long experiment and record the exact command, software versions, checkpoint hashes, and output CSV with the report.

## Demo

The web and CLI demos are retained as qualitative tools. Their four displayed views are not identical to the five-row controlled benchmark, so per-image outputs must not be interpreted as aggregate test metrics.

```bash
# Web interface
python app.py --port 5001

# Inspect checkpoints
python demo.py --inspect-weights

# Run one image
python demo.py --image sample_enhanced_images/1_raw_dark/image_1_raw.jpg

# Run all bundled samples
python demo.py --batch
```

## Dataset

- ExDark YOLO export: 7,345 images.
- Train: 5,142 images.
- Validation: 1,469 images.
- Test: 734 images.
- Classes: Bicycle, Boat, Bottle, Bus, Cat, Cup, Motorbike, People, Table, Car, Chair, and Dog.

See `DATA.md` for provenance, layout, and reproduction notes.

## Reporting rules

- Use `Results/comparisons_table.csv` as the single source of quantitative results.
- Do not mix the from-scratch Zero-DCE experiment with pretrained Zero-DCE++.
- Do not report expected, simulated, or formula-derived values as measured results.
- State the dataset split, checkpoint, detector configuration, and preprocessing order for every comparison.
- Use cautious language: the current table shows associations between pipelines and detection metrics; it does not by itself establish a physical cause.

## References

1. Loh, Y. P., and Chan, C. S. “Getting to Know Low-Light Images with the Exclusively Dark Dataset.” *Computer Vision and Image Understanding*, 178, 30–42, 2019. DOI: `10.1016/j.cviu.2018.10.010`.
2. Guo, C., Li, C., Guo, J., Loy, C. C., Hou, J., Kwong, S., and Cong, R. “Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement.” CVPR, 2020.
3. Li, C., Guo, C., and Loy, C. C. “Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation.” *IEEE TPAMI*, 44(8), 4225–4238, 2022.
4. Ultralytics. “YOLOv8.” https://github.com/ultralytics/ultralytics
