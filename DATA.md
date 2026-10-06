# Dataset and Reproduction Notes

## Dataset provenance

This project uses the Exclusively Dark (ExDark) object-detection dataset in YOLO format.

- Original project: https://github.com/cs-chan/ExDark-Dataset
- Paper: *Getting to Know Low-Light Images with the Exclusively Dark Dataset*
- DOI: https://doi.org/10.1016/j.cviu.2018.10.010
- Standardized YOLO export used by the project: https://universe.roboflow.com/project-h68de/exdark-kd37x/dataset/12
- License shown by the selected Roboflow release: CC BY 4.0. Confirm the release page when redistributing an archive.

Record the export version and download date with every experiment. Do not replace the dataset silently with another ExDark conversion because class order and split membership may differ.

## Local layout

```text
Dataset/exdark_yolo_dark/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The expected split counts are:

| Split | Images | Labels |
|---|---:|---:|
| Train | 5,142 | 5,142 |
| Validation | 1,469 | 1,469 |
| Test | 734 | 734 |
| Total | 7,345 | 7,345 |

The 12 classes used by the repository are Bicycle, Boat, Bottle, Bus, Cat, Cup, Motorbike, People, Table, Car, Chair, and Dog. Treat `Dataset/exdark_yolo_dark/data.yaml` as the authoritative class-index mapping for a run.

YOLO labels use normalized rows in the form:

```text
class_id x_center y_center width height
```

Image resizing is performed by the selected training or inference pipeline. Do not claim that every source file is physically stored at 640 × 640 unless this has been verified directly.

## Controlled input variants

The current result table compares five radiometric variants while preserving image geometry and bounding-box coordinates:

1. Raw dark.
2. CLAHE.
3. CLAHE + bilateral filtering.
4. Pretrained Zero-DCE++.
5. Pretrained Zero-DCE++ + bilateral filtering.

The local from-scratch Zero-DCE model is a separate learned-enhancement experiment. Its outputs and metrics must be labelled separately from pretrained Zero-DCE++.

### Classical preprocessing

`src/luong/preprocess_dip.py` applies CLAHE to the luminance channel in CIE LAB space. Bilateral filtering is an optional second operation and must be recorded as a distinct configuration.

### Learned preprocessing

`src/luong/model_zerodce.py` and `src/luong/loss_zerodce.py` implement the local from-scratch Zero-DCE branch. The default loss weights are:

```text
L_total = L_spa + 10 L_exp + 5 L_col + 200 L_tv
```

Pretrained Zero-DCE++ results must identify the exact external checkpoint and implementation used. Store the checkpoint hash with the experiment record.

## Verification and execution

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Verify the dataset structure and labels:

```bash
python run.py --phase 1
```

Train the local from-scratch Zero-DCE implementation and generate its enhanced dataset:

```bash
python run.py --phase 2 --epochs_dce 10
```

Run the detector stages only after checking the available options:

```bash
python run.py --help
python run.py --phase 3,4 --epochs_yolo 40 --batch_size 16
```

Run the classical preprocessor on a bundled sample:

```bash
python src/luong/preprocess_dip.py --input sample_enhanced_images/1_raw_dark/image_1_raw.jpg --output Results/clahe_output.jpg
```

Generated directories such as `Dataset/exdark_yolo_zerodce/` are not guaranteed to exist in a fresh clone. Create them through the corresponding pipeline and preserve the source split and labels.

## Experimental records

`Results/comparisons_table.csv` is the canonical table currently used by the README and report. It contains the controlled 734-image test results for the five variants listed above.

For each new result, record:

- source dataset version and split;
- preprocessing method and parameter values;
- enhancement checkpoint and hash;
- detector checkpoint and hash;
- YOLO version, image size, confidence threshold, IoU threshold, batch size, and device;
- exact command and random seed;
- precision, recall, mAP@0.5, and mAP@0.5:0.95;
- generated CSV, logs, and representative failure cases.

The functions named `calculate_niqe` and `calculate_brisque` in `src/luong/metrics.py` are project-specific proxy scores. They are not standard NIQE or BRISQUE measurements and must not be reported under those standard names without replacing or validating the implementation.

## Availability limitations

This repository contains sample images and selected checkpoints. It does not currently provide a verified direct archive URL for every processed dataset. Do not use a generic Kaggle search page as if it were a reproducible download link. Add an exact, accessible dataset URL only after verifying it from a clean environment.
