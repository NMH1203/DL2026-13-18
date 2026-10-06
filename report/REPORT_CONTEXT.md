# Project 18 report context

> Detailed result matrix, missing measurements, and reporting checks are maintained in `report/REPORT_RESULTS_CHECKLIST.md`.

Last updated: 2026-10-07

This file is the persistent reporting context for `Group13_Project18_Report.tex`.
Read it before writing or revising the Methods, Experimental Setup, Results,
Discussion, Error Analysis, or Conclusion sections.

## 1. Authoritative Project 18 requirements

Project 18 is **Low-Light Image Enhancement and Downstream Recognition**. The
two required outcomes are:

1. Enhance images captured under low-light conditions using deep-learning
   models.
2. Investigate whether improved visual quality also leads to better performance
   on downstream recognition tasks.

The report must answer both requirements. Enhancement quality alone is not a
sufficient result; every main image variant must also be evaluated through the
YOLO downstream detector under a clearly stated protocol.

## 2. Current study objective

The group is not evaluating only one enhancement method. The current study is
a multi-case comparison on ExDark. It aims to determine how classical, learned,
and hybrid enhancement pipelines affect YOLOv8n object detection.

The report must not assume that a visually brighter image is better for a
detector. Visual observations and downstream detection metrics must be reported
separately and then discussed together.

## 3. Experimental cases to preserve in the report

Use the following case names consistently. Do not merge cases that use different
architectures, checkpoints, datasets, or YOLO training protocols.

| ID | Case | Role in the comparison |
| --- | --- | --- |
| B0 | Raw dark ExDark | Downstream baseline without enhancement |
| C1 | CLAHE | Classical contrast-enhancement case |
| C2 | CLAHE + Bilateral Filter | Classical enhancement and denoising case |
| C3 | Zero-DCE trained from scratch | Deep-learning enhancement evaluated by the fixed raw-domain detector |
| C4 | Zero-DCE++ | Pretrained lightweight learned-enhancement case |
| C5 | Zero-DCE++ + Bilateral Filter | Hybrid learned-enhancement and denoising case |

The repository also contains an artifact explicitly named
`Zero-DCE++ + Bilateral` (`Results/tiep/test_zerodcepp_bilateral`). If this is
used as C4 in the final comparison, rename the case everywhere to
**Zero-DCE++ + Bilateral Filter**. Never call a Zero-DCE++ result a Zero-DCE
result. Confirm the exact checkpoint and generated dataset before freezing the
final table.

## 4. Downstream YOLO evaluation protocol

Each image case is passed to YOLOv8n. The report must state which of the
following protocol applies to every result:

- **Cascaded/fixed-detector evaluation:** a YOLO model trained on raw dark images
  is evaluated on an enhanced test set. This measures domain transfer without
  detector adaptation.
- **Domain-adapted evaluation:** YOLOv8n is fine-tuned on the corresponding
  enhanced training set and evaluated on the matching enhanced test set.

YOLO training uses `yolov8n.pt` pretrained weights in the current helpers. Call
this **fine-tuning**, not “training YOLO from scratch.” The phrase “from scratch”
applies to the locally trained Zero-DCE case only when the associated DCE
checkpoint and configuration support that statement.

For a fair comparison, lock the following across cases whenever possible:

- identical train/validation/test image IDs;
- identical 12-class mapping;
- identical detection labels and geometry;
- identical YOLO initialization, epochs, image size, batch size, seed, device,
  augmentation, confidence/IoU settings, and evaluation split;
- test metrics from an independent `model.val(..., split="test")` run;
- no values estimated from formulas or copied from another case.

Primary downstream metrics: Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
Report changes in **percentage points** when subtracting two percentages.

## 5. Dataset facts currently supported by the repository

- Local processed ExDark size: 7,345 images.
- Splits: 5,142 train, 1,469 validation, and 734 test images.
- Number of classes: 12.
- The original `exdark_yolo_dark` annotations include 9-field oriented boxes.
- The cleaned detection dataset converts them to 5-field axis-aligned YOLO
  labels; `preparation.json` records 23,146 conversions and 3 zero-area boxes
  dropped.
- Photometric enhancement/filtering preserves image geometry, but the report
  must not claim that the original 9-field labels were simply copied unchanged
  into the detection dataset.

## 6. Result evidence currently present

### 6.0 Confirmed five-case design and current CSV status

The group confirmed on 2026-10-07 that there are five enhancement cases plus the
raw baseline. Zero-DCE trained from scratch is one of those five cases. Its main
result uses the fixed raw-domain detector, whereas the separately adapted YOLO
result is only a domain-alignment ablation.

| Configuration | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| Raw dark | 0.6671 | 0.5883 | 0.6146 | 0.2862 |
| CLAHE | 0.6543 | 0.5935 | 0.6188 | 0.2872 |
| CLAHE + Bilateral | 0.6572 | 0.5800 | 0.6161 | 0.2856 |
| Zero-DCE++ | 0.6771 | 0.5281 | 0.5881 | 0.2705 |
| Zero-DCE++ + Bilateral | 0.7040 | 0.5677 | 0.6244 | 0.2921 |

The consolidated CSV supplies raw, CLAHE, CLAHE--bilateral, Zero-DCE++, and
Zero-DCE++--bilateral. The NML branch summary supplies the missing scratch
direct-cascade row: P=0.4131, R=0.2450, mAP50=0.2291, and mAP50-95=0.0970. Its
paired raw baseline is P=0.6802, R=0.5778, mAP50=0.6235, and mAP50-95=0.2909.
The scratch summary identifies a test-split evaluation but does not archive a
separate `measured_metrics.json`; keep this provenance limitation explicit.

### 6.1 Zero-DCE scratch-related recorded test logs

Sources:

- `Results/ninh/test_raw/measured_metrics.json`
- `Results/ninh/test_dce_cascaded/measured_metrics.json`
- `Results/ninh/test_dce_retrained/measured_metrics.json`

| Configuration | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| Raw detector on raw test images | 0.6627 | 0.5464 | 0.5980 | 0.2739 |
| Raw detector on Zero-DCE test images | 0.4062 | 0.2315 | 0.2092 | 0.0883 |
| YOLO fine-tuned and tested on Zero-DCE images | 0.6294 | 0.4742 | 0.5229 | 0.2363 |

These JSON files declare `split=test`, but their dataset and checkpoint paths
refer to another machine. The current checkout does not contain the referenced
`ninh_dark`/`ninh_zerodce` datasets or the associated checkpoints. Treat them as
recorded historical detector logs, not as the final 40-epoch scratch
comparison and not as locally reproduced results.

### 6.1a Integrated 40-epoch scratch detector logs

Sources on branch `codex/nml-integration`:

- `Results/yolo_runs/scenario1_dark_baseline/{args.yaml,results.csv}`
- `Results/yolo_runs/scenario4_zerodce_retrained/{args.yaml,results.csv}`

Both detector logs contain 40 epochs, batch size 16, image size 640, and
deterministic execution. Their exported arguments record seed 0; the raw run
uses `hsv_v=0.4` and the scratch-domain run uses `hsv_v=0.1`. The best archived
validation rows are raw at epoch 38 (P=0.6965, R=0.5871, mAP50=0.6350,
mAP50-95=0.2989) and scratch-domain at epoch 40 (P=0.6719, R=0.5438,
mAP50=0.5940, mAP50-95=0.2779). These are development validation values, not
held-out test values. They are excluded from the final controlled comparison
because the group-confirmed protocol uses seed 42 and identical augmentation
for every YOLO case. Traceable 40-epoch fixed-detector and matched-detector test
evaluations under that common protocol are still missing.

### 6.2 Zero-DCE++ + Bilateral recorded test log

Source: `Results/tiep/test_zerodcepp_bilateral/measured_metrics.json`.

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ---: | ---: | ---: | ---: |
| 0.7040 | 0.5677 | 0.6244 | 0.2921 |

The log declares `split=test` and references a YOLO model fine-tuned on
`exdark_yolo_zerodcepp_bilateral_d5_s25`. Its values match the final row of the
consolidated test matrix above.

### 6.3 CLAHE evidence status

The controlled test export now contains measured aggregate metrics for CLAHE
and CLAHE+bilateral. The older validation rows may still be discussed as
training evidence but are not the primary final comparison. The old value
0.6747 mAP@0.5 was generated by a formula in `src/luong/pipeline.py`; it is not
a measured CLAHE test result and must not appear as experimental evidence.

## 7. Reporting rules

1. Separate measured observations, qualitative observations, and causal
   hypotheses.
2. Do not claim that amplified high-ISO noise is the sole cause of a detection
   drop without a controlled denoising/noise ablation.
3. Domain shift is a plausible explanation when fixed-detector performance
   falls and matched-domain fine-tuning recovers performance, but word the
   conclusion as “consistent with” rather than “proves.”
4. Do not claim “false positives were eliminated” from aggregate precision.
5. Do not call the custom MSCN proxy implementation standard NIQE or BRISQUE.
6. Do not use `DATA.md`, `app.py`, `demo.py`, figures, checkpoints, or result
   tables as completed deliverables unless they exist in the submitted repo.
7. Use neutral report language. Avoid “dramatic,” “fatal,” “decisive proof,”
   “best of both worlds,” and “100% reproducible.”
8. Keep Zero-DCE, Zero-DCE++, CLAHE+bilateral, and learned-enhancement+bilateral
   cases distinct in text, tables, figures, and filenames.
9. The Abstract and Conclusion must treat scratch Zero-DCE as one of the five
   main cases and must not assign the ambiguous CSV row without provenance.

## 8. Intended report narrative

The report should follow this logic:

1. Project 18 asks for deep-learning low-light enhancement and downstream
   recognition analysis.
2. Build raw, classical, deep-learning, and hybrid image variants.
3. Evaluate visual/image characteristics separately from detector metrics.
4. Pass every variant through YOLOv8n under a documented protocol.
5. Compare raw versus enhanced cases and fixed-detector versus domain-adapted
   cases where available.
6. Analyze which cases help or hurt detection and show representative success
   and failure examples.
7. Conclude only within the tested ExDark/YOLOv8n configurations.
