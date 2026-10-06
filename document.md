# Technical Notes for the Project Report

## Why use ExDark?

ExDark contains real low-light scenes with object annotations across 12 classes. It supports the downstream detection study but does not provide a paired normally exposed reference for every dark image. Consequently, full-reference enhancement metrics cannot be the primary evidence for this dataset.

## Why use Zero-DCE?

Zero-DCE estimates image-specific light-enhancement curves and can be trained without paired dark/bright targets. The local implementation predicts curve maps and optimizes spatial consistency, exposure, color constancy, and illumination smoothness losses.

The repository contains two learned-enhancement settings that must remain distinct:

- **Zero-DCE trained from scratch:** the local model and loss implementation under `src/luong/`.
- **Pretrained Zero-DCE++:** a compact pretrained enhancement pipeline used in the current controlled result table.

They differ in architecture, initialization, and potentially training data. Their comparison therefore represents a comparison of complete pipelines unless every other factor is controlled.

## Why compare CLAHE and bilateral filtering separately?

CLAHE changes local luminance contrast. Bilateral filtering adds edge-aware smoothing. Reporting CLAHE-only and CLAHE-plus-bilateral as separate rows isolates the incremental effect of the filtering step under the same detector protocol.

## What is the research design?

All rows in the current main comparison use the same ExDark test split and matched YOLOv8n evaluation settings. The independent variable is the preprocessing pipeline:

1. Raw dark.
2. CLAHE.
3. CLAHE + bilateral.
4. Pretrained Zero-DCE++.
5. Pretrained Zero-DCE++ + bilateral.

The dependent variables are precision, recall, mAP@0.5, and mAP@0.5:0.95. This design supports a controlled comparison of recorded pipelines, but it does not by itself establish why a metric changes.

## How should the results be interpreted?

The current CSV records the following main pattern:

- Zero-DCE++ + bilateral has the highest precision and mAP values.
- CLAHE has the highest recall.
- Zero-DCE++ without bilateral filtering has lower mAP than the raw baseline.
- CLAHE and CLAHE + bilateral differ only modestly from the raw baseline.

The report may state that enhancement and denoising interact with downstream detection. It should not claim that a particular noise mechanism has been proven without noise measurements or a targeted ablation.

## How should visual quality be discussed?

Use representative side-by-side images to discuss brightness, contrast, color shift, noise, clipping, and lost detail. The current functions named `calculate_niqe` and `calculate_brisque` are lightweight proxy scores and must not be described as standard NIQE or BRISQUE measurements.

## What evidence is still needed?

- A controlled result row for the from-scratch Zero-DCE model if it is included in the final report.
- Exact checkpoint identity and hashes for every learned pipeline.
- Full YOLO evaluation configuration and software versions.
- Repeated runs or uncertainty estimates for small metric differences.
- Per-class AP and representative failure cases if space permits.
- Enhancement latency measured on a named device under a documented protocol.

## Recommended writing style

Use “the results indicate,” “under the evaluated settings,” and “is associated with” for observations. Avoid “proves,” “eliminates,” “mandatory,” and “best in general.” Keep measured results separate from hypotheses and qualitative explanations.
