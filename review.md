# Internal Report Review

This file is an internal quality-control review. It is not a substitute for the final report and should not be cited as experimental evidence.

## Overall assessment

The project addresses both requirements of Project 18: it implements deep-learning-based low-light enhancement and evaluates whether enhancement changes downstream object-detection performance. The experimental question is appropriate, but conclusions must remain limited to the controlled ExDark test results actually recorded in `Results/comparisons_table.csv`.

## Evidence currently available

| Input variant | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Raw dark | 0.6671 | 0.5883 | 0.6146 | 0.2862 |
| CLAHE | 0.6543 | **0.5935** | 0.6188 | 0.2872 |
| CLAHE + bilateral | 0.6572 | 0.5800 | 0.6161 | 0.2856 |
| Zero-DCE++ | 0.6771 | 0.5281 | 0.5881 | 0.2705 |
| Zero-DCE++ + bilateral | **0.7040** | 0.5677 | **0.6244** | **0.2921** |

Under the recorded protocol, Zero-DCE++ plus bilateral filtering produces the highest precision and mAP values, whereas CLAHE produces the highest recall. The differences are modest, so the report should describe them quantitatively and avoid words such as “dramatic,” “mandatory,” or “proven” unless uncertainty and repeated-run evidence are supplied.

## Strengths

- The project evaluates raw, classical, learned, and hybrid image variants.
- The five reported variants use the same test split and matched YOLOv8n settings.
- Precision, recall, mAP@0.5, and mAP@0.5:0.95 are reported together.
- The repository separates the local from-scratch Zero-DCE implementation from the pretrained Zero-DCE++ pipeline.
- A qualitative demo and model checkpoints are available.

## Required corrections before submission

1. Treat `Results/comparisons_table.csv` as the single quantitative source.
2. State explicitly that the `Zero-DCE` CSV row represents pretrained Zero-DCE++ if that mapping is retained.
3. Do not compare from-scratch Zero-DCE against Zero-DCE++ as a pure architectural ablation until both use a controlled protocol and have separate result rows.
4. Report the exact enhancement and detector checkpoints, their hashes, and software versions.
5. Label the current image-quality functions as proxy scores rather than standard NIQE/BRISQUE.
6. Separate aggregate test metrics from predictions shown for a single demo image.
7. Replace causal claims about noise amplification or domain shift with cautious interpretations unless targeted ablations support those causes.
8. Add repeated runs or uncertainty estimates if the report claims that small metric differences are meaningful.

## Recommended report conclusion

> On the controlled ExDark test split, preprocessing affected YOLOv8n detection differently across metrics. Zero-DCE++ followed by bilateral filtering achieved the highest precision, mAP@0.5, and mAP@0.5:0.95, while CLAHE achieved the highest recall. Zero-DCE++ without denoising underperformed the raw baseline in mAP, showing that visual enhancement alone did not consistently improve downstream recognition. These observations apply to the recorded dataset split, checkpoints, and evaluation settings; additional repeated runs and targeted ablations are required for broader causal conclusions.
