# Low-Light Image Enhancement and Downstream Object Detection

## Project outline

This course project studies whether enhancing low-light images improves object detection. It compares original ExDark images, a classical CLAHE and bilateral-filter baseline, and Zero-DCE or Zero-DCE++ enhancement before YOLOv8 detection. The proposed schedule is 18 days for a team of five or six, using Python, PyTorch, OpenCV, and a Colab or Kaggle GPU.

The core research question is whether an image that appears clearer to a human also improves detector accuracy. Enhancement may recover edges and local contrast, but it can also amplify noise or introduce artifacts. Each conclusion therefore requires detection measurements on a held-out test split.

> This document is a project plan and conceptual reference. The example metric tables below are illustrative targets, not measured results. For the actual supplied dataset, processing, and verified counts, see [DATA.md](../DATA.md).

## 1. Vision pipeline

1. **Low-level vision:** Enhance an input image using CLAHE plus bilateral filtering, Zero-DCE, or Zero-DCE++.
2. **High-level vision:** Run YOLOv8 object detection on the original or enhanced image.
3. **Evaluation:** Compare image quality, detection precision and recall, mAP, and processing speed under the same split and hardware conditions.

The ExDark dataset contains naturally dark scenes, such as streets, indoor settings, and nighttime outdoor scenes. Its original release is described as containing 7,363 images. The Roboflow version used by the current preparation pipeline contains 7,345 images; see [DATA.md](../DATA.md) for its exact provenance and split counts.

## 2. Enhancement methods

### CLAHE and bilateral filtering

The classical baseline converts BGR to LAB color space, applies contrast-limited adaptive histogram equalization to the L channel, converts back to BGR, and applies a bilateral filter. This can increase local contrast while retaining edges. It may also amplify sensor noise in very dark areas.

### Zero-DCE and Zero-DCE++

Zero-DCE learns per-pixel light-enhancement curves without paired bright reference images. A curve can be written as

```text
LE(I; A) = I + A * I * (1 - I)
```

where `I` is a normalized pixel value and `A` is a learned curve parameter. Repeating the operation allows stronger enhancement while adapting to image content. The model predicts curve maps instead of directly synthesizing RGB values.

The original DCE-Net uses seven convolution layers and skip connections. Zero-DCE++ uses depthwise separable convolutions and predicts curves at reduced resolution before upsampling. The implementation must follow its checkpoint's architecture and curve sign exactly. The local from-scratch implementation is documented in [`src/luong/model_zerodce.py`](../src/luong/model_zerodce.py); the pretrained Zero-DCE++ pipeline must additionally record its external implementation and checkpoint identity.

Zero-reference training may combine four losses:

- **Spatial consistency:** Retain local intensity differences and structural detail.
- **Exposure control:** Guide local regions toward a target mean brightness.
- **Color constancy:** Reduce imbalance between RGB channels.
- **Illumination smoothness:** Penalize abrupt changes in predicted curve maps.

The current preparation workflow applies an official pretrained Zero-DCE++ checkpoint. It does not train an enhancement model on ExDark.

## 3. YOLOv8 detection and alignment

YOLOv8 uses a convolutional backbone, feature aggregation, and a detection head to predict classes and bounding boxes. A detector trained only on dark images may see a different image distribution at inference time if images are enhanced first. Compare both a direct cascade and a detector trained on enhanced training images.

Recommended controlled scenarios:

| Scenario | Detector training images | Evaluation images |
| --- | --- | --- |
| Dark baseline | Original dark train split | Original dark test split |
| CLAHE cascade | Original dark train split | CLAHE test split |
| Zero-DCE++ cascade | Original dark train split | Zero-DCE++ test split |
| Zero-DCE++ retrained | Zero-DCE++ train split | Zero-DCE++ test split |

Use the same annotation conversion, source split membership, model size, optimization budget, and evaluation settings across scenarios. Record augmentation settings because strong brightness augmentation can alter the comparison.

## 4. Data and annotation format

The Roboflow export provides YOLO-oriented boxes. The preparation script converts each four-corner box to an enclosing axis-aligned detection box, clips bounds to the image, and writes normalized labels:

```text
class_id center_x center_y width height
```

The actual export has 12 classes: Bicycle, Boat, Bottle, Bus, Cat, Cup, Motorbike, People, Table, car, chair, and dog. Capitalization and class IDs are preserved. The current pipeline retains the supplied train, validation, and test membership rather than creating a new random split.

The three prepared variants share labels and image dimensions. This pairing lets changes in detector performance be attributed more clearly to image processing.

## 5. Proposed 18-day schedule

| Period | Work |
| --- | --- |
| Days 1–3 | Audit the dataset, inspect annotations, and verify class and split distributions. |
| Days 4–5 | Visualize boxes and complete the data loader and conversion checks. |
| Days 6–7 | Train and evaluate the YOLO dark-image baseline. |
| Days 8–9 | Implement and validate CLAHE and Zero-DCE enhancement. |
| Days 10–11 | Generate paired enhanced datasets and evaluate image quality. |
| Days 12–14 | Run cascade and retrained detector experiments. |
| Days 15–16 | Compare metrics and complete ablation studies. |
| Days 17–18 | Prepare the report, figures, slides, and demonstration. |

This schedule is a plan, not a record of completed experiments.

## 6. Evaluation

For image quality, use standard NIQE or BRISQUE only when validated implementations and evaluation conditions are documented. The current functions in `src/luong/metrics.py` are project-specific proxies and must be labelled as such. Image-quality scores do not by themselves establish better detection.

For detection, report precision, recall, mAP at IoU 0.50, and mAP averaged over IoU 0.50 to 0.95. Measure latency or frames per second on the same device and include preprocessing in pipeline timing. Inspect predictions visually, especially where bright signs, headlights, noise, or small objects may affect boxes.

Useful ablations include changing the exposure target, comparing enhancement methods, and changing detector brightness augmentation. Report actual measurements, uncertainty where possible, and qualitative failure cases. Do not present expected values as experimental results.

## 7. Reproducibility and deliverables

Keep source export details, exact model checkpoints, configuration, software versions, class names, and image and label hashes with the results. Store the prepared datasets and reports together. [`DATA.md`](../DATA.md) records the current source, layout, and reproduction procedure.

Deliverables may include the baseline and enhanced datasets, detector checkpoints, comparison tables, plots, side-by-side images, and a short demonstration. Detector training, image-quality scoring, and final scientific conclusions require separate experiment runs; the dataset preparation pipeline alone does not produce them.

## References

- [Official ExDark dataset repository](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- [Roboflow ExDark version 12](https://universe.roboflow.com/project-h68de/exdark-kd37x/dataset/12)
- [Official Zero-DCE repository](https://github.com/Li-Chongyi/Zero-DCE)
- [Official Zero-DCE++ repository and checkpoint](https://github.com/Li-Chongyi/Zero-DCE_extension)
