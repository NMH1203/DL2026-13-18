"""
Project 18: Low-Light Image Enhancement and Downstream Recognition
Interactive Local Demo & Comprehensive Weight Inspector (demo.py)

This script provides:
1. Detailed technical inspection of model weights stored in Results/weights/
2. End-to-end local inference across all 4 research scenarios:
   - Scenario 1: Raw Dark Baseline (Raw Dark Image -> Baseline Dark YOLOv8n)
   - Scenario 2: CLAHE Cascaded (CLAHE + Bilateral -> Baseline Dark YOLOv8n)
   - Scenario 3: Zero-DCE Cascaded (Zero-DCE Enhancement -> Baseline Dark YOLOv8n) [Domain Shift]
   - Scenario 4: Zero-DCE Retrained (Zero-DCE Enhancement -> Retrained YOLOv8n) [Co-Design]
3. Image Quality Assessment (NIQE, BRISQUE) and Downstream Detection Metrics
4. Side-by-side 4-panel visual comparison export
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Import internal project modules
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.luong.model_zerodce import DCENet, enhance_image
from src.luong.preprocess_dip import enhance_clahe_bilateral
from src.luong.metrics import calculate_niqe, calculate_brisque
from src.luong.visualize import CLASS_COLORS, draw_bounding_boxes

# Default Paths
RESULTS_DIR = PROJECT_ROOT / "Results"
WEIGHTS_DIR = RESULTS_DIR / "weights"
FIGURES_DIR = RESULTS_DIR / "figures"
SAMPLE_DIR = PROJECT_ROOT / "sample_enhanced_images" / "1_raw_dark"

EXDARK_CLASSES = {
    0: "Bicycle",
    1: "Boat",
    2: "Bottle",
    3: "Bus",
    4: "Cat",
    5: "Cup",
    6: "Motorbike",
    7: "People",
    8: "Table",
    9: "Car",
    10: "Chair",
    11: "Dog",
}


def get_device(requested: str = "auto") -> torch.device:
    """Detects and returns torch computing device."""
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==============================================================================
# WEIGHT INSPECTOR & DIAGNOSTICS
# ==============================================================================
def inspect_weights(weights_dir: Path = WEIGHTS_DIR):
    """
    Performs deep inspection on trained weights located in Results/weights/
    Extracts parameter counts, tensor shapes, layer counts, and file metadata.
    """
    print("\n" + "=" * 80)
    print("🔬 COMPREHENSIVE WEIGHT INSPECTION & DIAGNOSTICS REPORT")
    print("=" * 80)

    if not weights_dir.exists():
        print(f"❌ Weights directory does not exist: {weights_dir}")
        return

    # Checkpoint 1: Zero-DCE PyTorch State Dict
    zerodce_path = weights_dir / "zerodce_best.pth"
    print(f"\n📦 [1] Zero-DCE Enhancement Model Checkpoint")
    print(f"   - File Path: {zerodce_path}")
    if zerodce_path.exists():
        size_kb = zerodce_path.stat().st_size / 1024.0
        print(f"   - File Size: {size_kb:.2f} KB ({zerodce_path.stat().st_size:,} bytes)")

        try:
            state_dict = torch.load(zerodce_path, map_location="cpu")
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"   - Checkpoint Type: PyTorch State Dictionary")
            print(f"   - Total Trained Parameters: {total_params:,}")
            print(f"   - Parameter Tensors Breakdown:")
            for k, v in state_dict.items():
                print(f"       * {k:20s}: shape {str(list(v.shape)):18s} | dtype={v.dtype} | mean={v.float().mean().item():+.4f}")
            print(f"   - Integrity Status: ✅ Valid & Loaded successfully")
        except Exception as e:
            print(f"   - Integrity Status: ❌ Load Error: {e}")
    else:
        print("   - Integrity Status: ⚠️ File not found.")

    # Checkpoint 2: YOLOv8 Dark Baseline
    yolo_dark_path = weights_dir / "yolov8n_dark_best.pt"
    print(f"\n📦 [2] YOLOv8 Dark Baseline Checkpoint (Scenario 1 & 3)")
    print(f"   - File Path: {yolo_dark_path}")
    if yolo_dark_path.exists():
        size_mb = yolo_dark_path.stat().st_size / (1024.0 * 1024.0)
        print(f"   - File Size: {size_mb:.2f} MB ({yolo_dark_path.stat().st_size:,} bytes)")
        try:
            # PyTorch 2.6+ defaults to weights_only=True; YOLO checkpoints contain custom model definitions
            try:
                ckpt = torch.load(yolo_dark_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(yolo_dark_path, map_location="cpu")
            model = ckpt.get("model", None)
            total_params = sum(p.numel() for p in model.parameters()) if model else 0
            epoch = ckpt.get("epoch", "N/A")
            train_args = ckpt.get("train_args", {})
            imgsz = train_args.get("imgsz", 640)
            names = ckpt.get("names", {})

            print(f"   - Backbone / Architecture: YOLOv8 Nano (Anchor-Free)")
            print(f"   - Total Parameters: {total_params:,} (~3.0M params)")
            print(f"   - Training Epoch: {epoch}")
            print(f"   - Input Dimension: {imgsz}x{imgsz}")
            print(f"   - Class Count: {len(names)} classes")
            print(f"   - Classes: {list(names.values())[:6]}... ({len(names)} total)")
            print(f"   - Integrity Status: ✅ Valid Ultralytics PyTorch Checkpoint")
        except Exception as e:
            print(f"   - Integrity Status: ❌ Load Error: {e}")
    else:
        print("   - Integrity Status: ⚠️ File not found.")

    # Checkpoint 3: YOLOv8 Zero-DCE Retrained Checkpoint
    yolo_retrained_path = weights_dir / "yolov8n_zerodce_best.pt"
    print(f"\n📦 [3] YOLOv8 Zero-DCE Retrained Checkpoint (Scenario 4)")
    print(f"   - File Path: {yolo_retrained_path}")
    if yolo_retrained_path.exists():
        size_mb = yolo_retrained_path.stat().st_size / (1024.0 * 1024.0)
        print(f"   - File Size: {size_mb:.2f} MB ({yolo_retrained_path.stat().st_size:,} bytes)")
        try:
            try:
                ckpt = torch.load(yolo_retrained_path, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt = torch.load(yolo_retrained_path, map_location="cpu")
            model = ckpt.get("model", None)
            total_params = sum(p.numel() for p in model.parameters()) if model else 0
            epoch = ckpt.get("epoch", "N/A")
            train_args = ckpt.get("train_args", {})
            imgsz = train_args.get("imgsz", 640)
            names = ckpt.get("names", {})

            print(f"   - Strategy: Co-Design Retrained on Zero-DCE Enhanced Data")
            print(f"   - Total Parameters: {total_params:,}")
            print(f"   - Training Epoch: {epoch}")
            print(f"   - Input Dimension: {imgsz}x{imgsz}")
            print(f"   - Class Count: {len(names)} classes")
            print(f"   - Integrity Status: ✅ Valid Ultralytics PyTorch Checkpoint")
        except Exception as e:
            print(f"   - Integrity Status: ❌ Load Error: {e}")
    else:
        print("   - Integrity Status: ⚠️ File not found.")

    # Benchmark summary table check
    csv_path = RESULTS_DIR / "comparisons_table.csv"
    if csv_path.exists():
        print(f"\n📊 Benchmark Evaluation Summary (from comparisons_table.csv):")
        df = pd.read_csv(csv_path)
        print(df.to_string(index=False))

    print("\n" + "=" * 80 + "\n")


# ==============================================================================
# 4-SCENARIO DEMO INFERENCE ENGINE
# ==============================================================================
class FourScenarioDemoEngine:
    """
    Orchestrates the 4 research scenarios on a given low-light image:
    1. Raw Dark Baseline
    2. CLAHE + Bilateral Cascaded
    3. Zero-DCE Cascaded (Domain Shift demonstration)
    4. Zero-DCE Retrained (Co-Design recovery)
    """

    def __init__(self, device: torch.device):
        self.device = device
        print(f"⚙️ Initializing FourScenarioDemoEngine on {device}...")

        # 1. Load Zero-DCE model
        self.zerodce_model = DCENet().to(self.device)
        zerodce_weights = WEIGHTS_DIR / "zerodce_best.pth"
        if zerodce_weights.exists():
            self.zerodce_model.load_state_dict(
                torch.load(zerodce_weights, map_location=self.device)
            )
            self.zerodce_model.eval()
            print(f"   ✅ Zero-DCE model loaded from {zerodce_weights.name}")
        else:
            print(f"   ⚠️ Zero-DCE weights not found at {zerodce_weights}. Using uninitialized model.")

        # 2. Load YOLO models
        dark_weights = WEIGHTS_DIR / "yolov8n_dark_best.pt"
        zerodce_yolo_weights = WEIGHTS_DIR / "yolov8n_zerodce_best.pt"

        if dark_weights.exists():
            self.yolo_dark = YOLO(str(dark_weights))
            print(f"   ✅ Baseline Dark YOLOv8 loaded from {dark_weights.name}")
        else:
            print(f"   ⚠️ Dark YOLO weights not found. Falling back to yolov8n.pt")
            self.yolo_dark = YOLO("yolov8n.pt")

        if zerodce_yolo_weights.exists():
            self.yolo_retrained = YOLO(str(zerodce_yolo_weights))
            print(f"   ✅ Retrained Zero-DCE YOLOv8 loaded from {zerodce_yolo_weights.name}")
        else:
            print(f"   ⚠️ Retrained YOLO weights not found. Reusing dark YOLO.")
            self.yolo_retrained = self.yolo_dark

    def enhance_zerodce(self, image_bgr: np.ndarray) -> np.ndarray:
        """Enhances image using trained Zero-DCE network."""
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            enhanced_tensor, _ = self.zerodce_model(tensor)

        enhanced_np = (
            enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        ).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)

    def enhance_clahe(self, image_bgr: np.ndarray) -> np.ndarray:
        """Enhances image using traditional DIP CLAHE + Bilateral filter."""
        return enhance_clahe_bilateral(
            image_bgr,
            clip_limit=2.0,
            tile_grid_size=(8, 8),
            bilateral_d=7,
            bilateral_sigma_color=50.0,
            bilateral_sigma_space=50.0,
        )

    def detect(self, model: YOLO, image_bgr: np.ndarray, conf: float = 0.25):
        """Runs YOLO object detection and parses predictions."""
        results = model.predict(image_bgr, conf=conf, verbose=False)[0]
        boxes = results.boxes
        detections = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                name = EXDARK_CLASSES.get(cls_id, str(cls_id))
                detections.append({
                    "box": xyxy,
                    "cls_id": cls_id,
                    "name": name,
                    "confidence": confidence,
                })
        return results, detections

    def run_image(self, image_path: Path | str, conf: float = 0.25) -> dict:
        """
        Executes all 4 scenarios on the given image.
        Returns a dictionary with comprehensive results for all 4 scenarios.
        """
        image_path = Path(image_path)
        raw_bgr = cv2.imread(str(image_path))
        if raw_bgr is None:
            raise FileNotFoundError(f"Cannot read image at {image_path}")

        # Ensure reasonable dimension for inference
        h, w = raw_bgr.shape[:2]
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            raw_bgr = cv2.resize(raw_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # 1. Process enhancements
        t0 = time.perf_counter()
        clahe_bgr = self.enhance_clahe(raw_bgr)
        t_clahe = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        zerodce_bgr = self.enhance_zerodce(raw_bgr)
        t_zerodce = (time.perf_counter() - t0) * 1000.0

        # 2. Image Quality Assessment (NIQE, BRISQUE)
        iqa = {
            "raw": {"niqe": calculate_niqe(raw_bgr), "brisque": calculate_brisque(raw_bgr)},
            "clahe": {"niqe": calculate_niqe(clahe_bgr), "brisque": calculate_brisque(clahe_bgr)},
            "zerodce": {"niqe": calculate_niqe(zerodce_bgr), "brisque": calculate_brisque(zerodce_bgr)},
        }

        # 3. Downstream Detection across 4 scenarios
        # Scenario 1: Raw Dark -> Dark YOLO
        res_sc1, det_sc1 = self.detect(self.yolo_dark, raw_bgr, conf=conf)

        # Scenario 2: CLAHE -> Dark YOLO
        res_sc2, det_sc2 = self.detect(self.yolo_dark, clahe_bgr, conf=conf)

        # Scenario 3: Zero-DCE -> Dark YOLO (Domain Shift)
        res_sc3, det_sc3 = self.detect(self.yolo_dark, zerodce_bgr, conf=conf)

        # Scenario 4: Zero-DCE -> Retrained YOLO (Co-Design)
        res_sc4, det_sc4 = self.detect(self.yolo_retrained, zerodce_bgr, conf=conf)

        return {
            "image_name": image_path.name,
            "image_shape": (h, w),
            "timing_ms": {"clahe": t_clahe, "zerodce": t_zerodce},
            "iqa": iqa,
            "images_bgr": {
                "raw": raw_bgr,
                "clahe": clahe_bgr,
                "zerodce": zerodce_bgr,
            },
            "scenarios": [
                {
                    "id": 1,
                    "name": "Scenario 1: Raw Dark Baseline",
                    "enhancement": "None (Raw Dark)",
                    "detector": "yolov8n_dark_best.pt",
                    "input_image_bgr": raw_bgr,
                    "detections": det_sc1,
                    "niqe": iqa["raw"]["niqe"],
                    "brisque": iqa["raw"]["brisque"],
                    "academic_map50": "62.35%",
                    "academic_precision": "68.02%",
                    "academic_note": "Benchmark baseline on unenhanced nighttime data",
                },
                {
                    "id": 2,
                    "name": "Scenario 2: CLAHE Cascaded",
                    "enhancement": "CIE LAB + CLAHE + Bilateral",
                    "detector": "yolov8n_dark_best.pt",
                    "input_image_bgr": clahe_bgr,
                    "detections": det_sc2,
                    "niqe": iqa["clahe"]["niqe"],
                    "brisque": iqa["clahe"]["brisque"],
                    "academic_map50": "67.47%",
                    "academic_precision": "68.02%",
                    "academic_note": "Highest mAP (+5.1%); bilateral filter suppresses ISO noise",
                },
                {
                    "id": 3,
                    "name": "Scenario 3: Zero-DCE Cascaded",
                    "enhancement": "Deep Learning Zero-DCE",
                    "detector": "yolov8n_dark_best.pt (Unretrained)",
                    "input_image_bgr": zerodce_bgr,
                    "detections": det_sc3,
                    "niqe": iqa["zerodce"]["niqe"],
                    "brisque": iqa["zerodce"]["brisque"],
                    "academic_map50": "22.91%",
                    "academic_precision": "41.31%",
                    "academic_note": "Domain Shift failure! Amplified sensor noise confuses dark detector",
                },
                {
                    "id": 4,
                    "name": "Scenario 4: Zero-DCE Retrained",
                    "enhancement": "Deep Learning Zero-DCE",
                    "detector": "yolov8n_zerodce_best.pt (Retrained)",
                    "input_image_bgr": zerodce_bgr,
                    "detections": det_sc4,
                    "niqe": iqa["zerodce"]["niqe"],
                    "brisque": iqa["zerodce"]["brisque"],
                    "academic_map50": "59.22%",
                    "academic_precision": "69.17%",
                    "academic_note": "Co-design recovery (+36.3%); PEAK precision across entire study",
                },
            ],
        }


# ==============================================================================
# VISUAL COMPARISON EXPORT & CONSOLE FORMATTING
# ==============================================================================
def render_annotated_scenario_panel(scenario: dict) -> np.ndarray:
    """Renders a single scenario panel with banner, bounding boxes, and metadata."""
    img_bgr = scenario["input_image_bgr"].copy()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Draw boxes
    for det in scenario["detections"]:
        box = det["box"]
        cls_id = det["cls_id"]
        conf = det["confidence"]
        name = det["name"]
        color = CLASS_COLORS.get(cls_id, (0, 255, 0))

        x1, y1, x2, y2 = box
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)

        # Label background
        label_text = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lbl_y1 = max(y1 - 18, 0)
        lbl_y2 = max(y1, 18)
        cv2.rectangle(img_rgb, (x1, lbl_y1), (x1 + tw + 6, lbl_y2), color, -1)
        text_color = (255, 255, 255) if cls_id not in [1, 2] else (0, 0, 0)
        cv2.putText(
            img_rgb,
            label_text,
            (x1 + 3, max(y1 - 4, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

    # Convert back to BGR for canvas composition
    panel = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Add header banner
    banner_height = 80
    h, w = panel.shape[:2]
    canvas = np.zeros((h + banner_height, w, 3), dtype=np.uint8)
    canvas[banner_height:, :] = panel

    # Banner background color based on scenario
    banner_colors = {
        1: (45, 45, 45),     # Dark gray
        2: (30, 80, 40),     # Dark green (CLAHE best mAP)
        3: (30, 30, 100),    # Dark red (Domain shift drop)
        4: (90, 60, 20),     # Dark cyan/blue (Recovery & top precision)
    }
    b_color = banner_colors.get(scenario["id"], (40, 40, 40))
    cv2.rectangle(canvas, (0, 0), (w, banner_height), b_color, -1)

    # Banner texts
    title_text = scenario["name"]
    meta_text = f"Det: {len(scenario['detections'])} objs | NIQE: {scenario['niqe']:.2f} | mAP: {scenario['academic_map50']}"
    prec_text = f"Precision: {scenario['academic_precision']} ({scenario['detector']})"

    cv2.putText(canvas, title_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, meta_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, prec_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 230, 255), 1, cv2.LINE_AA)

    return canvas


def create_composite_figure(eval_result: dict, save_path: Path | str | None = None) -> np.ndarray:
    """Combines all 4 scenario panels into a high-resolution 1x4 comparison grid."""
    panels = [render_annotated_scenario_panel(sc) for sc in eval_result["scenarios"]]

    # Standardize height
    target_height = 480
    resized_panels = []
    for p in panels:
        ph, pw = p.shape[:2]
        new_w = int(pw * (target_height / ph))
        resized = cv2.resize(p, (new_w, target_height), interpolation=cv2.INTER_AREA)
        resized_panels.append(resized)

    # Concatenate horizontally
    composite = np.hstack(resized_panels)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), composite)
        print(f"💾 Saved 4-Scenario Composite Comparison to:\n   👉 {save_path.resolve()}")

    return composite


def print_console_summary(eval_result: dict):
    """Prints a beautiful, informative summary table in the terminal."""
    print("\n" + "=" * 90)
    print(f"📸 4-SCENARIO DEMO EVALUATION FOR: {eval_result['image_name']}")
    print("=" * 90)

    rows = []
    for sc in eval_result["scenarios"]:
        det_summary = f"{len(sc['detections'])} detected"
        if sc["detections"]:
            classes_detected = [d["name"] for d in sc["detections"]]
            det_summary += f" ({', '.join(set(classes_detected))})"

        rows.append({
            "Scenario": sc["name"].split(":")[1].strip(),
            "Enhancement": sc["enhancement"],
            "Detector": sc["detector"],
            "NIQE (IQA)": f"{sc['niqe']:.2f}",
            "BRISQUE": f"{sc['brisque']:.2f}",
            "Objects": det_summary,
            "Study mAP@0.5": sc["academic_map50"],
            "Study Precision": sc["academic_precision"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("=" * 90)
    print("💡 SCIENTIFIC TAKEAWAYS:")
    print("  1. Scenario 1 (Baseline): Establishes dark detection performance (mAP = 62.4%).")
    print("  2. Scenario 2 (CLAHE): Peak mAP (67.5%), bilateral filter dampens ISO noise.")
    print("  3. Scenario 3 (Zero-DCE Cascaded): Demonstrates DOMAIN SHIFT! mAP drops from 62.4% to 22.9%.")
    print("     (Amplified sensor noise confuses unretrained dark feature extractors).")
    print("  4. Scenario 4 (Zero-DCE Retrained): Demonstrates CO-DESIGN RECOVERY! mAP rebounds to 59.2%")
    print("     and achieves the HIGHEST PRECISION of the entire study (69.2%), suppressing false positives.")
    print("=" * 90 + "\n")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Project 18: Low-Light Image Enhancement & 4-Scenario Downstream Demo"
    )
    parser.add_argument(
        "--inspect-weights",
        action="store_true",
        help="Inspect model weights and parameters in Results/weights/ and exit",
    )
    parser.add_argument(
        "--image",
        "-i",
        type=str,
        default=None,
        help="Path to input low-light image (if not specified, auto-selects a sample image)",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO object detection (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Computing device (default: auto)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(FIGURES_DIR / "demo_4_scenarios_comparison.png"),
        help="Destination path for output composite comparison image",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run inference sequentially on all sample images in sample_enhanced_images/1_raw_dark/",
    )
    args = parser.parse_args()

    # Weight inspection mode
    if args.inspect_weights:
        inspect_weights(WEIGHTS_DIR)
        return

    # Initialize device and inference engine
    device = get_device(args.device)
    engine = FourScenarioDemoEngine(device=device)

    # Inspect weights briefly
    inspect_weights(WEIGHTS_DIR)

    # Determine input images
    if args.batch:
        if SAMPLE_DIR.exists():
            input_images = sorted(list(SAMPLE_DIR.glob("*.jpg")))
        else:
            test_dir = PROJECT_ROOT / "Dataset" / "exdark_yolo_dark" / "test" / "images"
            input_images = sorted(list(test_dir.glob("*.jpg")))[:5]
    elif args.image:
        input_images = [Path(args.image)]
    else:
        # Default sample image candidates
        candidates = [
            SAMPLE_DIR / "image_1_raw.jpg",
            SAMPLE_DIR / "image_2_raw.jpg",
            PROJECT_ROOT / "Doc" / "samples" / "sample_1.jpg",
        ]
        test_dir = PROJECT_ROOT / "Dataset" / "exdark_yolo_dark" / "test" / "images"
        if test_dir.exists():
            candidates.extend(list(test_dir.glob("*.jpg"))[:3])

        chosen = None
        for cand in candidates:
            if cand.exists():
                chosen = cand
                break

        if chosen is None:
            print("❌ No low-light images found. Specify an image via --image <path>")
            return
        input_images = [chosen]

    print(f"\n🚀 Running 4-Scenario Evaluation on {len(input_images)} image(s)...")

    for idx, img_path in enumerate(input_images):
        print(f"\n[{idx+1}/{len(input_images)}] Processing: {img_path}")
        eval_result = engine.run_image(img_path, conf=args.conf)
        print_console_summary(eval_result)

        if len(input_images) == 1:
            out_file = Path(args.output)
        else:
            out_file = FIGURES_DIR / f"demo_comparison_{img_path.stem}.png"

        create_composite_figure(eval_result, save_path=out_file)

    print("\n🎉 DEMO RUN COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
