"""Audit, prepare, and verify paired ExDark datasets for YOLO detection.

Example:
    python prepare_dataset.py --stage build --variant zerodcepp --scale-factor 4
"""

import argparse
import csv
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


SPLITS = {"train": "train", "valid": "val", "test": "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_URL = "https://github.com/NMH1203/Low-Light-Image-Enhancement-and-Downstream-Recognition"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def save_png(path, image):
    success, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise OSError(f"Cannot encode image: {path}")
    temporary = path.with_suffix(".png.tmp")
    encoded.tofile(temporary)
    temporary.replace(path)


def convert_labels(text, class_count):
    """Convert OBB corners to enclosing axis-aligned boxes; retain class IDs.

    Bounding boxes are clipped to the visible image. Every clipped or discarded
    annotation is returned in the audit trail. Invalid syntax is never ignored.
    """
    output, classes, issues = [], [], []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) not in (5, 9):
            raise ValueError(f"Line {line_number}: expected 5 or 9 fields, got {len(fields)}")
        values = np.asarray([float(field) for field in fields], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Line {line_number}: non-finite annotation")
        class_id = int(values[0])
        if values[0] != class_id or not 0 <= class_id < class_count:
            raise ValueError(f"Line {line_number}: invalid class ID {values[0]}")
        if len(fields) == 9:
            points = values[1:].reshape(4, 2)
            bounds = np.concatenate((points.min(axis=0), points.max(axis=0)))
        else:
            center, size = values[1:3], values[3:5]
            if np.any(size <= 0):
                raise ValueError(f"Line {line_number}: non-positive box size")
            bounds = np.concatenate((center - size / 2, center + size / 2))
        clipped = np.clip(bounds, 0, 1)
        if np.max(np.abs(bounds - clipped)) > 1e-8:
            issues.append({"line": line_number, "action": "clip_to_image", "original_bounds": bounds.tolist()})
        x_min, y_min, x_max, y_max = clipped
        width, height = x_max - x_min, y_max - y_min
        if min(width, height) <= 1e-10:
            issues.append({"line": line_number, "action": "drop_zero_area_box"})
            continue
        box = [(x_min + x_max) / 2, (y_min + y_max) / 2, width, height]
        row = f"{class_id} " + " ".join(f"{value:.10f}" for value in box)
        if row in output:
            issues.append({"line": line_number, "action": "drop_duplicate_box"})
            continue
        output.append(row)
        classes.append(class_id)
    return "\n".join(output) + ("\n" if output else ""), classes, issues


def inspect_record(item):
    source, source_split, split, path, class_count = item
    label_path = source / source_split / "labels" / (path.stem + ".txt")
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label: {label_path}")
    image_bytes = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    label_bytes = label_path.read_bytes()
    try:
        normalized, classes, issues = convert_labels(label_bytes.decode("utf-8-sig"), class_count)
    except ValueError as error:
        raise ValueError(f"{label_path}: {error}") from error
    return {
        "sample_id": path.stem, "original_id": path.stem.split(".rf.")[0],
        "split": split, "source_split": source_split,
        "source_image": path.relative_to(source).as_posix(),
        "source_label": label_path.relative_to(source).as_posix(),
        "width": image.shape[1], "height": image.shape[0],
        "source_sha256": digest(image_bytes), "source_label_sha256": digest(label_bytes),
        "pixel_sha256": digest(str(image.shape).encode() + image.tobytes()),
        "label_sha256": digest(normalized.encode()), "labels": normalized,
        "classes": classes, "box_count": len(classes), "issues": issues,
    }


def audit(source, workers):
    configuration = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8-sig"))
    names = configuration["names"]
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("Class IDs must be contiguous integers starting at zero.")
        names = [names[index] for index in range(len(names))]
    tasks = []
    for source_split, split in SPLITS.items():
        image_dir = source / source_split / "images"
        label_dir = source / source_split / "labels"
        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"Empty source split: {image_dir}")
        image_stems = [path.stem for path in images]
        if len(set(image_stems)) != len(image_stems):
            raise ValueError(f"Duplicate image stems in {image_dir}")
        orphan_labels = {path.stem for path in label_dir.glob("*.txt")} - set(image_stems)
        if orphan_labels:
            raise ValueError(f"Orphan labels in {label_dir}: {sorted(orphan_labels)[:5]}")
        tasks.extend((source, source_split, split, path, len(names)) for path in images)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(inspect_record, tasks))
    duplicate_groups = {}
    for key in ("sample_id", "original_id", "pixel_sha256"):
        groups = defaultdict(list)
        for record in records:
            groups[record[key]].append({"sample_id": record["sample_id"], "split": record["split"]})
        duplicate_groups[key] = [group for group in groups.values() if len(group) > 1]
    leakage = [
        {"kind": key, "samples": group}
        for key, groups in duplicate_groups.items() for group in groups
        if len({item["split"] for item in group}) > 1
    ]
    summary = {
        "source": str(source), "project_reference": PROJECT_URL,
        "image_count": len(records), "box_count": sum(record["box_count"] for record in records),
        "class_names": names, "class_id_policy": "Preserve source data.yaml IDs without remapping.",
        "split_counts": dict(Counter(record["split"] for record in records)),
        "image_sizes": dict(Counter(f"{r['width']}x{r['height']}" for r in records)),
        "annotation_actions": dict(Counter(issue["action"] for r in records for issue in r["issues"])),
        "empty_label_images": sum(not record["classes"] for record in records),
        "duplicate_groups": duplicate_groups, "cross_split_leakage": leakage,
    }
    return names, records, summary


def dataset_directories(output, variant):
    return {name: output / f"exdark_yolo_{name}" for name in ("dark", "clahe", variant)}


def output_image(root, record, kind):
    suffix = Path(record["source_image"]).suffix if kind == "dark" else ".png"
    return root / "images" / record["split"] / (record["sample_id"] + suffix)


def create_layout(directories, names, source):
    for root in directories.values():
        for split in SPLITS.values():
            for folder in ("images", "labels"):
                (root / folder / split).mkdir(parents=True, exist_ok=True)
        # Absolute paths make the configuration usable from any working directory.
        configuration = {"path": root.resolve().as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "nc": len(names), "names": dict(enumerate(names))}
        (root / "data.yaml").write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
        for filename in ("README.dataset.txt", "README.roboflow.txt"):
            if (source / filename).is_file():
                shutil.copy2(source / filename, root / ("SOURCE_" + filename))


def write_inventory(metadata, records, names, summary):
    write_json(metadata / "source_audit.json", summary)
    write_json(metadata / "annotation_changes.json", [
        {"sample_id": record["sample_id"], "split": record["split"], "changes": record["issues"]}
        for record in records if record["issues"]
    ])
    fields = ["sample_id", "original_id", "split", "source_image", "source_label", "width", "height", "box_count", "source_sha256", "source_label_sha256", "pixel_sha256", "label_sha256"]
    with (metadata / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    with (metadata / "class_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "class_id", "class_name", "images", "boxes"])
        for split in SPLITS.values():
            subset = [record for record in records if record["split"] == split]
            for class_id, name in enumerate(names):
                writer.writerow([split, class_id, name, sum(class_id in r["classes"] for r in subset), sum(r["classes"].count(class_id) for r in subset)])


def verify(directories, records):
    """Decode every output and verify pairing, labels, dimensions, and raw bytes."""
    started = time.monotonic()
    for kind, root in directories.items():
        for split in SPLITS.values():
            expected = {r["sample_id"] for r in records if r["split"] == split}
            actual_images = {p.stem for p in (root / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
            actual_labels = {p.stem for p in (root / "labels" / split).glob("*.txt")}
            if actual_images != expected or actual_labels != expected:
                raise ValueError(f"Image/label inventory mismatch: {root}, {split}")
        for record in records:
            path = output_image(root, record, kind)
            image = read_image(path)
            if image.shape != (record["height"], record["width"], 3):
                raise ValueError(f"Output geometry changed: {path}")
            label = root / "labels" / record["split"] / (record["sample_id"] + ".txt")
            if digest(label.read_bytes()) != record["label_sha256"]:
                raise ValueError(f"Output labels changed: {label}")
            if kind == "dark" and digest(path.read_bytes()) != record["source_sha256"]:
                raise ValueError(f"Raw image changed: {path}")
        print(f"Verified {kind}: {len(records)} images and matching labels.", flush=True)
    return {"status": "passed", "images_verified": len(records) * len(directories), "datasets_verified": list(directories), "elapsed_seconds": round(time.monotonic() - started, 2)}


def draw_preview(directories, records, names, output):
    """Show a deterministic set of source images with inherited ground-truth boxes."""
    candidates = [record for record in records if record["split"] == "val"]
    chosen = [candidates[index] for index in np.linspace(0, len(candidates) - 1, 4, dtype=int)]
    rows = []
    for record in chosen:
        cells = []
        for kind, root in directories.items():
            frame = cv2.resize(read_image(output_image(root, record, kind)), (360, 360))
            for line in record["labels"].splitlines():
                class_id, center_x, center_y, width, height = map(float, line.split())
                left, top = int((center_x - width / 2) * 360), int((center_y - height / 2) * 360)
                right, bottom = int((center_x + width / 2) * 360), int((center_y + height / 2) * 360)
                cv2.rectangle(frame, (left, top), (right, bottom), (60, 220, 100), 1)
                cv2.putText(frame, names[int(class_id)], (left, max(12, top - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 220, 100), 1, cv2.LINE_AA)
            cell = cv2.copyMakeBorder(frame, 40, 24, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
            title = {"dark": "Original dark", "clahe": "CLAHE + Bilateral", "zerodce": "Zero-DCE pretrained", "zerodcepp": "Zero-DCE++ pretrained"}[kind]
            cv2.putText(cell, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(cell, record["original_id"] + " | ground-truth boxes", (10, 416), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1, cv2.LINE_AA)
            cells.append(cell)
        rows.append(np.concatenate(cells, axis=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_png(output, np.concatenate(rows, axis=0))


def build(args, names, records, summary):
    import torch
    from src.enhancement import enhance_clahe_bilateral, enhance_neural, load_model

    checkpoint = args.checkpoint or Path("Results/weights") / f"{args.variant}_Epoch99.pth"
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {checkpoint}")
    if summary["cross_split_leakage"]:
        raise ValueError("Cross-split duplicates detected. Review metadata/source_audit.json before building.")
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model = load_model(checkpoint, args.variant, args.scale_factor, device)
    directories = dataset_directories(args.output, args.variant)
    metadata = args.output / "metadata"
    config = {
        "pipeline_version": 1, "variant": args.variant, "scale_factor": args.scale_factor,
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint.read_bytes()),
        "source_fingerprint": digest(json.dumps([(r["source_image"], r["source_sha256"], r["source_label_sha256"]) for r in records]).encode()),
        "output_format": "Original dark bytes; lossless PNG for enhanced images.",
        "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8], "bilateral_d": 7, "sigma_color": 50, "sigma_space": 50},
        "geometry": "Preserve source dimensions; no crop, resize, or spatial augmentation of saved images.",
        "split_policy": "Preserve the supplied Roboflow train/valid/test allocation; rename valid to val.",
        "model_source": "https://github.com/Li-Chongyi/Zero-DCE" + ("_extension" if args.variant == "zerodcepp" else ""),
        "training": "Official pretrained weights; no training or fitting on ExDark validation/test images.",
    }
    config_path = metadata / "build_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing output uses different source/model settings. Choose a new --output directory.")
    if not config_path.exists() and any(root.exists() for root in directories.values()):
        raise ValueError("Refusing to overwrite an output dataset without a matching build configuration.")
    write_json(config_path, config)
    create_layout(directories, names, args.source)
    completed_path = metadata / "completed.jsonl"
    completed = {}
    if completed_path.exists():
        for line in completed_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # Recover from an interrupted final journal write.
            completed[item["key"]] = item
    started = time.monotonic()
    processed = 0
    print(f"Enhancing {len(records)} images with {args.variant}, scale={args.scale_factor}, device={device}.", flush=True)

    def process_record(record):
        key = record["split"] + "/" + record["sample_id"]
        all_present = all(
            output_image(root, record, kind).is_file()
            and (root / "labels" / record["split"] / (record["sample_id"] + ".txt")).is_file()
            for kind, root in directories.items()
        )
        if key in completed and all_present:
            return completed[key], False
        image = read_image(args.source / record["source_image"])
        enhanced = {
            "dark": image,
            "clahe": enhance_clahe_bilateral(image),
            args.variant: enhance_neural(image, model, device),
        }
        brightness = {}
        for kind, root in directories.items():
            destination = output_image(root, record, kind)
            if kind == "dark":
                shutil.copy2(args.source / record["source_image"], destination)
            else:
                save_png(destination, enhanced[kind])
            label_path = root / "labels" / record["split"] / (record["sample_id"] + ".txt")
            label_path.write_bytes(record["labels"].encode("utf-8"))
            brightness[kind] = round(
                float(cv2.cvtColor(enhanced[kind], cv2.COLOR_BGR2GRAY).mean()), 4
            )
        return {"key": key, "mean_luminance_0_255": brightness}, True

    # The evaluation-only model has no mutable running state. Each worker owns
    # its tensors and output files; the main thread alone writes the journal.
    processing_workers = args.workers if device == "cpu" else 1
    with completed_path.open("a", encoding="utf-8") as journal, ThreadPoolExecutor(max_workers=processing_workers) as executor:
        for index, (item, is_new) in enumerate(executor.map(process_record, records), start=1):
            if not is_new:
                continue
            journal.write(json.dumps(item) + "\n")
            journal.flush()
            completed[item["key"]] = item
            processed += 1
            if processed == 1 or index % 100 == 0 or index == len(records):
                elapsed = time.monotonic() - started
                rate = processed / max(elapsed, 1e-9)
                print(f"[{index}/{len(records)}] {rate:.2f} images/s; estimated remaining {(len(records)-index)/rate/60:.1f} min", flush=True)
    validation = verify(directories, records)
    draw_preview(directories, records, names, Path("Results/figures/enhancement_comparison.png"))
    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_image_count": len(records), "total_output_images": len(records) * len(directories),
        "split_counts": summary["split_counts"], "boxes_per_dataset": summary["box_count"],
        "datasets": {kind: str(root.resolve()) for kind, root in directories.items()},
        "validation": validation, "runtime": {"torch": torch.__version__, "opencv": cv2.__version__, "numpy": np.__version__, "device": device, "threads": args.threads},
        "mean_luminance_0_255": {kind: round(float(np.mean([completed[r['split'] + '/' + r['sample_id']]["mean_luminance_0_255"][kind] for r in records])), 4) for kind in directories},
        "metrics_note": "Mean luminance is descriptive only. NIQE, BRISQUE, and detector mAP have not been measured.",
        "elapsed_seconds_this_build": round(time.monotonic() - started, 2),
    }
    write_json(metadata / "dataset_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("ExDark.v12i.yolov8-obb"))
    parser.add_argument("--output", type=Path, default=Path("Dataset"))
    parser.add_argument("--stage", choices=["audit", "build", "verify"], default="build")
    parser.add_argument("--variant", choices=["zerodce", "zerodcepp"], default="zerodcepp")
    parser.add_argument("--scale-factor", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if min(args.threads, args.workers, args.scale_factor) < 1:
        parser.error("Thread counts and scale factor must be positive.")
    if args.variant == "zerodce" and args.scale_factor != 1:
        parser.error("Use --scale-factor 1 with --variant zerodce.")
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if args.source == args.output or args.source in args.output.parents or args.output in args.source.parents:
        parser.error("Source and output directories must be separate, non-overlapping directories.")
    cv2.setNumThreads(1)
    print("Auditing source images, labels, class IDs, and split leakage...", flush=True)
    names, records, summary = audit(args.source, args.workers)
    metadata = args.output / "metadata"
    write_inventory(metadata, records, names, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "duplicate_groups"}, indent=2), flush=True)
    if args.stage == "build":
        build(args, names, records, summary)
    elif args.stage == "verify":
        config = json.loads((metadata / "build_config.json").read_text())
        result = verify(dataset_directories(args.output, config["variant"]), records)
        write_json(metadata / "verification.json", result)


if __name__ == "__main__":
    main()
