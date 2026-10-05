"""Project 18 training helper. Put this file beside the repository's run.py.

Commands: prepare, dce, enhance, train, evaluate. Run a command with --help.
Data and metrics are written to new folders; source files are never edited.
Reviewed against upstream commit 95b2e079b6361a7e3dc68e1b69651cfef1ac8890.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def local_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def names_dict(config):
    names = config["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    if not names or set(names) != set(range(len(names))):
        raise ValueError("Class IDs must be contiguous, starting at 0.")
    return names


def convert_label(text, class_count):
    """Convert 9-field OBB corners to 5-field enclosing detection boxes."""
    rows, actions = [], Counter()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = [float(v) for v in line.split()]
        if len(fields) not in (5, 9) or not all(math.isfinite(v) for v in fields):
            raise ValueError(f"Line {line_number}: expected 5 or 9 finite numbers.")
        cls = int(fields[0])
        if cls != fields[0] or not 0 <= cls < class_count:
            raise ValueError(f"Line {line_number}: invalid class ID.")
        if len(fields) == 9:
            xs, ys = fields[1::2], fields[2::2]
            bounds = [min(xs), min(ys), max(xs), max(ys)]
            actions["obb_to_detection"] += 1
        else:
            x, y, width, height = fields[1:]
            if width < 0 or height < 0:
                raise ValueError(f"Line {line_number}: negative box size.")
            bounds = [x - width / 2, y - height / 2, x + width / 2, y + height / 2]
        clipped = [max(0.0, min(1.0, v)) for v in bounds]
        if any(abs(a - b) > 1e-8 for a, b in zip(bounds, clipped)):
            actions["clipped_to_image"] += 1
        x1, y1, x2, y2 = clipped
        if x2 - x1 <= 1e-10 or y2 - y1 <= 1e-10:
            actions["dropped_zero_area"] += 1
            continue
        box = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        row = f"{cls} " + " ".join(f"{v:.10f}" for v in box)
        if row in rows:
            actions["dropped_duplicate_box"] += 1
            continue
        rows.append(row)
    return "\n".join(rows) + ("\n" if rows else ""), actions


def find_split(source, key):
    aliases = ("valid", "val") if key == "val" else (key,)
    for alias in aliases:
        for images, labels in [(source / alias / "images", source / alias / "labels"),
                               (source / "images" / alias, source / "labels" / alias)]:
            if images.is_dir() and labels.is_dir():
                return images, labels
    raise FileNotFoundError(f"Cannot find images and labels for {key} under {source}")


def image_files(folder):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)


def prepare(args):
    from PIL import Image

    source, output = local_path(args.source), local_path(args.output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-overlapping folders.")
    if output.exists():
        raise FileExistsError(f"Use the existing data.yaml or choose a NEW --output: {output}")
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8-sig"))
    names = names_dict(config)
    rng = random.Random(args.seed)
    records, actions, counts, seen_ids, seen_bytes = [], Counter(), {}, {}, {}
    keys = ("train", "val") if args.limit else tuple(SPLITS)
    for key in keys:
        images_dir, labels_dir = find_split(source, key)
        images = image_files(images_dir)
        if not images or len({p.stem for p in images}) != len(images):
            raise ValueError(f"Empty split or duplicate image stems: {images_dir}")
        if args.limit:
            images = sorted(rng.sample(images, min(args.limit, len(images))))
        counts[key] = len(images)
        for image in images:
            label = labels_dir / (image.stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Missing label: {label}")
            try:
                normalized, changes = convert_label(label.read_text(encoding="utf-8-sig"), len(names))
                with Image.open(image) as im:
                    im.verify()
            except Exception as error:
                raise ValueError(f"Invalid sample {image}: {error}") from error
            original_id = image.stem.split(".rf.")[0]
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            for value, seen in [(original_id, seen_ids), (digest, seen_bytes)]:
                if value in seen and seen[value] != key:
                    raise ValueError(f"Cross-split duplicate: {image}; previously in {seen[value]}")
                seen[value] = key
            records.append((key, image, normalized))
            actions.update(changes)
        print(f"Verified {key}: {len(images)} images", flush=True)
    output.mkdir(parents=True)
    for key in keys:
        (output / SPLITS[key] / "images").mkdir(parents=True)
        (output / SPLITS[key] / "labels").mkdir(parents=True)
    for key, image, normalized in records:
        split_dir = output / SPLITS[key]
        shutil.copy2(image, split_dir / "images" / image.name)
        (split_dir / "labels" / (image.stem + ".txt")).write_text(normalized, encoding="utf-8")
    new_config = {"path": output.as_posix(), "names": names}
    new_config.update({key: f"{SPLITS[key]}/images" for key in keys})
    (output / "data.yaml").write_text(yaml.safe_dump(new_config, sort_keys=False), encoding="utf-8")
    write_json(output / "preparation.json", {
        "source": str(source), "seed": args.seed, "smoke_only": bool(args.limit),
        "counts": counts, "label_actions": dict(actions),
        "samples": {key: [p.name for k, p, _ in records if k == key] for key in keys},
        "duplicate_check": "Original Roboflow ID and encoded image SHA-256 across selected splits.",
    })
    print("Ready:", output / "data.yaml")
    print("Label actions:", dict(actions))


def dataset_config(value):
    path = local_path(value)
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    root = Path(config.get("path", path.parent))
    if not root.is_absolute():
        raise ValueError("Use prepare first: this helper expects an absolute dataset path in data.yaml.")
    if not root.is_dir():
        raise FileNotFoundError(f"Update data.yaml 'path' after moving the dataset: {root}")
    return path, config, root


def select_device(value):
    import torch
    if value == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if value != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return value


def new_run(args):
    project = local_path(args.project)
    if Path(args.name).name != args.name:
        raise ValueError("--name must be a folder name, without path separators.")
    run = project / args.name
    if run.exists():
        raise FileExistsError(f"Choose a new --name to preserve the existing run: {run}")
    project.mkdir(parents=True, exist_ok=True)
    return project, run


def train(args):
    import torch
    import ultralytics
    from ultralytics import YOLO

    data, config, _ = dataset_config(args.data)
    if "train" not in config or "val" not in config:
        raise ValueError("Training requires separate train and val splits.")
    project, run = new_run(args)
    device = select_device(args.device)
    print(f"Device={device}, CUDA available={torch.cuda.is_available()}", flush=True)
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=device, workers=0, seed=args.seed, deterministic=True,
        hsv_v=args.hsv_v, close_mosaic=min(args.close_mosaic, args.epochs),
        patience=args.patience, project=str(project), name=args.name,
        exist_ok=False, save=True, plots=True,
    )
    run = Path(model.trainer.save_dir)
    write_json(run / "environment.json", {
        "python": sys.version, "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "device": device,
        "data": str(data), "seed": args.seed,
    })
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if frozen.returncode == 0:
        (run / "environment-freeze.txt").write_text(frozen.stdout, encoding="utf-8")
    print("Best:", run / "weights" / "best.pt")
    print("Last:", run / "weights" / "last.pt")
    print("Training log:", run / "results.csv")
    print("Only validation metrics were used during training. Run evaluate after model selection.")


def evaluate(args):
    from ultralytics import YOLO

    data, config, _ = dataset_config(args.data)
    weights = local_path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if args.split not in config:
        raise ValueError(f"No {args.split} split in {data}; a smoke dataset intentionally has no test split.")
    project, run = new_run(args)
    model = YOLO(str(weights))
    if names_dict({"names": model.names}) != names_dict(config):
        raise ValueError("Checkpoint class names/IDs differ from the dataset. Use your trained ExDark model.")
    result = model.val(
        data=str(data), split=args.split, imgsz=args.imgsz, batch=args.batch,
        device=select_device(args.device), workers=0, plots=True,
        project=str(project), name=args.name, exist_ok=False,
    )
    metrics = {"weights": str(weights), "data": str(data), "split": args.split,
               "precision": float(result.box.mp), "recall": float(result.box.mr),
               "map50": float(result.box.map50), "map50_95": float(result.box.map)}
    write_json(run / "measured_metrics.json", metrics)
    with (run / "measured_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    print(json.dumps(metrics, indent=2))


def dce(args):
    import numpy as np
    import torch
    from PIL import Image
    from src.luong.model_zerodce import DCENet
    from src.luong.loss_zerodce import ZeroDCELoss

    _, config, data_root = dataset_config(args.data)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if select_device(args.device) == "cpu" else f"cuda:{select_device(args.device)}")
    _, run = new_run(args)
    train_images = image_files(data_root / config["train"])
    val_images = image_files(data_root / config["val"])
    if not train_images or not val_images:
        raise ValueError("DCE requires nonempty, separate train and validation images.")
    run.mkdir()
    model = DCENet().to(device)
    criterion = ZeroDCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def batches(paths, shuffle=False):
        paths = list(paths)
        if shuffle:
            random.shuffle(paths)
        for start in range(0, len(paths), args.batch):
            tensors = []
            for path in paths[start:start + args.batch]:
                with Image.open(path) as im:
                    pixels = np.array(im.convert("RGB").resize((args.imgsz, args.imgsz)), dtype=np.float32) / 255.0
                tensors.append(torch.from_numpy(pixels).permute(2, 0, 1))
            yield torch.stack(tensors).to(device)

    best = float("inf")
    write_json(run / "config.json", {**vars(args), "device_used": str(device),
                                    "model_implementation": "src.luong.model_zerodce.DCENet",
                                    "torch_version": torch.__version__,
                                    "train_images": len(train_images), "val_images": len(val_images)})
    with (run / "loss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "validation_loss"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            total, count = 0.0, 0
            for x in batches(train_images, shuffle=True):
                optimizer.zero_grad(set_to_none=True)
                enhanced, curves = model(x)
                loss, _ = criterion(x, enhanced, curves)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite Zero-DCE training loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                total += loss.item() * x.size(0)
                count += x.size(0)
            model.eval()
            validation, nval = 0.0, 0
            with torch.inference_mode():
                for x in batches(val_images):
                    enhanced, curves = model(x)
                    loss, _ = criterion(x, enhanced, curves)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite Zero-DCE validation loss.")
                    validation += loss.item() * x.size(0)
                    nval += x.size(0)
            train_loss, val_loss = total / count, validation / nval
            writer.writerow([epoch, train_loss, val_loss])
            stream.flush()
            torch.save(model.state_dict(), run / "last.pth")
            if val_loss < best:
                best = val_loss
                torch.save(model.state_dict(), run / "best.pth")
            print(f"Epoch {epoch}/{args.epochs}: train={train_loss:.6f}, val={val_loss:.6f}", flush=True)
    print("Best validation-loss checkpoint:", run / "best.pth")
    print("Use this checkpoint only with src.luong.model_zerodce.DCENet.")


def enhance(args):
    import cv2
    import numpy as np
    import torch
    from src.luong.preprocess_dip import enhance_clahe_bilateral
    from src.luong.model_zerodce import DCENet

    _, config, source = dataset_config(args.data)
    output = local_path(args.output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate, non-overlapping folders.")
    if output.exists():
        raise FileExistsError(f"Choose a NEW --output folder: {output}")
    device = torch.device("cpu" if select_device(args.device) == "cpu" else f"cuda:{select_device(args.device)}")
    model, weights = None, None
    if args.method == "dce":
        if not args.weights:
            raise ValueError("--method dce requires --weights pointing to the dce command's best.pth.")
        weights = local_path(args.weights)
        model = DCENet().to(device)
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
        model.eval()
    # This stage expects the normalized layout produced by 'prepare'.
    for key, split in SPLITS.items():
        if key in config and config[key] != f"{split}/images":
            raise ValueError("Run prepare on this dataset first to normalize its layout.")
    output.mkdir(parents=True)
    counts = {}
    for key, split in SPLITS.items():
        if key not in config:
            continue
        image_dir = output / split / "images"
        image_dir.mkdir(parents=True)
        shutil.copytree(source / split / "labels", output / split / "labels")
        images = image_files(source / split / "images")
        counts[key] = len(images)
        with torch.inference_mode():
            for index, path in enumerate(images, 1):
                bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Unreadable image: {path}")
                if model is None:
                    result = enhance_clahe_bilateral(bgr)
                else:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    enhanced, _ = model(tensor)
                    if not torch.isfinite(enhanced).all():
                        raise ValueError(f"Non-finite pixels: {path}")
                    rgb = (enhanced[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
                    result = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if result.shape != bgr.shape:
                    raise ValueError(f"Unexpected geometry change: {path}")
                success, encoded = cv2.imencode(".png", result)
                if not success:
                    raise ValueError(f"PNG encoding failed: {path}")
                encoded.tofile(image_dir / (path.stem + ".png"))
                if index % 100 == 0 or index == len(images):
                    print(f"{args.method} {key}: {index}/{len(images)}", flush=True)
    config["path"] = output.as_posix()
    (output / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(output / "enhancement.json", {"source": str(source), "method": args.method,
               "weights": str(weights) if weights else None,
               "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest() if weights else None,
               "counts": counts, "geometry": "Original dimensions; unchanged normalized labels."})
    print("Ready:", output / "data.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Copy a dataset and normalize labels; --limit 5 creates a smoke dataset.")
    p.add_argument("--source", default="Dataset/exdark_yolo_dark")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    for command in ("train", "evaluate", "dce", "enhance"):
        p = sub.add_parser(command)
        p.add_argument("--data", required=True)
        p.add_argument("--device", default="auto", help="auto, cpu, or a single CUDA index such as 0")
        if command == "enhance":
            p.add_argument("--method", choices=["clahe", "dce"], required=True)
            p.add_argument("--weights")
            p.add_argument("--output", required=True)
            continue
        p.add_argument("--project", default="Results/ninh")
        p.add_argument("--name", required=True)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--imgsz", type=int, default=256 if command == "dce" else 640)
        if command == "evaluate":
            p.add_argument("--weights", required=True)
            p.add_argument("--split", choices=["val", "test"], default="test")
        else:
            p.add_argument("--epochs", type=int, default=20 if command == "dce" else 40)
            p.add_argument("--seed", type=int, default=42)
            if command == "train":
                p.add_argument("--hsv-v", type=float, default=0.4)
                p.add_argument("--close-mosaic", type=int, default=10)
                p.add_argument("--patience", type=int, default=10)
            else:
                p.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    for key in ("epochs", "batch", "imgsz"):
        if hasattr(args, key) and getattr(args, key) < 1:
            parser.error(f"--{key} must be positive")
    if getattr(args, "limit", 0) < 0:
        parser.error("--limit cannot be negative")
    if getattr(args, "imgsz", 256) < 16 and args.command == "dce":
        parser.error("DCE --imgsz must be at least 16")
    if hasattr(args, "lr") and args.lr <= 0:
        parser.error("--lr must be positive")
    if hasattr(args, "device"):
        if args.device not in ("auto", "cpu") and not args.device.isdigit():
            parser.error("--device accepts auto, cpu, or one CUDA index such as 0")
    {"prepare": prepare, "train": train, "evaluate": evaluate, "dce": dce, "enhance": enhance}[args.command](args)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
