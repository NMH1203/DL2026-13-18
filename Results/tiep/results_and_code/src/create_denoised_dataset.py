"""Build a YOLO dataset for: Zero-DCE/Zero-DCE++ -> Bilateral -> YOLOv8.

The filter changes pixel values only, so YOLO labels and the original
train/validation/test split are copied unchanged. Both common layouts are
supported: ``<split>/images`` and ``images/<split>``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import yaml


EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a bilateral-filtered YOLO dataset")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--diameter", type=int, default=5)
    parser.add_argument("--sigma-color", type=float, default=25.0)
    parser.add_argument("--sigma-space", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="First N images per split; smoke tests only")
    return parser.parse_args()


def find_images(source: Path, config: dict, split: str) -> Path:
    aliases = ("val", "valid") if split == "val" else (split,)
    candidates = []
    configured = config.get(split)
    if isinstance(configured, str):
        path = Path(configured)
        candidates.append(path if path.is_absolute() else source / path)
    for alias in aliases:
        candidates += [source / alias / "images", source / "images" / alias]
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    raise FileNotFoundError(f"Cannot locate {split} images under {source}")


def find_labels(source: Path, images: Path) -> Path:
    parts = list(images.relative_to(source).parts)
    if "images" not in parts:
        raise ValueError(f"No 'images' component in {images}")
    parts[parts.index("images")] = "labels"
    labels = source.joinpath(*parts)
    if not labels.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels}")
    return labels


def process(image_path: Path, labels: Path, out_images: Path, out_labels: Path, args) -> None:
    label = labels / f"{image_path.stem}.txt"
    if not label.is_file():
        raise FileNotFoundError(f"Missing label for {image_path.name}: {label}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV cannot read {image_path}")
    result = cv2.bilateralFilter(
        image,
        d=args.diameter,
        sigmaColor=args.sigma_color,
        sigmaSpace=args.sigma_space,
    )
    parameters = []
    if image_path.suffix.lower() in {".jpg", ".jpeg"}:
        parameters = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif image_path.suffix.lower() == ".png":
        parameters = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if not cv2.imwrite(str(out_images / image_path.name), result, parameters):
        raise RuntimeError(f"Cannot write {image_path.name}")
    shutil.copy2(label, out_labels / label.name)


def main() -> None:
    args = parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must not overlap")
    if args.diameter <= 0 or args.diameter % 2 == 0:
        raise ValueError("--diameter must be a positive odd integer")
    if min(args.sigma_color, args.sigma_space, args.workers) <= 0:
        raise ValueError("Sigma values and workers must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    config_path = source / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or "names" not in config:
        raise ValueError(f"Invalid data.yaml: {config_path}")

    counts = {}
    try:
        for split, folder in SPLITS.items():
            images = find_images(source, config, split)
            labels = find_labels(source, images)
            files = sorted(p for p in images.iterdir() if p.suffix.lower() in EXTENSIONS)
            files = files[: args.limit] if args.limit else files
            if not files:
                raise ValueError(f"No images in {images}")
            out_images, out_labels = output / folder / "images", output / folder / "labels"
            out_images.mkdir(parents=True)
            out_labels.mkdir(parents=True)
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(lambda p: process(p, labels, out_images, out_labels, args), files))
            if len(list(out_labels.glob("*.txt"))) != len(files):
                raise RuntimeError(f"Image/label count mismatch in {split}")
            counts[split] = len(files)
            print(f"Completed {split}: {len(files)} pairs", flush=True)

        output_config = {
            "path": str(output),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "names": config["names"],
        }
        (output / "data.yaml").write_text(
            yaml.safe_dump(output_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        metadata = {
            "source": str(source),
            "method": "OpenCV bilateralFilter",
            "diameter": args.diameter,
            "sigma_color": args.sigma_color,
            "sigma_space": args.sigma_space,
            "counts": counts,
            "smoke_test_limit": args.limit,
        }
        (output / "denoising_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise
    print(f"Ready: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
