"""Plot audited class counts and measured output luminance without model metrics."""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "Results" / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    metadata = root / "Dataset" / "metadata"
    destination = root / "Results" / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    with (metadata / "class_distribution.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["class_name"] for row in rows if row["split"] == "train"]
    colors = {"train": "#31688e", "val": "#35b779", "test": "#f5c344"}
    figure, axis = plt.subplots(figsize=(11, 5.5), layout="constrained")
    left = np.zeros(len(names))
    for split, color in colors.items():
        values = np.asarray([int(row["boxes"]) for row in rows if row["split"] == split])
        axis.barh(names, values, left=left, color=color, label=split.capitalize())
        left += values
    for index, total in enumerate(left):
        axis.text(total + left.max() * 0.01, index, f"{int(total):,}", va="center", fontsize=9)
    axis.set_xlim(0, left.max() * 1.13)
    axis.invert_yaxis()
    axis.set_xlabel("Valid object annotations")
    axis.set_title("ExDark: class distribution after label preparation\n7,345 images | 23,143 bounding boxes | source split preserved", loc="left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False)
    figure.savefig(destination / "class_distribution.png", dpi=180)
    plt.close(figure)
    print(f"Saved {destination / 'class_distribution.png'}")

    if not (metadata / "dataset_report.json").exists():
        print("The full build is still running; luminance plot will be available after completion.")
        return
    records = {}
    for line in (metadata / "completed.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[record["key"]] = record["mean_luminance_0_255"]
    report = json.loads((metadata / "dataset_report.json").read_text(encoding="utf-8"))
    if len(records) != report["source_image_count"]:
        raise ValueError("The progress journal does not match the completed dataset.")
    figure, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    titles = {"dark": "Original dark", "clahe": "CLAHE + Bilateral", "zerodcepp": "Zero-DCE++", "zerodce": "Zero-DCE"}
    for kind in report["datasets"]:
        values = [record[kind] for record in records.values()]
        axis.hist(values, bins=np.linspace(0, 255, 40), histtype="step", linewidth=2, label=titles[kind])
    axis.set(xlim=(0, 255), xlabel="Mean grayscale luminance per image (0-255)", ylabel="Number of images")
    axis.set_title("Measured luminance of all 7,345 paired images\nDescriptive brightness only; not a quality or detection score", loc="left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.savefig(destination / "luminance_distribution.png", dpi=180)
    plt.close(figure)
    print(f"Saved {destination / 'luminance_distribution.png'}")


if __name__ == "__main__":
    main()
