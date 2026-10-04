# Summary to run
# Tạo môi trường Python mới và cài thư viện
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Tải weights Zero-DCE/Zero-DCE++
.\.venv\Scripts\python.exe download_weights.py

# Kiểm tra dữ liệu nguồn
.\.venv\Scripts\python.exe prepare_dataset.py --stage audit

# Tạo dataset dark, CLAHE và Zero-DCE++
.\.venv\Scripts\python.exe prepare_dataset.py --stage build --variant zerodcepp --scale-factor 4 --threads 1 --workers 4

# ExDark Low-Light Dataset Preparation

Scripts for auditing the supplied ExDark Roboflow export and creating paired
YOLO detection datasets: original dark images, CLAHE-enhanced images, and
Zero-DCE++ enhanced images. The pipeline converts oriented boxes to enclosing
axis-aligned boxes and keeps the source class IDs and train/validation/test split.

The prepared data contains 7,345 images per variant (5,142 train, 1,469 val,
734 test). Enhanced images are lossless PNG at the original 640 x 640 size.
The build does not train the enhancement model or YOLO detector, and does not
claim improved detection accuracy.

## Requirements

- Windows PowerShell (commands below use PowerShell syntax)
- Python 3.12 recommended
- About 7 GB free space for all generated datasets, plus space for the source
- Internet access to download the source export and model weights

## Get the project files

Clone your GitHub repository, then open PowerShell in its folder:

```powershell
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
cd <YOUR-REPOSITORY>
```

The source Roboflow export should be added to this Git repository along with the
prepared datasets. If you clone a copy that does not include the source, download
the ExDark v12 YOLO OBB export from
[Roboflow](https://universe.roboflow.com/project-h68de/exdark-kd37x/dataset/12),
extract it at the project root, and make sure this file exists:

```text
ExDark.v12i.yolov8-obb/data.yaml
```

The source export and `Dataset/` are included when added to Git. They occupy
about 7 GB together, so Git LFS is required for image files. `Results/weights/`
is ignored because `download_weights.py` can fetch those checkpoints again.

## Set up Python

Run these commands from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download and checksum the official model weights:

```powershell
.\.venv\Scripts\python.exe download_weights.py
```

The checkpoints are saved under `Results/weights/`. If you install CPU-only
PyTorch, install its wheel from
[the PyTorch CPU index](https://download.pytorch.org/whl/cpu) before installing
the remaining requirements. `requirements-validated.txt` records versions used
for a verified build; the more flexible `requirements.txt` is the default setup.

## Run the pipeline

Audit the source images and labels without generating enhanced images:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --stage audit
```

Build the original, CLAHE, and Zero-DCE++ datasets (default settings):

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --stage build --variant zerodcepp --scale-factor 4 --threads 1 --workers 4
```

The command writes three datasets under `Dataset/` and audit/build records under
`Dataset/metadata/`. It can resume an interrupted build when the source and
settings are unchanged. Verify the generated images and labels with:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --stage verify
```

Optional: create the heavier original Zero-DCE variant in another output folder:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --variant zerodce --scale-factor 1 --output Dataset_ZeroDCE
```

To regenerate the plots from the completed dataset:

```powershell
.\.venv\Scripts\python.exe plot_dataset_statistics.py
```

## Train YOLOv8

Install Ultralytics in the environment used for detection training, then point
it to one dataset's `data.yaml`:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="Dataset/exdark_yolo_zerodcepp/data.yaml",
    epochs=40,
    imgsz=640,
    hsv_v=0.1,
    close_mosaic=10,
)
```

Choose `exdark_yolo_dark`, `exdark_yolo_clahe`, or
`exdark_yolo_zerodcepp` for the corresponding experiment. Each generated
`data.yaml` stores an absolute path for the machine that built it. When moving a
dataset, update its `path` value to the new dataset directory.

## Upload the project to GitHub

The `.gitignore` excludes `.venv/`, Python caches, generated model weights, and
Matplotlib cache. Keep `THIRD_PARTY_NOTICES.md`: it records dataset/model sources
and their license terms. Since the source and prepared datasets total roughly
7 GB, configure Git LFS before staging them. Install Git LFS first if needed,
then from the project root:

```powershell
git lfs install
git lfs track "*.jpg" "*.jpeg" "*.png" "*.bmp" "*.webp" "*.txt" "*.pth"
git add .gitattributes .gitignore README.md requirements.txt requirements-validated.txt THIRD_PARTY_NOTICES.md download_weights.py prepare_dataset.py plot_dataset_statistics.py src references tests Dataset ExDark.v12i.yolov8-obb
git status --short
git commit -m "Add ExDark data and preparation pipeline"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPOSITORY>.git
git push -u origin main
```

If `git remote add origin` says the remote already exists, inspect it with
`git remote -v` and update it with `git remote set-url origin <REPOSITORY-URL>`.
On later code changes, use:

```powershell
git add <changed-files>
git commit -m "Describe the change"
git push
```

`git status` should show the intended files before committing. GitHub has a
100 MB per-file limit for ordinary Git files; Git LFS stores matching image and
label files separately. Check that your GitHub account/repository has enough
LFS storage and transfer quota for roughly 7 GB before pushing. Cloning with
Git LFS installed downloads the actual file contents automatically.

## Outputs and method notes

- `Dataset/exdark_yolo_dark`: source images with normalized detection labels.
- `Dataset/exdark_yolo_clahe`: LAB luminance CLAHE followed by bilateral filtering.
- `Dataset/exdark_yolo_zerodcepp`: official pretrained Zero-DCE++, scale factor 4.
- `Dataset/metadata/`: source audit, sample manifest, annotation changes, build
  settings, progress journal, and verification report.
- `Results/figures/`: generated comparison and dataset statistics plots.
- `Results/weights/`: downloaded official checkpoints.

Three zero-area source boxes are removed; no images are removed. The outputs use
23,143 valid boxes each. The source `valid` split is named `val` in the outputs.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dataset, method, and
checkpoint attribution and license notes.
