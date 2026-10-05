# HƯỚNG DẪN CHẠY PROJECT 18 TRÊN KAGGLE (PHIÊN BẢN TỰ ĐỘNG NHẬN DIỆN MỌI TÊN DATASET)

---

## NGUYÊN NHÂN LỖI TRƯỚC ĐÓ:
Nhìn vào thanh **Add Input** bên phải màn hình của bạn:
- Dataset của bạn tên là: **`ExDark YOLO Low-Light Detection_NML(USTH)`**
- Nhưng code cũ tìm kiếm cố định thư mục: `/kaggle/input/exdark-yolo-dark`
$\rightarrow$ Vì tên khác nhau nên code báo lỗi `AssertionError: Không tìm thấy data.yaml`.

Ngoài ra, trên Kaggle thư mục `/kaggle/input/` là **Read-Only** (chỉ đọc, không cho sửa). Đoạn code mới dưới đây đã được sửa lại thông minh: **Tự động quét tìm `data.yaml` ở mọi thư mục mà không cần quan tâm bạn đặt tên dataset là gì**, đồng thời xử lý quyền ghi an toàn 100%.

---

## BỘ 4 CELL HOÀN CHỈNH (COPY LẦN LƯỢT VÀO KAGGLE NOTEBOOK)

### 🔹 Cell 1: Cài thư viện & Tự động giải nén mã nguồn vào `/kaggle/working`
```python
!pip install ultralytics opencv-python-headless -q

import os, zipfile, shutil
from pathlib import Path
import torch

# Tự động tìm file zip chứa code trong /kaggle/input
code_zips = list(Path("/kaggle/input").glob("**/*code*.zip")) + list(Path("/kaggle/input").glob("**/project18*.zip"))

if code_zips:
    code_zip_path = code_zips[0]
    print(f"📦 Tìm thấy file code: {code_zip_path.name}")
    with zipfile.ZipFile(code_zip_path, 'r') as zip_ref:
        zip_ref.extractall("/kaggle/working")
else:
    # Nếu Kaggle đã tự giải nén sẵn file code
    run_files = list(Path("/kaggle/input").glob("**/run.py"))
    if run_files:
        src_dir = run_files[0].parent
        for item in src_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, Path("/kaggle/working") / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, Path("/kaggle/working") / item.name)

%cd /kaggle/working
print("=" * 60)
print("🚀 GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "⚠️ Chưa bật GPU (Chọn GPU T4 ở cột phải)")
print("📁 Danh sách file trong working:", os.listdir("/kaggle/working"))
print("=" * 60)
```

---

### 🔹 Cell 2: Tự động quét Dataset và khớp nối dữ liệu an toàn
```python
import os, yaml
from pathlib import Path

# Tự động quét tìm file data.yaml trong TOÀN BỘ /kaggle/input (không phụ thuộc tên dataset bạn đặt)
yaml_files = list(Path("/kaggle/input").glob("**/data.yaml"))
assert len(yaml_files) > 0, "❌ Không tìm thấy data.yaml trong /kaggle/input! Hãy kiểm tra danh sách Add Input."

yaml_src = yaml_files[0]
dataset_root = yaml_src.parent
print("📂 Tìm thấy Dataset gốc tại:", dataset_root)

# Tạo thư mục Dataset chuẩn trong /kaggle/working
target_dark_dir = Path("/kaggle/working/Dataset/exdark_yolo_dark")
target_dark_dir.mkdir(parents=True, exist_ok=True)

# Tạo liên kết symbolic link đến các thư mục train, valid, test
for split in ["train", "valid", "test"]:
    src_split = dataset_root / split
    dst_split = target_dark_dir / split
    if src_split.exists() and not dst_split.exists():
        os.symlink(src_split, dst_split)
        print(f"🔗 Đã liên kết thư mục {split}")

# Đọc cấu hình data.yaml và lưu file mới vào thư mục working (tránh lỗi Read-Only của Kaggle)
with open(yaml_src, "r") as f:
    cfg = yaml.safe_load(f)

cfg["path"] = str(target_dark_dir)
cfg["train"] = "train/images"
cfg["val"] = "valid/images"
cfg["test"] = "test/images"

target_yaml = target_dark_dir / "data.yaml"
with open(target_yaml, "w") as f:
    yaml.safe_dump(cfg, f)

print(f"✅ Đã tạo cấu hình data.yaml thành công tại: {target_yaml}")

# Kiểm tra xác thực số lượng ảnh
n_train = len(list((target_dark_dir / "train" / "images").glob("*.*")))
n_val = len(list((target_dark_dir / "valid" / "images").glob("*.*")))
n_test = len(list((target_dark_dir / "test" / "images").glob("*.*")))
print(f"📊 Xác thực dữ liệu: Train={n_train} | Valid={n_val} | Test={n_test} ảnh")
print(f"🏷️ 12 Classes:", list(cfg.get("names", {}).values()))
```

---

### 🔹 Cell 3: CHẠY HUẤN LUYỆN TOÀN BỘ PIPELINE BẰNG `run.py`
```python
# Chạy tự động:
# Phase 0 & 1: Khởi tạo & Kiểm tra
# Phase 2: Train Zero-DCE (10 epochs) & Tăng sáng ảnh
# Phase 3: Train YOLOv8 trên GPU T4 (40 epochs)
# Phase 4: Sinh bảng comparisons_table.csv và vẽ biểu đồ mAP
!python run.py --phase 0,1,2,3,4 --epochs_dce 10 --epochs_yolo 40 --batch_size 16
```
*(Nếu muốn chạy thử nghiệm nhanh kiểm tra luồng trước, bạn có thể đổi `--epochs_yolo 40` thành `--epochs_yolo 2`)*.

---

### 🔹 Cell 4: Nén toàn bộ kết quả để tải về máy tính (1-Click)
```python
# Đóng gói toàn bộ thư mục Results (weights, biểu đồ, bảng csv) thành 1 file zip duy nhất
!zip -r /kaggle/working/Project18_Results_Final.zip /kaggle/working/Results

print("=" * 60)
print("🎉 ĐÃ ĐÓNG GÓI THÀNH CÔNG! File lưu tại: /kaggle/working/Project18_Results_Final.zip")
print("👉 Hãy bấm tải file này tại mục Output ở cột bên phải về máy tính.")
print("=" * 60)
```
