"""
Classic Digital Image Processing (DIP) Baseline Module.
Implements CIE LAB + CLAHE (Contrast Limited Adaptive Histogram Equalization)
coupled with Bilateral Filtering for edge-preserving denoising.
"""

import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def enhance_clahe_bilateral(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 50.0,
    bilateral_sigma_space: float = 50.0,
) -> np.ndarray:
    """
    Enhances a low-light image using traditional DIP:
    1. RGB/BGR to CIE LAB color space.
    2. CLAHE applied strictly to the L (Luminance) channel to avoid color distortion.
    3. Convert back to RGB/BGR.
    4. Bilateral filter to smooth sensor noise while preserving object boundaries.

    Args:
        image_bgr: Input BGR image (uint8, [0, 255]).
        clip_limit: Threshold for contrast limiting in CLAHE.
        tile_grid_size: Size of grid for histogram equalization (rows, cols).
        bilateral_d: Diameter of each pixel neighborhood in bilateral filter.
        bilateral_sigma_color: Filter sigma in the color space.
        bilateral_sigma_space: Filter sigma in the coordinate space.

    Returns:
        Enhanced BGR image (uint8, [0, 255]).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is None or empty.")

    # 1. Convert BGR to CIE LAB color space
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 2. Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)

    # 3. Merge back and convert to BGR
    lab_enhanced = cv2.merge((l_clahe, a_channel, b_channel))
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 4. Apply Bilateral Filtering for edge-preserving denoising
    if bilateral_d > 0:
        bgr_enhanced = cv2.bilateralFilter(
            bgr_enhanced,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space,
        )

    return bgr_enhanced


def batch_enhance_clahe(
    src_dir: str | Path,
    dst_dir: str | Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    **kwargs
) -> int:
    """
    Batch-enhances all images in a source directory and saves them to a destination directory.
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in src_path.iterdir() if f.suffix.lower() in extensions]
    count = 0

    for img_file in tqdm(image_files, desc=f"CLAHE enhancing {src_path.name}"):
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        enhanced = enhance_clahe_bilateral(img, **kwargs)
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trích xuất và xem thử ảnh làm sáng bằng CLAHE + Bilateral")
    parser.add_argument("--input", "-i", type=str, default=None, help="Đường dẫn file ảnh tối đầu vào")
    parser.add_argument("--output", "-o", type=str, default="Results/clahe_output.jpg", help="Đường dẫn lưu ảnh kết quả CLAHE")
    parser.add_argument("--clip_limit", type=float, default=2.0, help="Ngưỡng tương phản CLAHE (mặc định: 2.0)")
    parser.add_argument("--bilateral_d", type=int, default=7, help="Đường kính lân cận lọc Bilateral (mặc định: 7)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Nếu không truyền ảnh, tự động lấy 1 ảnh mẫu trong Doc/samples hoặc Dataset
    if args.input is None:
        sample_candidates = [
            project_root / "Doc" / "samples" / "sample_1.jpg",
            project_root / "Doc" / "samples" / "sample_2.jpg",
        ]
        test_images = list((project_root / "Dataset" / "exdark_yolo_dark" / "test" / "images").glob("*.jpg"))
        if test_images:
            sample_candidates.append(test_images[0])

        input_path = None
        for cand in sample_candidates:
            if cand.exists():
                input_path = cand
                break
        if input_path is None:
            print("❌ Không tìm thấy ảnh mẫu! Hãy truyền đường dẫn: python src/preprocess_dip.py --input <đường_dẫn_ảnh>")
            exit(1)
    else:
        input_path = Path(args.input)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"🔄 Đang xử lý CLAHE + Bilateral cho ảnh:")
    print(f"   📥 Đầu vào:  {input_path}")
    print(f"   ⚙️ Tham số:  clip_limit={args.clip_limit}, bilateral_d={args.bilateral_d}")

    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        print(f"❌ Không thể đọc file ảnh: {input_path}")
        exit(1)

    enhanced = enhance_clahe_bilateral(
        img_bgr,
        clip_limit=args.clip_limit,
        bilateral_d=args.bilateral_d
    )

    cv2.imwrite(str(output_path), enhanced)
    print(f"✅ ĐÃ XUẤT ẢNH CLAHE THÀNH CÔNG!")
    print(f"   📤 Ảnh lưu tại: {output_path.resolve()}")
    print("=" * 65)
