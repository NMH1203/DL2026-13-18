# Kế hoạch nghiên cứu Zero-DCE train-from-scratch và tác động lên YOLOv8

Ngày lập kế hoạch: 06/10/2026  
Phạm vi: project hiện tại và implementation chính thức tại <https://github.com/Li-Chongyi/Zero-DCE>

## 1. Câu hỏi nghiên cứu

Nghiên cứu cần trả lời riêng ba câu hỏi, không gộp chúng thành một kết luận:

1. Zero-DCE train-from-scratch trong project có khác implementation chính thức về kiến trúc, loss, khởi tạo, dữ liệu và lịch train hay không?
2. Khi chỉ thay bộ tiền xử lý ảnh, một detector YOLO cố định thay đổi kết quả như thế nào?
3. Khi huấn luyện lại YOLO trên từng miền ảnh, detector có thích nghi và thu hồi được phần hiệu năng bị mất hay không?

Trong kế hoạch này, "train from scratch" mặc định nói về **Zero-DCE**. Các run YOLO hiện tại không phải train from scratch: mã dùng `YOLO("yolov8n.pt")`, tức khởi tạo từ trọng số COCO. Nếu cần nghiên cứu YOLO from scratch, phải thêm một nhánh riêng dùng `YOLO("yolov8n.yaml")` và `pretrained=False`.

## 2. Kết luận sơ bộ từ artifact hiện có

Repo Zero-DCE gốc chỉ huấn luyện/tăng sáng ảnh; không chứa pipeline YOLO. Vì vậy không tồn tại một "kết quả YOLO chính thức của Zero-DCE repo" để so trực tiếp. Hai pipeline cần so trong project là:

- **Official-pretrained:** ảnh ExDark được tăng sáng bằng kiến trúc và `Epoch99.pth` chính thức.
- **Local-scratch:** ảnh ExDark được tăng sáng bằng `src/luong/model_zerodce.py`, checkpoint được train từ đầu trên train split ExDark bởi `src/LBN/train_ninh.py`.

Các số test đã lưu trong `Results/ninh` cho thấy:

| Pipeline | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| Raw, YOLO train trên raw | 0,6627 | 0,5464 | 0,5980 | 0,2739 |
| Local DCE cascaded, dùng cùng YOLO raw | 0,4062 | 0,2315 | 0,2092 | 0,0883 |
| Local DCE, YOLO retrain trên ảnh DCE | 0,6294 | 0,4742 | 0,5229 | 0,2363 |

So với raw, cascaded giảm 38,88 điểm phần trăm mAP@0.5; retrained còn thấp hơn 7,50 điểm phần trăm. Đây là **quan sát từ log**, chưa đủ để kết luận nguyên nhân. Dataset `ninh_zerodce` và checkpoint DCE scratch tạo ra nó hiện không có trong checkout này, nên chưa thể audit pixel, hash, class mapping và tính toàn vẹn của run cũ.

## 3. Khác biệt kỹ thuật đã xác định

| Thành phần | Zero-DCE chính thức | Project hiện tại | Tác động có thể có |
| --- | --- | --- | --- |
| Công thức curve | `x + r * (x^2 - x)` | `x + A * x * (1 - x)` | Hai công thức đảo dấu tham số. Với train từ đầu, không làm giảm không gian biểu diễn; nhưng checkpoint giữa hai implementation không tương thích nếu không đổi dấu/đúng forward. |
| Thứ tự skip concat | `[x3, x4]`, `[x2, x5]`, `[x1, x6]` | `[x4, x3]`, `[x5, x2]`, `[x6, x1]` | Tương đương về năng lực khi train từ đầu nhưng làm đổi thứ tự channel, nên không thể nạp checkpoint chính thức rồi giả định output giống nhau. |
| Clamp output | Không clamp trong forward | `torch.clamp(..., 0, 1)` | Có thể làm mất gradient tại pixel vượt biên và thay đổi quỹ đạo tối ưu. |
| Khởi tạo convolution | Normal mean 0, std 0,02 | PyTorch default | Khác điểm xuất phát, đặc biệt đáng kể khi chỉ train 10 epoch. |
| Exposure loss | Bình phương sai lệch tới 0,6 | Trị tuyệt đối sai lệch tới 0,6 | Gradient và mức phạt vùng rất tối/rất sáng khác nhau. |
| Color loss | Căn của tổng lũy thừa bậc bốn của chênh lệch kênh | Căn của tổng bình phương chênh lệch kênh | Cường độ phạt lệch màu khác. |
| TV loss | Tổng trên channel, nhân 2, chia spatial và batch | Chia thêm số channel | Với 24 curve maps và cùng hệ số 200, TV chính thức xấp xỉ mạnh hơn 48 lần theo cách chuẩn hóa hiện tại. |
| Dữ liệu DCE | Tập chính thức 2.422 ảnh; paper còn khảo sát các biến thể dữ liệu | 5.142 ảnh train + 1.469 ảnh validation ExDark | Domain, phân bố ánh sáng và nội dung vật thể khác. ExDark-specific training có thể thích nghi tốt hoặc overfit vào loss không phù hợp detection. |
| Lịch DCE train | 200 epoch, batch 8, 256 px, Adam 1e-4, weight decay 1e-4, clip 0,1 | Run đã lưu: 10 epoch, batch 8, 256 px, cùng optimizer/clip | Ngân sách train khác 20 lần; run local có thể chưa hội tụ. Tên thư mục `dce_e20_gpu` nhưng config thực tế ghi 10 epoch. |
| Chọn checkpoint | Snapshot theo iteration/epoch, README dùng Epoch99 | Local chọn loss validation thấp nhất | Tiêu chí chọn khác; loss tự giám sát thấp nhất không đảm bảo mAP YOLO cao nhất. |
| Runner local | N/A | `src/luong/pipeline.py` chỉ dùng tối đa 200 ảnh mỗi epoch; `src/LBN/train_ninh.py` dùng toàn bộ train split | Hai entry point local tạo ra hai thí nghiệm rất khác nhau. Nghiên cứu phải dùng một runner duy nhất. |
| Khởi tạo YOLO hiện tại | Không có YOLO trong upstream | `YOLO("yolov8n.pt")`, pretrained=True | Đây là fine-tuning, không phải YOLO from scratch. |

## 4. Giả thuyết giải thích chênh lệch YOLO

Sắp theo mức ưu tiên kiểm chứng:

1. **Domain shift ở kịch bản cascaded.** YOLO raw học đặc trưng trên ảnh tối nhưng được test trên phân phối màu/độ sáng/texture mới. Việc retrain giúp mAP@0.5 từ 0,2092 lên 0,5229 là bằng chứng mạnh rằng domain shift là một phần lớn nguyên nhân.
2. **Implementation/loss local không tương đương bản gốc.** Exposure, color và đặc biệt TV loss có thang đo khác; curve maps có thể kém mượt, tăng nhiễu hoặc tạo color cast dù tổng loss giảm.
3. **DCE local chưa hội tụ.** Run chỉ có 10 epoch trong khi code chính thức đặt 200 epoch; dùng khởi tạo khác làm khoảng cách này đáng kể hơn.
4. **Loss DCE không task-aware.** Loss tối ưu độ phơi sáng, màu và độ mượt, không tối ưu khả năng giữ biên/texture mà YOLO cần. Ảnh nhìn sáng hơn không đồng nghĩa mAP cao hơn.
5. **Khác augmentation YOLO.** Raw dùng `hsv_v=0.4`, retrained DCE dùng `hsv_v=0.1`. Vì hai yếu tố thay đổi cùng lúc, chưa thể quy toàn bộ chênh lệch cho ảnh DCE.
6. **Artifact/dataset chưa được audit.** Enhanced dataset và DCE checkpoint của run cũ vắng mặt; chưa xác nhận cùng image IDs, nhãn, kích thước, encoding, RGB/BGR và không có ảnh lỗi.
7. **Epoch YOLO quá ít.** Kết quả ninh dùng 10 epoch. Các run raw 40 epoch trong repo đạt validation mAP@0.5 khoảng 0,624-0,633, cho thấy 10 epoch chưa phải ngân sách tốt để kết luận cuối.
8. **Phiên bản môi trường.** Run ninh dùng Ultralytics 8.4.172, Torch 2.5.1 CUDA 12.1; các notebook/run khác không phải lúc nào cũng có environment artifact. Khác phiên bản có thể đổi augmentation, remapping và mặc định trainer.

## 5. Thiết kế thí nghiệm tối thiểu

### 5.1. Khóa protocol

- Dataset: đúng 5.142 train, 1.469 validation, 734 test; giữ nguyên image IDs và nhãn.
- Chọn model bằng validation; chỉ chạy test sau khi khóa cấu hình.
- Ảnh enhanced phải giữ nguyên kích thước, lưu lossless PNG, cùng tên stem.
- Ghi SHA-256 của checkpoint và manifest SHA-256 của danh sách ảnh/nhãn.
- YOLOv8n, `imgsz=640`, batch cố định theo GPU, 40 epoch tối thiểu, cùng optimizer/augmentation.
- Seed chính: 42. Cấu hình cuối chạy thêm seed 0 và 1143 để báo mean ± std.
- Mọi run lưu `args.yaml`, `results.csv`, `environment.json`, checkpoint, per-class AP và prediction theo ảnh.

### 5.2. Bốn bộ ảnh bắt buộc

| ID | Bộ ảnh | Mục đích |
| --- | --- | --- |
| I0 | Raw ExDark | Baseline |
| I1 | Official Zero-DCE + official `Epoch99.pth` | Đo pipeline upstream-pretrained |
| I2 | Official-equivalent Zero-DCE train từ đầu trên ExDark | Tách ảnh hưởng của dữ liệu/weights khỏi khác biệt implementation |
| I3 | Local Zero-DCE train từ đầu trên ExDark | Đo đúng phương pháp hiện tại |

Không dùng checkpoint Zero-DCE++ để đại diện cho Zero-DCE trong so sánh chính, vì đó là kiến trúc khác. Zero-DCE++ có thể là nhánh mở rộng I4.

### 5.3. Ma trận YOLO

Thực hiện hai lớp thí nghiệm:

**A. Detector cố định — đo riêng tác động tiền xử lý**

- Train một YOLO trên I0.
- Giữ nguyên checkpoint đó và evaluate trên I0, I1, I2, I3.
- Kết quả trả lời: chỉ đổi ảnh đầu vào có giúp hay gây hại?

**B. Detector thích nghi — đo khả năng retrain**

- Train bốn YOLO độc lập trên I0, I1, I2, I3.
- Mỗi model evaluate trên test thuộc đúng miền ảnh của nó.
- Giữ mọi hyperparameter giống nhau, kể cả `hsv_v`. Sau đó mới làm ablation `hsv_v={0.1, 0.4}` riêng.
- Kết quả trả lời: khi detector được thích nghi công bằng, pipeline nào tốt nhất?

Tổng run tối thiểu cho seed 42: 1 YOLO raw + 3 phép cascaded + 3 YOLO retrained = 7 phép train/evaluate chính. Official-equivalent DCE và local DCE phải được train trước đó.

### 5.4. Ablation để tìm nguyên nhân

Chỉ chạy sau ma trận tối thiểu:

1. **Loss parity:** lần lượt đổi local `L_exp`, `L_color`, `L_TV` về công thức official; mỗi lần chỉ đổi một yếu tố.
2. **Initialization parity:** PyTorch default so với normal(0, 0,02).
3. **Train budget:** DCE 10, 50, 100, 200 epoch; chọn theo validation DCE nhưng báo mAP YOLO ở checkpoint đã định trước.
4. **Exposure target:** 0,5; 0,6; 0,7, chọn bằng validation.
5. **YOLO augmentation:** `hsv_v=0.1` và 0,4 trong một factorial nhỏ, không thay cùng lúc với yếu tố khác.
6. **YOLO initialization:** chỉ khi cần trả lời riêng câu hỏi detector from scratch, so `yolov8n.pt` với `yolov8n.yaml` trên cùng I0 và phương án DCE tốt nhất.

## 6. Đo lường nguyên nhân thay vì chỉ đo mAP

Ngoài Precision, Recall, mAP@0.5 và mAP@0.5:0.95, cần lưu:

- AP theo 12 lớp và confusion matrix.
- Chênh lệch true positive/false positive/false negative theo từng ảnh giữa I0 và I1-I3.
- Thống kê luminance trung bình, percentile 1/50/99, tỷ lệ pixel clipping gần 0 và 1.
- Color shift theo trung bình RGB/Lab; noise proxy trong vùng tối; edge energy quanh bounding box.
- Ảnh minh họa được chọn bằng quy tắc cố định: top 20 tăng AP proxy, top 20 giảm, và 20 ảnh ngẫu nhiên.
- NIQE/BRISQUE chỉ là chỉ số bổ trợ. Không dùng PSNR/SSIM trên ExDark thật vì không có ảnh sáng ground truth tương ứng.

Phân tích nguyên nhân nên dùng paired bootstrap trên 734 ảnh test để tạo khoảng tin cậy cho chênh lệch AP/mAP, thay vì chỉ nhìn một số trung bình.

## 7. Trình tự thực hiện

### Giai đoạn 0 — Audit và đóng băng môi trường (0,5 ngày)

- Pin phiên bản Python, Torch, Ultralytics, OpenCV và commit của hai repo.
- Kiểm tra class mapping, split, ảnh/nhãn và manifest.
- Thu hồi hoặc train lại checkpoint local-scratch vì checkpoint run cũ đang thiếu.

Tiêu chí hoàn tất: một manifest chung và environment file có thể truy ngược mọi artifact.

### Giai đoạn 1 — Reproduce hai Zero-DCE (1-2 ngày GPU)

- Chạy official pretrained trên ExDark để tạo I1.
- Port công thức official sang runner hiện đại nhưng giữ đúng forward/loss/init, train trên ExDark để tạo I2.
- Train implementation local với cùng seed, epoch và data order để tạo I3.
- Kiểm tra 100% ảnh giữ nguyên geometry, không NaN, không thiếu label.

Tiêu chí hoàn tất: I1-I3 có metadata và checkpoint hash; có bảng loss và thống kê pixel.

### Giai đoạn 2 — YOLO controlled experiment (1-2 ngày GPU)

- Chạy ma trận A và B với seed 42.
- Không thay `hsv_v`, epoch, batch, image size hay pretrained flag giữa các nhánh chính.
- Khóa cấu hình bằng validation rồi chạy test đúng một lần cho bảng chính.

Tiêu chí hoàn tất: mỗi số trong bảng trỏ được tới run folder, checkpoint và raw log.

### Giai đoạn 3 — Ablation nguyên nhân (1-3 ngày GPU)

- Ưu tiên TV normalization, exposure loss, train budget và HSV augmentation.
- Chỉ chạy nhiều seed cho baseline và phương án tốt nhất/quan trọng nhất.

Tiêu chí hoàn tất: mỗi kết luận nguyên nhân có một cặp đối chứng chỉ khác đúng một yếu tố.

### Giai đoạn 4 — Báo cáo (0,5-1 ngày)

- Báo riêng kết quả cascaded và retrained.
- Tách quan sát, giả thuyết và kết luận đã kiểm chứng.
- Giữ cả kết quả âm; không chọn ảnh minh họa theo cảm tính.

## 8. Quy tắc diễn giải kết quả

- Nếu I1 tốt hơn I3 khi dùng cùng YOLO cố định: ưu tiên kiểm tra implementation/loss/checkpoint local.
- Nếu I2 gần I1 nhưng I3 kém: khác biệt implementation là nguyên nhân chính.
- Nếu I2 và I3 đều kém I1: dữ liệu hoặc ngân sách train scratch là nguyên nhân chính.
- Nếu cascaded kém nhưng retrained phục hồi: domain shift là nguyên nhân chính.
- Nếu retrained vẫn kém raw dù image-quality score tốt hơn: enhancement không bảo toàn đặc trưng phục vụ detection, hoặc loss không task-aware.
- Không kết luận từ khác split, khác seed, khác epoch hay khác augmentation.

## 9. Việc cần sửa trước khi chạy chính thức

1. Dùng `src/LBN/train_ninh.py` làm runner nền hoặc hợp nhất runner; không dùng nhánh giới hạn 200 ảnh/epoch trong `src/luong/pipeline.py`.
2. Thêm chế độ `official_exact` cho model/loss/init và test parity với official `Epoch99.pth` trên một ảnh cố định.
3. Tách cấu hình `dce_variant`, `weights_source`, `curve_formula`, `loss_formula`, `seed`, `epochs` vào metadata.
4. Bắt buộc cùng hyperparameter YOLO trong bảng chính; chuyển `hsv_v` thành ablation riêng.
5. Thêm audit image-ID/label/hash và kiểm tra RGB/BGR, pixel range, clipping.
6. Không dùng các số suy ra/fallback trong `src/luong/pipeline.py`; chỉ nhận metrics do `model.val()` sinh ra.

## 10. Nguồn sơ cấp

- Repository và README chính thức: <https://github.com/Li-Chongyi/Zero-DCE>
- Model chính thức: <https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/model.py>
- Train script chính thức: <https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/lowlight_train.py>
- Loss chính thức: <https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py>
- Trang dự án/paper và mô tả dữ liệu: <https://li-chongyi.github.io/Proj_Zero-DCE.html>

