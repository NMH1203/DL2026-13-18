# Checklist kết quả cần có cho báo cáo Project 18

## 1. Mục tiêu của file này

File này là nguồn kiểm tra trước khi điền số liệu vào báo cáo. Chỉ đưa một kết quả vào bảng so sánh chính khi có đủ:

1. cùng danh sách ID ảnh ở tập test;
2. cùng nhãn và cùng 12 lớp;
3. cùng cấu hình đánh giá YOLO;
4. đường dẫn checkpoint và file metrics đo trực tiếp;
5. tên phương pháp tăng cường ảnh không bị nhập nhằng;
6. ghi rõ detector được giữ cố định hay fine-tune trên miền ảnh tương ứng.

Không suy ra số liệu từ ảnh biểu đồ, không dùng giá trị dự kiến, và không trộn validation với test trong cùng một bảng kết luận.

## 2. Ma trận thí nghiệm chuẩn cần hoàn thành

| ID | Ảnh đầu vào | Nguồn enhancer | Bilateral | Detector protocol | Kết quả test hiện có | Trạng thái |
|---|---|---|---|---|---|---|
| B0 | Raw dark | Không có | Không | Fine-tune raw domain | Có artifact test khớp CSV | Đã đo |
| C1 | CLAHE | Cổ điển | Không | Fine-tune matching domain | Có trong CSV test chung | Đã đo |
| C2 | CLAHE | Cổ điển | d=7, sigma=50/50 | Fine-tune matching domain | Có trong CSV test chung | Đã đo |
| C3 | Zero-DCE scratch | Checkpoint nhóm tự train | Không | Raw detector giữ nguyên | Có summary test trên NML | Đã có, thiếu JSON riêng |
| C4 | Zero-DCE++ | Official pretrained | Không | Fine-tune matching domain | Có trong CSV test chung | Đã đo |
| C5 | Zero-DCE++ | Official pretrained | d=5, sigma=25/25 | Fine-tune matching domain | Có artifact test khớp CSV | Đã đo |

Raw là baseline; năm case enhancement là C1--C5. Kết quả scratch dùng fixed raw
detector. Nếu nhóm làm `Zero-DCE scratch + Bilateral`, phải ghi thành ablation
mới và không đổi tên kết quả `Zero-DCE++ + Bilateral` hiện tại.

## 3. Số liệu đã đo và có thể truy vết

### 3.1. Block Zero-DCE lịch sử, không thuộc protocol cuối

Nguồn: `Results/ninh/test_*/measured_metrics.json`.

| Case | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Raw + raw-domain detector | 0.6627 | 0.5464 | 0.5980 | 0.2739 |
| Zero-DCE + fixed raw detector | 0.4062 | 0.2315 | 0.2092 | 0.0883 |
| Zero-DCE + matched fine-tuning | 0.6294 | 0.4742 | 0.5229 | 0.2363 |

Các chênh lệch đã kiểm tra:

- Cascaded so với raw: mAP@0.5 giảm **38.88 điểm phần trăm**; mAP@0.5:0.95 giảm **18.57 điểm**.
- Fine-tune matching domain so với cascaded: mAP@0.5 tăng **31.38 điểm**; mAP@0.5:0.95 tăng **14.81 điểm**.
- Fine-tune matching domain vẫn thấp hơn raw: mAP@0.5 thấp hơn **7.50 điểm**; mAP@0.5:0.95 thấp hơn **3.76 điểm**.

Block này chỉ dùng như bằng chứng lịch sử và không đưa vào kết luận cuối cùng của
ma trận 40 epoch. Chỉ viết “consistent with domain mismatch”, không viết rằng
domain shift đã được chứng minh là nguyên nhân duy nhất.

### 3.2. Log validation ở epoch 40

Nguồn: hàng cuối của các file `results.csv`.

| Matching-domain run | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Raw | 0.6775 | 0.5847 | 0.6229 | 0.2913 |
| CLAHE only | 0.6728 | 0.5898 | 0.6338 | 0.2976 |
| CLAHE + Bilateral | 0.6944 | 0.5814 | 0.6292 | 0.2967 |
| Zero-DCE++ + Bilateral | 0.6941 | 0.5843 | 0.6311 | 0.2950 |

Các số này là **validation**, không phải test. Có thể dùng để mô tả quá trình huấn luyện, nhưng không dùng làm bảng kết luận cuối cùng khi các case khác đang dùng test metrics.

### 3.3. Ma trận test chung 40 epoch

Nguồn: `report/data/comparisons_table_test.csv`. Dòng `Zero-DCE` trong file này
được trường `method` của bảng tích hợp xác định là pretrained Zero-DCE++.

| Case | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Raw dark | 0.6671 | 0.5883 | 0.6146 | 0.2862 |
| CLAHE | 0.6543 | 0.5935 | 0.6188 | 0.2872 |
| CLAHE + Bilateral | 0.6572 | 0.5800 | 0.6161 | 0.2856 |
| Zero-DCE scratch + fixed raw YOLO (NML summary) | 0.4131 | 0.2450 | 0.2291 | 0.0970 |
| Zero-DCE++ | 0.6771 | 0.5281 | 0.5881 | 0.2705 |
| Zero-DCE++ + Bilateral | 0.7040 | 0.5677 | 0.6244 | 0.2921 |

Chênh lệch chính so với raw:

- CLAHE: mAP@0.5 **+0.42** điểm, mAP@0.5:0.95 **+0.10** điểm.
- CLAHE + Bilateral: mAP@0.5 **+0.15** điểm, mAP@0.5:0.95 **-0.06** điểm.
- Zero-DCE++: mAP@0.5 **-2.65** điểm, mAP@0.5:0.95 **-1.57** điểm.
- Zero-DCE++ + Bilateral: Precision **+3.69** điểm, Recall **-2.05** điểm,
  mAP@0.5 **+0.98** điểm, mAP@0.5:0.95 **+0.59** điểm.

CSV chỉ có metric tổng hợp; vẫn cần lưu manifest hash, checkpoint hash,
`args.yaml`, phiên bản môi trường và lệnh đánh giá để truy vết đầy đủ.

### 3.4. Hai artifact test 40 epoch đã đối chiếu với bảng tổng hợp

| Artifact | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| `Results/qal/test_raw` | 0.6671 | 0.5883 | 0.6146 | 0.2862 |
| `Results/tiep/test_zerodcepp_bilateral` | 0.7040 | 0.5677 | 0.6244 | 0.2921 |

Hai hàng này khớp chính xác với raw và hybrid trong CSV test chung. Chênh lệch
của hybrid so với raw là:

- Precision: +3.69 điểm phần trăm.
- Recall: -2.05 điểm phần trăm.
- mAP@0.5: +0.98 điểm phần trăm.
- mAP@0.5:0.95: +0.59 điểm phần trăm.

## 4. Kết quả còn thiếu cần chạy hoặc xuất lại

### 4.1. Bắt buộc cho bảng kết quả chính

- [x] Chọn một baseline raw duy nhất cho ma trận 40 epoch.
- [x] Tổng hợp test metrics cho raw baseline trên ma trận chung.
- [x] Tổng hợp test metrics cho CLAHE-only.
- [x] Tổng hợp test metrics cho CLAHE + Bilateral.
- [x] Có summary held-out test cho Zero-DCE scratch với fixed raw detector 40 epoch.
- [ ] Xuất lại dòng scratch thành `measured_metrics.json` kèm checkpoint/data path để truy vết độc lập.
- [x] Tổng hợp test metrics cho Zero-DCE++ standalone.
- [x] Tổng hợp test metrics cho Zero-DCE++ + Bilateral.
- [ ] Lưu `measured_metrics.json`, checkpoint hash và `args.yaml` riêng cho từng hàng trong CSV chung.
- [ ] Với mỗi enhanced domain, đánh giá cả `fixed raw detector` và `matching-domain detector` nếu thời gian cho phép.
- [ ] Lưu per-class AP, confusion matrix và PR curve cho từng case chính.

### 4.2. Manifest và khả năng so sánh

- [ ] Tạo file danh sách ID cho train/valid/test và hash SHA-256 của từng manifest.
- [ ] So sánh test IDs giữa `exdark_yolo_clean`, raw, CLAHE, Zero-DCE và Zero-DCE++ variants.
- [ ] Xác nhận số lượng mỗi split là 5,142 / 1,469 / 734 cho mọi variant.
- [ ] Xác nhận nhãn được copy nguyên vẹn sau khi chỉ đổi pixel.
- [ ] Lưu phiên bản dataset/Roboflow export hoặc ngày tải.
- [ ] Giải thích vì sao bản chính thức có 7,363 ảnh nhưng bản local dùng 7,345 ảnh.
- [ ] Tạo `DATA.md` theo yêu cầu đề thi: official URL, version, split, preprocessing và script tái tạo dữ liệu.

### 4.3. Cấu hình phải ghi cho từng run

- [ ] Đường dẫn checkpoint enhancer và SHA-256 của checkpoint.
- [ ] Enhancer là trained from scratch hay official pretrained.
- [ ] CLAHE: color space, clip limit, tile grid.
- [ ] Bilateral: diameter, sigmaColor, sigmaSpace và thứ tự áp dụng.
- [ ] YOLO: weights ban đầu, epoch, batch, image size, patience, optimizer, seed, deterministic, `hsv_v`, `close_mosaic`.
- [ ] Phiên bản Python, PyTorch, CUDA, OpenCV và Ultralytics.
- [ ] GPU/CPU và thời gian huấn luyện/suy luận nếu báo cáo hiệu năng tính toán.

## 5. Chất lượng ảnh cần đo thế nào

ExDark không có ảnh sáng ground truth ghép cặp, do đó không dùng PSNR/SSIM như bằng chứng chính. Nên có:

- một implementation chuẩn, có version, của NIQE hoặc BRISQUE nếu nhóm quyết định dùng no-reference IQA;
- thống kê độ sáng trung bình, tương phản hoặc histogram như mô tả bổ trợ, không gọi là “perceptual quality” nếu chưa có đánh giá phù hợp;
- thời gian suy luận enhancer trên cùng phần cứng;
- 6–10 ảnh định tính có cùng ID giữa raw và mọi variant;
- mô tả over-exposure, color cast, halo, noise amplification và over-smoothing.

Không dùng các hàm heuristic trong `src/luong/metrics.py` như NIQE/BRISQUE chuẩn. Không dùng giá trị CLAHE 0.6747 từng được tạo bằng công thức giả lập trong pipeline.

## 6. Error and Qualitative Analysis bắt buộc

Chọn ảnh theo quy tắc, không chỉ chọn ảnh đẹp:

1. 2 case cải thiện detection rõ nhất;
2. 2 case giảm detection rõ nhất;
3. 1–2 case false positive do noise/halo/reflection;
4. 1–2 case missed detection ở vật thể nhỏ hoặc che khuất;
5. ít nhất một case có precision tăng nhưng recall giảm.

Mỗi case phải có:

- image ID;
- ground-truth boxes;
- raw prediction;
- enhanced prediction;
- cùng confidence threshold và IoU setting;
- giải thích dựa trên chi tiết nhìn thấy trong ảnh;
- không khẳng định nguyên nhân nếu chỉ dựa vào một ví dụ.

## 7. Phân tích thống kê nên có nếu còn thời gian

- Chạy ít nhất 3 seed cho các case chính, hoặc ghi rõ hạn chế single-seed.
- Báo cáo mean ± standard deviation cho mAP nếu có nhiều seed.
- Nếu không thể train lại, dùng bootstrap trên test images cho chênh lệch metric, nhưng phải ghi rõ cách bootstrap.
- Không dùng từ “significant” nếu chưa có kiểm định hoặc khoảng tin cậy phù hợp.

## 8. Hình và bảng cần đưa vào report

- [ ] Figure pipeline: Input → Enhancer → YOLOv8n → Metrics.
- [ ] Table dataset: official/local counts, 12 classes, split.
- [ ] Table method parameters: từng enhancer và bilateral settings.
- [ ] Table main test results: cùng test split và cùng protocol.
- [ ] Table cascade vs matched-domain.
- [ ] Figure training curves hoặc PR curves của các case chính.
- [ ] Figure qualitative success/failure với image IDs.
- [ ] Per-class AP hoặc normalized confusion matrix.
- [ ] Table member contributions trong Appendix.

## 9. Những câu kết luận được phép viết ở trạng thái hiện tại

- Hai log validation detector 40 epoch trên nhánh tích hợp dùng seed và
  augmentation không khớp protocol cuối, nên không được dùng để kết luận.
- Kết quả fixed-detector cũ chỉ được nêu như bằng chứng lịch sử, không dùng
  để kết luận định lượng cho protocol 40 epoch.
- Các validation log 40 epoch cho thấy enhancement có thể tạo các thay đổi nhỏ và không đồng nhất giữa Precision, Recall và mAP.
- Trong ma trận test chung, Zero-DCE++ + Bilateral đạt Precision và hai mAP cao
  nhất, nhưng Recall thấp hơn raw; mức tăng AP so với raw là nhỏ.
- Chất lượng nhìn tốt hơn không đồng nghĩa chắc chắn với downstream detection tốt hơn.

## 10. Những câu chưa được phép viết

- “Zero-DCE++ + Bilateral tốt nhất trên mọi phương diện”; Recall của phương
  pháp này không cao nhất và chưa có đánh giá đa seed.
- “Enhancement cải thiện detection” như một kết luận chung cho mọi metric và mọi protocol.
- “Domain shift là nguyên nhân duy nhất” của kết quả cascaded.
- “NIQE/BRISQUE được cải thiện” nếu số liệu đến từ heuristic proxy.
- “Mô hình YOLO được train from scratch”; các run hiện tại khởi tạo từ `yolov8n.pt`.
- “Kết quả có ý nghĩa thống kê” khi chưa có nhiều seed, bootstrap hoặc kiểm định.

## 11. Yêu cầu hình thức của đề thi cần kiểm tra lần cuối

- Report PDF 10–15 trang, không tính References và Appendix.
- Abstract 150–200 từ.
- Introduction and Research Question 0.5–1 trang.
- Related Work 0.5–1 trang.
- Dataset section có official dataset URL và version.
- Methods có Baseline, Main Method và Comparison Strategy.
- Experimental Setup có Setup 1, Setup 2 và Setup 3/robustness/ablation.
- Results phải có diễn giải, không chỉ liệt kê số.
- Error and Qualitative Analysis phải có failure cases và phân tích lý do.
- Conclusion and Limitations khoảng 0.5 trang.
- Khoảng 5–10 tài liệu tham khảo liên quan.
- Appendix có Member Contribution Table.
- Repository có README tái tạo kết quả và `DATA.md` đầy đủ.
