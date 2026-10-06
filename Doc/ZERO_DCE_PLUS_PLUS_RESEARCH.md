# Nghiên cứu Zero-DCE++ và tác động lên YOLOv8

Ngày nghiên cứu: 06/10/2026  
Phạm vi: Zero-DCE++, Zero-DCE, các implementation hiện có trong project và pipeline YOLOv8 trên ExDark.

## 1. Kết luận chính

Zero-DCE++ không chỉ là Zero-DCE thay bằng convolution nhẹ hơn. Bản chính thức thay đổi đồng thời ba thành phần:

1. Thay convolution chuẩn bằng depthwise-separable convolution.
2. Giảm output từ 24 curve parameter maps xuống 3 maps RGB, sau đó dùng chung ba maps này cho cả tám lần lặp.
3. Cho phép ước lượng curve trên ảnh downsample rồi phóng curve về kích thước gốc trước khi áp dụng lên pixel full-resolution.

Theo paper, các thay đổi này giảm từ 79.416 xuống 10.561 tham số và từ khoảng 84,99G xuống 0,12G FLOPs trên ảnh 1200×900. Đổi lại, việc dùng chung curve và ước lượng curve ở độ phân giải thấp làm giảm khả năng điều chỉnh khác nhau theo từng vòng lặp và từng chi tiết không gian nhỏ. Đây là trade-off cần đo riêng trên các đối tượng nhỏ của ExDark, không thể suy ra chỉ từ chất lượng ảnh nhìn bằng mắt.

Trong project hiện tại:

- `src/hienanh/enhancement.py` là implementation phù hợp với kiến trúc và tên tham số của checkpoint Zero-DCE++ chính thức.
- `src/luong/model_zerodce.py::ZeroDCEpp` **không phải bản tái hiện chính xác Zero-DCE++ chính thức**: nó xuất 24 maps thay vì 3, dùng một map khác cho mỗi vòng lặp, không có scale factor và có tên layer khác checkpoint.
- Checkpoint chính thức đã có tại `Results/hienanh/weights/zerodcepp_Epoch99.pth`, kích thước 52.395 byte, SHA-256 `ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84`.
- Notebook đã tạo đủ ảnh Zero-DCE++ trên Google Drive, nhưng dataset enhanced không nằm trong checkout hiện tại và chưa có run/result YOLO Zero-DCE++ được lưu trong `Results/`. Vì vậy hiện chưa thể tuyên bố mAP YOLO của Zero-DCE++ trong repo này.

## 2. Zero-DCE++ khác Zero-DCE như thế nào?

| Thành phần | Zero-DCE | Zero-DCE++ chính thức | Ý nghĩa đối với detection |
| --- | --- | --- | --- |
| Convolution | Conv 3×3 chuẩn | Depthwise 3×3 + pointwise 1×1 | Giảm mạnh FLOPs/params nhưng giảm mức trộn thông tin không gian-kênh trong từng layer. |
| Output curve | 24 maps = 8 vòng × 3 RGB | 3 maps RGB dùng chung cho 8 vòng | Ít tự do hơn; curve mượt và nhất quán hơn nhưng có thể kém thích nghi với vùng sáng/tối phức tạp. |
| Curve iteration | Mỗi vòng có `A_n` riêng | Mọi vòng dùng cùng `A` | Vẫn là curve bậc cao nhờ lặp, nhưng không còn điều chỉnh riêng theo stage. |
| Độ phân giải ước lượng | Full-resolution | Có thể downsample; paper mặc định factor 12 | Nhanh hơn rất nhiều; curve map có tần số không gian thấp hơn, có thể ảnh hưởng chi tiết nhỏ. |
| Áp dụng curve | Trên ảnh gốc | Curve được upscale rồi áp dụng trên ảnh gốc | Geometry có thể giữ nguyên nếu implementation không crop. |
| Tham số | 79.416 | 10.561 | Giảm khoảng 7,5 lần. |
| FLOPs công bố | 84,99G | 0,12G | Lợi thế lớn cho pipeline thời gian thực/edge. |
| Runtime GPU công bố | 0,0025 giây | 0,0012 giây | Chỉ là số trên thiết bị và protocol của paper; phải benchmark lại trên máy mục tiêu. |
| Train script PyTorch | 200 epoch, input 256 | 100 epoch, input 512 | Không nên so checkpoint scratch nếu ngân sách và input size khác nhau. |
| TV coefficient trong code release | 200 | 1600 | Bù cho việc giảm từ 24 xuống 3 curve maps; dùng hệ số Zero-DCE cho Zero-DCE++ sẽ làm đổi độ mượt curve. |

Paper ghi Zero-DCE và Zero-DCE++ dùng cùng dữ liệu/cấu hình huấn luyện: 2.422 ảnh train và 600 ảnh validation từ 3.022 ảnh multi-exposure của SICE Part1, resize 512×512, batch 8 và Adam learning rate `1e-4`.

## 3. Paper và code release không hoàn toàn đồng nhất

Cần phân biệt hai chuẩn tham chiếu:

- **Checkpoint-compatible:** bám sát code PyTorch và checkpoint `Epoch99.pth` được tác giả phát hành.
- **Paper-equivalent:** bám sát công thức, loss weights và initialization được mô tả trong paper.

Các bất nhất cần ghi vào metadata:

| Thành phần | Paper | PyTorch release Zero-DCE++ |
| --- | --- | --- |
| Exposure loss | Công thức L1/absolute | `Myloss.py` dùng squared error/MSE. |
| Loss weights | Paper ghi `W_col=0,5`, `W_tv=20` | Script dùng color ×5, exposure ×10, TV ×1600. |
| Initialization | Gaussian mean 0, std 0,02 | Dòng `DCE_net.apply(weights_init)` bị comment, nên dùng PyTorch default. |
| Downsample factor | Mặc định 12 | Train CLI mặc định 1; test script đặt 12. |
| Geometry test | Paper mô tả output cùng kích thước | Test script crop chiều cao/rộng về bội số của 12 trước inference. |

Do đó, nghiên cứu phải đặt tên rõ:

- `zpp_official_ckpt`: code/checkpoint release, dùng inference tương thích.
- `zpp_paper_scratch`: tái hiện công thức paper từ đầu.
- Không gọi cả hai là cùng một model nếu chưa xác nhận output parity.

## 4. Audit implementation trong project

### 4.1. Bản `src/hienanh/enhancement.py`

Đây là implementation nên dùng với checkpoint chính thức vì:

- Layer có tên `e_conv*`, `depth_conv`, `point_conv` giống checkpoint.
- Layer cuối xuất 3 channels.
- Một curve RGB được dùng lại tám lần.
- Có `scale_factor` và resize curve về đúng `image.shape[-2:]`.
- Dùng đúng dấu forward của code chính thức: `x + curve * (x² - x)`.
- Nạp checkpoint bằng `strict=True`.
- Giữ nguyên kích thước ảnh thay vì crop về bội số của scale factor.

Khác biệt có chủ đích so với test script upstream là bảo toàn geometry. Đây là lựa chọn đúng cho YOLO vì crop ảnh nhưng giữ nguyên label sẽ làm bounding box sai. Tuy nhiên nó phải được ghi là một adaptation, không phải bit-exact upstream inference.

### 4.2. Bản `src/luong/model_zerodce.py::ZeroDCEpp`

Bản này là một custom lightweight DCE, không phải official Zero-DCE++:

- Layer cuối xuất 24 channels thay vì 3.
- Tám lần lặp dùng tám curve RGB khác nhau.
- Không có downsample/upsample curve map.
- Có 11.926 tham số thay vì 10.561.
- Dùng tên `d_conv*`, `depthwise`, `pointwise`, không tương thích state-dict chính thức.
- Dùng công thức `x + A*x*(1-x)`, đảo dấu tham số so với checkpoint release.
- Clamp output trong forward.

Vì vậy:

- Không được nạp checkpoint `zerodcepp_Epoch99.pth` vào class này.
- Không được dùng kết quả của class này để tuyên bố hiệu năng Zero-DCE++ chính thức.
- Nếu vẫn nghiên cứu, đổi tên thí nghiệm thành `custom_dcepp_24map`.

### 4.3. Loss local

`src/luong/loss_zerodce.py` dùng chung loss cho Zero-DCE và custom ZeroDCEpp:

- Exposure dùng L1, trong khi code release dùng MSE.
- Color loss khác công thức trong code release.
- TV chia thêm số channel; code release cộng trên channel.
- TV weight mặc định là 200, trong khi release Zero-DCE++ dùng 1600.

Theo cách chuẩn hóa hiện tại, độ mạnh TV hiệu dụng của local custom model nhỏ hơn khoảng 48 lần so với released PyTorch Zero-DCE++ nếu so trên trung bình gradient mỗi curve channel. Đây là ứng viên gây curve map kém mượt, artifacts hoặc noise amplification.

### 4.4. Notebook và pipeline hiện có

`Notebooks/NMH/exdark_yolo_zerodcepp.ipynb`:

- Tải đúng checkpoint và kiểm tra đúng SHA-256.
- Dùng implementation 3-map shared-curve tương thích official.
- Dùng `scale_factor=4`, không phải mặc định 12 trong paper.
- Đã xử lý 5.142 train, 1.469 valid và 734 test, lưu PNG và copy labels.
- Ghi metadata ra Google Drive.

`Notebooks/NMH/Yolo_Fine-tunning.ipynb` có khai báo dataset `zerodcepp`, nhưng trạng thái notebook đang lưu `DATASET_VARIANT="clahe"`. Các result được commit chỉ có raw và CLAHE; chưa có `yolov8n_zerodcepp_*`.

`src/hienanh/prepare_dataset.py` có workflow audit tốt nhưng chưa chạy trực tiếp được từ bố cục hiện tại nếu không sửa:

- Import đang dùng `from src.enhancement`, trong khi module thật là `src/hienanh/enhancement.py`.
- Checkpoint mặc định trỏ `Results/weights`, trong khi file hiện ở `Results/hienanh/weights`.

## 5. Vì sao YOLO trên Zero-DCE++ có thể khác Zero-DCE?

### H1. Shared curve làm mất tính linh hoạt theo stage

Zero-DCE có 24 maps nên mỗi vòng lặp có thể thay đổi hành vi. Zero-DCE++ dùng cùng 3 maps tám lần. Điều này có thể làm ánh sáng nhất quán hơn nhưng kém linh hoạt trong cảnh vừa có bóng tối sâu vừa có nguồn sáng mạnh.

Kiểm chứng: cùng checkpoint/dataset YOLO, so Zero-DCE và Zero-DCE++ ở scale 1; đo clipping, luminance theo vùng và AP theo ảnh.

### H2. Downsample curve map làm mượt chi tiết nhỏ

Scale factor 4 hoặc 12 làm curve map thay đổi chậm hơn theo không gian. Điều này có thể giảm noise, nhưng cũng có thể không tăng đủ contrast quanh xe, chai, mèo hoặc vật thể nhỏ.

Kiểm chứng: dùng cùng checkpoint Zero-DCE++, chạy `scale_factor={1,4,12}`; đo AP theo kích thước bounding box và edge energy quanh box.

### H3. Domain shift từ SICE sang ExDark

Checkpoint chính thức học từ multi-exposure SICE, còn ExDark chứa cảnh đêm thực, noise cảm biến và vật thể cụ thể. Model có thể tạo ảnh đẹp theo tiêu chí SICE nhưng thay đổi phân phối texture/màu không phù hợp backbone YOLO.

Kiểm chứng: so official checkpoint với Zero-DCE++ train-from-scratch chỉ trên train split ExDark, giữ mọi yếu tố khác cố định.

### H4. Cascaded detector không thích nghi

YOLO train trên raw ExDark có thể giảm mạnh khi nhận ảnh enhanced. Đây là domain shift ở detector, không nhất thiết là enhancement xấu.

Kiểm chứng: tách rõ `fixed-detector/cascaded` và `domain-matched/retrained`.

### H5. Hyperparameter YOLO không đồng nhất

Nếu raw dùng `hsv_v=0,4` còn enhanced dùng `hsv_v=0,1`, hai run khác cả dữ liệu lẫn augmentation. Không thể quy chênh lệch cho Zero-DCE++.

Kiểm chứng: bảng chính giữ cùng hyperparameter; `hsv_v` là ablation riêng.

### H6. Chất lượng thị giác không đồng nghĩa chất lượng detection

Paper chứng minh Zero-DCE++ hỗ trợ DSFD trên DARK FACE, nhưng đó là face detector, dataset và protocol khác. Paper không cung cấp YOLOv8/ExDark result. Kết quả face detection không thể chuyển trực tiếp thành kỳ vọng mAP của 12 lớp ExDark.

## 6. Thiết kế nghiên cứu đề xuất

### 6.1. Câu hỏi nghiên cứu

1. Zero-DCE++ official có giúp YOLOv8n trên ExDark so với raw và Zero-DCE không?
2. Scale factor nào cân bằng tốt nhất giữa mAP và latency?
3. Official pretrained và ExDark-scratch khác nhau do domain hay do implementation?
4. Custom 24-map local có lợi hơn shared 3-map chính thức không?

### 6.2. Các bộ ảnh cần tạo

| ID | Enhancement | Weights | Scale | Vai trò |
| --- | --- | --- | ---: | --- |
| I0 | Không | N/A | N/A | Raw baseline |
| I1 | Zero-DCE | Official | 1 | Baseline enhancement đầy đủ |
| I2 | Zero-DCE++ | Official | 1 | Tách ảnh hưởng shared curve/DW conv, chưa downsample |
| I3 | Zero-DCE++ | Official | 4 | Cấu hình notebook hiện tại |
| I4 | Zero-DCE++ | Official | 12 | Cấu hình mặc định paper/test upstream |
| I5 | Zero-DCE++ official architecture | Scratch trên ExDark train | Scale được chọn bằng validation | Đo domain adaptation |
| I6 | Custom DCE++ 24-map local | Scratch trên ExDark train | 1 | Ablation kiến trúc local; không gọi là official |

I2-I4 dùng cùng một checkpoint; scale factor không nằm trong weights. Chọn scale bằng validation trước khi đánh giá test.

### 6.3. Hai lớp thí nghiệm YOLO

**A. Fixed detector**

- Train một YOLOv8n duy nhất trên I0.
- Evaluate cùng checkpoint trên test I0-I6.
- Đo tác động thuần của enhancement và domain shift.

**B. Domain-matched detector**

- Train YOLOv8n riêng trên I0, I1, phương án Zero-DCE++ official tốt nhất, I5 và I6.
- Mọi run dùng cùng `yolov8n.pt`, epoch, batch, seed, augmentation và model-selection rule.
- Evaluate trên test cùng miền tương ứng.

Không cần train YOLO riêng cho cả scale 1/4/12 trước khi sàng lọc. Dùng fixed detector trên validation để chọn một scale; sau đó mới retrain detector.

### 6.4. Cross-domain matrix

Với ba miền chính Raw, Zero-DCE và Zero-DCE++, nên đánh giá ma trận 3×3:

| Train YOLO trên | Test raw | Test Zero-DCE | Test Zero-DCE++ |
| --- | ---: | ---: | ---: |
| Raw | ✓ | ✓ | ✓ |
| Zero-DCE | ✓ | ✓ | ✓ |
| Zero-DCE++ | ✓ | ✓ | ✓ |

Ma trận này chỉ cần ba detector nhưng chỉ ra trực tiếp model nào robust với domain shift và enhancement nào làm phân phối lệch nhiều nhất.

## 7. Protocol công bằng

- Split cố định: 5.142 train, 1.469 validation, 734 test.
- Chỉ train enhancement trên train split; validation để chọn checkpoint/scale; không dùng test để tuning.
- Giữ nguyên image stem, label, kích thước và class mapping.
- Lưu ảnh enhanced bằng PNG lossless.
- YOLO: `yolov8n.pt`, `imgsz=640`, 40 epoch tối thiểu, batch 16 nếu GPU cho phép, seed 42, cùng `hsv_v`, `close_mosaic`, patience và Ultralytics version.
- Cấu hình cuối chạy thêm seed 0 và 1143; báo mean ± std.
- Ghi checkpoint SHA-256, source manifest, environment và args cho mọi run.
- Dùng validation để chọn best checkpoint; test chỉ chạy sau khi khóa cấu hình.

## 8. Chỉ số cần báo cáo

### Detection

- Precision, Recall, mAP@0.5 và mAP@0.5:0.95.
- AP theo 12 lớp.
- AP theo kích thước box nhỏ/vừa/lớn hoặc ít nhất chia tercile theo diện tích box.
- False negative và false positive theo từng ảnh.
- Paired bootstrap confidence interval trên 734 ảnh test.

### Enhancement

- Mean luminance và percentile 1/50/99.
- Tỷ lệ pixel gần 0 và gần 1 để phát hiện crushing/clipping.
- Color shift trong RGB/Lab.
- Edge energy trong và quanh bounding box.
- Noise proxy ở vùng tối.
- NIQE/BRISQUE chuẩn chỉ dùng bổ trợ; không dùng PSNR/SSIM trên ExDark thật vì không có normal-light ground truth cùng cảnh.

### Hiệu năng

- Latency enhancement, YOLO và toàn pipeline tách riêng.
- Cùng thiết bị, batch, image size, warmup và số vòng lặp.
- Đồng bộ CUDA trước/sau timing.
- Peak VRAM, kích thước checkpoint và throughput.

## 9. Ablation ưu tiên

1. `scale_factor={1,4,12}` với cùng official checkpoint.
2. Shared 3-map so với distinct 24-map, giữ depthwise-separable backbone.
3. Official pretrained so với ExDark-scratch.
4. Released-code loss so với paper-equivalent loss.
5. TV coefficient/normalization parity.
6. `hsv_v={0,1; 0,4}` sau khi đã chọn enhancement.
7. Nếu mục tiêu là deployment: benchmark CPU/GPU thật, không dùng FPS paper như số của project.

## 10. Trình tự thực hiện

### Giai đoạn 0 — Sửa khả năng tái lập

- Sửa import/checkpoint path trong `src/hienanh/prepare_dataset.py`.
- Thêm tên variant rõ ràng cho official 3-map và custom 24-map.
- Ghi environment, git commit, checkpoint hash và dataset manifest.

Đầu ra: lệnh build dataset chạy được từ project root.

### Giai đoạn 1 — Scale-factor screening

- Tạo I2, I3, I4 từ cùng checkpoint.
- Validate bằng cùng raw-trained YOLO và đo image statistics/latency.
- Chọn scale trước khi mở test.

Đầu ra: một scale factor đã khóa bằng validation.

### Giai đoạn 2 — Main YOLO experiment

- Chạy fixed-detector và domain-matched detector.
- So raw, Zero-DCE và Zero-DCE++ đã chọn.
- Chạy cross-domain matrix 3×3.

Đầu ra: bảng chính trả lời tác động enhancement và adaptation.

### Giai đoạn 3 — Scratch và architecture ablation

- Train official Zero-DCE++ architecture trên ExDark train.
- Train custom 24-map nếu cần bảo vệ đóng góp local.
- So với official checkpoint trong cùng protocol.

Đầu ra: kết luận domain-vs-architecture có đối chứng.

### Giai đoạn 4 — Multi-seed và báo cáo

- Chạy thêm hai seed cho raw và phương án tốt nhất.
- Báo confidence interval, per-class AP, failure cases và latency.

Đầu ra: kết luận có độ biến động và artifact truy xuất được.

## 11. Tiêu chí kết luận

- Zero-DCE++ chỉ được xem là tốt hơn raw nếu cải thiện mAP trên cùng test/protocol và chênh lệch ổn định qua seed hoặc confidence interval.
- Nếu cascaded kém nhưng retrained tốt, kết luận chính là domain shift của detector.
- Nếu scale 12 nhanh nhưng AP-small giảm, chọn scale 4 hoặc 1 tùy mục tiêu accuracy/latency.
- Nếu official checkpoint tốt hơn ExDark-scratch, không tự động kết luận SICE tốt hơn; trước tiên kiểm tra loss parity, epoch, initialization và checkpoint selection.
- Nếu custom 24-map tốt hơn official 3-map, gọi đó là đóng góp custom; không đổi tên thành Zero-DCE++ chính thức.
- Kết quả DARK FACE/DSFD trong paper chỉ là bằng chứng phương pháp có thể hỗ trợ downstream detection, không phải baseline YOLOv8/ExDark.

## 12. Nguồn sơ cấp

- Repository Zero-DCE++ chính thức: <https://github.com/Li-Chongyi/Zero-DCE_extension>
- Model release: <https://github.com/Li-Chongyi/Zero-DCE_extension/blob/main/Zero-DCE%2B%2B/model.py>
- Train script release: <https://github.com/Li-Chongyi/Zero-DCE_extension/blob/main/Zero-DCE%2B%2B/lowlight_train.py>
- Loss release: <https://github.com/Li-Chongyi/Zero-DCE_extension/blob/main/Zero-DCE%2B%2B/Myloss.py>
- Paper TPAMI/arXiv: <https://arxiv.org/abs/2103.00860>
- Trang dự án: <https://li-chongyi.github.io/Proj_Zero-DCE++.html>

