# Tổng hợp thành quả Project 18 và nghiên cứu của NMH

Ngày rà soát: 04/10/2026. Nhánh: `NMH`, sau khi tích hợp cập nhật từ `main`.

Bản tổng hợp dựa trên mã nguồn, toàn bộ notebook hiện có, cấu hình, log CSV/JSON, kiểm tra toàn bộ ảnh/nhãn và 13 báo cáo nghiên cứu do NMH cung cấp. Không đọc nội dung bất kỳ file README nào. Không chạy lại huấn luyện hoặc sửa mã nguồn. Kết quả đọc trong báo cáo được phân biệt với kết quả có log thực tế trong repository.

## 1. Project đã làm được gì?

Project đã có nền tảng thực nghiệm cho bài toán phát hiện 12 loại đối tượng trong ảnh thiếu sáng: chuẩn bị dữ liệu ExDark, tăng sáng bằng phương pháp truyền thống hoặc mạng neural, huấn luyện YOLOv8n và đánh giá độ chính xác. Có kết quả đo thật cho ảnh tối gốc, Zero-DCE nối trước detector, và detector huấn luyện lại trên ảnh Zero-DCE. Chưa đủ bằng chứng để gọi bản hiện tại là hệ thống hoàn chỉnh đã triển khai thời gian thực trên thiết bị biên.

### Dữ liệu

Đã đọc và giải mã toàn bộ ảnh, đọc toàn bộ nhãn bằng hàm audit trong `src/hienanh/prepare_dataset.py`:

| Thành phần | Kết quả kiểm tra tại máy |
| --- | ---: |
| Tổng ảnh | 7.345 |
| Train | 5.142 |
| Validation (`valid`) | 1.469 |
| Test | 734 |
| Kích thước ảnh | Tất cả 640 × 640 |
| Nhãn ban đầu | 23.146 dòng, mỗi dòng 9 trường theo dạng OBB |
| Box hợp lệ sau chuyển đổi | 23.143 |
| Box diện tích bằng 0 cần loại bỏ | 3 |
| Ảnh không còn box hợp lệ sau chuyển đổi | 3 |
| Trùng giữa các split theo sample ID, original ID và pixel hash | Không phát hiện |

12 lớp theo đúng ID trong YAML: Bicycle, Boat, Bottle, Bus, Cat, Cup, Motorbike, People, Table, car, chair, dog. Bộ dữ liệu local này có 7.345 ảnh; không lấy con số mô tả dataset gốc trong tài liệu để thay cho số local đã kiểm tra.

Nhãn hiện có là 9 trường, vì vậy cần chuyển OBB thành box detection 5 trường trước khi dùng quy trình YOLO detection thông thường. Hai helper chuẩn bị dữ liệu đã có chức năng này. Việc audit ở trên chỉ đọc dữ liệu, chưa ghi bộ dữ liệu đã chuyển đổi.

### Tăng sáng và mô hình

- `src/luong/preprocess_dip.py`: CLAHE trên kênh L của LAB, sau đó Bilateral Filter. Có xử lý cả ảnh đơn và thư mục ảnh. Kiểm tra thực thi một ảnh thành công, giữ kích thước 640 × 640 và kiểu `uint8`.
- `src/luong/model_zerodce.py`: DCENet, biến thể dùng depthwise convolution và phép biến đổi đường cong ánh sáng lặp 8 lần.
- `src/luong/loss_zerodce.py`: bốn thành phần loss tự giám sát: spatial consistency, exposure, color constancy và smoothness/TV.
- Bản Zero-DCE tự xây có ràng buộc làm mượt: `L_TV_A` phạt biến thiên của bản đồ đường cong A, còn `L_spa` giữ quan hệ tương phản không gian. Đây là cơ chế kiểm soát độ ổn định khi tăng sáng. Trong code hiện được rà soát, TV áp dụng lên A, không trực tiếp lên nhiễu ảnh đầu ra; chưa thấy nhánh denoising hoặc loss noise riêng. Vì vậy cần đánh giá hiệu quả giảm nhiễu thực tế trước khi gọi model là bộ khử nhiễu. Bilateral Filter được gọi trong phương án CLAHE riêng. Nếu nhóm có bản Zero-DCE bổ sung xử lý nhiễu ngoài các file hiện có, cần đưa code/checkpoint/config đó vào đối chiếu.
- `src/hienanh/enhancement.py`: một implementation riêng cho Zero-DCE/Zero-DCE++ tương thích tên tham số, thứ tự skip connection và dấu đường cong của checkpoint tham chiếu. Có strict loading, kiểm tra pixel hữu hạn và giữ kích thước gốc.
- Hai implementation ở `luong` và `hienanh` không thể coi là thay thế trực tiếp cho nhau khi nạp checkpoint. Bản Zero-DCE++ của `luong` dự đoán 24 kênh; bản tương thích checkpoint tham chiếu ở `hienanh` dùng 3 kênh chia sẻ qua các vòng lặp.

### Huấn luyện, đánh giá và trực quan

- `run.py` điều phối các phase qua `src/luong/pipeline.py`: setup, kiểm tra dữ liệu, tăng sáng, YOLO, tổng hợp biểu đồ và demo ảnh đơn.
- `src/LBN/train_ninh.py` có các lệnh `prepare`, `dce`, `enhance`, `train`, `evaluate`. Helper tách train/val/test, lưu seed và môi trường, chọn checkpoint DCE theo validation loss, xuất kết quả test thành JSON/CSV.
- `src/hienanh/prepare_dataset.py` có audit ảnh/nhãn, hash nguồn, kiểm tra leakage, giữ hình học, xây các dataset dark/CLAHE/neural, journal để tiếp tục build và kiểm tra output.
- `src/luong/visualize.py` có vẽ box, ảnh đối chiếu và biểu đồ mAP. `src/hienanh/plot_dataset_statistics.py` có biểu đồ số box theo lớp và phân bố độ sáng.
- `Results/ninh/` có training curves, confusion matrices, ảnh dự đoán và log đo thực tế. Log DCE `dce_e20_gpu` ghi 10 epoch; validation loss giảm từ khoảng 1,5861 xuống 1,0657. Tên thư mục không phải bằng chứng đã chạy 20 epoch.
- `dark_e10` và `zerodce_e10` có 10 epoch. `dark_e40_v2` mới có log đến epoch 11, chưa có bằng chứng hoàn tất 40 epoch. `dce_e20/loss.csv` chỉ có header; các smoke run chỉ là kiểm tra quy trình.

## 2. Kết quả thực nghiệm có bằng chứng trong repository

Nguồn: `Results/ninh/test_raw/measured_metrics.json`, `test_dce_cascaded/measured_metrics.json`, `test_dce_retrained/measured_metrics.json`. Cả ba khai báo `split=test`.

| Kịch bản | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| --- | ---: | ---: | ---: | ---: |
| YOLO huấn luyện ảnh tối, test ảnh tối | 66,27% | 54,64% | **59,80%** | **27,39%** |
| Model ảnh tối, test ảnh Zero-DCE | 40,62% | 23,15% | 20,92% | 8,83% |
| YOLO huấn luyện lại trên ảnh Zero-DCE, test ảnh Zero-DCE | 62,94% | 47,42% | 52,29% | 23,63% |

Trong nhóm run này, baseline ảnh tối đạt kết quả tốt nhất. Nối Zero-DCE trước detector ảnh tối làm giảm mạnh hiệu năng; huấn luyện lại detector trên ảnh tăng sáng khôi phục một phần nhưng chưa vượt baseline. Đây là kết quả của các run cụ thể, không phải kết luận mọi kỹ thuật tăng sáng đều bất lợi.

Các đường dẫn trong log trỏ đến máy `C:\18.deeplearning\...`; bản local hiện tại không có dataset `ninh_dark`/`ninh_zerodce` hay checkpoint của các run đó để tái lập ngay. Vì vậy xác nhận được nội dung log, chưa tái đánh giá độc lập model.

## 3. Thành quả nghiên cứu NMH bổ sung cho nhóm

Nguồn: 13 ghi chú `00`–`12` trong thư mục Obsidian `Low-Light Object Detection Report` do NMH cung cấp.

- Đã hệ thống hóa vấn đề thiếu sáng: nhiễu, SNR thấp, tương phản, màu và ảnh hưởng đến đặc trưng nhận diện; tổng hợp các họ phương pháp enhancement và detection.
- Đã xây khung thiết kế hai hướng: enhancement rồi detection và huấn luyện liên kết; phân tích loss, dữ liệu, augmentation và đánh đổi độ chính xác/tốc độ.
- Đã ghi nhận baseline YOLO ở 640 và 1024, phân tích theo lớp và thời gian xử lý.
- Đã nghiên cứu ba cách khai thác biên: RGB+Edge 4 kênh, nhánh tần số/biên và Edge-Aware Loss. Báo cáo có code mẫu OpenCV/PyTorch cho trích biên và decomposition.
- Đã ghi nhiều cấu hình EdgeLoss: trọng số 0,05/0,10, batch, vị trí hook và số epoch; có phân tích Recall/mAP/timing.
- Đã tập hợp bảng tiền xử lý và master benchmark, đồng thời đề xuất TensorRT, quantization và các hướng mở rộng.

Các kết quả nổi bật được ghi trong nghiên cứu:

| Thử nghiệm | mAP@0.5 | mAP@0.5:0.95 | Trạng thái bằng chứng |
| --- | ---: | ---: | --- |
| Raw baseline 640, 30 epoch | 63,0% | 29,5% | Báo cáo 07; đánh giá 734 ảnh, chưa thấy raw run tương ứng trong repo |
| Raw baseline 1024, 30 epoch | 58,2% | 25,3% | Báo cáo 08; đánh giá 734 ảnh, chưa thấy raw run tương ứng trong repo |
| EdgeLoss `train-5`, λ=0,05 | 63,2% | 29,7% | Báo cáo 10 và output notebook tương ứng; đánh giá 1.469 ảnh |
| Các EdgeLoss batch 16 trong bảng raw của báo cáo 10 | 62,6% | 29,5% | Theo báo cáo; thiếu các raw run/checkpoint để đối chiếu riêng |
| Zero-DCE cascaded trong báo cáo 11 | 22,91% | 9,70% | Theo báo cáo; khác run test đã lưu trong repo |
| Retrained trong báo cáo 11 | 59,22% | 27,37% | Theo báo cáo; khác run test đã lưu trong repo |
| CLAHE trong báo cáo 11 | 67,47% | 32,04% | Chưa xác nhận là phép đo detector độc lập |

Đóng góp có thể đưa vào báo cáo nhóm ngay là nền tảng nghiên cứu, thiết kế thí nghiệm, baseline theo từng cấu hình và kết quả EdgeLoss đã ghi nhận. Notebook `model_EdgeawareLoss.ipynb` có log train 30 epoch và kết quả cuối 0,632/0,297 trên validation; đây là bằng chứng run đã chạy. Tuy nhiên chưa chứng minh riêng rằng phần EdgeLoss đã tham gia đúng vào tối ưu gradient.

## 4. Những điểm cần chỉnh trước khi chốt báo cáo

1. **Không so trực tiếp khác split.** Báo cáo baseline 07/08 dùng 734 ảnh (bằng số ảnh test local), EdgeLoss dùng 1.469 ảnh validation. Chênh lệch +0,2 điểm phần trăm mAP hoặc các mức tăng theo lớp chưa đủ để quy cho EdgeLoss. Cần đánh giá cả hai checkpoint trên cùng split và cùng thiết lập.
2. **Điểm CLAHE có dấu hiệu là số suy ra.** Pipeline đang gán `map50 = baseline × 1,05 + 0,02` và `map50_95 = baseline × 1,05 + 0,015`, đồng thời sao chép Precision/Recall. Với baseline trong báo cáo 11: `0,6235 × 1,05 + 0,02 = 0,674675`, làm tròn đúng `0,6747`; chỉ số còn lại cũng ra `0,3204`. Chưa nên dùng bảng này để kết luận CLAHE đã đạt cao nhất nếu không có log `model.val()` riêng.
3. **Bảng notebook NML 03 là số viết sẵn.** Bốn giá trị mAP/FPS trong `scenarios_summary` không được đo trong notebook. Pipeline còn có fallback cộng điểm khi thiếu dataset Zero-DCE; phải loại các dòng đó khỏi bảng kết quả thực nghiệm.
4. **NIQE/BRISQUE hiện tại là công thức xấp xỉ tự xây dựng.** Module `metrics.py` dùng vài thống kê MSCN, hằng số tham chiếu và clip score; chưa có mô hình/tham số chuẩn để gọi là phép đo NIQE/BRISQUE tiêu chuẩn.
5. **Benchmark EdgeLoss có bất nhất nội bộ.** Báo cáo 10 ghi các run batch 16 đều 0,679/0,593/0,626/0,295, nhưng bảng master 12 ghi L4 và L5 khác. Báo cáo 08 ghi P/R của baseline 1024 là 0,618/0,552, bảng 12 lại là 0,669/0,549. Cần lấy raw log từng run làm nguồn thống nhất.
6. **Tốc độ khác nhau chưa chứng minh loss làm model nhanh hơn.** Notebook `train-5` có tổng timing 5,2 ms, trong ghi chú 09/00 xuất hiện 4,8 ms; đó là các nguồn khác nhau. FPS tính nghịch đảo thời gian trung bình validation không tự tương đương FPS video thực tế trên Jetson. Cần cùng thiết bị, batch, warmup và quy trình đo; đo GPU cần xử lý đồng bộ khi timing thủ công.
7. **SOTA và edge deployment chưa được đối chiếu bằng artifact của nhóm.** Các số IA-YOLO 73,5%, SCI 70,2%, TensorRT/Jetson/Hailo và ablation trong báo cáo 05/06 chưa có code, run hoặc file export tương ứng trong repo. Nên xếp là thông tin trong nghiên cứu cần nguồn/kiểm chứng, chưa tuyên bố nhóm đã triển khai.
8. **Phân biệt quan sát và giải thích nguyên nhân.** Run 1024 có điểm thấp hơn 640 là quan sát được ghi trong báo cáo. Giải thích do khuếch đại nhiễu/SNR là giả thuyết cần thí nghiệm bổ sung. Tương tự, điểm Zero-DCE thấp không đủ để quy toàn bộ cho nhiễu khi chưa kiểm tra checkpoint, domain shift và cấu hình.
9. **“Retrained” cần ghi đúng đối tượng.** Trong helper LBN và pipeline, scenario retrained huấn luyện lại YOLO trên ảnh tăng sáng. Báo cáo 11 mô tả thành huấn luyện lại mạng Zero-DCE; cần sửa mô tả hoặc chỉ rõ run khác nếu có.

## 5. Phần nào dùng được ở bản hiện tại?

| Thành phần | Mức sử dụng hiện tại |
| --- | --- |
| Dataset gốc | Đọc được đầy đủ; cần chuyển nhãn 9 trường sang 5 trường cho detection |
| Audit và chuyển nhãn ở HienAnh | Hàm audit đã thực thi trên toàn bộ dataset, có thể tái dùng |
| CLAHE + Bilateral | Hàm ảnh đơn đã kiểm tra chạy thành công; chưa có mAP đo riêng |
| Kiến trúc/loss Zero-DCE | Có mã nguồn và log train; thiếu torch/checkpoint trên runtime hiện tại |
| Helper LBN | Có quy trình train/evaluate thực tế, nhưng cần sửa đường dẫn sau khi chuyển file |
| Notebook EdgeLoss | Có run/output thật; cần kiểm chứng hook và thời điểm cộng loss/backward |
| Notebook raw_model | Mới setup model và môi trường; output hiện dừng ở FileNotFoundError dữ liệu |
| Notebook NML 01–03 | Khung khảo sát/visualization; chưa có output thực thi và còn lỗi đường dẫn/import |
| Demo | Có ảnh output và code demo ảnh đơn; chưa có checkpoint để xác nhận demo detector ExDark đã train |
| Nghiên cứu Obsidian | Dùng được cho cơ sở lý thuyết, thiết kế và phân tích; cần sửa bảng số liệu trước khi dùng làm benchmark cuối |

Các trở ngại thực tế khi chạy:

- Python mặc định tại máy có OpenCV/NumPy/YAML nhưng chưa có `torch` và `ultralytics`. Đã parse cú pháp thành công 18 file Python; đây không thay thế kiểm tra runtime của mô hình.
- Không thấy checkpoint đã train trong `Results/` hoặc `src/`; file `yolov8n.pt` ở root là mô hình nền, không phải bằng chứng có detector 12 lớp đã train. `.gitignore` loại các file `.pt`/`.pth`, nên cần chia sẻ checkpoint riêng để tái lập.
- `src/LBN/train_ninh.py` đang lấy ROOT bằng thư mục chứa file; sau khi chuyển vào `src/LBN`, các đường dẫn tương đối dataset/results và import `src.luong` cần được chỉnh. Rà soát chỉ đọc, không sửa việc di chuyển này.
- `src/hienanh/prepare_dataset.py` import `src.enhancement`, trong khi module nằm ở `src/hienanh/enhancement.py`; test cũng dùng import và ROOT của bố cục cũ. Script download/plot có đường dẫn phụ thuộc thư mục của file. Các phần này cần thống nhất trước khi build/test toàn bộ.
- Notebook NML vẫn dùng đường dẫn `../Dataset`/`sys.path.append('..')` dù đã nằm sâu hơn trong `Notebooks/NML`. Notebook 02 thiếu import `Path` và đang so sánh bằng model khởi tạo chưa load checkpoint.
- Hook EdgeLoss lưu feature và gọi backward trong `on_train_batch_end`. Cần kiểm tra hook có gắn lên model trainer thực sự dùng, tensor batch có tồn tại và loss được cộng trước backward/optimizer step; log train thành công một mình chưa xác nhận điều đó.

## 6. Đoạn tóm tắt có thể gửi cho nhóm

Nhóm đã xây dựng nền tảng phát hiện đối tượng thiếu sáng trên ExDark gồm 7.345 ảnh, 12 lớp, các bước kiểm tra/chuyển nhãn, CLAHE + Bilateral, Zero-DCE và huấn luyện/đánh giá YOLOv8n. Có log thực nghiệm cho ảnh tối gốc, ảnh tăng sáng và detector huấn luyện lại trên ảnh tăng sáng. Trong ba run test đã lưu, baseline ảnh tối đạt mAP@0.5 59,80%, cao hơn cascaded Zero-DCE 20,92% và retrained 52,29%.

NMH bổ sung bộ 13 báo cáo nghiên cứu về nguyên nhân suy giảm ảnh thiếu sáng, các hướng thuật toán, thiết kế pipeline, baseline 640/1024 và thử nghiệm Edge-Aware Loss. Notebook EdgeLoss có kết quả validation mAP@0.5 63,2%, mAP@0.5:0.95 29,7%. Đóng góp này cung cấp cơ sở nghiên cứu và thiết kế thí nghiệm cho nhóm. Để chốt kết luận định lượng, cần thống nhất split/seed/cấu hình, đo thật CLAHE, kiểm chứng loss tham gia gradient, bổ sung checkpoint và sửa đường dẫn. Joint learning, các mô hình SOTA khác và triển khai thiết bị biên hiện là hướng nghiên cứu chưa được xác nhận bằng artifact trong repository.

## 7. Quy trình chỉnh sửa và hoàn thiện theo mục tiêu đề tài

Phần này là kế hoạch công việc đề xuất, chưa phải các thay đổi đã triển khai. Hướng chính được giữ là Zero-DCE kết hợp YOLOv8n trên ExDark để tận dụng dữ liệu, mã nguồn và kết quả hiện có. Mục tiêu là trả lời hai câu hỏi: (1) tăng cường bằng deep learning cải thiện chất lượng ảnh như thế nào; (2) sự thay đổi đó có giúp phát hiện đối tượng tốt hơn hay không. Task-aware enhancement và triển khai thiết bị biên là các hướng mở rộng sau khi thí nghiệm chính hoàn chỉnh.

### Bước 1. Khóa phạm vi và quy tắc thí nghiệm

Thống nhất ExDark, YOLOv8n, kích thước đầu vào 640 và một bộ train/validation/test chung. Ghi rõ class mapping, seed, số epoch, augmentation, cách chọn checkpoint và thiết lập đánh giá. Chọn cấu hình bằng validation; dùng test để đánh giá cuối sau khi khóa cấu hình. Các kết quả test đã xem được coi là kết quả thăm dò, không tiếp tục điều chỉnh để tối đa hóa điểm trên tập này.

**Đầu ra:** cấu hình chung và bản mô tả protocol. Hoàn tất khi mọi thành viên sử dụng cùng split và hiểu rõ tiêu chí chọn model.

### Bước 2. Sửa pipeline để chạy xuyên suốt và báo kết quả thật

| Vị trí | Nội dung chỉnh sửa |
| --- | --- |
| `src/LBN/train_ninh.py` | Xác định đúng root project sau khi chuyển file, sửa đường dẫn tương đối và cách chạy module |
| `src/hienanh/prepare_dataset.py`, `test_preparation.py` | Sửa import và đường dẫn theo bố cục hiện tại |
| Notebook NML | Sửa đường dẫn dữ liệu, import còn thiếu và nạp đúng checkpoint trước khi so sánh |
| `src/luong/pipeline.py` | Bỏ số mAP tự suy ra và fallback cộng điểm; thiếu dữ liệu/checkpoint phải báo lỗi hoặc đánh dấu chưa đánh giá |
| `src/luong/metrics.py` | Thay NIQE/BRISQUE xấp xỉ bằng implementation chuẩn; nếu giữ công thức cũ thì đổi tên và ghi rõ chỉ số thử nghiệm |

Chọn đúng implementation Zero-DCE tương ứng với checkpoint, kiểm tra RGB/BGR, miền giá trị pixel và output. Chạy thử một nhóm ảnh qua toàn bộ chuỗi `ảnh → tăng sáng → YOLO → lưu dự đoán` bằng model đã huấn luyện.

**Đầu ra:** một lệnh chạy được pipeline và xuất kết quả đo thật; không có số liệu tự cộng hoặc model khởi tạo ngẫu nhiên được trình bày như model đã train.

### Bước 3. Chuẩn hóa dữ liệu và thu hồi checkpoint

Xuất nhãn detection 5 trường từ OBB 9 trường, loại 3 box diện tích bằng 0 và ghi lại thao tác trong audit. Giữ ảnh không còn box hợp lệ làm ảnh nền nếu phù hợp với protocol. Giữ nguyên split, tên/ID và kích thước ảnh; kiểm tra pairing sau chuyển đổi.

Thu lại checkpoint Zero-DCE, YOLO ảnh tối và YOLO ảnh tăng sáng từ máy hoặc môi trường đã huấn luyện. Lưu kèm config, log, phiên bản thư viện và hash checkpoint để nhận diện đúng artifact. Chia sẻ checkpoint qua vị trí thống nhất của nhóm nếu không đưa vào Git.

**Đầu ra:** dataset detection đã chuẩn hóa và bộ checkpoint có nguồn gốc rõ ràng. Hoàn tất khi checkpoint nạp thành công và model/dataset có class mapping tương ứng.

### Bước 4. Tạo các bộ ảnh đối chiếu

Tạo bốn phương án từ cùng nguồn: Raw, Gamma, CLAHE + Bilateral và Zero-DCE. Giữ nguyên hình học để dùng cùng nhãn; lưu output enhanced bằng PNG. Ghi tham số, implementation và hash checkpoint trong metadata. Chọn Gamma và các tham số xử lý bằng validation rồi khóa trước khi đánh giá test.

**Đầu ra:** các bộ ảnh có cùng ID, split và nhãn; kiểm tra đủ ảnh, kích thước và pixel hợp lệ. Raw được giữ làm mốc đối chiếu.

### Bước 5. Đo tác động enhancement bằng cùng một detector

Giữ nguyên checkpoint YOLOv8n đã huấn luyện trên ảnh tối, đánh giá trên các phiên bản của cùng tập test:

| Thí nghiệm | Ảnh đầu vào | Detector |
| --- | --- | --- |
| Baseline | Raw | YOLO ảnh tối cố định |
| Gamma cascaded | Gamma | Cùng checkpoint YOLO ảnh tối |
| CLAHE cascaded | CLAHE + Bilateral | Cùng checkpoint YOLO ảnh tối |
| Zero-DCE cascaded | Zero-DCE | Cùng checkpoint YOLO ảnh tối |

Giữ cùng thiết lập đánh giá và đo Precision, Recall, mAP@0.5, mAP@0.5:0.95, kết quả theo lớp. Lưu dự đoán từng ảnh, confusion matrix và raw log. Đo thời gian trên cùng thiết bị, cùng batch và quy trình warmup; phân biệt latency riêng từng module với toàn pipeline.

**Đầu ra:** bảng đo thật trả lời câu hỏi “chỉ thay ảnh đầu vào có cải thiện nhận diện không?”. Bảng CLAHE trước đây được thay bằng kết quả đánh giá độc lập.

### Bước 6. Bổ sung đánh giá chất lượng ảnh

Đánh giá theo hai loại dữ liệu:

| Dữ liệu | Cách đánh giá và giới hạn |
| --- | --- |
| ExDark thật, không có ảnh sáng tương ứng | NIQE/BRISQUE chuẩn làm chỉ số bổ trợ; ảnh đối chiếu để phân tích nhiễu, màu, clipping và mất chi tiết |
| Ảnh sáng có nhãn được làm tối nhân tạo | Giữ ảnh sáng gốc để tính PSNR, SSIM, LPIPS; đánh giá detection trên cùng ảnh và cùng nhãn |

Thí nghiệm nhân tạo cần chuẩn bị nguồn ảnh sáng có nhãn, tách train/validation/test rõ ràng và giữ nguyên hình học khi làm tối. Điều chỉnh mức làm tối/nhiễu bằng train/validation rồi khóa trước test. Không coi ảnh làm tối nhân tạo là mô phỏng hoàn hảo ảnh đêm thật, và không coi ảnh ExDark vốn tối là ground truth đủ sáng.

Chọn ảnh minh họa bằng quy tắc cố định, bao gồm cả trường hợp cải thiện và suy giảm. Độ sáng trung bình chỉ mô tả độ sáng; PSNR/SSIM/LPIPS mô tả mức phục hồi so với tham chiếu, cần kết hợp quan sát để bàn về chất lượng thị giác. Nếu phân tích quan hệ giữa chất lượng và nhận diện, sử dụng dữ liệu của cùng các ảnh; không tương quan trực tiếp điểm chất lượng của dataset này với mAP của dataset khác.

**Đầu ra:** bảng chất lượng ảnh đi cùng bảng detection, ảnh đối chiếu và phân tích trường hợp enhancement giúp hoặc gây hại.

### Bước 7. Kiểm tra nguyên nhân suy giảm và tối ưu có kiểm soát

Trước khi thay đổi mô hình, kiểm tra checkpoint/implementation, RGB/BGR, pixel range, ID, nhãn và kích thước ảnh. Kiểm tra output có tăng nhiễu, cháy sáng hoặc đổi màu không. Chỉ kết luận nguyên nhân khi có bằng chứng, không quy toàn bộ suy giảm cho nhiễu dựa trên mAP.

Sau khi pipeline hợp lệ, thử số lượng nhỏ cấu hình có cơ sở, chẳng hạn target exposure khi huấn luyện Zero-DCE hoặc thêm khử nhiễu. Chọn bằng validation. Đánh giá detector huấn luyện lại trên ảnh Zero-DCE thành thí nghiệm riêng, với ngân sách train tương đương baseline, vì thí nghiệm này trả lời khả năng thích nghi của detector chứ không chỉ tác động của ảnh đầu vào. Nếu đủ tài nguyên, chạy nhiều seed cho baseline và phương án cuối để báo độ biến động.

**Đầu ra:** cấu hình cuối được chọn trước test, bảng cascaded/retrained tách rõ và phân tích lỗi có bằng chứng. Kết quả không cải thiện vẫn được giữ trong báo cáo.

### Bước 8. Hoàn thiện EdgeLoss như một phần mở rộng

Xác nhận hook gắn đúng model đang train và EdgeLoss được cộng đúng trước backward/optimizer step. Kiểm tra có gradient hữu hạn và ảnh hưởng đến tham số cần tối ưu. So baseline với EdgeLoss trên cùng split, số epoch, seed, batch và thiết lập còn lại; chỉ thay yếu tố cần khảo sát. Không lấy khác biệt validation/test làm mức tăng do EdgeLoss và không quy timing nhanh hơn cho loss khi chưa có benchmark đồng nhất.

**Đầu ra:** một đối chứng EdgeLoss hợp lệ; nếu chưa xác nhận được cơ chế hoặc thiếu artifact thì ghi trạng thái chưa hoàn tất. Ưu tiên bước này sau phần enhancement chính.

### Bước 9. Khóa kết quả, sửa báo cáo và kiểm tra tái lập

Mỗi số trong bảng và slide phải truy được về `run_id`, config, checkpoint và raw log. Bỏ các số viết sẵn hoặc suy ra. Thống nhất số liệu giữa báo cáo chi tiết và bảng tổng hợp; chuyển IA-YOLO, SCI hoặc triển khai edge chưa có artifact sang related work/future work với nguồn thích hợp.

Trình bày kết quả theo mạch: tăng sáng bằng Zero-DCE → đánh giá chất lượng ảnh → đánh giá cùng YOLO → phân tích ảnh được lợi/bị hại → đánh giá retrained hoặc EdgeLoss bổ sung. Chạy demo từ môi trường sạch và lưu phiên bản môi trường. Kết luận đúng phạm vi các run, nêu giới hạn dữ liệu nhân tạo, thiếu tham chiếu ở ảnh đêm thật và độ biến động thí nghiệm.

**Đầu ra:** bảng kết quả thống nhất, báo cáo, slide, demo và hướng dẫn chạy lại. Hoàn thành tốt là trả lời được hai mục tiêu bằng bằng chứng, không bắt buộc enhancement phải vượt baseline.

### Thứ tự ưu tiên và kế hoạch 5 ngày nếu thời gian còn ngắn

| Ngày | Công việc ưu tiên | Điều kiện hoàn tất |
| --- | --- | --- |
| 1 | Khóa protocol, sửa pipeline, chuẩn hóa nhãn và gom checkpoint | Chạy được nhóm ảnh thử từ đầu đến cuối |
| 2 | Đánh giá Raw/Gamma/CLAHE/Zero-DCE bằng cùng YOLO | Có log thật và bảng detection |
| 3 | Đánh giá chất lượng ảnh, chuẩn bị hoặc chạy phần synthetic, phân tích lỗi | Có bằng chứng chất lượng và các hạn chế ghi rõ |
| 4 | Hoàn thiện retrained/EdgeLoss hoặc nhiều seed nếu đủ tài nguyên | Có đối chứng hợp lệ, khóa kết quả |
| 5 | Đối chiếu số liệu, hoàn thiện báo cáo và chạy demo sạch | Mọi số liệu truy được về artifact |

Kế hoạch phụ thuộc việc thu đủ checkpoint, dữ liệu bổ sung và GPU. Ưu tiên bước 2–5 để có thí nghiệm detection đo thật, sau đó bước 6 để trả lời đầy đủ mối quan hệ chất lượng ảnh–nhận diện. Nếu quá tải, cắt số biến thể mở rộng và ghi rõ phần chưa thực hiện; không thay phần thiếu bằng số liệu dự kiến.
