# TÀI LIỆU CHUYÊN SÂU: THUẬT TOÁN ZERO-DCE (ZERO-REFERENCE DEEP CURVE ESTIMATION)
### Dự án: Low-Light Image Enhancement and Downstream Recognition (Đề tài 18)

---

## MỤC LỤC
1. [Bản Chất Cốt Lõi Của Thuật Toán Zero-DCE](#1-bản-chất-cốt-lõi-của-thuật-toán-zero-dce)
2. [Chi Tiết Toán Học & Kiến Trúc Mạng](#2-chi-tiết-toán-học--kiến-trúc-mạng)
   - [Đường cong ánh sáng LE-Curve](#21-đường-cong-ánh-sáng-le-curve-light-enhancement-curve)
   - [Kiến trúc mạng nơ-ron DCE-Net](#22-kiến-trúc-mạng-nơ-ron-dce-net)
   - [Hệ thống 4 hàm mất mát tự giám sát (Non-Reference Losses)](#23-hệ-thống-4-hàm-mất-mát-tự-giám-sát-non-reference-loss-functions)
3. [Top 3 Nguồn Học & Bài Báo Khoa Học Uy Tín Nhất (Kèm Link & Hướng Dẫn Học)](#3-top-3-nguồn-học--bài-báo-khoa-học-uy-tín-nhất)
   - [Nguồn 1: Bài báo gốc CVPR 2020 (Nền tảng lý thuyết)](#nguồn-1-bài-báo-gốc-cvpr-2020-zero-dce-bắt-buộc-đọc)
   - [Nguồn 2: Bài báo mở rộng IEEE TPAMI 2021 (Zero-DCE++ & Downstream Vision)](#nguồn-2-bài-báo-mở-rộng-ieee-tpami-2021-zero-dce-cực-kỳ-quan-trọng)
   - [Nguồn 3: Tutorial Thực Hành Chính Thức Của Keras / Papers With Code](#nguồn-3-tutorial-thực-hành-chuẩn-công-nghiệp-keras--papers-with-code)
4. [Đối Chiếu Trực Tiếp Với Mã Nguồn Dự Án (Thư Mục `src/`)](#4-đối-chiếu-trực-tiếp-với-mã-nguồn-dự-án-thư-mục-src)
5. [Bộ Câu Hỏi Vấn Đáp Bắt Buộc Phải Thuộc Khi Bảo Vệ Đồ Án](#5-bộ-câu-hỏi-vấn-đáp-khi-bảo-vệ-đồ-án)

---

## 1. Bản Chất Cốt Lõi Của Thuật Toán Zero-DCE

### 1.1. Vấn đề của các phương pháp cũ
Trong bài toán tăng cường ảnh chụp thiếu sáng (Low-Light Image Enhancement - LLIE):
- **Phương pháp truyền thống (DIP - Histogram Equalization, CLAHE):** Dễ làm cháy sáng các vùng vốn đã đủ sáng, khuếch đại nhiễu cảm biến ISO và gây lệch màu giả tạo.
- **Phương pháp học sâu có giám sát (RetinexNet, KinD, LLNet):** Bắt buộc phải có **bộ dữ liệu cặp đối ứng (Paired Dataset)** gồm 1 ảnh chụp tối và 1 ảnh chụp sáng ở cùng góc máy và cùng thời điểm. Tuy nhiên, trong thực tế (đặc biệt là bộ dữ liệu thực nghiệm **ExDark**), **hoàn toàn không có ảnh sáng chuẩn làm Ground Truth**.
- **Phương pháp sinh pixel trực tiếp (U-Net, Pix2Pix, CycleGAN):** Ép mạng CNN sinh trực tiếp giá trị pixel màu RGB rất dễ gây méo cấu trúc, sinh ảo ảnh (artifacts) và tốn tài nguyên tính toán cực lớn.

### 1.2. Ý tưởng đột phá của Zero-DCE (CVPR 2020)
Thay vì ép mạng CNN vẽ lại toàn bộ bức ảnh, Zero-DCE tiếp cận theo cách hoàn toàn mới:
1. **Zero-Reference (Không cần ảnh mẫu):** Huấn luyện mô hình mà không cần bất kỳ ảnh sáng chuẩn nào.
2. **Học đường cong ánh sáng (Curve Estimation):** Mạng CNN (**DCE-Net**) không sinh pixel, mà chỉ dự đoán **bản đồ hệ số điều chỉnh đường cong ánh sáng (Curve Parameter Maps $\mathcal{A}$)** cho từng pixel độc lập.
3. **Bảo toàn dải động tự nhiên:** Giá trị pixel sau biến đổi luôn được đảm bảo nằm trọn vẹn trong khoảng $[0, 1]$, không bao giờ bị tràn (clipping/overflow) hay bão hòa màu sắc.
4. **Siêu nhẹ & Thời gian thực (Real-time):** Mạng chỉ có ~79.000 tham số (79K params), tốc độ xử lý > 100 FPS trên GPU, có thể nhúng trực tiếp vào camera an ninh và xe tự hành.

---

## 2. Chi Tiết Toán Học & Kiến Trúc Mạng

### 2.1. Đường cong ánh sáng LE-Curve (Light-Enhancement Curve)
Zero-DCE thiết kế đường cong bậc hai tự thích nghi theo từng pixel, lấy cảm hứng từ phép biến đổi Gamma nhưng có khả năng thay đổi độ dốc cục bộ:

$$LE_1(I(x)) = I(x) + \mathcal{A}(x) \cdot I(x) \cdot (1 - I(x))$$

Trong đó:
- $x$ là tọa độ pixel $(u, v)$.
- $I(x) \in [0, 1]$ là giá trị pixel đầu vào chuẩn hóa.
- $\mathcal{A}(x) \in [-1, 1]$ là tham số độ cong do mạng nơ-ron dự đoán cho pixel đó:
  - Nếu $\mathcal{A}(x) > 0$: Điểm ảnh được làm sáng lên.
  - Nếu $\mathcal{A}(x) = 0$: Điểm ảnh giữ nguyên.
  - Nếu $\mathcal{A}(x) < 0$: Điểm ảnh giảm độ sáng (tránh cháy sáng).

Để tăng dải động (dynamic range) cho các vùng tối sâu mịt mù, thuật toán lặp lại công thức trên qua $n$ bước lặp (chuẩn mực $n = 8$ bước):

$$LE_n(x) = LE_{n-1}(x) + \mathcal{A}_n(x) \cdot LE_{n-1}(x) \cdot (1 - LE_{n-1}(x))$$

```
              Đặc tính toán học của đường cong LE-Curve
       1.0 ┌───────────────────────────────────────────/
           │                                      _--/
           │                                  _--/   (Vùng sáng vốn có: ít thay đổi)
  Output   │                             _--/
  LE_8(x)  │                        _--/
           │                   _--/
           │              _--/   (Vùng tối: được đẩy sáng mạnh mẽ)
           │         _--/
       0.0 └────────/──────────────────────────────────
          0.0          Input Pixel I(x)               1.0
```

**Ưu điểm toán học vượt trội:**
1. Khi $I(x) = 0 \implies LE(I) = 0$ (điểm đen tuyệt đối không bị nhiễu nhảy số).
2. Khi $I(x) = 1 \implies LE(I) = 1$ (điểm trắng tối đa không bị tràn lóa).
3. Đạo hàm liên tục bậc nhất và bậc hai, giúp quá trình lan truyền ngược (Backpropagation) cực kỳ mượt mà.

---

### 2.2. Kiến Trúc Mạng Nơ-ron DCE-Net
Kiến trúc mạng được thiết kế tối giản gồm 7 tầng tích chập đối xứng (Conv layers) có các đường kết nối tắt (Skip Connections):
- **Đầu vào:** Ảnh tối 3 kênh $H \times W \times 3$.
- **Tầng 1 đến 4 (Feature Extraction):** 4 tầng `Conv2d(32 filters, 3x3, stride=1, padding=1) + ReLU`.
- **Tầng 5 đến 7 (Reconstruction with Skip Connections):**
  - Tầng 5 nhận đầu vào là ghép nối `Concatenate(Conv4, Conv3)` $\rightarrow$ 64 channels.
  - Tầng 6 nhận đầu vào là ghép nối `Concatenate(Conv5, Conv2)` $\rightarrow$ 64 channels.
  - Tầng 7 nhận đầu vào là ghép nối `Concatenate(Conv6, Conv1)` $\rightarrow$ ra 24 channels với hàm kích hoạt `Tanh` ($\in [-1, 1]$).
- **Đầu ra:** Bản đồ kích thước $H \times W \times 24$ ứng với 8 bước lặp $\times$ 3 kênh màu RGB.

---

### 2.3. Hệ Thống 4 Hàm Mất Mát Tự Giám Sát (Non-Reference Loss Functions)
Vì không có ảnh tham chiếu chuẩn (Ground Truth), Zero-DCE được tối ưu hóa toàn bộ dựa trên 4 định luật vật lý và cảm nhận thị giác:

$$\mathcal{L}_{total} = \mathcal{L}_{spa} + \mathcal{L}_{exp} + W_{col} \mathcal{L}_{col} + W_{tv\_A} \mathcal{L}_{tv\_A}$$

*(Trọng số chuẩn tác giả đề xuất: $W_{col} = 0.5$, $W_{tv\_A} = 20.0$)*

#### 1. Spatial Consistency Loss ($\mathcal{L}_{spa}$ - Giữ gìn cấu trúc biên cạnh):
Bảo toàn sự chênh lệch mức xám giữa các vùng lân cận (4 hướng: Trên, Dưới, Trái, Phải) giữa ảnh gốc $I$ và ảnh sau tăng sáng $Y$:
$$\mathcal{L}_{spa} = \frac{1}{K} \sum_{i=1}^K \sum_{j \in \Omega(i)} \left( |(Y_i - Y_j)| - |(I_i - I_j)| \right)^2$$
*(Ý nghĩa: Ngăn chặn triệt để hiện tượng nhòe viền, mờ biên hoặc xuất hiện hào quang/halo artifacts quanh vật thể).*

#### 2. Exposure Control Loss ($\mathcal{L}_{exp}$ - Kiểm soát mức độ phơi sáng):
Đo khoảng cách giữa cường độ sáng trung bình của từng vùng cục bộ $16 \times 16$ với mức phơi sáng lý tưởng $E = 0.6$:
$$\mathcal{L}_{exp} = \frac{1}{M} \sum_{k=1}^M |Y_k - E|$$
*(Ý nghĩa: Vùng nào quá tối thì ép sáng lên, vùng nào vốn đã sáng hoặc có đèn xe thì hãm lại, không để bị cháy sáng).*

#### 3. Color Constancy Loss ($\mathcal{L}_{col}$ - Chống méo màu):
Dựa trên giả thuyết thế giới xám (Gray-World Assumption): trong một bức ảnh tự nhiên, giá trị trung bình giữa các kênh màu R, G, B có xu hướng cân bằng:
$$\mathcal{L}_{col} = \sum_{\forall (p, q) \subset \{(R,G), (R,B), (G,B)\}} (J^p - J^q)^2$$
*(Ý nghĩa: Triệt tiêu hiện tượng ngả màu, ám vàng hoặc bệt màu khi đẩy độ sáng).*

#### 4. Illumination Smoothness Loss ($\mathcal{L}_{tv\_A}$ - Làm mịn trường tham số):
Áp dụng hàm Total Variation (TV) lên bản đồ tham số $\mathcal{A}$ để đảm bảo độ sáng chuyển dịch trơn tru giữa các pixel lân cận:
$$\mathcal{L}_{tv\_A} = \frac{1}{N} \sum_{n=1}^8 \sum_{c \in \{R,G,B\}} \left( \|\nabla_x \mathcal{A}_n^c\|^2 + \|\nabla_y \mathcal{A}_n^c\|^2 \right)$$
*(Ý nghĩa: Ngăn hiện tượng đốm sáng cục bộ hoặc các vệt sáng bị phân tách đột ngột).*

---

## 3. Top 3 Nguồn Học & Bài Báo Khoa Học Uy Tín Nhất

Dưới đây là 3 tài liệu chính thống và uy tín nhất thế giới về thuật toán này:

---

### NGUỒN 1: Bài Báo Gốc CVPR 2020 (Zero-DCE) [BẮT BUỘC ĐỌC]
Đây là bài báo khoa học đầu tiên khai sinh ra phương pháp, được công bố tại hội nghị thị giác máy tính số 1 thế giới **CVPR 2020 (Oral Presentation)**.

- **Tên bài báo:** *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*
- **Tác giả:** Chongyi Li, Chunle Guo, Chen Change Loy (Nanyang Technological University & Nankai University).
- **Liên kết chính thức:**
  - 📄 **ArXiv Paper (PDF đầy đủ miễn phí):** [https://arxiv.org/abs/2001.06826](https://arxiv.org/abs/2001.06826)
  - 🌐 **Trang chủ dự án (Project Page):** [https://li-chongyi.github.io/Proj_Zero-DCE.html](https://li-chongyi.github.io/Proj_Zero-DCE.html)
  - 💻 **Kho mã nguồn gốc (Official PyTorch GitHub):** [https://github.com/Li-Chongyi/Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE)
  - 🏛️ **Bản CVF Open Access:** [CVF CVPR 2020 Record](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.html)
- **Nội dung bạn cần tập trung học trong bài này:**
  1. Đọc **Mục 3 (Methodology):** Nắm chắc 3 tính chất của phương trình LE-Curve và công thức 4 hàm loss.
  2. Đọc **Mục 4.2 (Ablation Study):** Xem tác giả chứng minh khi bỏ từng hàm loss (ví dụ bỏ $\mathcal{L}_{spa}$ hay bỏ $\mathcal{L}_{exp}$) thì bức ảnh bị lỗi méo màu và lóa sáng như thế nào.

---

### NGUỒN 2: Bài Báo Mở Rộng IEEE TPAMI 2021 (Zero-DCE++) [CỰC KỲ QUAN TRỌNG CHO ĐỀ TÀI]
Đây là phiên bản nâng cấp hoàn thiện được xuất bản trên tạp chí khoa học đỉnh cao **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)** (Tạp chí số 1 ngành AI thế giới với Impact Factor > 20).

- **Tên bài báo:** *Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation*
- **Tác giả:** Chunle Guo, Chongyi Li, Jichang Guo, Chen Change Loy, Junhui Hou, Sam Kwong, Runmin Cong.
- **Liên kết chính thức:**
  - 📄 **ArXiv Paper (PDF):** [https://arxiv.org/abs/2103.00860](https://arxiv.org/abs/2103.00860)
  - 🏛️ **IEEE Xplore Digital Library:** [https://ieeexplore.ieee.org/document/9369102](https://ieeexplore.ieee.org/document/9369102)
  - 💻 **Kho mã nguồn Zero-DCE++:** [https://github.com/Li-Chongyi/Zero-DCE_extension](https://github.com/Li-Chongyi/Zero-DCE_extension)
- **Nội dung bạn cần tập trung học trong bài này:**
  1. **Depthwise Separable Convolution:** Cách thay thế Convolution tiêu chuẩn để giảm số tham số từ 79K xuống chỉ còn **10K params**, tăng tốc độ lên hơn 1000 FPS.
  2. **Mục "Downstream Task Evaluation":** Nhóm tác giả thực hiện thí nghiệm nhận diện khuôn mặt (Face Detection) trong bóng tối. **Đây chính là cơ sở học thuật trực tiếp củng cố cho Đề tài 18 của bạn khi nghiên cứu tác vụ YOLOv8 nhận diện vật thể phía sau!**

---

### NGUỒN 3: Tutorial Thực Hành Chuẩn Công Nghiệp (Keras / Papers With Code)
Nếu muốn học cách viết code từng dòng một cách mạch lạc, dễ hiểu và trực quan hóa từng bước thì đây là tài liệu tốt nhất.

- **Tên bài hướng dẫn:** *Zero-DCE for low-light image enhancement*
- **Tác giả:** Soumik Rakshit (Keras Community Expert & Weights & Biases).
- **Liên kết chính thức:**
  - 📖 **Keras Code Example Tutorial:** [https://keras.io/examples/vision/zero_dce/](https://keras.io/examples/vision/zero_dce/)
  - 📊 **Papers With Code Benchmark Leaderboard:** [https://paperswithcode.com/paper/zero-reference-deep-curve-estimation-for-low](https://paperswithcode.com/paper/zero-reference-deep-curve-estimation-for-low)
- **Nội dung bạn cần tập trung học trong bài này:**
  1. Hướng dẫn code từng hàm Loss tùy biến bằng Tensor operations.
  2. Cách trực quan hóa ma trận tham số đường cong (Parameter Maps Visualization) để xem mạng nơ-ron "suy nghĩ" và phân bổ ánh sáng vào từng pixel như thế nào.

---

## 4. Đối Chiếu Trực Tiếp Với Mã Nguồn Dự Án (Thư Mục `src/`)

Trong dự án của bạn, thuật toán Zero-DCE đã được lập trình sẵn và lưu tại:

### 1. Kiến trúc mô hình: `src/luong/model_zerodce.py`
- Hàm `enhance_image(x, A, iterations=8)`: Thực thi phương trình LE-Curve qua 8 bước lặp theo đúng công thức toán học.
- Lớp `DCENet(nn.Module)`: Mạng nơ-ron tích chập 7 tầng với các đường nối tắt `torch.cat([x6, x1], dim=1)` để xuất ra ma trận tham số 24 channels.

### 2. Bộ 4 hàm Loss: `src/luong/loss_zerodce.py`
- Lớp `L_spa`: Sử dụng 4 ma trận tích chập (conv filters) 4 hướng để kiểm tra gradient cục bộ.
- Lớp `L_exp`: Dùng `F.avg_pool2d(kernel_size=16)` so khớp với giá trị mục tiêu $E = 0.6$.
- Lớp `L_color`: Tính sai khác trung bình giữa 3 kênh $(R - G)^2 + (R - B)^2 + (G - B)^2$.
- Lớp `L_TV_A`: Tính tổng bình phương sai phân ngang và dọc của các ma trận tham số $\mathcal{A}$.

### 3. Trọng số mô hình đã huấn luyện: `Results/weights/zerodce_best.pth`
- File trọng số PyTorch chỉ nặng **~320 KB** chứa toàn bộ các trọng số tối ưu sau khi huấn luyện trên tập ExDark.

---

## 5. Bộ Câu Hỏi Vấn Đáp Khi Bảo Vệ Đồ Án

Khi đứng trước Hội đồng bảo vệ, thầy cô thường sẽ hỏi sâu vào 5 câu hỏi trọng tâm sau:

#### Câu 1: Tại sao gọi là "Zero-Reference"?
> **Trả lời:** Vì mô hình hoàn toàn không cần bất kỳ ảnh sáng chuẩn (Ground Truth/Reference image) nào để tính hàm Loss (như L1, MSE hay SSIM thông thường). Toàn bộ quá trình học chỉ dựa vào 4 hàm loss tự giám sát được thiết kế theo các định luật vật lý về ánh sáng và độ tương phản cục bộ.

#### Câu 2: Tại sao mô hình không sinh ra ảnh trực tiếp mà lại dự đoán ma trận tham số đường cong $\mathcal{A}$?
> **Trả lời:** Nếu sinh pixel RGB trực tiếp, mạng rất dễ tạo ra các đốm nhiễu, artifacts giả tạo và làm biến dạng biên cạnh của vật thể. Khi học đường cong LE-Curve, công thức toán học đảm bảo giá trị pixel luôn bị chặn trong $[0, 1]$, giữ nguyên mối quan hệ thứ tự mức xám giữa các pixel, bảo toàn màu sắc và giúp mạng cực kỳ nhẹ (~79K tham số).

#### Câu 3: Khi ảnh có cả đèn đường rất sáng lẫn góc tối mịt mù thì Zero-DCE xử lý thế nào để không bị cháy sáng?
> **Trả lời:** Trong công thức $LE(I) = I + \mathcal{A} \cdot I \cdot (1 - I)$, khi điểm ảnh $I$ tiến sát giá trị 1.0 (vùng sáng lóa), thành phần $(1 - I)$ sẽ tiến về 0, làm cho mức độ gia tăng ánh sáng bị triệt tiêu tự nhiên. Đồng thời, hàm Loss $\mathcal{L}_{exp}$ phạt nặng các vùng có cường độ trung bình vượt quá 0.6, giúp kìm hãm vùng đèn lóa.

#### Câu 4: Tại sao trong đề tài của bạn, khi đưa ảnh Zero-DCE vào YOLOv8 chưa huấn luyện lại (Kịch bản 3) thì mAP bị sụt giảm từ 62.4% xuống 22.9%?
> **Trả lời:** Đây chính là hiện tượng **Lệch miền dữ liệu (Domain Shift)** giữa Thị giác Người và Thị giác Máy. Zero-DCE nâng độ sáng để mắt người dễ quan sát, nhưng đồng thời khuếch đại các hạt nhiễu cảm biến ISO ẩn sâu trong bóng tối thành các đốm hạt rõ rệt. Bộ trích xuất đặc trưng của YOLOv8 gốc vốn chỉ quen với kết cấu ảnh đêm mờ mịn, nên bị nhiễu đánh lừa dẫn đến bắt trượt vật thể.

#### Câu 5: Bạn đã giải quyết vấn đề Domain Shift đó như thế nào?
> **Trả lời:** Nhóm đã áp dụng chiến lược **Đồng thiết kế (Co-design & Retraining)** ở Kịch bản 4: Dùng chính tập ảnh đã được làm sáng bởi Zero-DCE để huấn luyện lại mạng YOLOv8n. Kết quả là mAP đã hồi phục ngoạn mục lên **59.22%** (+36.3%), và đặc biệt **Precision tăng vọt lên 69.17% (đạt đỉnh cao nhất toàn bộ nghiên cứu)**, giúp loại bỏ hầu như toàn bộ các cảnh báo giả.
