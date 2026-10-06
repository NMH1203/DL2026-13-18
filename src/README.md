# Quy ước cộng tác trong `src`

Mỗi thành viên tạo một Python package riêng bằng tên không dấu, viết thường, ví
dụ `src/luong/`, `src/hoang/`. Mỗi package phải có file `__init__.py` và nên tự
chứa module, pipeline hoặc thử nghiệm mà người đó phụ trách.

Quy ước này giảm khả năng nhiều người cùng sửa một file, nhưng Git vẫn có thể
xung đột nếu hai người sửa `run.py`, tài liệu hoặc file dùng chung. Logic tái sử
dụng nên nằm trong package của từng thành viên. `run.py` hiện là entry point tích
hợp của dự án và vẫn chứa một phần logic điều phối; khi refactor, chuyển logic
nghiệp vụ vào package trước rồi mới làm mỏng entry point.

Chạy pipeline hiện tại từ thư mục gốc:

```powershell
python run.py --phase 1
```

Cũng có thể chạy trực tiếp package của Lương:

```powershell
python -m src.luong.pipeline --phase 1
```

Các module dùng chung của pipeline hiện tại nằm trong `src/luong/`. Không tạo
bản sao cùng tên trực tiếp dưới `src/`; import theo dạng
`from src.luong.<module> import ...` để tránh hai nguồn triển khai khác nhau.
