# Quy ước cộng tác trong `src`

Mỗi thành viên tạo một Python package riêng bằng tên không dấu, viết thường, ví
dụ `src/luong/`, `src/hoang/`. Mỗi package phải có file `__init__.py` và nên tự
chứa module, pipeline hoặc thử nghiệm mà người đó phụ trách.

Quy ước này giảm khả năng nhiều người cùng sửa một file, nhưng Git vẫn có thể
xung đột nếu hai người sửa `run.py`, tài liệu hoặc file dùng chung. Vì vậy
`run.py` ở thư mục gốc được giữ thật mỏng và ổn định; logic chính nằm trong
package của từng thành viên.

Chạy pipeline hiện tại từ thư mục gốc:

```powershell
python run.py --phase 0,1
```

Cũng có thể chạy trực tiếp package của Lương:

```powershell
python -m src.luong.pipeline --phase 0,1
```
