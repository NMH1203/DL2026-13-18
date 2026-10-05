# THƯ MỤC GIAO DIỆN DEMO (PROJECT 18)

Thư mục này gom toàn bộ các tệp tin liên quan đến Giao diện Web Demo Tương tác vào một nơi gọn gàng:

```text
Demo/
├── app.py              # Server Flask backend, nạp trọng số và cung cấp API suy luận
├── templates/
│   └── index.html      # Giao diện web UI 3 Tab (Demo trực quan, Giám định trọng số, Đồ thị 40 epochs)
└── static/
    ├── css/
    │   └── style.css   # Giao diện Dark Theme hiện đại, glassmorphism
    └── js/
        └── app.js      # Xử lý logic tương tác, API, vẽ đồ thị Chart.js
```

---

## Cách Khởi Động Giao Diện Demo

Từ thư mục gốc dự án:
```bash
python run_demo.py
# hoặc
python Demo/app.py
```

Sau đó mở trình duyệt web truy cập:
👉 **http://127.0.0.1:5000**
