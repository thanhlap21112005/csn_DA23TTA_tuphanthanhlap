# Dự án FaceNet – Nhận diện gương mặt (Python 3.11)

Dự án này minh họa cách xây dựng hệ thống **nhận diện gương mặt** sử dụng:
- `facenet-pytorch` (MTCNN + InceptionResnetV1 – FaceNet),
- Trích xuất **embedding 512 chiều** cho từng khuôn mặt,
- Huấn luyện bộ phân loại (SVM) trên embedding,
- Nhận diện realtime bằng webcam.

## 1. Yêu cầu môi trường

- Python **3.11**
- Hệ điều hành: Windows / Linux / macOS
- Đã cài driver + CUDA (nếu muốn dùng GPU, không bắt buộc)

### Tạo môi trường ảo (khuyến nghị)

Trên Windows:

```bash
py -3.11 -m venv venv311
venv311\Scripts\activate
```

Linux / macOS:

```bash
python3.11 -m venv venv311
source venv311/bin/activate
```

Cài thư viện:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Cấu trúc thư mục

```text
facenet_project/
├─ dataset/                 # dữ liệu gốc: mỗi thư mục là 1 người
│  ├─ alice/
│  │  ├─ 1.jpg
│  │  └─ 2.jpg
│  └─ bob/
│     └─ ...
├─ cache/                   # (tùy chọn) lưu ảnh đã crop / align
├─ embeddings/              # lưu file embeddings + nhãn
├─ models/
│  └─ classifier.joblib     # model SVM sau khi train
├─ scripts/
│  ├─ extract_embeddings.py # trích xuất embedding từ ảnh trong dataset/
│  ├─ train_classifier.py   # huấn luyện SVM trên embedding
│  └─ recognize.py          # chạy nhận diện realtime từ webcam
├─ README.md
└─ requirements.txt
```

> Bạn cần **tự tạo thư mục `dataset/`** và thêm ảnh của từng người vào trước khi chạy.

## 3. Chuẩn bị dữ liệu

- Tạo thư mục `dataset/` cùng cấp với `scripts/`.
- Trong `dataset/`, mỗi người 1 thư mục con, tên thư mục chính là **nhãn** (label) để nhận diện.

Ví dụ:

```text
dataset/
├─ khau_van_nhut/
│  ├─ 1.jpg
│  ├─ 2.jpg
│  └─ 3.jpg
├─ nguyen_van_a/
│  ├─ a1.jpg
│  └─ a2.jpg
└─ ...
```

Yêu cầu:
- Ảnh rõ mặt, chính diện càng tốt.
- Mỗi người nên có **≥ 5 ảnh**.

## 4. Các bước chạy chính

### Bước 1 – Trích xuất embeddings

Chạy:

```bash
python scripts/extract_embeddings.py
```

Script sẽ:
- Duyệt toàn bộ `dataset/`,
- Dùng MTCNN để detect & crop mặt,
- Cho qua model InceptionResnetV1 (`pretrained='vggface2'`) để lấy embedding 512 chiều,
- Lưu:
  - `embeddings/embeddings.npy`
  - `embeddings/labels.npy`

### Bước 2 – Huấn luyện bộ phân loại (SVM)

Chạy:

```bash
python scripts/train_classifier.py
```

Script sẽ:
- Load `embeddings.npy` + `labels.npy`,
- Mã hoá nhãn bằng `LabelEncoder`,
- Chia train / test (80/20),
- Huấn luyện **SVM kernel linear**,
- In ra `classification_report` (precision, recall, f1-score),
- Lưu model vào: `models/classifier.joblib`.

### Bước 3 – Nhận diện realtime bằng webcam

Chạy:

```bash
python scripts/recognize.py
```

Script sẽ:
- Mở webcam (`VideoCapture(0)`),
- Dùng MTCNN phát hiện tất cả khuôn mặt (`keep_all=True`),
- Với mỗi mặt:
  - Cắt / chuẩn hoá,
  - Đưa qua FaceNet để lấy embedding,
  - Cho vào SVM để dự đoán tên + xác suất,
  - Vẽ bounding box + tên + độ tự tin trên frame.

Nhấn phím `q` để thoát.

## 5. Một số cải tiến gợi ý cho đồ án

- **Thêm ngưỡng nhận diện "unknown"**: nếu xác suất cao nhất < 0.5 → gán là `unknown`.
- **Augmentation dữ liệu**: xoay, dịch, thay đổi sáng/tối để tăng robust.
- **Lưu log kết quả**: mỗi lần nhận diện, lưu tên + thời gian vào file CSV (ứng dụng điểm danh).
- **Xây GUI đơn giản**: dùng `tkinter` hoặc web (Flask / FastAPI) làm giao diện.

## 6. Gợi ý mục lục báo cáo (cơ sở ngành)

1. Giới thiệu đề tài
2. Lý do chọn đề tài
3. Tổng quan về nhận diện gương mặt
4. Giới thiệu FaceNet và facenet-pytorch
5. Thiết kế hệ thống (luồng xử lý, sơ đồ)
6. Cài đặt – môi trường – công cụ
7. Thực nghiệm (dataset, kết quả, nhận xét)
8. Kết luận và hướng phát triển

Bạn có thể dùng thư mục này như skeleton, sau đó mở rộng thêm theo yêu cầu của môn học.
