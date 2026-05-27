# 🚦 Traffic Sign Recognition Project

Hệ thống nhận diện biển báo giao thông Việt Nam sử dụng Deep Learning kết hợp **YOLOv8** (Detection) và **CNN** (Classification).

## 📁 Cấu trúc thư mục

```
Project_DL/                             # Thư mục gốc
│
├── Traffic_Sign_Project/              # 📦 CODE DỰ ÁN
│   ├── Data_Pipeline/                 # Xử lý và chuẩn bị dữ liệu
│   │   ├── Yolov8_Data.ipynb
│   │   ├── CNN_Data.ipynb
│   │   └── convert-zd-yolov8-cnn.ipynb
│   │
│   ├── Training/                      # Huấn luyện models
│   │   ├── yolov8-detectsign.ipynb
│   │   └── cnn-detectsign.ipynb
│   │
│   ├── Inference/                     # Chạy thực tế (Production)
│   │   ├── Model_hybrid.py
│   │   ├── best.pt
│   │   └── traffic_sign_cnn.keras
│   │
│   ├── Utils/                         # Các hàm phụ trợ
│   │   ├── __init__.py
│   │   └── helper_functions.py
│   │
│   └── README.md
│
├── Data_Raw/                          # 📥 DỮ LIỆU GỐC (chưa xử lý)
│   ├── YOLO_Dataset/                  # Dataset YOLO format
│   │   ├── train/images/, labels/
│   │   └── val/images/, labels/
│   └── CNN_Dataset/                   # Dataset CNN format
│       ├── train/0_Cam_nguoc_chieu/...
│       └── val/0_Cam_nguoc_chieu/...
│
├── Data_Processed/                    # 📤 DỮ LIỆU ĐÃ XỬ LÝ
│   ├── YOLO_Processed/                # YOLO sau khi clean, augment
│   │   ├── train/images/, labels/
│   │   ├── val/images/, labels/
│   │   └── test/images/, labels/
│   └── CNN_Processed/                 # CNN sau khi resize, dedupe
│       ├── train/0_Cam_nguoc_chieu/...
│       ├── val/0_Cam_nguoc_chieu/...
│       └── test/0_Cam_nguoc_chieu/...
│
├── Pictures/                          # 🖼️ ẢNH VISUALIZATION
│   ├── YOLO_Processing/               # Ảnh quá trình xử lý YOLO
│   │   ├── 01_Raw_Samples/
│   │   ├── 02_Class_Distribution/
│   │   ├── 03_Blur_Detection/
│   │   ├── 04_Duplicate_Detection/
│   │   ├── 05_Augmentation/
│   │   ├── 06_Final_Distribution/
│   │   └── 07_Sample_Annotations/
│   └── CNN_Processing/                # Ảnh quá trình xử lý CNN
│       ├── 01_Raw_Samples/
│       ├── 02_Class_Distribution/
│       ├── 03_Resize_Padding/
│       ├── 04_Duplicate_Detection/
│       ├── 05_Augmentation/
│       ├── 06_Final_Distribution/
│       └── 07_Sample_Per_Class/
│
└── Data/                              # (Legacy - tương thích ngược)
    ├── yolo_dataset/
    └── cnn_dataset/
```

## 🔄 Workflow (Quy trình làm việc)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. DATA PREP   │ ──▶ │   2. TRAINING   │ ──▶ │  3. INFERENCE   │
│  Data_Pipeline/ │     │    Training/    │     │    Inference/   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 1️⃣ Data Pipeline (Xử lý dữ liệu)

**Chạy trên máy local:**

```bash
cd Traffic_Sign_Project/Data_Pipeline/
```

- **Yolov8_Data.ipynb**: Xử lý dữ liệu YOLO format
  - Grouping classes
  - Blur filtering
  - Deduplication
  - Data augmentation
  
- **CNN_Data.ipynb**: Xử lý dữ liệu CNN format
  - Resize với padding (giữ tỷ lệ)
  - Deduplication bằng pHash
  - Train/Val/Test split

### 2️⃣ Training (Huấn luyện)

**Chạy trên Kaggle/Colab (cần GPU):**

1. Upload dataset đã xử lý lên Kaggle
2. Mở notebook trong `Training/`
3. Chạy các cells theo thứ tự
4. Download model weights về `Inference/`

### 3️⃣ Inference (Chạy thực tế)

**Chạy trên máy local:**

```bash
cd Traffic_Sign_Project/Inference/
python Model_hybrid.py
```

## �️ Pictures - Ảnh Visualization

Tất cả ảnh trong quá trình xử lý được lưu vào thư mục `Pictures/` với tên rõ ràng:

### YOLO Processing:
| Thư mục | Mô tả |
|---------|-------|
| 01_Raw_Samples | Mẫu ảnh gốc trước xử lý |
| 02_Class_Distribution | Biểu đồ phân bố classes |
| 03_Blur_Detection | Ảnh bị mờ được phát hiện |
| 04_Duplicate_Detection | Ảnh trùng lặp được loại bỏ |
| 05_Augmentation | Mẫu ảnh sau augmentation |
| 06_Final_Distribution | Phân bố cuối cùng |
| 07_Sample_Annotations | Mẫu ảnh với annotations |

### CNN Processing:
| Thư mục | Mô tả |
|---------|-------|
| 01_Raw_Samples | Mẫu ảnh gốc trước xử lý |
| 02_Class_Distribution | Biểu đồ phân bố classes |
| 03_Resize_Padding | So sánh trước/sau resize |
| 04_Duplicate_Detection | Ảnh trùng lặp được loại bỏ |
| 05_Augmentation | Mẫu ảnh sau augmentation |
| 06_Final_Distribution | Phân bố cuối cùng |
| 07_Sample_Per_Class | Mẫu từng loại biển báo |

## 🛠️ Cài đặt

```bash
# Môi trường cơ bản
pip install opencv-python numpy matplotlib seaborn tqdm pyyaml

# Cho Data Pipeline
pip install albumentations imagehash scikit-learn

# Cho Training & Inference
pip install ultralytics tensorflow
```

## 📋 Classes (7 loại biển báo)

| ID | Tên class | Mô tả |
|----|-----------|-------|
| 0 | Cam_nguoc_chieu | Biển cấm ngược chiều |
| 1 | Cam_dung_va_do | Biển cấm dừng và đỗ |
| 2 | Cam_re | Biển cấm rẽ |
| 3 | Gioi_han_toc_do | Biển giới hạn tốc độ |
| 4 | Cam_con_lai | Các biển cấm khác |
| 5 | Nguy_hiem | Biển báo nguy hiểm |
| 6 | Hieu_lenh | Biển hiệu lệnh |

## 🚀 Sử dụng nhanh

### Chạy Inference trên video:

```python
from Inference.Model_hybrid import main
main()
```

### Import helper functions:

```python
from Utils.helper_functions import (
    resize_with_padding,
    calculate_iou,
    draw_detection,
    TRAFFIC_SIGN_CLASSES,
)
```

## 📝 Ghi chú

- **Utils/helper_functions.py**: Chứa tất cả hàm dùng chung, dễ maintain
- Các notebooks tự động detect đường dẫn dựa trên `PROJECT_ROOT`
- Model_hybrid.py hỗ trợ fallback cho cả đường dẫn mới và cũ
- Training notebooks được thiết kế để chạy trên Kaggle

## 👤 Author

Tran Trang Sang HCMUTE
