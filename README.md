# Vietnamese Traffic Sign Recognition | Data Preparation & Hybrid Computer Vision

A computer-vision project for **Vietnamese traffic-sign dataset preparation and recognition**, combining a **YOLOv8 object detector** with a **CNN classifier**. The main emphasis is on creating a traceable image-data workflow: data cleaning, class organization, annotation inspection, train/validation/test splitting, augmentation, model training, and inference.

**Maintainer:** Tran Trang Sang  
**Main stack:** Python, OpenCV, NumPy, YOLOv8, TensorFlow/Keras, Jupyter Notebook  
**Project period:** May–July 2026

## Project highlights

- **Dataset quality checks:** image-blur filtering, perceptual-hash (pHash) duplicate detection, class-distribution analysis, and visualization of detection annotations.
- **Dataset preparation:** YOLO-format labels, aspect-ratio-preserving resizing and padding, augmentation, and train/validation/test splits.
- **Two-stage recognition:** detect a traffic-sign region using YOLOv8, then classify the cropped region with a CNN.
- **Reproducible notebooks:** separate data-preparation and model-training workflows, with reusable utilities in `Utils/`.
- **Seven classification groups:** Vietnamese traffic signs organized into seven categories (listed below).

> **Scope note:** The CNN uses seven classification groups. The YOLO data-preparation notebook also describes grouping detection labels into three broader super-classes. Detection and classification label spaces should not be assumed to be identical.

## Processing workflow

```text
Raw road images + labels                Cropped traffic-sign images
              |                                      |
              v                                      v
  YOLO dataset preparation                  CNN dataset preparation
  - inspect box labels                      - check class folders
  - filter blur / duplicates                - pHash deduplication
  - review distributions                    - resize with padding
  - split and augment                       - split and balance
              |                                      |
              v                                      v
         YOLOv8 detector                        CNN classifier
              |                                      ^
              +--> detected sign crop ---------------+
                              |
                              v
                   Class + bounding box
```

## Repository structure

The notebooks and utilities are located **at the repository root** (not inside a nested `Traffic_Sign_Project/` directory).

```text
.
├── Data_Pipeline/
│   ├── Yolov8_Data.ipynb
│   └── CNN_Data.ipynb
├── Training/
│   ├── yolov8-detectsign.ipynb
│   └── cnn-detectsign.ipynb
├── Utils/
│   └── helper_functions.py
├── requirements.txt
└── README.md
```

The notebooks and helper functions also refer to these **local data/output directories** (which may need to be created and populated separately):

```text
Data_Raw/
├── YOLO_Dataset/
└── CNN_Dataset/

Data_Processed/
├── YOLO_Processed/
└── CNN_Processed/

Pictures/
├── YOLO_Processing/
└── CNN_Processing/
```

Large raw datasets, processed datasets, model weights, and locally generated visualizations are **not guaranteed to be included** in the GitHub checkout. Review the paths in the notebooks and `Utils/helper_functions.py` before running them.

## Dataset and annotation workflow

### 1. YOLO detection dataset

The detection-data notebook is [`Data_Pipeline/Yolov8_Data.ipynb`](Data_Pipeline/Yolov8_Data.ipynb). It covers:

1. Class mapping and exploratory analysis of class and bounding-box distributions.
2. Blur detection and duplicate-image filtering.
3. Inspection of annotations, including invalid or low-quality bounding boxes.
4. Train/validation/test splitting (configured as **70% / 20% / 10%** in the notebook).
5. Augmentation and final dataset visualization.

A YOLO detection annotation is typically stored as a text row:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the image dimensions. Image files must be paired with matching label files, and the chosen class mapping must remain consistent across all splits.

### 2. CNN classification dataset

The classification-data notebook is [`Data_Pipeline/CNN_Data.ipynb`](Data_Pipeline/CNN_Data.ipynb). It includes:

- Cleaning and splitting images by category.
- pHash-based detection of near-duplicate images.
- Resizing with padding to preserve sign geometry.
- Reviewing class balance and augmentation results.
- Before/after visualization to inspect preparation decisions.

**Important:** Horizontal flipping is intentionally avoided in the CNN notebook because direction-sensitive traffic signs can change meaning when mirrored.

### Classification categories

| ID | Vietnamese category | English interpretation |
|---|---|---|
| 0 | Cam_nguoc_chieu | No entry / wrong-way restriction |
| 1 | Cam_dung_va_do | No stopping or parking |
| 2 | Cam_re | Turn prohibited |
| 3 | Gioi_han_toc_do | Speed limits |
| 4 | Cam_con_lai | Other prohibition signs |
| 5 | Nguy_hiem | Warning signs |
| 6 | Hieu_lenh | Mandatory instruction signs |

These seven categories describe the **classification task**; consult the YOLO notebook for the specific detection-label mapping.

## Getting started

### Prerequisites

- Python 3 and Jupyter Notebook
- Data provided in the expected raw-data folders
- Additional model-training dependencies if running the training notebooks

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/verasol23/vietnam-traffic-sign-recognition.git
cd vietnam-traffic-sign-recognition

python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

### 2. Configure dataset paths

Prepare your raw data in the expected YOLO and CNN formats. Before execution, check notebook path settings and `Utils/helper_functions.py`: parts of the code originated from a **nested local project layout** and may expect data folders relative to the parent directory. Adjust paths to match your machine.

Do not train using images that overlap between training and evaluation splits. When using pHash filtering, inspect potential near-duplicate groups before removal.

### 3. Run data-preparation notebooks

Launch Jupyter from `Data_Pipeline/` so the notebooks' working-directory assumptions are easier to follow:

```bash
cd Data_Pipeline
jupyter notebook
```

Open `Yolov8_Data.ipynb` or `CNN_Data.ipynb` and execute cells in order after verifying the inputs and configuration.

### 4. Model training

Training notebooks are located at:

- [YOLOv8 training notebook](Training/yolov8-detectsign.ipynb)
- [CNN training notebook](Training/cnn-detectsign.ipynb)

They are intended for GPU-capable environments such as Kaggle or Colab. Supply the generated datasets and review the notebook configuration before starting training.

### 5. Inference

The intended final architecture is **YOLO detection → sign crop / preprocessing → CNN classification → visual overlay**. The original local workflow refers to `Inference/Model_hybrid.py` and the model artifacts `best.pt` and `traffic_sign_cnn.keras`. **These inference artifacts are not confirmed in the current repository checkout**, so this README does not claim that a fresh clone can run end-to-end inference without additional files.

## Quality checks and reproducibility

Before trusting a dataset or model result:

- Check class IDs, matching image/label stems, and bounding-box coordinate ranges.
- Look for identical or near-duplicate images crossing data splits.
- Review blurry images and difficult examples rather than relying only on automatic filtering.
- Inspect representative annotation visualizations and class-distribution plots.
- Keep track of dataset versions, split seeds, augmentation settings, and training configurations.

**Evaluation status:** Numerical benchmark values (such as mAP, accuracy, precision, recall, and FPS) are intentionally omitted here until they can be backed by a clearly defined evaluation dataset and recorded experiment outputs.

## Project scope and limitations

This is a **student engineering project**, not a production-validated traffic-safety system. Performance may vary with night scenes, motion blur, occlusion, small signs, camera viewpoint, and class imbalance. The repository documents the preparation and training workflows; deployment readiness requires reproducible evaluation and the necessary model artifacts.

## Author

**Tran Trang Sang**  
Robotics & Artificial Intelligence Engineering — HCM-UTE  
[GitHub profile](https://github.com/verasol23)
