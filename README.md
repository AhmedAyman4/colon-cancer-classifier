# Colon Cancer Histopathology Classification
### Team Gradiators

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![EfficientNetB3](https://img.shields.io/badge/Backbone-EfficientNetB3-blue)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.19%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-LC25000-lightgrey)
![License](https://img.shields.io/badge/License-Academic-orange)

> Binary classification of colon histopathology images into **Colon Adenocarcinoma**
> vs **Benign Tissue** using transfer learning on the LC25000 dataset.


## Repository Structure
```text
├── final-saved-models/   # Production-ready trained models (.keras)
│   └── best_*.keras      # Best performing model checkpoints
├── notebooks/            # Jupyter notebooks for development
│   ├── final_*.ipynb     # Final training and evaluation pipeline
│   ├── v0, v1_*.ipynb    # Experimental and versioned iterations
│   └── reference/        # Reference materials and research
└── README.md             # Project documentation
```

---

## Dataset & Pipeline
- **Dataset:** [LC25000](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) — 10,000 images (5,000 `colon_aca`, 5,000 `colon_n`).
- **Splits:** 80% Train, 10% Val, 10% Test.
- **Preprocessing:** Augmentation (flip, rotation, zoom) + `tf.data` optimization (cache & prefetch).

## Model Architecture
Using **EfficientNetB3** (frozen backbone) with a custom classification head:
- `GlobalAveragePooling2D` -> `Dense(128, ReLU)` -> `Dropout(0.3)` -> `Dense(2, Softmax)`.
- **Trainable Parameters:** 197,250 (~1.8%).

## Training & Results
- **Optimizer:** Adam (LR: 1e-4) | **Batch Size:** 16 | **Epochs:** 50 (Early Stopping).
- **Performance:** 99.19% Test Accuracy (F1-Score: 0.99).

## Quick Start
```python
# 1. Install dependencies
pip install tensorflow kagglehub scikit-learn

# 2. Load the final model
from tensorflow import keras
model = keras.models.load_model("best_colon_cancer_efficientnet_transfer_learning.keras")
```


---
*Academic project. Dataset credit: Andrew MVD.*
