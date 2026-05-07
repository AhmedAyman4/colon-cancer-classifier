# Colon Cancer Histopathology Classification
### Team Gradiators

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![EfficientNetB3](https://img.shields.io/badge/Backbone-EfficientNetB3-blue)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.19%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-LC25000-lightgrey)
![License](https://img.shields.io/badge/License-Academic-orange)

> Binary classification of colon histopathology images into **Colon Adenocarcinoma**
> vs **Benign Tissue** using transfer learning on the LC25000 dataset.


---

## Dataset

**LC25000 — Lung and Colon Cancer Histopathological Images**
- **Source:** [Kaggle — andrewmvd/lung-and-colon-cancer-histopathological-images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
- **Classes used:** `colon_aca` (Colon Adenocarcinoma), `colon_n` (Benign Tissue)
- **Total images:** 10,000 (5,000 per class)
- **Image size:** 224x224 px

---

## Pipeline Overview

### 1. Data Splitting
| Split | Size | Purpose |
|-------|------|---------|
| Train | 80% (8,000) | Model learning |
| Validation | 10% (1,000) | Hyperparameter tuning |
| Test | 10% (~992) | Final evaluation |

### 2. Augmentation (training only)
- Random horizontal flip
- Random rotation +/-15 degrees
- Random translation +/-5%
- Random zoom +/-10%
- ResNet50-style preprocessing (ImageNet zero-centering)

### 3. Performance Optimization
- tf.data pipeline with .cache() and .prefetch(AUTOTUNE)
- Parallel map operations for GPU utilization

---

## Model Architecture

**Backbone:** EfficientNetB3 (frozen, pretrained on ImageNet)

```
Input (224x224x3)
  -> Data Augmentation
  -> EfficientNetB3 [frozen -- 10.78M params]
  -> GlobalAveragePooling2D -> (1536,)
  -> Dense(128, ReLU)
  -> BatchNormalization
  -> Dropout(0.3)
  -> Dense(2, Softmax)
```

| Parameter group | Count |
|-----------------|-------|
| Total params | 10,981,041 |
| Trainable | 197,250 (~1.8%) |
| Non-trainable | 10,783,791 |

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | 1e-4 (initial) |
| Loss | Sparse Categorical Crossentropy |
| Batch size | 16 |
| Max epochs | 50 |

**Callbacks:**
- EarlyStopping — patience=10, monitors val_loss
- ReduceLROnPlateau — patience=3, factor=0.3 (1e-4 -> 3e-5 -> 9e-6 -> 2.7e-6)
- ModelCheckpoint — saves best weights as efficientnetB3_RMSprop.keras

---

## Results

**Best checkpoint reached at Epoch 5.**

| Metric | Value |
|--------|-------|
| Val accuracy (best) | 99.40% |
| Test accuracy | 99.19% |

### Classification Report (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| colon_aca | 0.9878 | 0.9959 | 0.9918 | 487 |
| colon_n | 0.9960 | 0.9881 | 0.9920 | 505 |
| weighted avg | 0.9920 | 0.9919 | 0.9919 | 992 |

---

## Repository Structure

```
├── notebook.ipynb                                          # Main Colab notebook
├── efficientnetB3_RMSprop.keras                            # Best checkpoint (by val_loss)
├── best_colon_cancer_efficientnet_transfer_learning.keras  # Final saved model
└── README.md
```

---

## Getting Started

```python
# Install dependencies
pip install tensorflow kagglehub scikit-learn tqdm matplotlib seaborn

# Download dataset
import kagglehub
path = kagglehub.dataset_download(
    "andrewmvd/lung-and-colon-cancer-histopathological-images"
)

# Load saved model
from tensorflow import keras
model = keras.models.load_model(
    "best_colon_cancer_efficientnet_transfer_learning.keras"
)
```

---

## Dependencies

```
tensorflow >= 2.x
numpy
matplotlib
seaborn
scikit-learn
tqdm
kagglehub
```

---

## License

This project is developed for academic purposes.
Dataset credit: [LC25000 — Andrew MVD](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)
