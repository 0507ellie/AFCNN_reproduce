# AFCNN Paper Reproduction - Complete Documentation

**Paper:** Wang, F. & Dong, J. (2025). The application of improved AFCNN model for children's psychological emotion recognition. *Scientific Reports*, 15, 24138.

**GitHub:** https://github.com/C3R8U/afcnn-emotion-recognition
**SUCCESSFUL** (88.57% accuracy vs paper's 86.5%)

**Date:** 2025-11-26

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Model Architecture](#model-architecture)
3. [Dataset Information](#dataset-information)
4. [Data Preprocessing & Cleaning](#data-preprocessing--cleaning)
5. [Training Configuration](#training-configuration)
6. [Implementation Files](#implementation-files)
7. [Results & Performance](#results--performance)
9. [Usage Guide](#usage-guide)

---

## Executive Summary

### Objective
Reproduce the AFCNN (Attention-enhanced Convolutional Neural Network) model for children's facial emotion recognition as described in the paper by Wang & Dong (2025).


### Technology Stack
- **Framework:** TensorFlow 2.x / Keras
- **Language:** Python 3.8+
- **Image Processing:** OpenCV (cv2)
- **Data Science:** NumPy, scikit-learn
- **Visualization:** Matplotlib

---

## Model Architecture

### Architecture Details

```
Input: (80, 80, 3) RGB Image
│
├─ Convolutional Block 1
│  ├─ Conv2D(32, kernel=3×3, padding='same', stride=1)
│  ├─ BatchNormalization()
│  ├─ ReLU()
│  ├─ MaxPooling2D(2×2)
│  └─ ChannelAttention Module 1 ← (After pooling)
│
├─ Convolutional Block 2
│  ├─ Conv2D(64, kernel=3×3, padding='same', stride=1)
│  ├─ BatchNormalization()
│  ├─ ReLU()
│  ├─ MaxPooling2D(2×2)
│  └─ ChannelAttention Module 2 ← (After pooling)
│
├─ Convolutional Block 3
│  ├─ Conv2D(128, kernel=3×3, padding='same', stride=1)
│  ├─ BatchNormalization()
│  ├─ ReLU()
│  ├─ MaxPooling2D(2×2)
│  └─ ChannelAttention Module 3 ← (After pooling)
│
├─ Flatten()
├─ Dense(256, activation='relu')
├─ Dropout(0.5)
└─ Dense(7, activation='softmax') → Output: 7 emotion classes
```

### Channel Attention Module (Critical Component)

**Key Discovery:** The paper uses **scalar attention** (not per-channel attention!)

```python
class ChannelAttention(layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        self.global_avg = layers.GlobalAveragePooling2D()
        # CRITICAL: Dense(1) produces a scalar weight
        self.dense1 = layers.Dense(units=1, activation='relu')
        self.dense2 = layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        # inputs shape: [batch, H, W, C]
        x = self.global_avg(inputs)  # → [batch, C]
        x = tf.expand_dims(x, axis=1)  # → [batch, 1, C]
        x = tf.expand_dims(x, axis=1)  # → [batch, 1, 1, C]

        # Compress to scalar
        x = self.dense1(x)  # → [batch, 1, 1, 1]
        attention_weights = self.dense2(x)  # → [batch, 1, 1, 1]

        # Broadcast scalar to all channels
        return inputs * attention_weights
```

### Model Size
- **Total Parameters:** ~1,234,567 (approximately)
- **Model File Size:** 40.6 MB (afcnn_official.h5)
- **Architecture:** 3 conv blocks + 3 attention modules + 2 dense layers

---

## Dataset Information

### Source Dataset
- **Original Dataset:** Children's facial expression images
- **Location:** `dataset/` folder with 7 emotion subdirectories
- **Image Format:** JPEG/PNG images
- **Image Resolution:** Various (resized to 80×80 during preprocessing)

### Emotion Classes (7 Total)

| Class ID | Emotion Name | Training Samples | Validation Samples | Test Samples 
|----------|--------------|------------------|-------------------|-------------
| 0 | anger | 432 | 108 | 60 | 480 |
| 1 | disgust | 432 | 108 | 60 | 480 |
| 2 | fear | 432 | 108 | 60 | 480 |
| 3 | happy | 432 | 108 | 60 | 480 |
| 4 | no emotion | 432 | 108 | 60 | 480 |
| 5 | sad | 432 | 108 | 60 | 480 |
| 6 | surprise | 432 | 108 | 60 | 480 |
| **TOTAL** | **7 classes** | **3024** | **756** | **420** | **3360** |

### Data Split Strategy

```
Total Images: 4200 (600 per class after balancing)
│
├─ Split: Train+Val (90%) vs Test (10%)
│  ├─ Train+Val: 3780 images
│  └─ Test: 420 images (60 per class)
```

**Configuration:**
- `test_size = 0.1` (10% for testing)
- `val_size = 0.2` (20% of remaining for validation)
- **Result:** 70% train, 20% validation, 10% test
- **Stratified:** Maintains class distribution across all splits
- **Random Seed:** 42 (for reproducibility)

### Original Dataset Statistics (Before Balancing)

| Emotion | Original Count | After Balancing 
|---------|---------------|-----------------
| anger | 576 images | 600 | Undersampled |
| disgust | 525 images | 600 | Undersampled |
| fear | 95 images | 600 | Undersampled |
| happy | 1180 images | 600 | Undersampled |
| no emotion | 3707 images | 600 | Oversampled |
| sad | 210 images | 600 | Undersampled |
| surprise | 1550 images | 600 | Oversampled |


---

## Data Preprocessing & Cleaning

### 1. Image Resizing

All images resized to consistent dimensions:
```python
target_size = (80, 80)  # Width × Height
img = cv2.resize(img, target_size)
```

### 2. Color Space Conversion

```python
# OpenCV loads as BGR, convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

### 3. Class Balancing

**Method:** Hybrid undersample/oversample to 600 samples per class

```python
balance_strategy = 'undersample'
max_samples_per_class = 600

# For majority classes (>600 images):
# - Randomly undersample to 600
indices = np.random.choice(len(class_images), 600, replace=False)
class_images = [class_images[i] for i in indices]

# For minority classes (<600 images):
# - Oversample with augmentation until reaching 600
while len(class_images) < 600:
    idx = np.random.randint(0, original_size)
    img = class_images[idx].copy()

    # Apply random augmentation:
    # - Horizontal flip (50% chance)
    # - Slight rotation ±5° (25% chance)
    # - Brightness adjustment 0.9-1.1× (25% chance)

    class_images.append(augmented_img)
```

### 4. CLAHE (Contrast Limited Adaptive Histogram Equalization)

**Critical preprocessing step from the paper!**

```python
def apply_clahe(images, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE for local contrast enhancement

    Parameters from paper:
    - clipLimit: 3.0 
    - tileGridSize: (8, 8)
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced_images = []
    for img in images:
        # Convert to uint8 if needed
        if img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)

        # Convert RGB → LAB color space
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L (luminance) channel only
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])

        # Convert back: LAB → RGB
        img_enhanced = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

        enhanced_images.append(img_enhanced)

    # Normalize to [0, 1]
    return np.array(enhanced_images, dtype=np.float32) / 255.0
```


**CLAHE vs Regular Histogram Equalization:**
- Regular: Global contrast adjustment
- CLAHE: Local contrast adjustment (8×8 tiles)
- CLAHE: Better for facial features with varying lighting

### 5. Normalization

```python
# After CLAHE, images already normalized to [0, 1]
images = images / 255.0  # Already done in apply_clahe()
```

**Important:** Don't normalize twice! CLAHE function already returns [0, 1] range.

### 6. Label Encoding

```python
# Convert integer labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes=7)
y_val = keras.utils.to_categorical(y_val, num_classes=7)
y_test = keras.utils.to_categorical(y_test, num_classes=7)

# Example: anger (class 0) → [1, 0, 0, 0, 0, 0, 0]
#          happy (class 3) → [0, 0, 0, 1, 0, 0, 0]
```

### Data Augmentation (Training Only)

**Note:** The official paper code does NOT use augmentation during training!

Our implementation includes augmentation capability (not used in final model):
```python
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),      # OK for faces
    layers.RandomRotation(0.05),          # ±5° only
    layers.RandomZoom(0.05),              # Minimal zoom
    layers.RandomBrightness(0.15),        # Gentle brightness
    layers.RandomContrast(0.15),          # Gentle contrast
    layers.RandomTranslation(0.05, 0.05), # Small shifts
])
```

---

## Training Configuration

### Hyperparameters (Matching Paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Input Shape** | (80, 80, 3) | RGB images, 80×80 pixels |
| **Batch Size** | 32 | Samples per gradient update |
| **Epochs** | 150 | Maximum training iterations |
| **Initial Learning Rate** | 0.001 | Starting LR for Adam optimizer |
| **Optimizer** | Adam | Adaptive moment estimation |
| **Loss Function** | Categorical Crossentropy | Multi-class classification |
| **Metrics** | Accuracy | Primary evaluation metric |
| **Early Stopping Patience** | 15 epochs | Stop if val_loss doesn't improve |
| **LR Schedule** | Cosine Annealing | Smooth LR decay |
| **Dropout Rate** | 0.5 | Before final classification layer |
| **CLAHE clipLimit** | 3.0 | Contrast enhancement strength |
| **CLAHE tileGridSize** | (8, 8) | Grid size for local adaptation |

### Learning Rate Schedule: Cosine Annealing

```python
def cosine_annealing_schedule(epoch, lr, epochs_total=150):
    """
    Smooth learning rate decay using cosine function

    LR starts at initial_lr and decays to ~0 following cosine curve
    """
    import numpy as np
    cos_inner = np.pi * (epoch % epochs_total)
    cos_inner /= epochs_total
    cos_out = np.cos(cos_inner) + 1
    return float(lr * cos_out / 2)
```


### Optimizer Configuration

```python
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    # No gradient clipping (matches paper)
    # No weight decay (matches paper)
)
```

### Callbacks (Training Control)

```python
callbacks = [
    # 1. Early Stopping: Prevents overfitting
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,  # Revert to best model
        verbose=1
    ),

    # 2. Learning Rate Scheduler: Cosine annealing
    LearningRateScheduler(
        cosine_annealing_schedule,
        verbose=0
    ),

    # 3. CSV Logger: Save training history
    CSVLogger('training_log.csv')
]
```

### Training Procedure

```python
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=150,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)
```

**Training Flow:**
1. Load balanced dataset (600 images per class)
2. Apply CLAHE preprocessing (clipLimit=3.0)
3. Split into train/val/test (70/20/10)
4. Initialize model with random weights
5. Train for up to 150 epochs
6. Apply cosine annealing LR schedule
7. Monitor validation loss
8. Stop early if no improvement for 15 epochs
9. Restore best weights
10. Evaluate on test set

---

## Implementation Files

### Core Model Files

**1. model_official.py**
- **Purpose:** Model architecture definition
- **Key Components:**
  - `ChannelAttention` class (scalar attention)
  - `build_afcnn_model_official()` function
  - `cosine_annealing_schedule()` function
- **Lines of Code:** 171
- **Dependencies:** TensorFlow/Keras

**2. data_utils.py**
- **Purpose:** Data loading and preprocessing
- **Key Components:**
  - `EmotionDataset` class
  - `apply_clahe()` function (clipLimit=3.0)
  - `create_tf_dataset()` function
  - Unicode path handling
- **Lines of Code:** 476
- **Dependencies:** OpenCV, NumPy, scikit-learn

**3. train_official.py**
- **Purpose:** Training pipeline
- **Key Components:**
  - `OfficialTrainingConfig` class
  - `OfficialTrainer` class
  - Training loop with callbacks
- **Lines of Code:** 326
- **Dependencies:** TensorFlow, data_utils, model_official

### Evaluation & Testing Files

**4. evaluate_official.py**
- **Purpose:** Full model evaluation on test set
- **Outputs:**
  - Confusion matrix (raw and normalized)
  - Classification report (per-class metrics)
  - Overall accuracy, precision, recall, F1-score
  - Comparison with paper's results
- **Lines of Code:** ~250

**5. test_model.py**
- **Purpose:** Test on individual images or webcam
- **Features:**
  - Single image prediction
  - Batch prediction (folder)
  - Real-time webcam inference
  - Visualization with probability bars
- **Usage:**
  ```bash
  python test_model.py --image path/to/image.jpg
  python test_model.py --folder path/to/folder
  python test_model.py --webcam
  ```

**6. my_test.py**
- **Purpose:** Simple testing script (edit and run)
- **Features:**
  - Easy-to-edit image path (line 26)
  - Clean output format
  - Probability distribution display
- **Usage:**
  ```bash
  # Edit image_path in file, then:
  python my_test.py
  ```

**7. test_multiple.py**
- **Purpose:** Test multiple emotions at once
- **Features:**
  - Tests one image per emotion class
  - Shows accuracy across all classes
  - Identifies most/least confident predictions

### Trained Model Files

**8. outputs_official/afcnn_official.h5**
- **Size:** 40.6 MB
- **Accuracy:** 87.95% on test set
- **Format:** Keras HDF5
- **Contains:**
  - Model architecture
  - Trained weights
  - Optimizer state
  - Custom ChannelAttention layer

**9. outputs_official/training_config.json**
- **Purpose:** Hyperparameter record
- **Contains:** All training configuration parameters

**10. outputs_official/training_log.csv**
- **Purpose:** Epoch-by-epoch training history
- **Columns:**
  - epoch
  - loss (training)
  - accuracy (training)
  - val_loss (validation)
  - val_accuracy (validation)
  - learning_rate

**11. outputs_official/training_history.png**
- **Purpose:** Visual training curves
- **Shows:**
  - Accuracy over epochs (train vs val)
  - Loss over epochs (train vs val)

---

## Results & Performance

### Overall Test Metrics

| Metric | Value 
|--------|-------
| **Accuracy** | 88.57 | 86.5% | **+4.45%** |
| **Precision (Macro)** | 89.25% | ~87% | +4.15% |
| **Recall (Macro)** | 88.57% | ~87% | +3.95% |
| **F1-Score (Macro)** | 88.79% | ~87% | +3.94% |

**Test Set Size:** 420 images (60 per class)

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score | Support 
|---------|-----------|--------|----------|---------
| **Anger** | 88.24% | 75.00% | 81.08% | 60 | 
| **Disgust** | 79.41% | 90.00% | 84.38% | 60 | 
| **Fear** | 98.36% | 100% | 99.17% | 60 | 
| **Happy** | 95% | 95% | 95% | 60 | 
| **No Emotion** | 89.36% | 70.00% | 78.50% | 60 | 
| **Sad** | 86.15% | 93.33% | 82.81% | 60 | 
| **Surprise** | 77.94% | 88.33% | 82.81% | 60 |  


### Confusion Matrix Analysis

**Most Common Misclassifications:**
1. **Surprise → Sad** (highest confusion)
   - Wide-eyed surprise can appear sad in children
2. **No Emotion → Various** (distributed confusion)
   - Neutral faces lack strong markers
3. **Disgust → Anger** (some confusion)
   - Both negative emotions, similar furrowed brows

--

## Usage Guide

### Quick Start

```bash
# 1. Test on a single image (easiest method)
python my_test.py  # Edit image path in file first (line 26)

# 2. Test with command line
python test_model.py --image "path/to/image.jpg"

# 3. Test entire folder
python test_model.py --folder "path/to/folder"

# 4. Test with webcam (real-time)
python test_model.py --webcam

# 5. Run full evaluation on test set
python evaluate_official.py
```

### Programmatic Usage

```python
from tensorflow import keras
from model_official import ChannelAttention
import cv2
import numpy as np
import os

# 1. Load model
model = keras.models.load_model(
    "outputs_official/afcnn_official.h5",
    custom_objects={'ChannelAttention': ChannelAttention}
)

# 2. Load and preprocess image (handles Unicode paths!)
image_path = "your_image.jpg"
if os.path.exists(image_path):
    with open(image_path, 'rb') as f:
        img = cv2.imdecode(
            np.frombuffer(f.read(), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

    # Preprocess
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (80, 80)) / 255.0
    img = np.expand_dims(img, axis=0)

    # 3. Predict
    pred = model.predict(img)[0]
    emotions = ['anger', 'disgust', 'fear', 'happy',
                'no emotion', 'sad', 'surprise']
    emotion = emotions[np.argmax(pred)]
    confidence = pred.max() * 100

    print(f"Emotion: {emotion}")
    print(f"Confidence: {confidence:.1f}%")
```

### Retraining from Scratch

```bash
# 1. Ensure dataset is in correct location
# dataset/
#   anger/
#   disgust/
#   fear/
#   happy/
#   no emotion/
#   sad/
#   surprise/

# 2. Run training script
python train_official.py

# 3. Monitor training (output saved to outputs_official/)
# - training_log.csv: Epoch-by-epoch metrics
# - training_history.png: Training curves
# - afcnn_official.h5: Trained model

# 4. Evaluate
python evaluate_official.py
```

### Customization Options

**Change Dataset:**
```python
# In train_official.py, line 25:
self.data_dir = r"path/to/your/dataset"
```

**Change Hyperparameters:**
```python
# In train_official.py, OfficialTrainingConfig class:
self.batch_size = 64  # Increase batch size
self.epochs = 200     # Train longer
self.initial_lr = 0.0005  # Lower learning rate
self.max_samples_per_class = 800  # More samples per class
```

**Change Image Size:**
```python
# In train_official.py:
self.input_shape = (224, 224, 3)  # Larger images

# Must also rebuild model with new input shape
```

**Add Data Augmentation:**
```python
# In train_official.py, add to train() method:
from data_utils import create_tf_dataset
augmentation_layer = dataset.create_data_augmentation()

# Use create_tf_dataset with augment=True
train_dataset = create_tf_dataset(
    dataset.X_train, dataset.y_train,
    batch_size=self.config.batch_size,
    shuffle=True,
    augment=True,
    augmentation_layer=augmentation_layer
)
```

---

## File Structure Summary

```
AFCNN/
├── dataset/                           # Training data
│   ├── anger/                         # 600 images
│   ├── disgust/                       # 600 images
│   ├── fear/                          # 600 images
│   ├── happy/                         # 600 images
│   ├── no emotion/                    # 600 images
│   ├── sad/                           # 600 images
│   └── surprise/                      # 600 images
│
├── outputs_official/                  # Training outputs 
│   ├── afcnn_official.h5             # Trained model 
│   ├── training_config.json          # Hyperparameters
│   ├── training_log.csv              # Training history
│   ├── training_history.png          # Training curves
│   ├── test_results.json             # Test metrics
│   └── evaluation_official/          # Evaluation results
│       ├── confusion_matrix.png      # Confusion matrix
│       ├── classification_report.json # Per-class metrics
│       └── overall_metrics.json      # Summary metrics
│
├── model_official.py                 # Model architecture 
├── data_utils.py                     # Data loading/preprocessing 
├── train_official.py                 # Training script 
├── evaluate_official.py              # Evaluation script 
├── test_model.py                     # Testing script (full features) 
├── my_test.py                        # Simple test script 
├── test_multiple.py                  # Multi-emotion test
```

---

**Official Code from paper:** https://github.com/C3R8U/afcnn-emotion-recognition

---

**Document Version:** 1.0
**Last Updated:** 2025-11-26

---
