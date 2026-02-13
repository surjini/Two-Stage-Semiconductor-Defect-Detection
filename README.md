## Two-Stage-Semiconductor-Defect-Detection

## Program DataPath - https://www.kaggle.com/code/surjini/two-stage-semiconductor-clean-and-defect-detection/edit

## Dataset Datapath - /kaggle/input/two-stage-semiconductor-defect-dataset-sem-img
# Two-Stage Semiconductor Defect Classification
Project Overview

This project proposes a Two-Stage Edge-AI framework for automated defect inspection in SEM (Scanning Electron Microscope) images used in semiconductor manufacturing.
The system first performs binary screening (Clean vs Defective) and then applies fine-grained defect classification only on defective samples.
This two-stage design reduces computation, latency, and power consumption, making it suitable for real-time edge deployment without cloud dependency.

Dataset Description

Dataset Source
A custom-curated SEM image dataset organized specifically for a two-stage classification pipeline. The dataset is structured to reflect realistic industrial wafer inspection conditions, including grayscale textures, SEM noise, and diverse defect patterns.

DATASET ORGANIZATION BY STAGE
| Dataset         | Included Samples    | Excluded Samples |
| --------------- | ------------------- | ---------------- |
| Master Dataset  | All 8 classes       | None             |
| Stage 1 Dataset | Clean + All Defects | —                |
| Stage 2 Dataset | Defect classes only | Clean            |

CLASS DESIGN

| Stage                 | Classification Type        | Classes                                                |
| --------------------- | -------------------------- | ------------------------------------------------------ |
| Stage 1               | Binary Classification      | Clean, Defective                                       |
| Stage 2               | Multi-Class Classification | Bridges, Cracks, Opens, Shorts, Scratches, Vias, Other |
| Master Dataset | 8 (7 defect + 1 clean)     |                                                        |


DATASET SPLIT:

  Training: 70%

  Validation: 15%

  Testing: 15%

Two-Stage Model Architecture
   
   Stage-1: Binary Classification (Clean vs Defective)
   Lightweight CNN
   Fast screening to reduce Stage-2 load

Stage-2: Multi-Class Defect Classification

   Model Used: DenseNet121
   Framework: TensorFlow / Keras
   Training Approach: Transfer Learning
   
MODEL DETAILS & BASELINE RESULTS

| Parameter      | Value             |
| -------------- | ----------------- |
| Input Size     | 224 × 224 × 3     |
| Model          | DenseNet121       |
| Training Type  | Transfer Learning |
| Model Size     | ~33 MB            |
| Framework      | TensorFlow        |
| No. of Classes | 8                 |

TEST METRICS 

Test Accuracy: 85%

Validation Accuracy (best): 80%

Precision (avg):  ~0.30

Recall (avg): ~0.27


## Code Structure and run insructions

## CODE STRUCTURE

1. Imports & Configuration

TensorFlow / Keras

ImageDataGenerator

DenseNet121 backbone

Callbacks for stability

2. Data Loading

Augmentation for training

Only rescaling for validation & test

3. Model Architecture

DenseNet121 (ImageNet pretrained)

Global Average Pooling

Dense + Dropout classifier head

4. Training Strategy

Phase-1: Feature extraction (frozen backbone)

Phase-2: Fine-tuning (last layers unfrozen)

## RUN INSTRUCTIONS

to Run (Step-by-Step)

1️. Upload dataset to Kaggle

2️. Verify folder structure

3️. Open new Kaggle notebook

4️. Paste full code above

5️. Run cells top to bottom

6️. Training happens in two phases automatically

OUTPUTS GENERATED:

| Output              | Description                    |
| ------------------- | ------------------------------ |
| Trained Model       | `DenseNet121` Stage-2          |
| Validation Accuracy | ~80% (current dataset size)    |
| Test Accuracy       | ~85%                           |
| Ready for ONNX      | Yes                            |

## code
```
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

BASE_PATH = "/kaggle/input/two-stage-semiconductor-defect-dataset-sem-img/semiconductor_defect_dataset"

STAGE1_PATH = BASE_PATH + "/stage1_binary_dataset"
STAGE2_PATH = BASE_PATH + "/stage2_multiclass_dataset"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 15
EPOCHS_FINE = 10

train_gen1 = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen1 = ImageDataGenerator(rescale=1./255)

train_stage1 = train_gen1.flow_from_directory(
    STAGE1_PATH + "/Train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_stage1 = val_gen1.flow_from_directory(
    STAGE1_PATH + "/Validation",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)


base_model1 = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model1.trainable = False

x = base_model1.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output1 = Dense(1, activation="sigmoid")(x)

stage1_model = Model(inputs=base_model1.input, outputs=output1)

stage1_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks1 = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("stage1_best_model.keras", save_best_only=True)
]

history_stage1 = stage1_model.fit(
    train_stage1,
    validation_data=val_stage1,
    epochs=EPOCHS_INITIAL,
    callbacks=callbacks1
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

val_gen1_eval = ImageDataGenerator(rescale=1./255)

val_stage1_eval = val_gen1_eval.flow_from_directory(
    STAGE1_PATH + "/Validation",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

val_stage1_eval.reset()

#predictions
y_pred_probs1 = stage1_model.predict(val_stage1_eval)
y_pred1 = (y_pred_probs1 > 0.5).astype(int).flatten()
y_true1 = val_stage1_eval.classes

# Confusion Matrix
cm1 = confusion_matrix(y_true1, y_pred1)

plt.figure(figsize=(6,6))
sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Clean", "Defect"],
            yticklabels=["Clean", "Defect"])
plt.title("Stage 1 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("Stage 1 Classification Report:")
print(classification_report(y_true1, y_pred1, zero_division=0))

train_gen2 = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen2 = ImageDataGenerator(rescale=1./255)

train_stage2 = train_gen2.flow_from_directory(
    STAGE2_PATH + "/Train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_stage2 = val_gen2.flow_from_directory(
    STAGE2_PATH + "/Validation",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

base_model2 = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model2.trainable = False

x = base_model2.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output2 = Dense(train_stage2.num_classes, activation="softmax")(x)

stage2_model = Model(inputs=base_model2.input, outputs=output2)

stage2_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
class_weights = compute_class_weight(
class_weight="balanced",
    classes=np.unique(train_stage2.classes),
    y=train_stage2.classes
)

class_weights_dict = dict(enumerate(class_weights))

callbacks2 = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint("stage2_best_model.keras", save_best_only=True)
]

history_stage2 = stage2_model.fit(
    train_stage2,
    validation_data=val_stage2,
    epochs=EPOCHS_INITIAL,
    callbacks=callbacks2,
    class_weight=class_weights_dict
)

val_gen2_eval = ImageDataGenerator(rescale=1./255)

val_stage2_eval = val_gen2_eval.flow_from_directory(
    STAGE2_PATH + "/Validation",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

val_stage2_eval.reset()

# Predictions
predictions2 = stage2_model.predict(val_stage2_eval)
y_pred2 = np.argmax(predictions2, axis=1)
y_true2 = val_stage2_eval.classes

class_names2 = list(val_stage2_eval.class_indices.keys())

# Confusion Matrix
cm2 = confusion_matrix(y_true2, y_pred2)

plt.figure(figsize=(8,8))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names2,
            yticklabels=class_names2)
plt.title("Stage 2 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()

# Classification Report
print("Stage 2 Classification Report:")
print(classification_report(
    y_true2,
    y_pred2,
 target_names=class_names2,
    zero_division=0
))

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0   # same normalization as training
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def two_stage_predict(img_path):

    img = preprocess_image(img_path)

    # ----- Stage 1 Prediction -----
    stage1_pred = stage1_model.predict(img, verbose=0)
    
    if stage1_pred[0][0] > 0.5:
        print("Stage-1 Result: DEFECT detected")
        
        # ----- Stage 2 Prediction -----
        stage2_pred = stage2_model.predict(img, verbose=0)
        class_index = np.argmax(stage2_pred)
        
        class_names = list(train_stage2.class_indices.keys())
        defect_type = class_names[class_index]
        
        print("Stage-2 Result: Defect Type =", defect_type)
        
    else:
        print("Stage-1 Result: NORMAL wafer")

two_stage_predict("/kaggle/input/two-stage-semiconductor-defect-dataset-sem-img/semiconductor_defect_dataset/stage2_multiclass_dataset/Test/cracks/3.jpg")






