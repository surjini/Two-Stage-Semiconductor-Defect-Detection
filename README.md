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

# CODE STRUCTURE

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

# RUN INSTRUCTIONS

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
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

BASE_PATH = "/kaggle/input/two-stage-semiconductor-defect-dataset-sem-img/semiconductor_defect_dataset/stage2_multiclass_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)


train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_PATH, "Train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(BASE_PATH, "Validation"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(BASE_PATH, "Test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
print("Class labels:", train_generator.class_indices)


base_model = tf.keras.applications.DenseNet121(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

x = base_model.output
x =x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

x = base_model.output
x =x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6)
]

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)


for layer in base_model.layers[-40:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=callbacks
)
```

