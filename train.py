import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input  # ✅ FIX 1: proper preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout,
    BatchNormalization, Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime

# ── Settings ───────────────────────────────────────────
DATASET_PATH    = "dataset"
IMAGE_SIZE      = (224, 224)
BATCH_SIZE      = 32
EPOCHS_FROZEN   = 20
EPOCHS_UNFREEZE = 30
MODEL_SAVE      = "model/leather_model.h5"

os.makedirs("model", exist_ok=True)

# ── Data Pipelines ─────────────────────────────────────
# FIX 1: Use preprocess_input instead of raw pixels.
#         EfficientNet expects inputs scaled by its own function,
#         NOT plain [0,255] or [0,1]. Skipping this caused the
#         accuracy noise and slow convergence you saw.
#
# FIX 2: Reduced augmentation aggressiveness.
#         rotation_range=30 + zoom=0.25 + channel_shift on a small
#         leather dataset creates images too far from reality.
#         Leather defects have specific textures — over-augmenting
#         destroys those features and confuses the model.
print("📂 Loading leather defect dataset...")

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,   # ✅ FIX 1
    rotation_range=20,          # was 30  ✅ FIX 2
    width_shift_range=0.15,     # was 0.2
    height_shift_range=0.15,    # was 0.2
    shear_range=0.1,            # was 0.2
    zoom_range=0.15,            # was 0.25
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.85, 1.15],  # was [0.75, 1.25]
    channel_shift_range=10.0,   # was 20.0
    fill_mode='reflect',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,   # ✅ FIX 1: val must use same preprocessing
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

class_names = list(train_data.class_indices.keys())
num_classes = len(class_names)
print(f"✅ Found {num_classes} classes: {class_names}")

# ── Class Weights (handles imbalanced datasets) ────────
# FIX 3: If one defect type has more images than others, the model
#         ignores rare classes. Class weights fix this automatically.
total_samples = train_data.samples
class_counts = np.bincount(train_data.classes)
class_weight_vals = total_samples / (num_classes * class_counts)
class_weights = {i: w for i, w in enumerate(class_weight_vals)}
print(f"📊 Class weights: { {class_names[i]: round(w,2) for i,w in class_weights.items()} }")

# ── Build Model ────────────────────────────────────────
print("\n🧠 Building EfficientNetB0 model...")

inputs = Input(shape=(224, 224, 3))

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_tensor=inputs
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)   # FIX 4: reduced from 512→256→output
x = Dropout(0.4)(x)                    # Simpler head = less overfitting on small datasets
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# ── PHASE 1: Train Head Only ───────────────────────────
print("\n" + "="*60)
print("🚀 PHASE 1 — Training head (base fully frozen)")
print("="*60)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase1 = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        "model/phase1_best.h5",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history1 = model.fit(
    train_data,
    epochs=EPOCHS_FROZEN,
    validation_data=val_data,
    class_weight=class_weights,    # ✅ FIX 3
    callbacks=callbacks_phase1
)

phase1_best_acc = max(history1.history['val_accuracy'])
print(f"\n✅ Phase 1 done — best val accuracy: {phase1_best_acc:.4f}")

# ── Load best Phase 1 weights before fine-tuning ──────
print("\n📂 Loading best Phase 1 weights before fine-tuning...")
model.load_weights("model/phase1_best.h5")

# ── PHASE 2: Conservative Fine-Tuning ─────────────────
print("\n" + "="*60)
print("🔓 PHASE 2 — Conservative fine-tuning (last 30 layers only)")
print("="*60)

base_model.trainable = True

FREEZE_UNTIL = 208
for layer in base_model.layers[:FREEZE_UNTIL]:
    layer.trainable = False

# Always keep BatchNorm frozen — critical for small datasets
# Unfrozen BN shifts running mean/variance and destabilises training
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(1 for l in model.layers if l.trainable)
print(f"   Unfrozen layers  : {trainable_count}")
print(f"   Fine-tune LR     : 2e-6")
print(f"   Freeze until     : layer {FREEZE_UNTIL}/238")

model.compile(
    optimizer=Adam(
        learning_rate=2e-6,
        clipnorm=1.0
    ),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_SAVE,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-8,
        verbose=1
    )
]

history2 = model.fit(
    train_data,
    epochs=EPOCHS_UNFREEZE,
    validation_data=val_data,
    class_weight=class_weights,    # ✅ FIX 3
    callbacks=callbacks_phase2
)

phase2_best_acc = max(history2.history['val_accuracy'])
print(f"\n✅ Phase 2 done — best val accuracy: {phase2_best_acc:.4f}")
print(f"   Δ improvement    : +{(phase2_best_acc - phase1_best_acc)*100:.2f}%")

# ── Save Outputs ───────────────────────────────────────
with open("model/class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")

metadata = {
    "model_architecture"  : "EfficientNetB0 + custom head",
    "image_size"          : list(IMAGE_SIZE),
    "num_classes"         : num_classes,
    "class_names"         : class_names,
    "phase1_best_val_acc" : round(phase1_best_acc, 4),
    "phase2_best_val_acc" : round(phase2_best_acc, 4),
    "freeze_until_layer"  : FREEZE_UNTIL,
    "phase2_learning_rate": "2e-6",
    "preprocessing"       : "EfficientNet preprocess_input (correct)",
    "trained_at"          : datetime.now().isoformat()
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("📋 Saved: class_names.txt, metadata.json,", MODEL_SAVE)

# ── Plot ───────────────────────────────────────────────
all_acc      = history1.history['accuracy']     + history2.history['accuracy']
all_val_acc  = history1.history['val_accuracy'] + history2.history['val_accuracy']
all_loss     = history1.history['loss']         + history2.history['loss']
all_val_loss = history1.history['val_loss']     + history2.history['val_loss']
split        = len(history1.epoch)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EfficientNetB0 — Fixed Training (No Catastrophic Forgetting)",
             fontsize=13, fontweight='bold')

for ax, tr, val, title, ylabel in zip(
    axes,
    [all_acc, all_loss],
    [all_val_acc, all_val_loss],
    ['Model Accuracy', 'Model Loss'],
    ['Accuracy', 'Loss']
):
    ep = range(1, len(tr) + 1)
    ax.plot(ep, tr,  label='Train', linewidth=2)
    ax.plot(ep, val, label='Val',   linewidth=2)
    ax.axvline(x=split, color='green', linestyle='--',
               alpha=0.7, label='Fine-tune start')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("model/training_results.png", dpi=150)
plt.show()
print("✅ Training complete!")