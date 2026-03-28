# robust_worm_training.py
import os, shutil, random
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report

# 1) Clean hidden folders automatically 
DATA_DIR = "dataset"  # update if different
hidden = os.path.join(DATA_DIR, '.ipynb_checkpoints')
if os.path.exists(hidden):
    print("Removing leftover hidden folder:", hidden)
    shutil.rmtree(hidden)

#2) Build explicit file list (no hidden dirs, no alphabetical surprises)
classes = ['earthworm', 'flatworm']
filepaths = []
labels = []

for cls in classes:
    cls_dir = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(cls_dir):
        raise FileNotFoundError(f"Expected folder {cls_dir} not found.")
    imgs = []
    for ext in ('*.jpg','*.jpeg','*.png','*.bmp'):
        imgs.extend(glob(os.path.join(cls_dir, ext)))
    imgs = sorted(imgs)
    if len(imgs) == 0:
        raise RuntimeError(f"No images found in {cls_dir}.")
    print(f"Found {len(imgs)} images for class '{cls}'")
    for p in imgs:
        filepaths.append(p)
        labels.append(cls)

df = pd.DataFrame({'filename': filepaths, 'label': labels})
print("Total images:", len(df))
print(df['label'].value_counts())

# 3) Stratified split to train/val (exact split from your small dataset) 
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
print("Train size:", len(train_df), "Val size:", len(val_df))
print("Train distribution:\n", train_df['label'].value_counts())
print("Val distribution:\n", val_df['label'].value_counts())

# 4) ImageDataGenerators (augmentation only on train) 
IMG_SIZE = (128,128)
BATCH = 4

train_gen_tf = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True
)

val_gen_tf = ImageDataGenerator(rescale=1./255)

train_gen = train_gen_tf.flow_from_dataframe(
    train_df,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH,
    shuffle=True
)

val_gen = val_gen_tf.flow_from_dataframe(
    val_df,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    class_mode='binary',
    batch_size=BATCH,
    shuffle=False
)

print("Class indices (explicit):", train_gen.class_indices)


inv_classes = {v:k for k,v in train_gen.class_indices.items()}

# 5) Build model (transfer learning) 
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#  6) Callbacks & class weights (optional, but safe) 
# compute class weights to reduce bias toward any class
from sklearn.utils import class_weight
cw = class_weight.compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
class_weights = {train_gen.class_indices[c]: w for c, w in zip(np.unique(train_df['label']), cw)}
print("Class weights:", class_weights)

es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
chk = ModelCheckpoint('best_worm_model.h5', save_best_only=True, monitor='val_loss')

#  7) Train 
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25,
    class_weight=class_weights,
    callbacks=[es, chk]
)

#  8) Evaluate and get predictions on validation set 
model = tf.keras.models.load_model('best_worm_model.h5')
val_steps = int(np.ceil(len(val_df)/BATCH))
preds = model.predict(val_gen, steps=val_steps)
# preds are floats in [0,1]
y_true = val_gen.labels  # numeric labels according to class_indices
y_probs = preds.ravel()

# default threshold
threshold = 0.5
y_pred = (y_probs > threshold).astype(int)

print("--- Classification report (threshold {:.2f}) ---".format(threshold))
# map numeric labels to names
y_true_names = [inv_classes[int(x)] for x in y_true]
y_pred_names = [inv_classes[int(x)] for x in y_pred]

print(classification_report(y_true_names, y_pred_names))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

# 9) Show misclassified files & their probs
mis_idx = np.where(y_pred != y_true)[0]
print(f"Number of misclassified examples: {len(mis_idx)}")
for i in mis_idx:
    # val_gen.filenames aligns with the generator order
    fname = val_gen.filepaths[i] if hasattr(val_gen, 'filepaths') else val_gen.filenames[i]
    print(f"MISCLASSIFIED: {fname} | true={inv_classes[int(y_true[i])]} | prob={y_probs[i]:.3f} | pred={inv_classes[int(y_pred[i])]}")

#10) If many errors, try fine-tuning last layers (quick) 
print("\nIf accuracy is low, you can try unfreezing last layers of base and retrain for a few epochs.")
print("Example:")
print("for layer in base.layers[-30:]: layer.trainable = True")
print("recompile and model.fit(..., epochs=5)")

# 11) Suggest threshold tuning if model leans toward one class
# show distribution of probabilities per true class
for cls_val, cls_name in inv_classes.items():
    cls_probs = y_probs[y_true == cls_val]
    print(f"Class '{cls_name}' prob mean={cls_probs.mean():.3f} std={cls_probs.std():.3f} n={len(cls_probs)}")
