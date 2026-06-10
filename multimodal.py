import os, glob, random
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, Concatenate,
    Embedding, LSTM, Bidirectional, TextVectorization
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

# ---------------- Config ----------------
base_dir = r"D:\diseeNN"
image_train_dir = os.path.join(base_dir, r"data\Train\Labeled")
tweets_csv = os.path.join(base_dir, r"data\multimodal_tweets_images.csv")
manifest_csv = os.path.join(base_dir, r"data\multimodal_manifest.csv")
save_path = os.path.join(base_dir, r"models\multimodalmodel.h5")

os.makedirs(os.path.dirname(save_path), exist_ok=True)
os.makedirs(os.path.dirname(manifest_csv), exist_ok=True)

image_size = (224, 224)
batch_size = 16
num_classes = 4
label_names = ["No damage","Mild","Severe","Help needed"]
label2id = {l:i for i,l in enumerate(label_names)}

# Text config
max_tokens = 10000
max_len = 128

# ---------------- Helpers ----------------
def list_images_by_class(root_dir):
    flooded = glob.glob(os.path.join(root_dir, "Flooded", "", "*.jpg"), recursive=True)
    flooded += glob.glob(os.path.join(root_dir, "Flooded", "", "*.png"), recursive=True)
    nonflood = glob.glob(os.path.join(root_dir, "Non-Flooded", "", "*.jpg"), recursive=True)
    nonflood += glob.glob(os.path.join(root_dir, "Non-Flooded", "", "*.png"), recursive=True)
    return nonflood, flooded

def load_tweets_df(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        text_col = next((c for c in ["text","tweet","tweet_text","content","message"] if c in df.columns), None)
        if text_col is None:
            return None
        label_col = next((c for c in ["label","category","target","severity"] if c in df.columns), None)
        if label_col is not None:
            df[label_col] = df[label_col].astype(str).str.strip()
        df[text_col] = df[text_col].astype(str)
        return {"df": df, "text_col": text_col, "label_col": label_col}
    return None

DEFAULT_TEXTS = {
    "No damage": [
        "No standing water observed, streets are clear.",
        "Light rain with normal traffic, no flooding."
    ],
    "Mild": [
        "Water accumulation on roads, minor delays.",
        "Localized flooding in low-lying areas, caution advised."
    ],
    "Severe": [
        "Widespread flooding with property damage reported.",
        "Roads submerged, vehicles stranded, severe impact."
    ],
    "Help needed": [
        "People stranded on rooftops, immediate rescue required.",
        "Emergency supplies needed, trapped individuals requesting help."
    ]
}

def sample_text_for_label(tweets_info, label):
    if tweets_info is None:
        return random.choice(DEFAULT_TEXTS[label])
    df = tweets_info["df"]
    text_col = tweets_info["text_col"]
    label_col = tweets_info["label_col"]
    if label_col and df[label_col].str.lower().isin(
        [label.lower(), label.replace(" ", "_").lower()]
    ).any():
        subset = df[df[label_col].str.lower().isin(
            [label.lower(), label.replace(" ", "_").lower()]
        )]
        if len(subset) > 0:
            return subset.sample(1, random_state=random.randint(0, 10_000))[text_col].iloc[0]
    return df.sample(1, random_state=random.randint(0, 10_000))[text_col].iloc[0] if len(df)>0 else random.choice(DEFAULT_TEXTS[label])

def ensure_manifest(manifest_csv, image_train_dir, tweets_csv):
    if os.path.exists(manifest_csv):
        return pd.read_csv(manifest_csv)
    nonflood, flooded = list_images_by_class(image_train_dir)
    rows = []
    tweets_info = load_tweets_df(tweets_csv)
    for path in nonflood:
        rows.append((path, sample_text_for_label(tweets_info, "No damage"), "No damage"))
    for i, path in enumerate(flooded):
        target = ["Mild","Severe","Help needed"][i % 3]
        rows.append((path, sample_text_for_label(tweets_info, target), target))
    df = pd.DataFrame(rows, columns=["image_path","tweet_text","label"])
    df.to_csv(manifest_csv, index=False)
    print(f"Created manifest with {len(df)} rows at {manifest_csv}")
    return df

# ---------------- Build or load manifest ----------------
df = ensure_manifest(manifest_csv, image_train_dir, tweets_csv)
df = df.dropna(subset=["image_path","tweet_text","label"])
df = df[df["label"].isin(label_names)].reset_index(drop=True)

# ---------------- Split and class weights ----------------
y = df["label"].map(label2id).values
y_onehot = to_categorical(y, num_classes=num_classes)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(df, y))
df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]

cls_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y[train_idx])
class_weight = {i: float(w) for i,w in enumerate(cls_weights)}

# ---------------- Text vectorization ----------------
# Adapt on training texts
text_vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_len
)
text_vectorizer.adapt(df_train["tweet_text"].astype(str).values)

# ---------------- Vision pipeline ----------------
def decode_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

def make_ds(sub_df, y_oh, training):
    paths = sub_df["image_path"].values
    texts = sub_df["tweet_text"].astype(str).values
    labels = y_oh.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, texts, labels))

    def map_fn(p, t, lab):
        img = decode_img(p)
        if training:
            img = augment(img)
        # Vectorize text to integer sequence
        txt_vec = text_vectorizer(tf.reshape(t, [1]))
        txt_vec = tf.reshape(txt_vec, [-1])  # shape [max_len]
        return (img, txt_vec), lab

    if training:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_ds(df_train, y_train, True)
val_ds   = make_ds(df_val,   y_val,   False)

# ---------------- Build multimodal model ----------------
# Vision branch
base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(image_size[0], image_size[1], 3))
base.trainable = False

img_in = Input(shape=(image_size[0], image_size[1], 3), name="image_input")
xv = base(img_in, training=False)
xv = GlobalAveragePooling2D()(xv)
xv = Dropout(0.4)(xv)
xv = Dense(256, activation="relu")(xv)
xv = Dropout(0.3)(xv)
xv = Dense(128, activation="relu", name="vision_feat")(xv)

# Text branch - LSTM encoder
txt_in = Input(shape=(max_len,), dtype=tf.int32, name="text_input")
xt = Embedding(input_dim=max_tokens, output_dim=128, mask_zero=True)(txt_in)
xt = Bidirectional(LSTM(128, return_sequences=False))(xt)
xt = Dropout(0.4)(xt)
xt = Dense(256, activation="relu")(xt)
xt = Dropout(0.3)(xt)
xt = Dense(128, activation="relu", name="text_feat")(xt)

# Fusion
xf = Concatenate(name="fusion_concat")([xv, xt])
xf = Dropout(0.5)(xf)
xf = Dense(256, activation="relu")(xf)
xf = Dropout(0.3)(xf)
out = Dense(num_classes, activation="softmax", name="multimodal_output")(xf)

model = Model(inputs=[img_in, txt_in], outputs=out)

# ---------------- Compile ----------------
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
)

print(model.summary())

# ---------------- Callbacks ----------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
]

history1 = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=2, 
    class_weight=class_weight, 
    callbacks=callbacks
)

# Final save
model.save(save_path)
print(f"\n✅ Saved multimodal model to: {save_path}")