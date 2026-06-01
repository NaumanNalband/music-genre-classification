"""
train_model.py — CNN Model Trainer for Music Genre Classification
Author  : Third-Year CS Student
Dataset : GTZAN Music Genre Dataset
Usage   : python train_model.py
          Expects data/genres/<genre>/*.wav
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(__file__), '..', 'data', 'genres')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'genre_classifier.keras')
ENCODER_PATH    = os.path.join(os.path.dirname(__file__), 'label_encoder.pkl')

SAMPLE_RATE   = 22050
DURATION      = 30        # seconds per clip
N_MELS        = 128       # mel-spectrogram frequency bins
HOP_LENGTH    = 512
N_FFT         = 2048
TARGET_SHAPE  = (128, 128)   # CNN input (H, W)

GENRES = [
    'blues', 'classical', 'country', 'disco',
    'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'
]

EPOCHS      = 20
BATCH_SIZE  = 32
VAL_SPLIT   = 0.2
RANDOM_SEED = 42

# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_melspectrogram(file_path: str):
    """
    Load a WAV file and return a normalised mel-spectrogram resized to
    TARGET_SHAPE, ready to be used as a single CNN channel (H x W x 1).
    Returns None on any loading error so the caller can skip bad files.
    """
    try:
        # Load audio using soundfile directly — avoids audioread/aifc which
        # was removed in Python 3.13. soundfile handles GTZAN's WAV/AIFF files.
        import soundfile as sf
        try:
            y, sr_orig = sf.read(file_path, always_2d=False)
        except Exception:
            # Last-resort fallback for any exotic format
            y, sr_orig = librosa.load(file_path, sr=None, mono=True, duration=DURATION)

        # Convert stereo → mono
        if y.ndim > 1:
            y = y.mean(axis=1).astype(np.float32)
        else:
            y = y.astype(np.float32)

        # Resample to target SR if the file differs
        if sr_orig != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=int(sr_orig), target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

        # Trim to DURATION seconds
        target_len = SAMPLE_RATE * DURATION
        if len(y) > target_len:
            y = y[:target_len]

        # Guard: skip clips shorter than 1 s
        if len(y) < sr:
            print(f"  [SKIP] Too short: {file_path}")
            return None

        # Mel-spectrogram → log scale (dB)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Resize to fixed shape so the CNN always sees 128×128
        mel_spec_resized = np.resize(mel_spec_db, TARGET_SHAPE)

        # Min-max normalisation → [0, 1]
        mel_min, mel_max = mel_spec_resized.min(), mel_spec_resized.max()
        if mel_max - mel_min == 0:
            return None
        mel_spec_norm = (mel_spec_resized - mel_min) / (mel_max - mel_min)

        return mel_spec_norm

    except Exception as exc:
        print(f"  [ERROR] {file_path}: {exc}")
        return None


# ─── Dataset Loading ──────────────────────────────────────────────────────────

def load_dataset():
    """
    Walk DATA_DIR, extract mel-spectrograms for every .wav file, and return
    (X, y) where X has shape (N, 128, 128, 1) and y contains integer genre IDs.
    """
    features, labels = [], []
    total_files = 0

    print("=" * 60)
    print("Loading GTZAN dataset …")
    print("=" * 60)

    for genre in GENRES:
        genre_path = os.path.join(DATA_DIR, genre)
        if not os.path.isdir(genre_path):
            print(f"  [WARN] Directory not found: {genre_path}")
            continue

        wav_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        print(f"\n  Genre: {genre:12s} — {len(wav_files)} files")

        for wav_file in wav_files:
            file_path = os.path.join(genre_path, wav_file)
            spec = extract_melspectrogram(file_path)
            if spec is not None:
                features.append(spec)
                labels.append(genre)
                total_files += 1

    print(f"\nTotal samples loaded: {total_files}")

    if total_files == 0:
        raise RuntimeError(
            "No audio files were loaded. "
            "Make sure data/genres/<genre>/*.wav exists."
        )

    # Stack and add channel dimension → (N, 128, 128, 1)
    X = np.array(features)[..., np.newaxis]
    y_raw = np.array(labels)

    return X, y_raw


# ─── CNN Architecture ─────────────────────────────────────────────────────────

def build_cnn(num_classes: int, input_shape=(128, 128, 1)) -> Sequential:
    """
    Build the CNN classifier.
    Architecture:
        Conv2D(32) → BN → MaxPool
        Conv2D(64) → BN → MaxPool
        Conv2D(128) → BN → MaxPool
        Flatten
        Dense(128) + Dropout(0.5)
        Dense(num_classes, softmax)
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=input_shape, name='conv1'),
        BatchNormalization(),
        MaxPooling2D((2, 2), name='pool1'),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        BatchNormalization(),
        MaxPooling2D((2, 2), name='pool2'),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        BatchNormalization(),
        MaxPooling2D((2, 2), name='pool3'),

        # Classifier head
        Flatten(name='flatten'),
        Dense(128, activation='relu', name='dense1'),
        Dropout(0.5, name='dropout'),
        Dense(num_classes, activation='softmax', name='output'),
    ], name='MusicGenreCNN')

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    # 1 — Load data
    X, y_raw = load_dataset()

    # 2 — Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    y_cat     = to_categorical(y_encoded)
    num_classes = len(le.classes_)

    print(f"\nClasses ({num_classes}): {le.classes_}")

    # 3 — Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y_encoded
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # 4 — Build model
    model = build_cnn(num_classes)
    model.summary()

    # 5 — Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

    # 6 — Fit
    print("\nStarting training …")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # 7 — Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✔ Test Accuracy : {acc * 100:.2f}%")
    print(f"✔ Test Loss     : {loss:.4f}")

    # 8 — Save artefacts
    model.save(MODEL_SAVE_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"\n✔ Model   saved → {MODEL_SAVE_PATH}")
    print(f"✔ Encoder saved → {ENCODER_PATH}")

    # 9 — Plot training curves
    _plot_history(history)

    return history, acc


def _plot_history(history):
    """Save training/validation accuracy & loss plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    plot_path = os.path.join(os.path.dirname(__file__), 'training_history.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"✔ Plot saved    → {plot_path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Set seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    train()
