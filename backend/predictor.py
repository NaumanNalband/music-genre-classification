"""
predictor.py — Audio Prediction Engine
Loads the trained CNN model and label encoder, extracts mel-spectrograms from
uploaded WAV files, and returns the predicted genre with confidence scores.
"""

import os
import json
import numpy as np
import librosa
import joblib
import tensorflow as tf

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR     = os.path.join(BASE_DIR, '..', 'model')
MODEL_PATH    = os.path.join(MODEL_DIR, 'genre_classifier.keras')
ENCODER_PATH  = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# ─── Audio Config (must match train_model.py) ─────────────────────────────────
SAMPLE_RATE  = 22050
DURATION     = 30
N_MELS       = 128
HOP_LENGTH   = 512
N_FFT        = 2048
TARGET_SHAPE = (128, 128)


class GenrePredictor:
    """
    Singleton-style predictor.
    Loads the model once when the Flask app starts; reuses it for every request.
    """

    _instance = None   # module-level cache

    def __init__(self):
        self.model   = None
        self.encoder = None
        self._loaded = False

    # ── Public interface ──────────────────────────────────────────────────────

    def load(self):
        """Load model + encoder from disk (called once at startup)."""
        if self._loaded:
            return

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run model/train_model.py first."
            )
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(
                f"Label encoder not found at {ENCODER_PATH}. "
                "Run model/train_model.py first."
            )

        print("  [Predictor] Loading model …")
        self.model   = tf.keras.models.load_model(MODEL_PATH)
        self.encoder = joblib.load(ENCODER_PATH)
        self._loaded = True
        print("  [Predictor] Model ready ✔")

    def predict(self, file_path: str) -> dict:
        """
        Predict the genre of an audio file.

        Parameters
        ----------
        file_path : str   absolute path to a .wav file

        Returns
        -------
        dict  {
            'genre'            : str,   # top predicted genre
            'confidence'       : float, # 0–1
            'all_probabilities': dict   # {genre: probability}
        }
        """
        if not self._loaded:
            self.load()

        # 1 — Extract features
        spec = self._extract_features(file_path)
        if spec is None:
            raise ValueError(
                "Could not extract features from the audio file. "
                "Ensure it is a valid WAV file of at least 1 second."
            )

        # 2 — Shape → (1, 128, 128, 1) for the CNN
        spec_input = spec[np.newaxis, ..., np.newaxis]

        # 3 — Inference
        probs = self.model.predict(spec_input, verbose=0)[0]   # shape (10,)

        # 4 — Decode
        top_idx    = int(np.argmax(probs))
        top_genre  = self.encoder.classes_[top_idx]
        confidence = float(probs[top_idx])

        all_probs = {
            genre: round(float(prob), 4)
            for genre, prob in zip(self.encoder.classes_, probs)
        }

        return {
            'genre':             top_genre,
            'confidence':        confidence,
            'all_probabilities': all_probs,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_features(file_path: str):
        """Extract a normalised mel-spectrogram from a WAV file."""
        try:
            import soundfile as sf
            try:
                y, sr_orig = sf.read(file_path, always_2d=False)
            except Exception:
                y, sr_orig = librosa.load(file_path, sr=None, mono=True, duration=DURATION)

            if y.ndim > 1:
                y = y.mean(axis=1).astype(np.float32)
            else:
                y = y.astype(np.float32)

            if sr_orig != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=int(sr_orig), target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

            target_len = SAMPLE_RATE * DURATION
            if len(y) > target_len:
                y = y[:target_len]

            if len(y) < sr:
                return None

            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_resized = np.resize(mel_db, TARGET_SHAPE)

            mn, mx = mel_resized.min(), mel_resized.max()
            if mx - mn == 0:
                return None
            return (mel_resized - mn) / (mx - mn)

        except Exception as exc:
            print(f"  [Predictor] Feature extraction error: {exc}")
            return None


# ─── Module-level singleton ───────────────────────────────────────────────────
predictor = GenrePredictor()
