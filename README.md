# 🎵 Music Genre Classification System

> A Deep Learning web application that identifies music genres from audio files using a Convolutional Neural Network (CNN) trained on the GTZAN dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=flat-square&logo=flask)
![SQLite](https://img.shields.io/badge/SQLite-3-blue?style=flat-square&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Folder Structure](#folder-structure)
- [CNN Architecture](#cnn-architecture)
- [API Documentation](#api-documentation)
- [Setup Instructions](#setup-instructions)
- [Screenshots](#screenshots)
- [Dataset](#dataset)

---

## Overview

This project classifies music audio files into 10 genres:

| Genre | Genre | Genre |
|-------|-------|-------|
| 🎸 Blues | 🎻 Classical | 🤠 Country |
| 🕺 Disco | 🎤 Hip-Hop | 🎷 Jazz |
| 🤘 Metal | 🎶 Pop | 🌴 Reggae |
| 🎸 Rock | | |

The system extracts **Mel Spectrograms** from audio files, feeds them into a trained **CNN model**, and returns a predicted genre with a confidence score.

---

## Features

- **User Authentication** — JWT-based register/login
- **Audio Upload** — WAV, MP3, OGG, FLAC support
- **Real-Time Prediction** — Genre + confidence in seconds
- **Audio Preview Player** — Listen before predicting
- **Prediction History** — Paginated table of all past predictions
- **Dashboard Analytics** — Stats at a glance
- **Statistics Charts** — Doughnut + bar charts (Chart.js)
- **Dark UI** — Neon-industrial aesthetic with responsive sidebar

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Deep Learning | TensorFlow / Keras |
| Audio Processing | Librosa |
| Backend | Flask, Flask-SQLAlchemy, Flask-CORS |
| Database | SQLite |
| Auth | JWT (PyJWT) + bcrypt |
| Frontend | HTML5, CSS3, Vanilla JS |
| Charts | Chart.js 4 |

---

## Folder Structure

```
music-genre-classification/
├── model/
│   ├── train_model.py        ← CNN training script
│   ├── genre_classifier.keras← Saved model (after training)
│   ├── label_encoder.pkl     ← Saved label encoder
│   └── training_history.png  ← Loss/accuracy plot
│
├── backend/
│   ├── app.py                ← Flask application & routes
│   ├── models.py             ← SQLAlchemy ORM (User, Prediction)
│   ├── predictor.py          ← Inference engine
│   ├── database.db           ← SQLite database (auto-created)
│   └── uploads/              ← Uploaded audio files
│
├── data/
│   └── genres/
│       ├── blues/            ← 100 × 30s WAV files
│       ├── classical/
│       ├── country/
│       ├── disco/
│       ├── hiphop/
│       ├── jazz/
│       ├── metal/
│       ├── pop/
│       ├── reggae/
│       └── rock/
│
├── frontend/
│   ├── index.html            ← Login page
│   ├── register.html         ← Register page
│   ├── dashboard.html        ← Dashboard
│   ├── upload.html           ← Upload & predict
│   ├── history.html          ← Prediction history
│   ├── stats.html            ← Statistics & charts
│   ├── style.css             ← Global stylesheet
│   └── script.js             ← Shared JS utilities
│
├── requirements.txt
└── README.md
```

---

## CNN Architecture

```
Input: (128, 128, 1)  ← normalised mel-spectrogram

Conv2D(32, 3×3, relu)
BatchNormalization
MaxPooling2D(2×2)

Conv2D(64, 3×3, relu)
BatchNormalization
MaxPooling2D(2×2)

Conv2D(128, 3×3, relu)
BatchNormalization
MaxPooling2D(2×2)

Flatten
Dense(128, relu)
Dropout(0.5)
Dense(10, softmax)         ← 10 genre classes
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Epochs | 20 (EarlyStopping) |
| Batch Size | 32 |
| Val Split | 20% |
| Input Shape | 128 × 128 × 1 |

---

## API Documentation

### Auth

#### `POST /api/register`
```json
// Request
{ "username": "alice", "email": "alice@example.com", "password": "secret123" }

// Response 201
{ "message": "Account created", "token": "<jwt>", "user": { ... } }
```

#### `POST /api/login`
```json
// Request
{ "username": "alice", "password": "secret123" }

// Response 200
{ "token": "<jwt>", "user": { "id": 1, "username": "alice", ... } }
```

### Predictions

All prediction endpoints require:
```
Authorization: Bearer <jwt>
```

#### `POST /api/upload`
- Content-Type: `multipart/form-data`
- Field: `audio` (file)
```json
// Response 200
{ "message": "File uploaded successfully", "stored_filename": "abc123_track.wav" }
```

#### `POST /api/predict`
```json
// Request
{ "stored_filename": "abc123_track.wav", "original_name": "my_track.wav" }

// Response 200
{
  "genre": "jazz",
  "confidence": 91.23,
  "all_probabilities": { "blues": 1.2, "jazz": 91.23, "rock": 4.1, ... },
  "prediction_id": 42
}
```

#### `GET /api/history?page=1&per_page=20`
```json
{
  "predictions": [ { "id": 1, "filename": "...", "predicted_genre": "jazz", ... } ],
  "total": 42,
  "page": 1,
  "pages": 3
}
```

#### `GET /api/stats`
```json
{
  "total_predictions": 42,
  "most_predicted": "jazz",
  "average_confidence": 87.5,
  "genre_distribution": { "jazz": 18, "rock": 12, ... },
  "recent_predictions": [ ... ]
}
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the GTZAN Dataset

Download from: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Extract and place the genre folders under:
```
data/genres/blues/
data/genres/classical/
... etc.
```

Each folder should contain **100 WAV files** of 30 seconds each.

### 5. Train the Model

```bash
cd model
python train_model.py
```

This will:
- Extract mel-spectrograms from all audio files
- Train the CNN for up to 20 epochs
- Save `genre_classifier.keras` and `label_encoder.pkl`
- Generate `training_history.png`

Training takes approximately **5–20 minutes** depending on hardware.

### 6. Start the Flask Backend

```bash
cd backend
python app.py
```

The server starts at `http://localhost:5000`

### 7. Open the Frontend

The Flask app serves the frontend directly. Open your browser:

```
http://localhost:5000
```

Or open `frontend/index.html` directly in your browser while the Flask server runs.

---

## Environment Variables (Optional)

```bash
export SECRET_KEY="your-super-secret-key"   # JWT signing key
```

---

## Screenshots

### Login Page
Dark auth card with logo, username/password fields, and animated background glow.

### Dashboard
4 stat cards (total predictions, top genre, avg. confidence, model type), waveform animation, and recent predictions table.

### Upload & Predict
Drag-and-drop zone with audio preview player. After prediction: large genre display with confidence bar and all-genre probability breakdown.

### History
Paginated table with colour-coded genre badges and mini confidence bars.

### Statistics
Doughnut chart (genre distribution) + bar chart (count by genre), both styled for the dark theme.

---

## Sample Accuracy

Expected performance on GTZAN (80/20 split):

| Metric | Value |
|--------|-------|
| Training Accuracy | ~88% |
| Validation Accuracy | ~72–78% |

*(Accuracy varies by epoch count, hardware, and data augmentation)*

---

## Dataset

**GTZAN Music Genre Dataset**
- 10 genres × 100 tracks × 30 seconds = 1,000 total files
- Sample rate: 22,050 Hz, Mono, WAV format
- Originally compiled by George Tzanetakis

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

Third-Year Computer Science Student  
Project: Deep Learning — Music Genre Classification  
Built with ❤️ using TensorFlow, Flask, and Librosa
