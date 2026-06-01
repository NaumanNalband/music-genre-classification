"""
app.py — Flask Backend for Music Genre Classification System
API Endpoints:
    POST /register        — create a new account
    POST /login           — authenticate and receive a JWT
    POST /upload          — upload a WAV file
    POST /predict         — predict the genre of an uploaded file
    GET  /history         — list prediction history for the logged-in user
    GET  /stats           — genre distribution & summary statistics
    GET  /health          — simple health-check
"""

import os
import json
import uuid
from datetime import datetime, timezone, timedelta
from functools import wraps

import bcrypt
import jwt
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

from models import db, User, Prediction
from predictor import predictor

# ─── App & Config ─────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR  = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, '..', 'frontend'))
app.config.update(
    SECRET_KEY            = os.environ.get('SECRET_KEY', 'mgc-super-secret-key-change-in-prod'),
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'database.db')}",
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    MAX_CONTENT_LENGTH    = 50 * 1024 * 1024,   # 50 MB upload limit
    UPLOAD_FOLDER         = UPLOAD_DIR,
)

CORS(app, resources={r'/api/*': {'origins': '*'}})
db.init_app(app)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}

# ─── JWT Helpers ──────────────────────────────────────────────────────────────

def _encode_token(user_id: int, username: str) -> str:
    payload = {
        'user_id':  user_id,
        'username': username,
        'exp':      datetime.now(timezone.utc) + timedelta(days=7),
        'iat':      datetime.now(timezone.utc),
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')


def token_required(f):
    """Decorator that enforces a valid JWT in the Authorization header."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or malformed token'}), 401
        token = auth_header.split(' ', 1)[1]
        try:
            payload = jwt.decode(
                token, app.config['SECRET_KEY'], algorithms=['HS256']
            )
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        # Attach user info to request context
        request.current_user_id = payload['user_id']
        request.current_username = payload['username']
        return f(*args, **kwargs)
    return decorated


def _allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Routes ───────────────────────────────────────────────────────────────────

# ── Serve frontend static files ───────────────────────────────────────────────
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')

@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_frontend(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ── Health ────────────────────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now(timezone.utc).isoformat()})


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.route('/api/register', methods=['POST'])
def register():
    """
    POST /api/register
    Body: { username, email, password }
    Returns: { message, user }
    """
    data = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    email    = data.get('email', '').strip().lower()
    password = data.get('password', '')

    # Validation
    if not username or not email or not password:
        return jsonify({'error': 'username, email and password are required'}), 400
    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if '@' not in email:
        return jsonify({'error': 'Invalid email address'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken'}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409

    # Hash password with bcrypt
    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    user = User(username=username, email=email, password_hash=pw_hash)
    db.session.add(user)
    db.session.commit()

    token = _encode_token(user.id, user.username)
    return jsonify({'message': 'Account created', 'token': token, 'user': user.to_dict()}), 201


@app.route('/api/login', methods=['POST'])
def login():
    """
    POST /api/login
    Body: { username, password }
    Returns: { token, user }
    """
    data     = request.get_json(silent=True) or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    user = User.query.filter_by(username=username).first()
    if not user or not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
        return jsonify({'error': 'Invalid credentials'}), 401

    token = _encode_token(user.id, user.username)
    return jsonify({'token': token, 'user': user.to_dict()})


# ── Upload ────────────────────────────────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
@token_required
def upload():
    """
    POST /api/upload   (multipart/form-data, field: audio)
    Returns: { message, filename, stored_filename }
    """
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file in request'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not _allowed_file(file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}'}), 415

    # Unique filename to avoid collisions
    ext             = file.filename.rsplit('.', 1)[1].lower()
    safe_original   = secure_filename(file.filename)
    stored_filename = f"{uuid.uuid4().hex}_{safe_original}"
    file_path       = os.path.join(UPLOAD_DIR, stored_filename)
    file.save(file_path)

    return jsonify({
        'message':          'File uploaded successfully',
        'original_name':    safe_original,
        'stored_filename':  stored_filename,
    }), 200


# ── Predict ───────────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
@token_required
def predict():
    """
    POST /api/predict
    Body: { stored_filename }
    Returns: { genre, confidence, all_probabilities, prediction_id }
    """
    data             = request.get_json(silent=True) or {}
    stored_filename  = data.get('stored_filename', '').strip()
    original_name    = data.get('original_name', stored_filename)

    if not stored_filename:
        return jsonify({'error': 'stored_filename required'}), 400

    file_path = os.path.join(UPLOAD_DIR, stored_filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'Uploaded file not found on server'}), 404

    try:
        result = predictor.predict(file_path)
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 422
    except Exception as exc:
        app.logger.error(f"Prediction error: {exc}")
        return jsonify({'error': 'Prediction failed. Please try a different file.'}), 500

    # Persist to database
    pred_record = Prediction(
        user_id          = request.current_user_id,
        filename         = original_name,
        predicted_genre  = result['genre'],
        confidence_score = result['confidence'],
        all_probabilities = json.dumps(result['all_probabilities']),
    )
    db.session.add(pred_record)
    db.session.commit()

    return jsonify({
        'prediction_id':   pred_record.id,
        'genre':           result['genre'],
        'confidence':      round(result['confidence'] * 100, 2),
        'all_probabilities': {
            k: round(v * 100, 2) for k, v in result['all_probabilities'].items()
        },
        'filename': original_name,
    })


# ── History ───────────────────────────────────────────────────────────────────

@app.route('/api/history', methods=['GET'])
@token_required
def history():
    """
    GET /api/history?page=1&per_page=20
    Returns paginated prediction history for the logged-in user.
    """
    page     = request.args.get('page',     1,  type=int)
    per_page = request.args.get('per_page', 20, type=int)

    paginated = (
        Prediction.query
        .filter_by(user_id=request.current_user_id)
        .order_by(Prediction.prediction_time.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )

    return jsonify({
        'predictions': [p.to_dict() for p in paginated.items],
        'total':       paginated.total,
        'page':        page,
        'per_page':    per_page,
        'pages':       paginated.pages,
    })


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.route('/api/stats', methods=['GET'])
@token_required
def stats():
    """
    GET /api/stats
    Returns genre distribution and summary stats for the logged-in user.
    """
    user_preds = (
        Prediction.query
        .filter_by(user_id=request.current_user_id)
        .all()
    )

    total = len(user_preds)
    if total == 0:
        return jsonify({
            'total_predictions':  0,
            'most_predicted':     None,
            'average_confidence': 0,
            'genre_distribution': {},
            'recent_predictions': [],
        })

    # Genre distribution
    genre_counts: dict[str, int] = {}
    conf_sum = 0.0
    for p in user_preds:
        genre_counts[p.predicted_genre] = genre_counts.get(p.predicted_genre, 0) + 1
        conf_sum += p.confidence_score

    most_predicted  = max(genre_counts, key=genre_counts.get)
    avg_confidence  = round((conf_sum / total) * 100, 2)

    # Last 7 predictions (most recent first)
    recent = sorted(user_preds, key=lambda p: p.prediction_time, reverse=True)[:7]

    return jsonify({
        'total_predictions':  total,
        'most_predicted':     most_predicted,
        'average_confidence': avg_confidence,
        'genre_distribution': genre_counts,
        'recent_predictions': [p.to_dict() for p in recent],
    })


# ─── Init & Run ───────────────────────────────────────────────────────────────

with app.app_context():
    db.create_all()           # create tables if they don't exist
    predictor.load()          # pre-load the model at startup

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
