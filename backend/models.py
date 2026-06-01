"""
models.py — SQLAlchemy ORM Models
Tables : User, Prediction
"""

from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

# Shared db instance — imported and initialised in app.py
db = SQLAlchemy()


class User(db.Model):
    """
    Stores registered users.
    password_hash is a bcrypt hash — the raw password is never stored.
    """
    __tablename__ = 'users'

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship: one user → many predictions
    predictions   = db.relationship('Prediction', backref='user', lazy=True,
                                    cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id':         self.id,
            'username':   self.username,
            'email':      self.email,
            'created_at': self.created_at.isoformat(),
        }

    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    """
    Stores every prediction request made by a user.
    confidence_score is a float between 0 and 1 (e.g. 0.93 = 93 %).
    """
    __tablename__ = 'predictions'

    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename         = db.Column(db.String(256), nullable=False)
    predicted_genre  = db.Column(db.String(50),  nullable=False)
    confidence_score = db.Column(db.Float,        nullable=False)
    # JSON string of all genre probabilities, e.g. '{"blues":0.02,"rock":0.93,...}'
    all_probabilities = db.Column(db.Text, nullable=True)
    prediction_time  = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        import json
        probs = {}
        if self.all_probabilities:
            try:
                probs = json.loads(self.all_probabilities)
            except Exception:
                pass
        return {
            'id':               self.id,
            'user_id':          self.user_id,
            'filename':         self.filename,
            'predicted_genre':  self.predicted_genre,
            'confidence_score': round(self.confidence_score * 100, 2),   # percentage
            'all_probabilities': probs,
            'prediction_time':  self.prediction_time.isoformat(),
        }

    def __repr__(self):
        return f'<Prediction {self.filename} → {self.predicted_genre}>'
