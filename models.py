from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

# Initialize DB instance
db = SQLAlchemy()

# User model with admin support
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    phone = db.Column(db.String(20))
    role_id = db.Column(db.Integer, db.ForeignKey('role.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    files = db.relationship('File', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Role model for user roles
class Role(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    users = db.relationship('User', backref='role', lazy=True)

# Frequently Asked Questions
class FAQ(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    embedding = db.Column(db.Text,nullable=True)  # Store JSON-encoded embedding string

# Uploaded file details
class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    embedding = db.Column(db.Text)  # Store the embedding for semantic search
    file_identifier = db.Column(db.String(100))  # ID used to locate embedding files
    original_filename = db.Column(db.String(255))  # Store the uploaded or derived name

# Chat query tracking
class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=True)  # Store the generated answer
    answer_found = db.Column(db.Boolean, default=False)
    happy = db.Column(db.Boolean, default=False)
    language = db.Column(db.String(2), default='en')  # Store language code (en/fi)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Unanswered visitor questions
class NewQuestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    embedding = db.Column(db.Text,nullable=True)  # Store the embedding for semantic search

# Settings model to store site-wide configurable text fields
class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=False)
