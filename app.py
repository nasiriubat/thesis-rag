import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'admin_login'

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    phone = db.Column(db.String(20))
    role_id = db.Column(db.Integer, db.ForeignKey('role.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Role(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    users = db.relationship('User', backref='role', lazy=True)

class FAQ(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    ai = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    answer_found = db.Column(db.Boolean, default=False)
    happy = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Context processor for settings
@app.context_processor
def inject_settings():
    settings = {s.key: s.value for s in Settings.query.all()}
    return dict(settings=settings)

# Create database tables and initial data
with app.app_context():
    db.create_all()
    
    # Create initial admin user if not exists
    if not User.query.filter_by(email='admin@gmail.com').first():
        admin = User(
            name='Admin',
            email='admin@gmail.com',
            role_id=1
        )
        admin.set_password('123456')
        db.session.add(admin)
        
        # Create admin role
        admin_role = Role(name='admin')
        db.session.add(admin_role)
        
        # Create default settings
        default_settings = [
            ('logo', os.getenv('APP_NAME', 'ChatBot')),
            ('copyright', f'© {datetime.now().year} {os.getenv("APP_NAME", "ChatBot")}. All rights reserved.'),
            ('about', 'Welcome to our AI-powered chatbot service.'),
            ('contact', 'Contact us for any questions or support.'),
            ('openai_key', os.getenv('OPENAI_API_KEY', ''))
        ]
        for key, value in default_settings:
            if not Settings.query.filter_by(key=key).first():
                setting = Settings(key=key, value=value)
                db.session.add(setting)
        
        db.session.commit()

# Update OpenAI API key when settings change
@app.before_request
def update_openai_key():
    if current_user.is_authenticated:
        openai.api_key = Settings.query.filter_by(key='openai_key').first().value

# Import routes after models to avoid circular imports
from routes import *

if __name__ == '__main__':
    app.run(debug=True) 