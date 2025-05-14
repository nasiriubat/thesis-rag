import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///chatbot.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    APP_NAME = os.getenv('APP_NAME', 'ChatBot')

    @staticmethod
    def get_default_settings() -> list:
        """Get default application settings."""
        return [
            ('logo', Config.APP_NAME),
            ('copyright', f'© {datetime.now().year} {Config.APP_NAME}. All rights reserved.'),
            ('about', 'Welcome to our AI-powered chatbot service.'),
            ('contact', 'Contact us for any questions or support.'),
            ('openai_key', Config.OPENAI_API_KEY or '')
        ]

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    # Add production-specific settings here
    # SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 