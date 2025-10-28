import os
from flask import Flask
from flask_login import LoginManager, current_user
from flask_migrate import Migrate
import openai
from config import config
from models import db

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.middleware.proxy_fix import ProxyFix

# Initialize login manager
login_manager = LoginManager()
login_manager.login_view = 'admin_login'

def create_app(config_name='default'):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    behindProxy = bool(os.getenv('BEHIND_PROXY', False))
    if behindProxy == True:
        # Apply ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Initialize extensions with app
    db.init_app(app)
    migrate = Migrate(app, db)  # Initialize Flask-Migrate
    login_manager.init_app(app)
    
    # Import models after db initialization
    from models import User, Role, FAQ, File, Query, Settings, NewQuestion
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Context processor for settings
    @app.context_processor
    def inject_settings():
        settings = {s.key: s.value for s in Settings.query.all()}
        return dict(settings=settings)
    
    # Update OpenAI API key when settings change
    @app.before_request
    def update_openai_key():
        if current_user.is_authenticated:
            openai.api_key = Settings.query.filter_by(key='openai_key').first().value
    
    # Import and register blueprints
    from routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app

if __name__ == '__main__':
    port=int(os.getenv('PORT', 4001))
    basepath=os.getenv('BASEPATH', "/")
    app = create_app(os.getenv('FLASK_ENV', 'production'))

    application = DispatcherMiddleware(Flask('dummy'), {
        basepath: app
    })
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', port, application)
