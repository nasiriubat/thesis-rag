from app import create_app
from models import db, User, Role, Settings
from config import Config

# Create app instance
app = create_app()

# Initialize database
with app.app_context():
    # Create all tables
    db.create_all()
    
    # Check if admin user exists
    admin = User.query.filter_by(email='admin@gmail.com').first()
    if not admin:
        # Create admin role if it doesn't exist
        admin_role = Role.query.filter_by(name='admin').first()
        if not admin_role:
            admin_role = Role(name='admin')
            db.session.add(admin_role)
            db.session.commit()
        
        # Create admin user
        admin = User(
            name='Admin',
            email='admin@gmail.com',
            role_id=admin_role.id
        )
        admin.set_password('123456')
        db.session.add(admin)
    
    # Add default settings if they don't exist
    default_settings = Config.get_default_settings()
    for key, value in default_settings:
        setting = Settings.query.filter_by(key=key).first()
        if not setting:
            setting = Settings(key=key, value=value)
            db.session.add(setting)
    
    db.session.commit()
    print("Database initialized successfully!") 