from app import create_app
from models import db, User, Role, Settings, KnowledgeGraph
from config import Config
from sqlalchemy import inspect

# Create app instance
app = create_app()

# Initialize database
with app.app_context():
    # Check if knowledge_graph table exists
    inspector = inspect(db.engine)
    existing_tables = inspector.get_table_names()
    
    # Create all tables (this will skip existing tables)
    db.create_all()
    
    # If knowledge_graph table didn't exist, log it
    if 'knowledge_graph' not in existing_tables:
        print("Created knowledge_graph table")
    else:
        print("knowledge_graph table already exists, skipping creation")
    
    # Check if admin user exists
    admin = User.query.filter_by(email='admin@example.com').first()
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
            email='admin@example.com',
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
    
    # Add retrieval_method setting if it doesn't exist
    retrieval_setting = Settings.query.filter_by(key="retrieval_method").first()
    if not retrieval_setting:
        retrieval_setting = Settings(key="retrieval_method", value="Vector")
        db.session.add(retrieval_setting)
    
    db.session.commit()
    print("Database initialized successfully!") 