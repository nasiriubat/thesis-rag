# Database Management Guide

This guide provides step-by-step instructions for common database operations in the RAG Chatbot project.

## Prerequisites
- Make sure you have activated your virtual environment:
  ```bash
  # Windows
  .\venv\Scripts\activate
  
  # Linux/Mac
  source venv/bin/activate
  ```

## 1. Initialize Database with Demo User

### Fresh Start (Empty Database)
```bash
# 1. Delete existing database (if it exists)
# First ensure Flask app is not running
rm instance/chatbot.db

# 2. Initialize migrations (first time only)
python -m flask db init

# 3. Apply all migrations
python -m flask db stamp head
python -m flask db upgrade

# 4. Initialize with demo data
python init_db.py
```

This will create:
- Admin user (email: admin@gmail.com, password: 123456)
- Initial settings
- Required database tables

## 2. Adding a New Column

1. First, modify the model in `models.py` with your new column:
```python
class YourModel(db.Model):
    # ... existing columns ...
    new_column = db.Column(db.String(100), nullable=True)  # Add your new column
```

2. Create and apply the migration:
```bash
# Create migration
python -m flask db migrate -m "Add new_column to YourModel"

# Apply migration
python -m flask db upgrade
```

## 3. Emptying Database (Preserving User Table)

1. Create a cleanup script `clean_db.py`:
```python
from app import create_app
from models import db, FAQ, File, Query, NewQuestion

app = create_app()

with app.app_context():
    # Delete all records except users and roles
    NewQuestion.query.delete()
    Query.query.delete()
    File.query.delete()
    FAQ.query.delete()
    db.session.commit()
```

2. Run the cleanup:
```bash
python clean_db.py
```

## 4. Deleting a Column

1. First, modify the model in `models.py` to remove the column you want to delete.

2. Create and apply the migration:
```bash
# Create migration
python -m flask db migrate -m "Remove column_name from ModelName"

# Apply migration
python -m flask db upgrade
```

## 5. Common Issues and Solutions

### Database is Locked
If you get "database is locked" error:
1. Stop the Flask application
2. Make sure no other process is using the database
3. Try operations again

### Migration Issues
If migrations fail:
```bash
# Reset migration state
python -m flask db stamp head

# Remove migration files (if needed)
rm -r migrations/

# Reinitialize migrations
python -m flask db init
python -m flask db migrate -m "Initial migration"
python -m flask db upgrade
```

### Data Backup
Before major operations, backup your database:
```bash
# Copy database file
cp instance/chatbot.db instance/chatbot_backup.db
```

## 6. Database Structure Reference

### Main Tables
- `user` - User accounts and authentication
- `role` - User roles and permissions
- `faq` - Frequently asked questions
- `file` - Uploaded documents
- `query` - Chat history and user interactions
- `new_question` - Unanswered questions for review
- `settings` - Application settings

### Important Columns
- `file.file_identifier` - Links to embeddings files
- `query.answer` - Stores GPT-generated responses
- `faq.embedding` - Stores semantic search data

## 7. Best Practices

1. Always backup before migrations
2. Test migrations in development first
3. Make migrations reversible when possible
4. Document database changes
5. Keep track of migration files

## 8. Troubleshooting

### Reset Everything (Complete Fresh Start)
```bash
# 1. Stop Flask application
# 2. Delete database and migrations
rm instance/chatbot.db
rm -r migrations/

# 3. Reinitialize everything
python -m flask db init
python -m flask db migrate -m "Initial migration"
python -m flask db upgrade
python init_db.py
```

### Fix Inconsistent Migration State
```bash
# Reset migration head
python -m flask db stamp head

# Reapply migrations
python -m flask db upgrade
``` 