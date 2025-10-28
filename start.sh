#!/bin/sh

if [ -f /data/chatbot.db ]; then echo "Database file found, skip initialize"; \
else echo "Creating database..."; python init_db.py; \
fi

python app.py