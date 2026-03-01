#!/bin/sh
# Run the app locally (development). Use: ./run-local.sh

set -e
cd "$(dirname "$0")"

# Use venv if present
if [ -d "venv" ]; then
  . venv/bin/activate
fi

# Create DB and default admin if needed
if [ ! -f "chatbot.db" ]; then
  echo "Creating database and default admin..."
  python init_db.py
fi

# Optional: run migrations (adds new columns if any)
if command -v flask >/dev/null 2>&1 && [ -d "migrations" ]; then
  export FLASK_APP=app.py
  flask db upgrade 2>/dev/null || true
fi

PORT="${PORT:-4001}"
echo "Starting app at http://127.0.0.1:$PORT"
echo "Admin: http://127.0.0.1:$PORT/admin  (default: admin@example.com / 123456)"
python app.py
