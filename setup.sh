#!/bin/bash
set -e
echo "🚀 Starting Flask app setup on Ubuntu..."

# Update and install required packages
# Ensure your server is up to date and has necessary build tools
# sudo apt update && sudo apt upgrade -y
# sudo apt install -y python3 python3-venv python3-pip npm git
# Note: nodejs might be needed if npm is not installed as a standalone package or if pm2 has other node dependencies.
# If you encounter issues with npm, try: sudo apt install -y nodejs

# Install pm2 globally if not already installed
# sudo npm install -g pm2

echo "🐍 Setting up Python backend..."
# Create virtual environment in the project root
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "🔑 Initializing database..."
venv/bin/python init_db.py

# Database migrations (if you use Flask-Migrate)
# You might want to run migrations here, e.g.:
# flask db upgrade
# Consider your deployment strategy for the database.
# The original script noted: "# flask db upgrade # this will not work rather upload the instance folder directly from local to server"
# Ensure your 'instance' folder or database is correctly set up on the server.

echo "🚀 Starting Flask app with PM2..."

pm2 start venv/bin/python --name thesis-rag -- app.py

echo "💾 Saving PM2 process list..."
pm2 save

echo "🚀 Enabling PM2 startup on boot..."
# This will generate a command you need to run with sudo privileges.
# Example: sudo env PATH=$PATH:/usr/bin /usr/local/lib/node_modules/pm2/bin/pm2 startup systemd -u your_user --hp /home/your_user
pm2 startup

echo "✅ Setup complete. Your Flask application 'thesis-rag' should be running on port 8080 and managed by PM2."
echo "👉 You might need to run the command output by 'pm2 startup' manually if this is the first time."
echo "👀 Monitor your app with: pm2 list"
echo "🪵 View logs with: pm2 logs thesis-rag"
