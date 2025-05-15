#!/bin/bash
set -e
echo "🔄 Updating application..."

# Pull latest code
git reset --hard HEAD
git pull origin main


# Backend update
source venv/bin/activate
pip install -r requirements.txt
flask db upgrade
pm2 restart thesis-rag


echo "✅ App updated and restarted!"
