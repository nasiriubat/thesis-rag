🔧 Step-by-Step:
Make sure Node.js and npm are installed first:

sudo apt update
sudo apt install -y nodejs npm

Then install PM2 globally:

sudo npm install -g pm2

====================================
First time setup:
chmod +x setup.sh

# Run the setup script
./setup.sh
=============================
Update after pushing to github (includes Zotero and other updates):
chmod +x update.sh

# Run the update script (pulls code, installs deps, runs DB migrations, restarts app)
./update.sh
# Note: Zotero uses the existing "requests" package (no new pip packages). 
# The migration adds File.source and File.external_id; update.sh runs "flask db upgrade" so you're covered.
===========================
Update after pushing to github:
chmod +x git.sh

# Run the update script
./git.sh

