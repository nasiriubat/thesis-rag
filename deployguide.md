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
Update after pushing to github ():
chmod +x update.shjust file update no db upgrade

# Run the update script
./update.sh
===========================
Update after pushing to github:
chmod +x git.sh

# Run the update script
./git.sh

