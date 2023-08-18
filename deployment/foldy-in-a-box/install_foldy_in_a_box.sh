#!/bin/bash

cd /

echo "Downloading Foldy code."
sudo git clone http://github.com/jbei/foldy

echo "Create directories for:"
echo " * the AlphaFold, pfam, and DiffDock databases"
echo " * Foldy outputs"
echo " * The Foldy postgres database"
sudo mkdir foldydbs
sudo mkdir aftmp
sudo mkdir pgdb

echo "Install docker compose."
sudo apt-get -y update
sudo apt-get install -y docker-compose-plugin

echo "Create a service that runs Foldy on startup."
sudo systemctl link /foldy/deployment/foldy-in-a-box/foldy.service
sudo systemctl enable foldy.service

echo "Start the Foldy service..."
echo "  If this fails, you can check the service status with:"
echo "    systemctl status foldy.service"
echo "  or check the longer logs with"
echo "    sudo journalctl -u foldy.service -f"
sudo systemctl start foldy.service