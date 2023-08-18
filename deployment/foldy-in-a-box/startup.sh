#!/bin/bash

MY_URL=`curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip`
echo "MY_URL: $MY_URL"
FOLDY_BOX_URL=$MY_URL echo "FOLDY_BOX_URL: $FOLDY_BOX_URL"

# Rebuild the images with the URL.
FOLDY_BOX_URL=$MY_URL /usr/bin/docker compose \
  -f deployment/foldy-in-a-box/docker-compose.yml \
  --project-directory . \
  build

# Start the service.
FOLDY_BOX_URL=$MY_URL /usr/bin/docker compose \
  -f deployment/foldy-in-a-box/docker-compose.yml \
  --project-directory . \
  up -d --force-recreate

# Make sure DBs are installed.
FOLDY_BOX_URL=$MY_URL /usr/bin/docker compose \
  -f deployment/foldy-in-a-box/docker-compose.yml \
  --project-directory . \
  exec backend flask db upgrade

if [ -f "/foldydbs/FINISHED" ]; then
    echo "Foldy DBs already finished downloading."
else 
    echo "Downloading Foldy Databases... might take up to 48 hours."
    /usr/bin/docker compose \
        -f deployment/foldy-in-a-box/docker-compose.yml \
        --project-directory /foldy \
        exec worker /worker/download_databases.sh
    if [ $? -eq 0 ]
    then
        echo "Finished downloading Foldy Databases."
        touch /foldydbs/FINISHED
    else
        echo "Failed to download Foldy Databases, will retry next time"
        echo "the foldy.service process starts. To restart the download"
        echo "now, you can either restart the machine or you can call"
        echo "  sudo systemctl restart foldy.service"
    fi
fi
