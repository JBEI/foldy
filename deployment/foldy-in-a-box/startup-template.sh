#!/bin/bash

VM_EXTERNAL_IP_ADDRESS=`curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip`

###############################################################################
# SET ENVIRONMENT VARIABLES.
###############################################################################

# If using an SSH tunnel to access Foldy, change this to
#   export FOLDY_BOX_URL=localhost
# If putting Foldy behind a URL, change this to that URL, eg
#   export FOLDY_BOX_URL=foldy.myinstitution.edu
export FOLDY_BOX_URL="http://$VM_EXTERNAL_IP_ADDRESS"

# Choose a random string.
export SECRET_KEY=superrandomandveryhardtoguessstring

# If enabling Google OAuth, you can change "DISABLE_OAUTH_AUTHENTICATION" to False
# and uncomment and fill out the below environment variables.
# export GOOGLE_CLIENT_ID=
# export GOOGLE_CLIENT_SECRET=
export DISABLE_OAUTH_AUTHENTICATION=True

# Name of the institution, to go on the frontend.
export INSTITUTION=Local

###############################################################################
# END SET ENVIRONMENT VARIABLES
###############################################################################

echo "Connecting to $FOLDY_BOX_URL"


# Rebuild the images with the URL.
/usr/bin/docker compose \
  --build-arg BACKEND_URL=$FOLDY_BOX_URL \
  --build-arg INSTITUTION=$INSTITUTION \
  -f deployment/foldy-in-a-box/docker-compose.yml \
  --project-directory . \
  build

# Start the service.
/usr/bin/docker compose \
  -f deployment/foldy-in-a-box/docker-compose.yml \
  --project-directory . \
  up -d --force-recreate

# Make sure DBs are installed.
/usr/bin/docker compose \
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
