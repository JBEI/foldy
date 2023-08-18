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