#!/bin/bash

MY_URL=`curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip`
echo "MY_URL: $MY_URL"
FOLDY_BOX_URL=$MY_URL echo "FOLDY_BOX_URL: $FOLDY_BOX_URL"
FOLDY_BOX_URL=$MY_URL /usr/bin/docker compose -f foldy-in-a-box.yml build
FOLDY_BOX_URL=$MY_URL /usr/bin/docker compose -f foldy-in-a-box.yml up -d --force-recreate