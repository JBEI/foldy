#!/bin/bash

set FOLDY_BOX_URL=`curl -s -H "Metadata-Flavor: Google" http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip`
/usr/bin/docker compose -f foldy-in-a-box.yml build
/usr/bin/docker compose -f foldy-in-a-box.yml up -d --force-recreate