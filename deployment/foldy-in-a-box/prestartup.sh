#!/bin/bash

/usr/bin/docker compose \
  -f deployment/foldy-in-a-box/docker-compose.yml \
  --project-directory . \
  down

/opt/deeplearning/install-driver.sh