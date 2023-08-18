#!/bin/bash

/usr/bin/docker compose \
  -f deployment/foldy-in-a-boxi/docker-compose.yml \
  --project-directory . \
  down

/opt/deeplearning/install-driver.sh