#!/bin/bash

set -e
set -o pipefail

# LBL-specific configuration
export BACKEND_URL=https://foldy.lbl.gov
export INSTITUTION=LBL

# Call the local build script with lbnl-latest tag
./deployment/local/build_and_deploy_containers.sh lbnl-latest "$@"
