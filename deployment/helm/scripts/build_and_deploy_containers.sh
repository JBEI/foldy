#!/bin/bash

set -e
set -o pipefail  # Also fail if any part of a pipe fails

if [ "$#" -gt 1 ]; then
  echo "Illegal number of parameters."
  echo ""
  echo "Example invokation:"
  echo "build_and_deploy_containers.sh" # TODO: delete [optional tag for foldy alphafold]"
  exit 2
fi

VALUES_YAML_PATH="$(dirname $0)/../values.yaml"

GOOGLE_CLOUD_PROJECT_ID=$(yq eval '.GoogleProjectId' $VALUES_YAML_PATH -e)
GOOGLE_CLOUD_REGION=$(yq eval '.GoogleCloudRegion' $VALUES_YAML_PATH -e)
GOOGLE_CLOUD_ZONE=$(yq eval '.GoogleCloudZone' $VALUES_YAML_PATH -e)
GOOGLE_CLOUD_ARTIFACT_REPO=$(yq eval '.ArtifactRepo' $VALUES_YAML_PATH -e)
VERSION=$(yq eval '.ImageVersion' $VALUES_YAML_PATH -e)
BACKEND_URL="https://$(yq eval '.FoldyDomain' $VALUES_YAML_PATH -e)"
INSTITUTION=$(yq eval '.Institution' $VALUES_YAML_PATH -e)
TESELAGEN_BACKEND_URL=$(yq eval '.TeselagenBackendUrl' $VALUES_YAML_PATH)

# TODO: delete
# FOLDY_ALPHAFOLD_TAG=foldyalphafold
# if [ "$#" -eq 1 ]; then
#   FOLDY_ALPHAFOLD_TAG="$1"
# fi

echo "Using the following variables:"
echo "  GOOGLE_CLOUD_PROJECT_ID: $GOOGLE_CLOUD_PROJECT_ID"
echo "  GOOGLE_CLOUD_REGION: $GOOGLE_CLOUD_REGION"
echo "  GOOGLE_CLOUD_ZONE: $GOOGLE_CLOUD_ZONE"
echo "  GOOGLE_CLOUD_ARTIFACT_REPO: $GOOGLE_CLOUD_ARTIFACT_REPO"
echo "  VERSION: $VERSION"
echo "  BACKEND_URL: $BACKEND_URL"
echo "  INSTITUTION: $INSTITUTION"
echo "  TESELAGEN_BACKEND_URL: $TESELAGEN_BACKEND_URL"

FRONTEND_TAG=$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT_ID/$GOOGLE_CLOUD_ARTIFACT_REPO/frontend:$VERSION
BACKEND_TAG=$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT_ID/$GOOGLE_CLOUD_ARTIFACT_REPO/backend:$VERSION
WORKER_TAG=$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT_ID/$GOOGLE_CLOUD_ARTIFACT_REPO/worker:$VERSION
WORKER_ESM_TAG=$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT_ID/$GOOGLE_CLOUD_ARTIFACT_REPO/worker_esm:$VERSION
WORKER_BOLTZ_TAG=$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT_ID/$GOOGLE_CLOUD_ARTIFACT_REPO/worker_boltz:$VERSION

echo "Building backend..."
DOCKER_BUILDKIT=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $BACKEND_TAG -f backend/Dockerfile \
  .
echo "Building worker..."
DOCKER_BUILDKIT=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $WORKER_TAG -f worker/Dockerfile \
  .

echo "Building worker ESM..."
DOCKER_BUILDKIT=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $WORKER_ESM_TAG -f worker/Dockerfile.esm \
  .
echo "Building worker BOLTZ..."
DOCKER_BUILDKIT=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $WORKER_BOLTZ_TAG -f worker/Dockerfile.boltz \
  .
echo "Building frontend..."
# Build frontend with conditional Teselagen integration
if [ -n "$TESELAGEN_BACKEND_URL" ] && [ "$TESELAGEN_BACKEND_URL" != "null" ]; then
  echo "  Including Teselagen integration: $TESELAGEN_BACKEND_URL"
  DOCKER_BUILDKIT=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $FRONTEND_TAG \
    --build-arg BACKEND_URL=$BACKEND_URL \
    --build-arg INSTITUTION=$INSTITUTION \
    --build-arg TESELAGEN_BACKEND_URL=$TESELAGEN_BACKEND_URL \
    frontend
else
  echo "  Building without Teselagen integration"
  DOCKER_BUILDKIT=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $FRONTEND_TAG \
    --build-arg BACKEND_URL=$BACKEND_URL \
    --build-arg INSTITUTION=$INSTITUTION \
    frontend
fi

docker push $BACKEND_TAG
docker push $WORKER_TAG
docker push $WORKER_ESM_TAG
docker push $WORKER_BOLTZ_TAG
docker push $FRONTEND_TAG
