#!/bin/bash

set -e

if [ "$#" -ne 0 ]; then
  echo "Illegal number of parameters."
  echo ""
  echo "Example invokation:"
  echo "build_and_deploy_containers.sh"
  exit 2
fi

GCLOUD_PROJECT_ID=$(yq eval '.GoogleProjectId' foldy/values.yaml -e)
GCLOUD_ARTIFACT_REPO=$(yq eval '.ArtifactRepo' foldy/values.yaml -e)
VERSION=$(yq eval '.ImageVersion' foldy/values.yaml -e)
BACKEND_URL="https://$(yq eval '.FoldyDomain' foldy/values.yaml -e)"
INSTITUTION=$(yq eval '.Institution' foldy/values.yaml -e)

echo "Using the following variables:"
echo "  GCLOUD_PROJECT_ID: $GCLOUD_PROJECT_ID"
echo "  GCLOUD_ARTIFACT_REPO: $GCLOUD_ARTIFACT_REPO"
echo "  VERSION: $VERSION"
echo "  BACKEND_URL: $BACKEND_URL"
echo "  INSTITUTION: $INSTITUTION"

FRONTEND_IMAGE=us-central1-docker.pkg.dev/$GCLOUD_PROJECT_ID/$GCLOUD_ARTIFACT_REPO/frontend:$VERSION
BACKEND_IMAGE=us-central1-docker.pkg.dev/$GCLOUD_PROJECT_ID/$GCLOUD_ARTIFACT_REPO/backend:$VERSION
WORKER_IMAGE=us-central1-docker.pkg.dev/$GCLOUD_PROJECT_ID/$GCLOUD_ARTIFACT_REPO/worker:$VERSION

echo "Building Alphafold (required for worker)..."
docker build -t foldyalphafold -f worker/alphafold/docker/Dockerfile worker/alphafold 
echo "Building backend..."
docker build -t $BACKEND_IMAGE backend
echo "Building worker..."
docker build -t $WORKER_IMAGE -f worker/Dockerfile .
echo "Building frontend..."
docker build -t $FRONTEND_IMAGE \
  --build-arg BACKEND_URL=$BACKEND_URL \
  --build-arg INSTITUTION=$INSTITUTION \
  frontend

docker push $BACKEND_IMAGE &&
docker push $WORKER_IMAGE &&
docker push $FRONTEND_IMAGE
