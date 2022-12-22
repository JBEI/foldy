#!/bin/bash

set -e

if [ "$#" -ne 3 ]; then
  echo "Illegal number of parameters.\n\nExample invokation:\n  build_and_deploy_containers.sh <gcloud project ID> <gcloud artifact repo> <version number>"
  exit 2
fi

GCLOUD_PROJECT_ID=$1
GCLOUD_ARTIFACT_REPO=$2
VERSION=$3

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
docker build -t $FRONTEND_IMAGE frontend

docker push $BACKEND_IMAGE &&
docker push $WORKER_IMAGE &&
docker push $FRONTEND_IMAGE
