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

GOOGLE_CLOUD_PROJECT_ID=$(yq eval '.GoogleProjectId' deployment/helm/values.yaml -e)
GOOGLE_CLOUD_REGION=$(yq eval '.GoogleCloudRegion' deployment/helm/values.yaml -e)
GOOGLE_CLOUD_ZONE=$(yq eval '.GoogleCloudZone' deployment/helm/values.yaml -e)
GOOGLE_CLOUD_ARTIFACT_REPO=$(yq eval '.ArtifactRepo' deployment/helm/values.yaml -e)
VERSION=$(yq eval '.ImageVersion' deployment/helm/values.yaml -e)
BACKEND_URL="https://$(yq eval '.FoldyDomain' deployment/helm/values.yaml -e)"
INSTITUTION=$(yq eval '.Institution' deployment/helm/values.yaml -e)

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
# TODO: delete
# echo "  FOLDY ALPHAFOLD TAG: $FOLDY_ALPHAFOLD_TAG"

FRONTEND_TAG=$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT_ID/$GOOGLE_CLOUD_ARTIFACT_REPO/frontend:$VERSION
BACKEND_TAG=$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT_ID/$GOOGLE_CLOUD_ARTIFACT_REPO/backend:$VERSION
WORKER_TAG=$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT_ID/$GOOGLE_CLOUD_ARTIFACT_REPO/worker:$VERSION
WORKER_ESM_TAG=$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT_ID/$GOOGLE_CLOUD_ARTIFACT_REPO/worker_esm:$VERSION
WORKER_BOLTZ_TAG=$GOOGLE_CLOUD_REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT_ID/$GOOGLE_CLOUD_ARTIFACT_REPO/worker_boltz:$VERSION

# TODO: delete
# echo "Building Alphafold (required for worker)..."
# DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $FOLDY_ALPHAFOLD_TAG -f worker/alphafold/docker/Dockerfile worker/alphafold 
# # docker tag FOLDY_ALPHAFOLD_TAG_OLD FOLDY_ALPHAFOLD_TAG

echo "Building backend..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $BACKEND_TAG backend
echo "Building worker..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $WORKER_TAG -f worker/Dockerfile \
  .

# TODO: delete
#   --build-arg FOLDY_ALPHAFOLD_TAG=$FOLDY_ALPHAFOLD_TAG \
  
echo "Building worker ESM..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $WORKER_ESM_TAG -f worker/Dockerfile.esm \
  .
echo "Building worker BOLTZ..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $WORKER_BOLTZ_TAG -f worker/Dockerfile.boltz \
  .
echo "Building frontend..."
DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $FRONTEND_TAG \
  --build-arg BACKEND_URL=$BACKEND_URL \
  --build-arg INSTITUTION=$INSTITUTION \
  frontend

docker push $BACKEND_TAG
docker push $WORKER_TAG
docker push $WORKER_ESM_TAG
docker push $WORKER_BOLTZ_TAG
docker push $FRONTEND_TAG

# TODO: delete  
# if [ "$#" -eq 1 ]; then
#   docker push $FOLDY_ALPHAFOLD_TAG
# fi

