#!/bin/bash

set -e
set -o pipefail  # Also fail if any part of a pipe fails

if [ "$#" -eq 0 ]; then
  VERSIONS=("latest")
else
  VERSIONS=("$@")
fi

# DockerHub configuration - change these to your DockerHub username/organization
DOCKERHUB_USER=${DOCKERHUB_USER:-keasling}
DOCKERHUB_REPO_PREFIX=${DOCKERHUB_REPO_PREFIX:-foldy}

# Function to build and tag images for a specific version
build_and_tag_images() {
  local VERSION=$1
  local FRONTEND_TAG=${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-frontend:${VERSION}
  local BACKEND_TAG=${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-backend:${VERSION}
  local WORKER_ESM_TAG=${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-worker-esm:${VERSION}
  local WORKER_BOLTZ_TAG=${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-worker-boltz:${VERSION}

  # Build images (only once for the first version)
  if [ "$VERSION" == "${VERSIONS[0]}" ]; then
    echo "Building backend..."
    DOCKER_BUILDKIT=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $BACKEND_TAG -f backend/Dockerfile .

    echo "Building worker ESM..."
    DOCKER_BUILDKIT=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $WORKER_ESM_TAG -f worker/Dockerfile.esm .

    echo "Building worker BOLTZ..."
    DOCKER_BUILDKIT=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $WORKER_BOLTZ_TAG -f worker/Dockerfile.boltz .

    echo "Building frontend..."
    DOCKER_BUILDKIT=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build -t $FRONTEND_TAG \
      --build-arg BACKEND_URL=$BACKEND_URL \
      --build-arg INSTITUTION="$INSTITUTION" \
      frontend
  else
    # Tag existing images with additional tags
    echo "Tagging images for version: $VERSION"
    docker tag ${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-backend:${VERSIONS[0]} $BACKEND_TAG
    docker tag ${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-worker-esm:${VERSIONS[0]} $WORKER_ESM_TAG
    docker tag ${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-worker-boltz:${VERSIONS[0]} $WORKER_BOLTZ_TAG
    docker tag ${DOCKERHUB_USER}/${DOCKERHUB_REPO_PREFIX}-frontend:${VERSIONS[0]} $FRONTEND_TAG
  fi

  # Store tags for later pushing
  ALL_TAGS+=("$FRONTEND_TAG" "$BACKEND_TAG" "$WORKER_ESM_TAG" "$WORKER_BOLTZ_TAG")
}

# Build arguments for frontend
BACKEND_URL=${BACKEND_URL:-http://localhost:3000}
INSTITUTION=${INSTITUTION:-Local}

echo "Building and deploying Foldy containers to DockerHub..."
echo "Using the following configuration:"
echo "  DOCKERHUB_USER: $DOCKERHUB_USER"
echo "  VERSIONS: ${VERSIONS[*]}"
echo "  BACKEND_URL: $BACKEND_URL"
echo "  INSTITUTION: $INSTITUTION"
echo ""

# Navigate to project root (assuming script is in deployment/local/)
cd "$(dirname "$0")/../.."

# Initialize array for all tags
ALL_TAGS=()

# Build and tag images for each version
for VERSION in "${VERSIONS[@]}"; do
  build_and_tag_images "$VERSION"
done

echo ""
echo "Pushing images to DockerHub..."
echo "Note: Make sure you're logged in to DockerHub (docker login)"

# Push all tags
for TAG in "${ALL_TAGS[@]}"; do
  echo "Pushing $TAG..."
  docker push "$TAG"
done

echo ""
echo "Successfully built and pushed all images!"
echo ""
echo "Images pushed:"
for TAG in "${ALL_TAGS[@]}"; do
  echo "  - $TAG"
done
