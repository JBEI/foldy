#!/bin/bash
# Upload FolDE data to Google Cloud Storage bucket
# This script uploads ~152GB of data required for running FolDE simulations

set -e  # Exit on error

BUCKET="gs://foldedata"
DATA_DIR="backend/folde/data"

echo "==================================="
echo "FolDE Data Upload to GCS"
echo "==================================="
echo ""
echo "This will sync approximately 200GB of data to ${BUCKET}"
echo "Breakdown:"
echo "  - DMS_ProteinGym_substitutions: ~1.0GB"
echo "  - embeddings: ~150GB"
echo "  - naturalness: ~1GB"
echo "  - DMS_substitutions.csv: ~300KB"
echo ""
echo "Using gsutil rsync - only new/changed files will be uploaded."
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 1
fi

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo "ERROR: gsutil not found. Please install Google Cloud SDK:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    echo "Please run this script from the repository root."
    exit 1
fi

echo ""
echo "Starting rsync..."
echo ""

# Use rsync for efficient upload
# -m: parallel operations
# -r: recursive
# -d: delete remote files that don't exist locally

gsutil -m rsync -r -d "${DATA_DIR}" "${BUCKET}"

echo ""
echo "==================================="
echo "Sync complete!"
echo "==================================="
echo ""
echo "Verifying upload..."
gsutil ls -lh "${BUCKET}"

echo ""
echo "Done! Data is now available at ${BUCKET}"
echo ""
echo "To make the bucket publicly readable (recommended for open release):"
echo "  gsutil iam ch allUsers:objectViewer ${BUCKET}"
