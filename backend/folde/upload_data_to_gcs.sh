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

# Public read access. The README documents this bucket as public, and the data is useless
# to external users without it, so offer the grant here instead of only printing it -- a
# printed reminder is easy to miss, and the result is a bucket that 401s for everyone.
# Still prompted rather than automatic: this makes the data world-readable, which should be
# a deliberate choice and not a side effect of re-running an upload.
if gsutil iam get "${BUCKET}" 2>/dev/null | grep -q "allUsers"; then
    echo "Bucket already grants access to allUsers - nothing to do."
else
    echo "This bucket does NOT currently grant public read access."
    echo "Without it, external users cannot download the data documented in folde/README.md:"
    echo "  401 Anonymous caller does not have storage.objects.list access"
    echo ""
    read -p "Grant public read access (allUsers:objectViewer) now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gsutil iam ch allUsers:objectViewer "${BUCKET}"
        echo "Granted. Verifying anonymous read..."
        if curl -sf -o /dev/null "https://storage.googleapis.com/${BUCKET#gs://}/DMS_substitutions.csv"; then
            echo "  OK - anonymous read confirmed."
        else
            echo "  WARNING - anonymous read still failing; check bucket IAM and public access prevention."
        fi
    else
        echo "Skipped. To grant it later:"
        echo "  gsutil iam ch allUsers:objectViewer ${BUCKET}"
    fi
fi
