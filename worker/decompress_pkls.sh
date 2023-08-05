#!/usr/bin/env bash

set -Eeuo pipefail

if [ "$#" -lt 2 ]; then
    die "Illegal number of parameters"
fi

ID=$1
PADDED_ID=`printf %06d $ID`
STORAGE_TYPE=$2

if [ "$STORAGE_TYPE" = "Local" ]; then
    if [ $# -eq 2 ]; then
        echo "Using local storage."
    else
        echo "Invalid: There are five arguments, but the fifth argument is not 'Local'."
        exit 1
    fi
elif [ "$STORAGE_TYPE" = "Cloud" ]; then
    if [ $# -eq 3 ]; then
        GS_OUT_FOLDER=$3
        echo "Using cloud storage at $GS_OUT_FOLDER"
    else
        echo "Invalid: There are six arguments, but the fifth argument is not 'Cloud'."
        exit 1
    fi
else
    echo "Invalid storage type: $STORAGE_TYPE"
    exit 1
fi

OUT_DIR=/aftmp

##############################################################
# Download pkls.
echo "Downloading PKLs..."
# Make sure the receiving directory already exists.
mkdir -p $OUT_DIR/$PADDED_ID
if [ "$STORAGE_TYPE" = "Cloud" ]; then
  /google-cloud-sdk/bin/gsutil rsync -r -x '.*\.pdb$|logs/.*|msa.*' \
    $GS_OUT_FOLDER/$PADDED_ID/ $OUT_DIR/$PADDED_ID
fi

##############################################################
# Decompress.
echo "Running decompression..."
# We used to use the worker env, but decompress pkls needs jax...
# /opt/conda/envs/worker/bin/python
/opt/conda/bin/python /worker/decompress_pkls.py \
  $OUT_DIR/$PADDED_ID \
  $OUT_DIR/$PADDED_ID

##############################################################
# Rsync.
if [ "$STORAGE_TYPE" = "Cloud" ]; then
  echo "Running final rsync to $GS_OUT_FOLDER/$PADDED_ID"
  /google-cloud-sdk/bin/gsutil rsync -r $OUT_DIR/$PADDED_ID $GS_OUT_FOLDER/$PADDED_ID
fi