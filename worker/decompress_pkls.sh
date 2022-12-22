#!/usr/bin/env bash

set -Eeuo pipefail

ID=$1
PADDED_ID=`printf %06d $ID`
GS_OUT_FOLDER=$2

OUT_DIR=/aftmp

##############################################################
# Download pkls.
echo "Downloading PKLs..."
# Make sure the receiving directory already exists.
mkdir -p $OUT_DIR/$PADDED_ID
/google-cloud-sdk/bin/gsutil rsync -r -x '.*\.pdb$|logs/.*|msa.*' \
  $GS_OUT_FOLDER/$PADDED_ID/ $OUT_DIR/$PADDED_ID

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
echo "Running final rsync to $GS_OUT_FOLDER/$PADDED_ID"
/google-cloud-sdk/bin/gsutil rsync -r $OUT_DIR/$PADDED_ID $GS_OUT_FOLDER/$PADDED_ID