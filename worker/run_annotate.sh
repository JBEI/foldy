#!/usr/bin/env bash

set -Eeuo pipefail

set -o xtrace

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
# Download fastas.
echo "Downloading fastas..."
# Make sure the receiving directory already exists.
mkdir -p $OUT_DIR/$PADDED_ID/logs
if [ "$STORAGE_TYPE" = "Cloud" ]; then
    # Download just the fasta file
    /google-cloud-sdk/bin/gsutil cp $GS_OUT_FOLDER/$PADDED_ID/${PADDED_ID}.fasta $OUT_DIR/$PADDED_ID/
fi

##############################################################
# Clear prior Annotations output...

rm -r -f $OUT_DIR/$PADDED_ID/pfam
mkdir $OUT_DIR/$PADDED_ID/pfam

##############################################################
# Run annotation tools.

/opt/conda/envs/annotations/bin/hmmscan \
    --noali \
    --cut_ga \
    --cpu 1 \
    --domtblout $OUT_DIR/$PADDED_ID/pfam/pfam.txt \
    /foldydbs/pfam/Pfam-A.hmm \
    $OUT_DIR/$PADDED_ID/${PADDED_ID}.fasta
# For debugging purposes...
cat $OUT_DIR/$PADDED_ID/pfam/pfam.txt
/opt/conda/envs/worker/bin/python \
    /worker/parse_hmmscan.py \
    $OUT_DIR/$PADDED_ID/pfam/pfam.txt \
    $OUT_DIR/$PADDED_ID/pfam/pfam.json

##############################################################
# Rsync.
if [ "$STORAGE_TYPE" = "Cloud" ]; then
    echo "Running final rsync to $GS_OUT_FOLDER/$PADDED_ID/pfam"
    /google-cloud-sdk/bin/gsutil rsync -r $OUT_DIR/$PADDED_ID/pfam $GS_OUT_FOLDER/$PADDED_ID/pfam
fi
