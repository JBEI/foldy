#!/usr/bin/env bash

set -Eeuo pipefail

set -o xtrace

if [ "$#" -ne 3 ]; then
    die "Illegal number of parameters"
fi

ID=$1
PADDED_ID=`printf %06d $ID`
GS_OUT_FOLDER=$2
RUN_ANTISMASH=$3

OUT_DIR=/aftmp

##############################################################
# Download fastas.
echo "Downloading fastas..."
# Make sure the receiving directory already exists.
mkdir -p $OUT_DIR/$PADDED_ID/logs
/google-cloud-sdk/bin/gsutil rsync -r -x '.*\.pdb$|logs/.*|msa.*|.*pkl|.*npy' \
  $GS_OUT_FOLDER/$PADDED_ID/ $OUT_DIR/$PADDED_ID

##############################################################
# Clear prior Annotations output...
rm -r -f $OUT_DIR/$PADDED_ID/antismash
mkdir $OUT_DIR/$PADDED_ID/antismash

rm -r -f $OUT_DIR/$PADDED_ID/pfam
mkdir $OUT_DIR/$PADDED_ID/pfam

##############################################################
# Run annotation tools.
if [[ "$RUN_ANTISMASH" == "True" ]]
then
    PATH="/opt/conda/envs/antismash/bin:$PATH" && \
        /opt/conda/envs/antismash/bin/antismash \
        --databases /foldydbs/antismash \
        --genefinding-tool=prodigal \
        --output-dir=$OUT_DIR/$PADDED_ID/antismash \
        $OUT_DIR/$PADDED_ID/${PADDED_ID}_dna.fasta
    /opt/conda/envs/antismash/bin/python \
        /worker/parse_antismash.py \
        $OUT_DIR/$PADDED_ID/antismash \
        $OUT_DIR/$PADDED_ID/antismash/parsed/simple.json
fi

/opt/conda/envs/antismash/bin/hmmscan \
    --noali \
    --cut_ga \
    --cpu 1 \
    --domtblout $OUT_DIR/$PADDED_ID/pfam/pfam.txt \
    /foldydbs/pfam/Pfam-A.hmm \
    $OUT_DIR/$PADDED_ID/${PADDED_ID}.fasta
# For debugging purposes...
cat $OUT_DIR/$PADDED_ID/pfam/pfam.txt
/opt/conda/bin/python \
    /worker/parse_hmmscan.py \
    $OUT_DIR/$PADDED_ID/pfam/pfam.txt \
    $OUT_DIR/$PADDED_ID/pfam/pfam.json

##############################################################
# Rsync.
echo "Running final rsync to $GS_OUT_FOLDER/$PADDED_ID"
/google-cloud-sdk/bin/gsutil rsync -r $OUT_DIR/$PADDED_ID $GS_OUT_FOLDER/$PADDED_ID