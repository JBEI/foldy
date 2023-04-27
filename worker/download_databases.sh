#!/bin/bash

# pfam.
mkdir -p /foldydbs/pfam
rm -rf /foldydbs/pfam/*
cd /foldydbs/pfam
wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-A.hmm.gz
gunzip Pfam-A.hmm.gz
hmmpress Pfam-A.hmm


# DiffDock.
mkdir -p /foldydbs/diffdockdbs/torch
rm -rf /foldydbs/diffdockdbs/*
cd /worker/diffdock/DiffDock
TORCH_HOME=/foldydbs/diffdockdbs/torch \
  /opt/conda/envs/diffdock/bin/python \
  /worker/diffdock/DiffDock/inference.py \
   || true
cp /worker/diffdock/DiffDock/.*.npy /foldydbs/diffdockdbs/

# Alphafold.
mkdir -p /foldydbs/afdbs
rm -rf /foldydbs/afdbs/*
/app/alphafold/scripts/download_all_data.sh /foldydbs/afdbs reduced_dbs
