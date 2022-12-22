#!/bin/bash

mkdir -p /foldydbs/antismash
rm -rf /foldydbs/antismash/*
/opt/conda/envs/antismash/bin/download-antismash-databases --database-dir /foldydbs/antismash

mkdir -p /foldydbs/pfam
rm -rf /foldydbs/pfam/*
cd /foldydbs/pfam
wget ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam35.0/Pfam-A.hmm.gz
gunzip Pfam-A.hmm.gz
hmmpress Pfam-A.hmm

mkdir -p /foldydbs/afdbs
rm -rf /foldydbs/afdbs/*
/app/alphafold/scripts/download_all_data.sh /foldydbs/afdbs reduced_dbs