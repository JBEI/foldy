#!/usr/bin/env bash
set -o xtrace
set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

# set -o xtrace

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

function usage() {
    cat << EOF # remove the space between << and EOF, this is due to web plugin issue
        Usage: $(basename "${BASH_SOURCE[0]}") <fold_id> <aa_sequence> <run_amber_relax> <storage type {Local|Cloud}> [<gs_out_folder>]

        Script description here.

        Available options:
EOF
    exit
}

cleanup() {
    trap - SIGINT SIGTERM ERR EXIT
    # script cleanup here
}

msg() {
    echo >&2 -e "${1-}"
}

die() {
    local msg=$1
    local code=${2-1} # default exit status 1
    msg "$msg"
    exit "$code"
}

##############################################################
# The bulk of the code is below...
##############################################################

if [ "$#" -lt 5 ]; then
    die "Illegal number of parameters"
fi

ID=$1
PADDED_ID=`printf %06d $ID`
STAGE=$2
MODEL_PRESET=$3
RUN_AMBER_RELAX=$4
STORAGE_TYPE=$5

if [ "$STORAGE_TYPE" = "Local" ]; then
    if [ $# -eq 5 ]; then
        echo "Using local storage."
    else
        echo "Invalid: There are five arguments, but the fifth argument is not 'Local'."
        exit 1
    fi
elif [ "$STORAGE_TYPE" = "Cloud" ]; then
    if [ $# -eq 6 ]; then
        GS_OUT_FOLDER=$6
        echo "Using cloud storage at $GS_OUT_FOLDER"
    else
        echo "Invalid: There are six arguments, but the fifth argument is not 'Cloud'."
        exit 1
    fi
else
    echo "Invalid storage type: $STORAGE_TYPE"
    exit 1
fi


echo "Starting feature generation with parameters:";
echo "  Fold ID: $ID";
echo "  Padded Fold ID: $PADDED_ID";
echo "  Stage: $STAGE";
echo "  Model preset: $MODEL_PRESET";
echo "  Run amber relax: $RUN_AMBER_RELAX";
echo "  Storage Type: $STORAGE_TYPE";


FASTA_PATH=/aftmp/$PADDED_ID/$PADDED_ID.fasta
OUT_DIR=/aftmp
DATA_DIR=/foldydbs/afdbs

##############################################################
# Copy fasta and any results that are available.
mkdir -p $OUT_DIR/$PADDED_ID/logs

if [ "$STORAGE_TYPE" = "Cloud" ]; then
    /google-cloud-sdk/bin/gsutil -m rsync -r $GS_OUT_FOLDER/$PADDED_ID $OUT_DIR/$PADDED_ID
fi

##############################################################
# Run Alphafold.


# Apparently we're supposed to disable unified memory now.
# https://github.com/deepmind/alphafold/issues/406
# export TF_FORCE_UNIFIED_MEMORY='0'
# unset TF_FORCE_UNIFIED_MEMORY
#
# But it worked well for me in the past!
# https://github.com/deepmind/alphafold/blob/main/docker/run_docker.py#L246
export TF_FORCE_UNIFIED_MEMORY='1'
export XLA_PYTHON_CLIENT_MEM_FRACTION='4.0'
export XLA_PYTHON_CLIENT_PREALLOCATE=false


if [[ "$MODEL_PRESET" == "multimer" ]]
then
  SPECIAL_DB_ARGS="--pdb_seqres_database_path=$DATA_DIR/pdb_seqres/pdb_seqres.txt --uniprot_database_path=$DATA_DIR/uniprot/uniprot.fasta"
else
  SPECIAL_DB_ARGS="--pdb70_database_path=$DATA_DIR/pdb70/pdb70"
fi

case "$STAGE" in
 models) STOP_AFTER_MSA=False; CLEAR_GPU=True ;;
 features) STOP_AFTER_MSA=True; CLEAR_GPU=False ;;
 *) echo "Invalid stage '$STAGE'"; exit 1 ;;
esac


# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
ldconfig

echo "Contents of afdbs:"
ls -lah /foldydbs/afdbs/
ls -lah /foldydbs/afdbs/small_bfd/

    #--mgnify_database_path=$DATA_DIR/mgnify/mgy_clusters_2018_12.fa \
time /opt/conda/bin/python /app/alphafold/run_alphafold.py \
    --data_dir="$DATA_DIR" \
    --output_dir="$OUT_DIR" \
    --fasta_paths="$FASTA_PATH" \
    --max_template_date="2022-12-31" \
    --db_preset="reduced_dbs" \
    --model_preset=$MODEL_PRESET \
    --small_bfd_database_path=$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta \
    --mgnify_database_path=$DATA_DIR/mgnify/mgy_clusters_2022_05.fa \
    --template_mmcif_dir=$DATA_DIR/pdb_mmcif/mmcif_files \
    --obsolete_pdbs_path=$DATA_DIR/pdb_mmcif/obsolete.dat \
    --uniref90_database_path=$DATA_DIR/uniref90/uniref90.fasta \
    $SPECIAL_DB_ARGS \
    --benchmark="false" \
    --logtostderr \
    --num_multimer_predictions_per_model=1 \
    --use_precomputed_msas=True \
    --stop_after_msa=$STOP_AFTER_MSA \
    --run_relax=$RUN_AMBER_RELAX \
    --tmp_dir="/tmp" \
    --clear_gpu=$CLEAR_GPU \
    --use_gpu_relax=True

##############################################################
# Rsync.
if [ "$STORAGE_TYPE" = "Cloud" ]; then
    /google-cloud-sdk/bin/gsutil -m rsync -r $OUT_DIR/$PADDED_ID $GS_OUT_FOLDER/$PADDED_ID
fi

rm -r $OUT_DIR/$PADDED_ID