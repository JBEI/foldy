#!/usr/bin/env bash
set -Eeuo pipefail
set -o xtrace

if [ "$#" -lt 5 ]; then
  echo "Too few arguments. Expected format:"
  echo "run_dock.sh <fold_id> <ligand_name> <ligand_smiles> <dock_tool> <storage location> [<gs_out_folder>] [optional other arguments passed to dock.py]"
  exit 1
fi

ID=$1
PADDED_ID=`printf %06d $ID`
LIGAND_NAME="${2}"
LIGAND_SMILES="${3}"
DOCK_TOOL="${4}"
STORAGE_TYPE=$5

if [ "$STORAGE_TYPE" = "Local" ]; then
    if [ $# -eq 5 ]; then
        echo "Using local storage."
        OTHER_ARGS=("${@:6}")
    else
        echo "Invalid: There are five arguments, but the fifth argument is not 'Local'."
        exit 1
    fi
elif [ "$STORAGE_TYPE" = "Cloud" ]; then
    if [ $# -eq 6 ]; then
        GS_OUT_FOLDER=$6
        OTHER_ARGS=("${@:7}")
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
# Download pdbs.
echo "Downloading PDBs..."
# Make sure the receiving directory already exists.
mkdir -p $OUT_DIR/$PADDED_ID/logs
if [ "$STORAGE_TYPE" = "Cloud" ]; then
  /google-cloud-sdk/bin/gsutil cp $GS_OUT_FOLDER/$PADDED_ID/ranked_0.pdb $OUT_DIR/$PADDED_ID/
fi

##############################################################
# Make output folder
mkdir -p $OUT_DIR/$PADDED_ID/dock/${LIGAND_NAME}

##############################################################
# Make dock batch files.
if [[ "$DOCK_TOOL" == "vina" ]]
then
  /opt/conda/envs/dock/bin/python \
      /worker/docking/dock.py \
      --adfrsuite_path /adfrsuite \
      "${OTHER_ARGS[@]}" \
      $OUT_DIR/$PADDED_ID/ranked_0.pdb \
      $LIGAND_SMILES \
      $OUT_DIR/$PADDED_ID/dock/${LIGAND_NAME}
else
  # DiffDock creates and accesses data in the "cwd", which we
  # prepopulated in /foldydbs/diffdockdbs.
  cd /worker/diffdock/DiffDock
  cp /foldydbs/diffdockdbs/.*.npy .
  TORCH_HOME=/foldydbs/diffdockdbs/torch \
      /opt/conda/envs/diffdock/bin/python \
      /worker/diffdock/DiffDock/inference.py \
      --protein_path $OUT_DIR/$PADDED_ID/ranked_0.pdb \
      --ligand "$LIGAND_SMILES" \
      --complex_name ${LIGAND_NAME} \
      --out_dir $OUT_DIR/$PADDED_ID/dock \
      --inference_steps 20 \
      --samples_per_complex 40 \
      --batch_size 10 \
      --actual_steps 18 \
      --no_final_step_noise 
  cd -

  # Combine all the ranked SDFs into one SDF for easy downloading.
  cd $OUT_DIR/$PADDED_ID/dock/${LIGAND_NAME}
  ls *confidence*.sdf | sort -k1.5n | xargs cat > poses.sdf
  cd -
fi

##############################################################
# Rsync.
if [ "$STORAGE_TYPE" = "Cloud" ]; then
  echo "Running final rsync to $GS_OUT_FOLDER/$PADDED_ID"
  /google-cloud-sdk/bin/gsutil rsync -r $OUT_DIR/$PADDED_ID/dock/${LIGAND_NAME} $GS_OUT_FOLDER/$PADDED_ID/dock/${LIGAND_NAME}
fi