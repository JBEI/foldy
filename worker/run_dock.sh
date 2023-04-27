#!/usr/bin/env bash
set -Eeuo pipefail
set -o xtrace

if [ "$#" -lt 5 ]; then
  echo "Too few arguments. Expected format:"
  echo "run_dock.sh fold_id gs_out_folder ligand_name ligand_smiles dock_tool [optional other arguments passed to dock.py]"
  exit 1
fi

ID=$1
PADDED_ID=`printf %06d $ID`
GS_OUT_FOLDER=$2
LIGAND_NAME="${3}"
LIGAND_SMILES="${4}"
DOCK_TOOL="${5}"

OUT_DIR=/aftmp

##############################################################
# Download pdbs.
echo "Downloading PDBs..."
# Make sure the receiving directory already exists.
mkdir -p $OUT_DIR/$PADDED_ID/logs
/google-cloud-sdk/bin/gsutil cp $GS_OUT_FOLDER/$PADDED_ID/ranked_0.pdb $OUT_DIR/$PADDED_ID/

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
      "${@: 5}" \
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
fi

##############################################################
# Rsync.
echo "Running final rsync to $GS_OUT_FOLDER/$PADDED_ID"
/google-cloud-sdk/bin/gsutil rsync -r $OUT_DIR/$PADDED_ID/dock/${LIGAND_NAME} $GS_OUT_FOLDER/$PADDED_ID/dock/${LIGAND_NAME}