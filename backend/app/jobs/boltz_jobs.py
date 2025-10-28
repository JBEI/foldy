import io
import logging
import string
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from Bio.PDB.MMCIFParser import (
    MMCIFParser,  # type: ignore[reportPrivateImportUsage] # Bio.PDB module structure quirk
)
from Bio.PDB.PDBIO import (
    PDBIO,  # type: ignore[reportPrivateImportUsage] # Bio.PDB module structure quirk
)
from werkzeug.exceptions import BadRequest

from app.helpers.boltz_yaml_helper import BoltzYamlHelper
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.jobs_util import (
    LoggingRecorder,
    get_torch_cuda_is_available_and_add_logs,
)
from app.models import Fold, Invokation


def try_check_smiles_string_validity(smiles_string):
    """Try to check if a smiles string is valid."""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            logging.error(f"Invalid SMILES: {smiles_string}")
    except Exception as e:
        logging.error(f"Error checking SMILES: {smiles_string} {e}")


def run_boltz(fold_id, invokation_id):
    """Run boltz workflow."""
    fold = Fold.get_by_id(fold_id)
    if not fold:
        raise BadRequest(f"Fold {fold_id} not found")
    invokation = Invokation.get_by_id(invokation_id)
    if not invokation:
        raise BadRequest(f"Invokation {invokation_id} not found")

    with LoggingRecorder(invokation):
        logging.info(
            "Starting Boltz execution...",
        )

        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)

        for ligand in boltz_yaml_helper.get_ligands():
            if "smiles" in ligand:
                try_check_smiles_string_validity(ligand["smiles"])

        # Create a foldstoragemanager.
        padded_fold_id = "%06d" % fold_id
        # fasta_relative_path = f"{padded_fold_id}.fasta"

        # Make a temporary directory for running Boltz.
        with TemporaryDirectory() as temp_dir:
            logging.info(f"Got temp directory at {temp_dir}")

            # Download the fasta file to the temporary directory.
            fsm = FoldStorageManager()
            fsm.setup()
            # binary_fasta_str = fsm.storage_manager.get_binary(
            #     fold_id, fasta_relative_path
            # )
            # fasta_file_path = Path(temp_dir) / fasta_relative_path
            # fasta_file_path.write_bytes(binary_fasta_str)
            yaml_file_str = fold.yaml_config
            yaml_file_path = Path(temp_dir) / "input.yml"
            yaml_file_path.write_text(yaml_file_str)
            fsm.storage_manager.write_file(fold_id, "boltz_input.yaml", yaml_file_str)
            logging.info(f"YAML file contents: {yaml_file_str}")

            diffusion_samples = fold.diffusion_samples or 1

            # Run Boltz.
            #
            # Note that we keep running out of shared memory (shm) when running Boltz
            # on A100s on Google Cloud.
            #
            # We increased shared memory to 20Gi but it didn't help.
            #
            # Based on the comments in this issue, it seems like we can improve
            # performance by reducing the number of dataworkers.
            # https://github.com/pytorch/pytorch/issues/5040#issuecomment-439590544
            #
            # Boltz API: https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md
            gpu_available = get_torch_cuda_is_available_and_add_logs(logging.info)
            accelerator = "gpu" if gpu_available else "cpu"
            boltz_command = [
                "/opt/conda/envs/boltzenv/bin/boltz",
                "predict",
                str(yaml_file_path),
                "--out_dir",
                str(temp_dir),
                "--use_msa_server",
                "--diffusion_samples",
                str(diffusion_samples),
                "--accelerator",
                accelerator,
                "--cache",
                "/hf-cache/",
                "--num_workers",
                "0",  # Should this be 1 or 0? 1 seems to work ok, but zero doesnt spin up any workers (a behavior which seems to cause a "pin memory" issue for foldy-in-a-box).
                "--use_potentials",
                "--write_full_pae",
                "--write_full_pde",
            ]
            logging.info(f"Running boltz with command: {boltz_command}")

            process = subprocess.Popen(
                boltz_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(process.stdout.readline, ""):
                logging.info(line.strip())

            process.stdout.close()
            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args)

            logging.info(f'Uploading files {list(Path(temp_dir).glob("*"))}')
            fsm.storage_manager.upload_folder(fold_id, temp_dir, "boltz")
            logging.info(f"Now converting mmCIF to PDB")

            # Use glob to find all files matching the pattern
            cif_files = list(Path(temp_dir).glob("boltz_results*/predictions/*/*_model_0.cif"))
            logging.info(f"Found {len(cif_files)} cif files: {cif_files}")
            if len(cif_files) == 0:
                logging.error(f"No CIF files found in {temp_dir}")
                raise BadRequest(f"No CIF files found in {temp_dir}")

            cif_file = cif_files[0]
            logging.info(f"Copying {cif_file} to ranked_0.cif")

            try:
                fsm.storage_manager.write_file(fold_id, "ranked_0.cif", cif_file.read_text())
            except Exception as e:
                logging.error(f"Error writing CIF to cif: {e}")
                raise e

            logging.info("We no longer convert CIF to PDB. In this case, CIF format is superior!!!")
            logging.info(f"Finished!")
