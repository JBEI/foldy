import time
from io import BytesIO
from datetime import datetime, UTC, timedelta
import traceback
import json
import io
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from werkzeug.exceptions import BadRequest
from pathlib import Path
import joblib
import tempfile
import subprocess
from tempfile import TemporaryDirectory
import os
import glob

from Bio.PDB import MMCIFParser, PDBIO

from app.models import Fold, Evolution, Invokation
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.sequence_util import (
    get_measured_and_unmeasured_mutant_seq_ids,
    get_loci_set,
    process_and_validate_evolve_input_files,
)
from app.helpers.jobs_util import (
    _live_update_tail,
    _psql_tail,
    try_run_job_with_logging,
    get_torch_cuda_is_available_and_add_logs,
)


def cif_to_pdb(cif_file: str, structure_id: str):
    """
    Convert mmCIF file to PDB format using Biopython.

    Parameters
    ----------
    cif_file : str
        Path to the input mmCIF file.

    Returns:
        PDB file contents as a string.
    """
    # Create a parser for mmCIF
    parser = MMCIFParser()

    # Read the structure from the mmCIF file
    structure = parser.get_structure(structure_id, cif_file)

    # Initialize PDBIO for writing PDB files
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)

    # Write out to PDB
    pdb_file_contents = io.StringIO()
    pdb_io.save(pdb_file_contents)
    pdb_file_contents.seek(0)
    return pdb_file_contents.read()


def run_boltz(fold_id, invokation_id):
    """Run boltz workflow."""
    fold = Fold.get_by_id(fold_id)
    if not fold:
        raise BadRequest(f"Fold {fold_id} not found")
    invokation = Invokation.get_by_id(invokation_id)
    if not invokation:
        raise BadRequest(f"Invokation {invokation_id} not found")

    def run_boltz_with_logger(add_log):
        add_log(
            "Starting Boltz execution...",
        )

        # Create a foldstoragemanager.
        padded_fold_id = "%06d" % fold_id
        # fasta_relative_path = f"{padded_fold_id}.fasta"

        # Make a temporary directory for running Boltz.
        with TemporaryDirectory() as temp_dir:
            add_log(f"Got temp directory at {temp_dir}")

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
            add_log(f"YAML file contents: {yaml_file_str}")

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
            gpu_available = get_torch_cuda_is_available_and_add_logs(add_log)
            accelerator = "gpu" if gpu_available else "cpu"
            boltz_command = [
                "/opt/conda/envs/worker/bin/boltz",
                "predict",
                str(yaml_file_path),
                "--out_dir",
                str(temp_dir),
                "--use_msa_server",
                "--accelerator",
                accelerator,
                "--cache",
                "/hf-cache/",
                "--num_workers",
                "0",  # Should this be 1 or 0? 1 seems to work ok, but zero doesnt spin up any workers (a behavior which seems to cause a "pin memory" issue for foldy-in-a-box).
            ]
            add_log(
                f"Running boltz with command: {boltz_command}",
                command=" ".join(boltz_command),
            )

            process = subprocess.Popen(
                boltz_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in iter(process.stdout.readline, ""):
                add_log(line.strip())

            process.stdout.close()
            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args)

            add_log(f'Uploading files {list(Path(temp_dir).glob("*"))}')
            fsm.storage_manager.upload_folder(fold_id, temp_dir, "boltz")
            add_log(f"Now converting mmCIF to PDB")

            # Use glob to find all files matching the pattern
            cif_files = list(
                Path(temp_dir).glob("boltz_results*/predictions/*/*_model_0.cif")
            )
            add_log(f"Found {len(cif_files)} cif files: {cif_files}")
            if len(cif_files) > 0:
                cif_file = cif_files[0]
                add_log(f"Copying {cif_file} to ranked_0.pdb")

                pdb_file_contents = cif_to_pdb(str(cif_file), "structure")
                fsm.storage_manager.write_file(
                    fold_id, "ranked_0.pdb", pdb_file_contents
                )
            add_log(f"Finished!")

    try_run_job_with_logging(run_boltz_with_logger, invokation)
