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

from app.models import Fold, Evolution, Invokation
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.sequence_util import (
    get_measured_and_unmeasured_mutant_seq_ids,
    get_loci_set,
    process_and_validate_evolve_input_files,
    train_and_predict_activities,
)
from app.helpers.jobs_util import (
    _live_update_tail,
    _psql_tail,
    try_run_job_with_logging,
)
from app.helpers.boltz_yaml_helper import BoltzYamlHelper


def run_evolvepro(evolve_id: int):
    """Run the evolvepro workflow."""
    evolve = Evolution.get_by_id(evolve_id)
    if not evolve:
        raise BadRequest(f"Evolution {evolve_id} not found")
    fold = Fold.get_by_id(evolve.fold_id)
    if not fold:
        raise BadRequest(f"Fold {evolve.fold_id} not found")
    invokation = Invokation.get_by_id(evolve.invokation_id)
    if not invokation:
        raise BadRequest(f"Invokation {evolve.invokation_id} not found")

    def run_evolvepro_with_logger(add_log):
        """Helper function to run evolvepro with a logger."""
        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)
        if len(boltz_yaml_helper.get_protein_sequences()) != 1:
            raise ValueError(
                f"Fold has {len(boltz_yaml_helper.get_protein_sequences())} protein sequences, which is not supported for evolvepro yet."
            )
        wt_aa_seq = boltz_yaml_helper.get_protein_sequences()[0][1]

        fsm = FoldStorageManager()
        fsm.setup()

        # 1. Get the activity file.
        evolve_directory = Path("evolve") / evolve.name
        activity_file_path = evolve_directory / "activity.xlsx"
        add_log(f"Getting the activity file {activity_file_path}")
        activity_file = fsm.storage_manager.get_binary(
            evolve.fold_id, str(activity_file_path)
        )
        raw_activity_df = pd.read_excel(BytesIO(activity_file))

        # 2. Read and merge all embedding CSVs
        embedding_paths = evolve.embedding_files.split(",")
        add_log(f"Reading {len(embedding_paths)} embedding files")
        embedding_dfs = []
        chunk_size = 10000  # Adjust based on memory constraints

        for path in embedding_paths:
            # Get the CSV content as a string
            csv_blob = fsm.storage_manager.get_blob(evolve.fold_id, path)

            with csv_blob.open("r") as csv_f:
                # Create chunks iterator
                chunks = pd.read_csv(csv_f, chunksize=chunk_size)

                # Process each chunk
                path_dfs = []
                for chunk in chunks:
                    path_dfs.append(chunk)

                # Combine chunks for this path
                if path_dfs:
                    embedding_dfs.append(pd.concat(path_dfs, ignore_index=True))

        # Combine all embeddings
        raw_embedding_df = pd.concat(embedding_dfs, ignore_index=True)
        add_log(f"Found {raw_embedding_df.shape[0]} embeddings")

        # 3. Process the activity and embedding data.
        activity_df, embedding_df = process_and_validate_evolve_input_files(
            wt_aa_seq, raw_activity_df, raw_embedding_df
        )
        add_log(
            f"Found {activity_df.shape[0]} activity measurements among {activity_df.seq_id.unique().shape[0]} mutants"
        )

        (measured_mutants, unmeasured_mutants, model, predicted_activity_df) = (
            train_and_predict_activities(activity_df, embedding_df)
        )

        add_log(
            f"Finished fitting model. {len(measured_mutants)} measured mutants and {len(unmeasured_mutants)} unmeasured mutants, all of which had activity predicted."
        )

        # 6. Store model, visualizations, and predicted activities in storage manager.
        add_log(
            f"Storing model, visualizations, and predicted activities in {evolve_directory}"
        )
        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        serialized_model_binary_string = model_buffer.getvalue()

        predicted_activity_csv_str = predicted_activity_df.to_csv(index=False)

        fsm.storage_manager.write_file(
            evolve.fold_id,
            str(evolve_directory / "model.joblib"),
            serialized_model_binary_string,
        )
        fsm.storage_manager.write_file(
            evolve.fold_id,
            str(evolve_directory / "predicted_activity.csv"),
            predicted_activity_csv_str,
        )

    try_run_job_with_logging(run_evolvepro_with_logger, invokation)
