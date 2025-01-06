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
from app.helpers.mutation_util import (
    get_measured_and_unmeasured_mutant_seq_ids,
    get_loci_set,
    process_and_validate_evolve_input_files,
)
from app.helpers.jobs_util import _live_update_tail, _psql_tail


def run_evolvepro(evolve_id: int):
    """Run the evolvepro workflow."""
    final_state = "failed"
    start_time = time.time()
    logs = []

    evolve = Evolution.get_by_id(evolve_id)
    if not evolve:
        raise BadRequest(f"Evolution {evolve_id} not found")
    fold = Fold.get_by_id(evolve.fold_id)
    if not fold:
        raise BadRequest(f"Fold {evolve.fold_id} not found")
    invokation = Invokation.get_by_id(evolve.invokation_id)
    if not invokation:
        raise BadRequest(f"Invokation {evolve.invokation_id} not found")

    def add_log(msg, tail_function=_live_update_tail, **kwargs):
        timestamp = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
        timestamped_msg = f"{timestamp} - {msg}"
        logs.append(timestamped_msg)
        print(timestamped_msg)
        invokation.update(
            log=tail_function("\n".join(logs)),
            timedelta=timedelta(seconds=time.time() - start_time),
            **kwargs,
        )

    try:
        add_log(
            "Starting evolvepro execution...",
            state="running",
            starttime=datetime.fromtimestamp(start_time),
            command="EvolvePro",
        )
        wt_aa_seq = fold.sequence

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
            fold.sequence, raw_activity_df, raw_embedding_df
        )

        measured_mutants, unmeasured_mutants = (
            get_measured_and_unmeasured_mutant_seq_ids(activity_df, embedding_df)
        )
        add_log(
            f"{len(measured_mutants)} measured mutants and {len(unmeasured_mutants)} unmeasured mutants"
        )

        # 4. Fit the random forest model.
        add_log("Fitting the random forest model")
        X_train = np.vstack(
            [json.loads(x) for x in embedding_df.loc[activity_df.index].embedding]
        )
        y_train = activity_df.activity.to_numpy()
        model = RandomForestRegressor(
            n_estimators=100,
            criterion="friedman_mse",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=1,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
        )
        model.fit(X_train, y_train)
        add_log("Model fit complete")

        # 5. Predict activities for unmeasured mutants.
        add_log("Predicting activities for all mutants")
        all_mutants_embedding_array = np.vstack(
            # [json.loads(x) for x in embedding_df.loc[unmeasured_mutants].embedding]
            [
                json.loads(x)
                for x in embedding_df.loc[
                    measured_mutants + unmeasured_mutants
                ].embedding
            ]
        )
        print(all_mutants_embedding_array.shape)

        y_all_pred = model.predict(all_mutants_embedding_array)

        predicted_activity_df = pd.DataFrame(
            {
                "seq_id": measured_mutants + unmeasured_mutants,
                "predicted_activity": y_all_pred,
            }
        )
        predicted_activity_df.index = predicted_activity_df.seq_id
        predicted_activity_df["relevant_measured_mutants"] = (
            predicted_activity_df.seq_id.apply(
                lambda seq_id: " ".join(
                    [
                        m
                        for m in measured_mutants
                        if get_loci_set(m) & get_loci_set(seq_id)
                    ]
                )
            )
        )
        predicted_activity_df["actual_activity"] = predicted_activity_df.join(
            activity_df.groupby(level=0).activity.mean(), how="left"
        ).activity
        predicted_activity_df = predicted_activity_df.sort_values(
            "predicted_activity", ascending=False
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
            str(evolve_directory / "predicted_activity_df.csv"),
            predicted_activity_csv_str,
        )

        # 7. Updated Evolution record with model and visualizations.
        final_state = "finished"
    except Exception as e:
        # Capture the full traceback
        full_traceback = traceback.format_exc()

        add_log(f"Job failed with exception:\n\n{e} {full_traceback}")
    finally:
        # This will get executed regardless of the exceptions raised in try
        # or except statements.
        add_log(
            f"Invokation ending with final state {final_state}",
            tail_function=_psql_tail,
            state=final_state,
        )

        if final_state != "finished":
            print(
                f'Job finished in state {final_state} with logs:\n\n{_psql_tail("\n".join(logs))}',
                flush=True,
            )
            assert False, _psql_tail("\n".join(logs))
