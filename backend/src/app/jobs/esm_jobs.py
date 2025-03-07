from datetime import datetime, UTC, timedelta
from flask import current_app
import signal
import subprocess
import sys
import time
from io import StringIO
import json
import traceback
import os
import re
import tempfile
from pathlib import Path
from io import BytesIO
from werkzeug.exceptions import BadRequest
import logging

import docker
import pandas as pd

from app.database import db
from app.extensions import rq
from app.models import Fold, Invokation, Dock, Embedding, Evolution
from app.helpers.fold_storage_manager import FoldStorageManager
from app import email_to
from app.models import Fold, Invokation, Dock, Logit
from app.helpers.sequence_util import (
    get_seq_ids_for_deep_mutational_scan,
    seq_id_to_seq,
    maybe_get_seq_id_error_message,
    process_and_validate_evolve_input_files,
    get_loci_set,
)
from app.helpers.jobs_util import (
    _live_update_tail,
    _psql_tail,
    try_run_job_with_logging,
    get_torch_cuda_is_available_and_add_logs,
    LoggingRecorder,
)
from app.helpers.boltz_yaml_helper import BoltzYamlHelper
from app.helpers.esm_util import get_naturalness
from app.helpers.esm_client import FoldyESMClient


def get_esm_embeddings(
    embed_id: int,
):
    """Compute the ESM embeddings and store them with the storage manager.

    Arguments:
        embed_id: ID of the embedding record to run.
    """
    # 1. Get records.
    embed_record = Embedding.get_by_id(embed_id)
    if not embed_record:
        raise KeyError(f"Embedding ID {embed_id} ({embed_id}) not found!")

    embed_name = embed_record.name
    embedding_model = embed_record.embedding_model
    dms_starting_seq_ids = (
        embed_record.dms_starting_seq_ids.split(",")
        if embed_record.dms_starting_seq_ids
        else []
    )
    extra_seq_ids = (
        embed_record.extra_seq_ids.split(",") if embed_record.extra_seq_ids else []
    )

    fold = embed_record.fold
    if not fold:
        raise KeyError(
            f"Embedding ID {embed_id} ({embed_name}) does not have an associated fold!"
        )
    invokation = Invokation.get_by_id(embed_record.invokation_id)
    if not invokation:
        raise KeyError(
            f"Embedding ID {embed_id} ({embed_name}) does not have an associated invokation!"
        )

    with LoggingRecorder(invokation):
        logging.info(
            "Starting embedding...",
        )

        # 3. Validate seq_ids.
        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)
        if len(boltz_yaml_helper.get_protein_sequences()) > 1:
            raise ValueError(
                "Fold has multiple protein sequences, which is not supported for ESM embeddings yet."
            )
        wt_aa_seq = boltz_yaml_helper.get_protein_sequences()[0][1]

        for extra_seq_id in extra_seq_ids:
            error = maybe_get_seq_id_error_message(wt_aa_seq, extra_seq_id)
            if error:
                raise ValueError(
                    f"Invalid seq_id in extra seq_ids: '{extra_seq_id}': {error}"
                )
        for dms_starting_seq_id in dms_starting_seq_ids:
            error = maybe_get_seq_id_error_message(wt_aa_seq, dms_starting_seq_id)
            if error:
                raise ValueError(
                    f"Invalid seq_id in DMS starting seq_ids: '{dms_starting_seq_id}': {error}"
                )

        # 4. Get the WT sequence.
        if ":" in wt_aa_seq or ";" in wt_aa_seq:
            raise KeyError(
                f"Fold ID {fold.id} seems to be a multimer which is not supported for ESM embeddings yet."
            )

        logging.info(
            f"Getting all sequence IDs (dms_starting_seq_ids: {dms_starting_seq_ids}; extra_seq_ids: {extra_seq_ids})"
        )
        dms_seq_ids = get_seq_ids_for_deep_mutational_scan(
            wt_aa_seq, dms_starting_seq_ids, extra_seq_ids
        )
        logging.info(f"Will be embedding {len(dms_seq_ids)} sequences")

        # 5. Import ESM and create client.
        logging.info(f"Importing ESM and creating client for {embedding_model}")

        gpu_available = get_torch_cuda_is_available_and_add_logs(logging.info)

        foldy_esm_client = FoldyESMClient.get_client(embedding_model)

        def get_embedding_dict(seq_id, seq):
            return {
                "seq_id": seq_id,
                "seq": seq,
                "embedding": json.dumps(foldy_esm_client.embed(seq)),
            }

        embedding_dicts = []

        for ii, seq_id in enumerate(dms_seq_ids):
            embedding_dicts.append(
                get_embedding_dict(seq_id, seq_id_to_seq(wt_aa_seq, seq_id))
            )
            if ii % 100 == 0:
                logging.info(f"Finished embedding {ii}/{len(dms_seq_ids)}")

        embedding_df = pd.DataFrame(embedding_dicts)

        # Convert the DataFrame to a CSV string
        csv_buffer = StringIO()
        embedding_df.to_csv(
            csv_buffer, index=False
        )  # Use index=False to exclude the index
        embedding_csv_string = csv_buffer.getvalue()

        # Create a FoldStorageManager and store the embeddings.
        padded_fold_id = "%06d" % fold.id
        embedding_path = (
            f"embed/{padded_fold_id}_embeddings_{embedding_model}_{embed_name}.csv"
        )

        logging.info(f"Saving output to {embedding_path}")
        fsm = FoldStorageManager()
        fsm.setup()
        fsm.storage_manager.write_file(fold.id, embedding_path, embedding_csv_string)

    try_run_job_with_logging(run_get_esm_embeddings_with_logger, invokation)


def get_esm_logits(logit_id: int):
    """Compute the ESM logits and store them with the storage manager.

    Arguments:
        logit_id: ID of the logit record to run.
    """
    logit_record = Logit.get_by_id(logit_id)
    if not logit_record:
        raise KeyError(f"Logit ID {logit_id} not found!")

    logit_name = logit_record.name
    logit_model = logit_record.logit_model
    fold = logit_record.fold
    if not fold:
        raise KeyError(
            f"Logit ID {logit_id} ({logit_name}) does not have an associated fold!"
        )
    invokation = Invokation.get_by_id(logit_record.invokation_id)
    if not invokation:
        raise KeyError(
            f"Logit ID {logit_id} ({logit_name}) does not have an associated invokation!"
        )

    with LoggingRecorder(invokation):
        logging.info("Starting logit...")

        fsm = FoldStorageManager()
        fsm.setup()

        # 3. Validate seq_ids.
        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)

        protein_input = None
        if len(boltz_yaml_helper.get_protein_sequences()) > 1:
            protein_input = boltz_yaml_helper.get_protein_sequences()
        else:
            protein_input = boltz_yaml_helper.get_protein_sequences()[0][1]

        get_depth_two_logits = logit_record.get_depth_two_logits or False

        with tempfile.TemporaryDirectory() as temp_dir:
            if logit_record.use_structure:
                pdb_binary = fsm.storage_manager.get_binary(fold.id, "ranked_0.pdb")
                with open(os.path.join(temp_dir, "ranked_0.pdb"), "wb") as f:
                    f.write(pdb_binary)
                pdb_file_path = os.path.join(temp_dir, "ranked_0.pdb")
            else:
                pdb_file_path = None

            if logit_model == "esm1v_t33_650M_UR90S_ensemble":
                logits_dicts_list = []
                melted_df_list = []
                for ii in range(1, 6):
                    submodel = f"esm1v_t33_650M_UR90S_{ii}"
                    logits_json, melted_df = get_naturalness(
                        protein_input, submodel, get_depth_two_logits, pdb_file_path
                    )
                    logits_dicts_list.append(json.loads(logits_json))
                    melted_df_list.append(melted_df.assign(model=ii))
                logits_json = json.dumps(logits_dicts_list)
                melted_df = pd.concat(melted_df_list)
            else:
                logits_json, melted_df = get_naturalness(
                    protein_input, logit_model, get_depth_two_logits, pdb_file_path
                )

        melted_csv_buffer = StringIO()
        melted_df.to_csv(melted_csv_buffer, index=False)
        melted_csv_string = melted_csv_buffer.getvalue()

        # Save both formats using FoldStorageManager
        logging.info("Saving logits to storage")
        padded_fold_id = "%06d" % fold.id
        logits_path = f"naturalness/logits_{logit_name}.json"
        melted_path = f"naturalness/logits_{logit_name}_melted.csv"

        fsm.storage_manager.write_file(fold.id, logits_path, logits_json)
        fsm.storage_manager.write_file(fold.id, melted_path, melted_csv_string)

        logging.info("Logits computation and storage complete")


def finetune_esm_model(evolve_id: int):
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

    with LoggingRecorder(invokation):
        logging.info("Starting finetuning...")

        logging.info("Loading training code.")
        from app.helpers.finetuning.training import train_per_protein, score_sequences
        import torch

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
        logging.info(f"Getting the activity file {activity_file_path}")
        activity_file = fsm.storage_manager.get_binary(
            evolve.fold_id, str(activity_file_path)
        )
        raw_activity_df = pd.read_excel(BytesIO(activity_file))

        # 3. Process the activity and embedding data.
        if all(
            [v in raw_activity_df.columns for v in ["sequence", "seq_id_w", "seq_id_l"]]
        ):
            loss = "dpo"
            # TODO: do some validation...
            activity_df = raw_activity_df
            for seq_id in activity_df.seq_id_w.tolist() + activity_df.seq_id_l.tolist():
                for locus in get_loci_set(seq_id):
                    if locus >= 1023:
                        raise BadRequest(
                            f"One of the seq_ids is for a protein that is too big: ESM only goes up to 1024AAs, not {seq_id}."
                        )

        elif all([v in raw_activity_df.columns for v in ["seq_id", "activity"]]):
            loss = "entropy"
            activity_df = process_and_validate_evolve_input_files(
                wt_aa_seq, raw_activity_df
            )
            # Convert activity_df, which has seq_id and activity, into train and valid dfs with an 80/20 split and columns sequence and label.
            activity_df["sequence"] = activity_df["seq_id"].apply(
                lambda seq_id: seq_id_to_seq(wt_aa_seq, seq_id)
            )
            activity_df["label"] = activity_df["activity"]
        else:
            raise ValueError(
                f"Activity file has invalid columns, got {raw_activity_df.columns}"
            )
        logging.info(f"Have {activity_df.shape[0]} rows in activity_df")

        if "use_for_validation" in activity_df.columns:
            logging.info(
                f'Using "use_for_validation" column to split into train and valid'
            )
            train_df = activity_df[activity_df["use_for_validation"] == False]
            valid_df = activity_df[activity_df["use_for_validation"] == True]
        else:
            logging.info(f'No "use_for_validation" column, so splitting randomly')
            train_df = activity_df.sample(frac=0.8, random_state=42)
            valid_df = activity_df.drop(train_df.index)

        logging.info(
            f"Train df has {train_df.shape[0]} rows and valid df has {valid_df.shape[0]} rows"
        )

        gpu_available = get_torch_cuda_is_available_and_add_logs(logging.info)

        epochs = 10
        learning_rate = 3e-4
        possible_params = evolve.name.split("_")
        for possible_param in possible_params:
            parts = possible_param.split("=")
            if len(parts) == 2:
                key, value = parts
                if key == "epochs":
                    epochs = int(value)
                elif key == "learningrate":
                    learning_rate = float(value)

        # Save model outputs
        padded_fold_id = "%06d" % fold.id
        model_dir = f"evolve/{evolve.name}/model"

        # Declare these outside the with block
        tokenizer = None
        model = None
        history = None

        with tempfile.TemporaryDirectory() as temp_dir:
            # Make output directory.
            temp_training_subdir = Path(temp_dir) / "training"
            temp_training_subdir.mkdir(parents=True, exist_ok=True)

            # Example: enable ranking loss
            tokenizer, model, history = train_per_protein(
                checkpoint=evolve.finetuning_model_checkpoint,
                train_df=train_df,
                valid_df=valid_df,
                device=torch.device("cuda" if gpu_available else "cpu"),
                train_batch_size=10,
                grad_accum_steps=2,
                val_batch_size=10,
                loss=loss,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=42,
                mixed_precision=False,  # This causes an error "Attempting to unscale FP16 gradients" when set to "gpu_available",
                train_full=True,
                output_dir=str(temp_training_subdir),
            )
            logging.info("Finetuning complete.")

            # Save tokenizer and model
            logging.info(f"Saving tokenizer and model to {model_dir}")
            tokenizer.save_pretrained(str(Path(temp_dir) / "tokenizer"))
            model.save_pretrained(str(Path(temp_dir) / "model"))
            fsm.storage_manager.upload_folder(fold.id, temp_dir, model_dir)

        # Save training history
        history_json = json.dumps(history)
        fsm.storage_manager.write_file(
            fold.id, f"{model_dir}/history.json", history_json
        )

        # Get all sequences to score
        logging.info(f"Getting all sequences to score")
        dms_seq_ids = get_seq_ids_for_deep_mutational_scan(wt_aa_seq, ["WT"], [])

        # Score sequences and save results
        logging.info(f"Scoring {len(dms_seq_ids)} sequences")
        scores_df = score_sequences(model, tokenizer, wt_aa_seq, dms_seq_ids)
        scores_fpath = f"evolve/{evolve.name}/scores.csv"
        logging.info(f"Saving scores to {scores_fpath}")
        scores_csv = scores_df.to_csv(index=False)
        fsm.storage_manager.write_file(fold.id, scores_fpath, scores_csv)

        logging.info(f"Finished finetuning and scoring.")
