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
)
from app.helpers.jobs_util import (
    _live_update_tail,
    _psql_tail,
    try_run_job_with_logging,
    get_torch_cuda_is_available_and_add_logs,
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
        raise KeyError(f"Embedding ID {embed_id} ({embed_name}) not found!")

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

    def run_get_esm_embeddings_with_logger(add_log):
        add_log(
            "Starting embedding...",
            state="running",
            command="ESMC",
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

        add_log(
            f"Getting all sequence IDs (dms_starting_seq_ids: {dms_starting_seq_ids}; extra_seq_ids: {extra_seq_ids})"
        )
        dms_seq_ids = get_seq_ids_for_deep_mutational_scan(
            wt_aa_seq, dms_starting_seq_ids, extra_seq_ids
        )
        add_log(f"Will be embedding {len(dms_seq_ids)} sequences")

        # 5. Import ESM and create client.
        add_log(f"Importing ESM and creating client for {embedding_model}")

        gpu_available = get_torch_cuda_is_available_and_add_logs(add_log)

        foldy_esm_client = FoldyESMClient.get_client(embedding_model)

        def get_embedding_dict(seq_id, seq):
            return {
                "seq_id": seq_id,
                "seq": seq,
                "embedding": json.dumps(foldy_esm_client.embed(sequence=seq)),
            }

        embedding_dicts = []

        for ii, seq_id in enumerate(dms_seq_ids):
            embedding_dicts.append(
                get_embedding_dict(seq_id, seq_id_to_seq(wt_aa_seq, seq_id))
            )
            if ii % 100 == 0:
                add_log(f"Finished embedding {ii}/{len(dms_seq_ids)}")

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

        add_log(f"Saving output to {embedding_path}")
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

    def get_esm_logits_with_logger(add_log):
        add_log(
            "Starting logit...",
            state="running",
            command="ESMC",
        )

        fsm = FoldStorageManager()
        fsm.setup()

        # 3. Validate seq_ids.
        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)
        if len(boltz_yaml_helper.get_protein_sequences()) > 1:
            raise ValueError(
                "Fold has multiple protein sequences, which is not supported for ESM embeddings yet."
            )
        wt_aa_seq = boltz_yaml_helper.get_protein_sequences()[0][1]

        with tempfile.TemporaryDirectory() as temp_dir:
            if logit_record.use_structure:
                pdb_binary = fsm.storage_manager.get_binary(fold.id, "ranked_0.pdb")
                with open(os.path.join(temp_dir, "ranked_0.pdb"), "wb") as f:
                    f.write(pdb_binary)
                pdb_file_path = os.path.join(temp_dir, "ranked_0.pdb")
            else:
                pdb_file_path = None

            logits_json, melted_df = get_naturalness(
                wt_aa_seq, logit_model, add_log, pdb_file_path
            )

        melted_csv_buffer = StringIO()
        melted_df.to_csv(melted_csv_buffer, index=False)
        melted_csv_string = melted_csv_buffer.getvalue()

        # Save both formats using FoldStorageManager
        add_log("Saving logits to storage")
        padded_fold_id = "%06d" % fold.id
        logits_path = f"naturalness/logits_{logit_name}.json"
        melted_path = f"naturalness/logits_{logit_name}_melted.csv"

        fsm.storage_manager.write_file(fold.id, logits_path, logits_json)
        fsm.storage_manager.write_file(fold.id, melted_path, melted_csv_string)

        add_log("Logits computation and storage complete")

    try_run_job_with_logging(get_esm_logits_with_logger, invokation)


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

    def finetune_with_logger(add_log):
        add_log(
            "Starting finetuning...",
            state="running",
        )

        add_log("Loading training code.")
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
        add_log(f"Getting the activity file {activity_file_path}")
        activity_file = fsm.storage_manager.get_binary(
            evolve.fold_id, str(activity_file_path)
        )
        raw_activity_df = pd.read_excel(BytesIO(activity_file))

        # 3. Process the activity and embedding data.
        activity_df = process_and_validate_evolve_input_files(
            wt_aa_seq, raw_activity_df
        )
        add_log(f"Have {activity_df.shape[0]} rows in activity_df")

        # Convert activity_df, which has seq_id and activity, into train and valid dfs with an 80/20 split and columns sequence and label.
        activity_df["sequence"] = activity_df["seq_id"].apply(
            lambda seq_id: seq_id_to_seq(wt_aa_seq, seq_id)
        )
        activity_df["label"] = activity_df["activity"]

        train_df = activity_df.sample(frac=0.8, random_state=42)
        valid_df = activity_df.drop(train_df.index)
        add_log(
            f"Train df has {train_df.shape[0]} rows and valid df has {valid_df.shape[0]} rows"
        )

        gpu_available = get_torch_cuda_is_available_and_add_logs(add_log)

        class InterceptHandler(logging.Handler):
            def __init__(self, add_log):
                super().__init__()
                self.add_log = add_log

            def emit(self, record):
                self.add_log(record.msg)

        # Later in your finetune_with_logger function:
        handler = InterceptHandler(add_log)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        epochs = 10
        learning_rate = 3e-4
        possible_params = evolve.name.split("_")
        for possible_param in possible_params:
            parts = possible_param.split("=")
            if len(parts) == 2:
                key, value = parts
                if key == "epochs":
                    epochs = int(value)
                elif key == "learning_rate":
                    learning_rate = float(value)

        # Example: enable ranking loss
        tokenizer2, model2, history2 = train_per_protein(
            checkpoint=evolve.finetuning_model_checkpoint,
            train_df=train_df,
            valid_df=valid_df,
            device=torch.device("cuda" if gpu_available else "cpu"),
            num_labels=1,  # still a single numeric target
            use_ranking_loss=True,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=42,
            mixed_precision=False,  # This causes an error "Attempting to unscale FP16 gradients" when set to "gpu_available",
            train_full=True,
        )
        add_log("Finetuning complete.")

        # Save model outputs
        padded_fold_id = "%06d" % fold.id
        model_dir = f"evolve/{evolve.name}/model"

        # Save tokenizer and model
        add_log(f"Saving tokenizer and model to {model_dir}")
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer2.save_pretrained(f"{temp_dir}/tokenizer")
            model2.save_pretrained(f"{temp_dir}/model")
            fsm.storage_manager.upload_folder(fold.id, temp_dir, model_dir)

        # Save training history
        history_json = json.dumps(history2)
        fsm.storage_manager.write_file(
            fold.id, f"{model_dir}/history.json", history_json
        )

        # Get all sequences to score
        add_log(f"Getting all sequences to score")
        dms_seq_ids = get_seq_ids_for_deep_mutational_scan(wt_aa_seq, ["WT"], [])
        add_log(f"Scoring {len(dms_seq_ids)} sequences")

        # Score sequences and save results
        scores_df = score_sequences(model2, tokenizer2, wt_aa_seq, dms_seq_ids)
        scores_csv = scores_df.to_csv(index=False)
        fsm.storage_manager.write_file(
            fold.id, f"evolve/{evolve.name}/scores.csv", scores_csv
        )

    try_run_job_with_logging(finetune_with_logger, invokation)
