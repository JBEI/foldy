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

import docker
import pandas as pd

from app.database import db
from app.extensions import rq
from app.models import Fold, Invokation, Dock, Embedding
from app.helpers.fold_storage_manager import FoldStorageManager
from app import email_to
from app.models import Fold, Invokation, Dock, Logit
from app.helpers.sequence_util import (
    get_seq_ids_for_deep_mutational_scan,
    seq_id_to_seq,
    maybe_get_seq_id_error_message,
)
from app.helpers.jobs_util import (
    _live_update_tail,
    _psql_tail,
    try_run_job_with_logging,
    get_torch_cuda_is_available_and_add_logs,
)
from app.helpers.boltz_yaml_helper import BoltzYamlHelper


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
        import torch
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig, SamplingConfig

        cuda_available = get_torch_cuda_is_available_and_add_logs(add_log)
        device = torch.device("cuda" if cuda_available else "cpu")
        add_log(f"Using device for ESMC embeddings: {device}")
        client = ESMC.from_pretrained(embedding_model).to(device)  # or "cpu" or "cuda"

        # 6. Define the embedding function.
        def get_embedding(seq):
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor,
                LogitsConfig(
                    sequence=False, return_embeddings=True
                ),  # We don't need the sequence logits.
            )
            # Compute the average across the residue dimension (dim=1)
            residue_pooled_avg_embedding = logits_output.embeddings.mean(
                dim=1
            )  # Shape: [1, N-dimension]

            # Remove the batch dimension to get [960]
            residue_pooled_avg_embedding = residue_pooled_avg_embedding.squeeze(
                0
            )  # Shape: [N-dimension]
            return json.dumps(residue_pooled_avg_embedding.tolist())

        def get_embedding_dict(seq_id, seq):
            return {"seq_id": seq_id, "seq": seq, "embedding": get_embedding(seq)}

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

        # 3. Validate seq_ids.
        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)
        if len(boltz_yaml_helper.get_protein_sequences()) > 1:
            raise ValueError(
                "Fold has multiple protein sequences, which is not supported for ESM embeddings yet."
            )
        wt_aa_seq = boltz_yaml_helper.get_protein_sequences()[0][1]

        # 4. Import ESM and create client
        add_log(f"Importing ESM and creating client for {logit_model}")
        import torch
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        from esm.utils.constants import esm3 as esm3_constants

        add_log(f"ESM3 SEQUENCE_VOCAB: {esm3_constants.SEQUENCE_VOCAB}")

        # Log cache directories
        torch_cache_dir = torch.hub.get_dir()
        add_log(f"Torch cache directory: {torch_cache_dir}")

        # Try to get Hugging Face cache dir if available
        try:
            from huggingface_hub import get_cache_dir

            hf_cache_dir = get_cache_dir()
            add_log(f"Hugging Face cache directory: {hf_cache_dir}")
        except Exception as e:
            add_log(f"Hugging Face cache directory command failed: {e}")

        add_log(
            f"Starting to load model {logit_model} - this may take several minutes on first run"
        )
        cuda_available = get_torch_cuda_is_available_and_add_logs(add_log)
        device = torch.device("cuda" if cuda_available else "cpu")
        add_log(f"Using device for ESMC logits: {device}")
        client = ESMC.from_pretrained(logit_model).to(device)
        add_log(f"Model {logit_model} loaded successfully")

        # 5. Create ESMProtein object and get logits
        add_log("Creating ESM protein object and computing logits")
        protein = ESMProtein(sequence=wt_aa_seq)
        protein_tensor = client.encode(protein)

        # Get logits for sequence prediction
        logits_output = client.logits(
            protein_tensor,
            LogitsConfig(
                sequence=True,  # We want sequence logits
                return_embeddings=False,  # We don't need embeddings
            ),
        )

        # 6. Process and store logits
        add_log("Processing logits and preparing to save")
        # Convert logits to probabilities using softmax
        sequence_probs = torch.softmax(logits_output.logits.sequence, dim=-1)
        add_log(f"Shape of sequence_probs: {sequence_probs.shape}")
        add_log(f"Length of protein sequence: {len(wt_aa_seq)}")

        # Create a list to store per-position amino acid probabilities
        position_probs = []

        # Process each position
        assert (
            sequence_probs.size(1) == len(wt_aa_seq) + 2
        ), f"We expect to get one logit vector per position, plus two special tokens (start and end)."

        # Create lists for both the original format and the melted format
        position_probs = []
        melted_rows = []

        for pos in range(sequence_probs.size(1)):  # Iterate over sequence length
            # Skip the first and last positions, which are special tokens.
            if pos == 0 or pos == sequence_probs.size(1) - 1:
                continue

            wt_aa = wt_aa_seq[pos - 1]
            probs = sequence_probs[0, pos, :].tolist()  # [0] for batch dimension

            # Original format
            pos_dict = {
                "locus": pos,  # 1-based position
                "wt_aa": wt_aa,
                "probabilities": probs,
            }
            position_probs.append(pos_dict)

            # Melted format
            for vocab_idx, vocab_char in enumerate(esm3_constants.SEQUENCE_VOCAB):
                prob = probs[vocab_idx]
                seq_id = f"{wt_aa}{pos}{vocab_char}"
                melted_rows.append({"seq_id": seq_id, "probability": prob})

        # Convert to JSON string for original format
        logits_json = json.dumps(position_probs)

        # Create melted CSV
        melted_df = pd.DataFrame(melted_rows)
        melted_csv_buffer = StringIO()
        melted_df.to_csv(melted_csv_buffer, index=False)
        melted_csv_string = melted_csv_buffer.getvalue()

        # Save both formats using FoldStorageManager
        add_log("Saving logits to storage")
        padded_fold_id = "%06d" % fold.id
        logits_path = f"naturalness/logits_{logit_name}.json"
        melted_path = f"naturalness/logits_{logit_name}_melted.csv"

        fsm = FoldStorageManager()
        fsm.setup()
        fsm.storage_manager.write_file(fold.id, logits_path, logits_json)
        fsm.storage_manager.write_file(fold.id, melted_path, melted_csv_string)

        add_log("Logits computation and storage complete")

    try_run_job_with_logging(get_esm_logits_with_logger, invokation)
