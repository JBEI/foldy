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
from app.models import Fold, Invokation, Dock
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
        wt_aa_seq = fold.sequence
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
        if ":" in fold.sequence or ";" in fold.sequence:
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
