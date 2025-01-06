from datetime import datetime, UTC
from flask import current_app
import signal
import subprocess
import sys
import time
from io import StringIO
import json
import traceback

import docker
import pandas as pd

from app.database import db
from app.extensions import rq
from app.models import Fold, Invokation, Dock
from app.helpers.fold_storage_manager import FoldStorageManager
from app import email_to
from app.models import Fold, Invokation, Dock
from app.helpers.mutation_util import (
    get_seq_ids_for_deep_mutational_scan,
    seq_id_to_seq,
)
from app.helpers.jobs_util import _live_update_tail, _psql_tail


def get_esm_embeddings(
    fold_id: int,
    batch_name: str,
    embedding_model: str,
    dms_starting_seq_ids: list[str],
    extra_seq_ids: list[str],
    invokation_id: int,
):
    """Compute the ESM embeddings and store them with the storage manager.

    Arguments:
        fold_id: ID of the fold to mutate.
        embedding_model: one of the ESMC models to use.
        dms_starting_seq_ids: starting sequences for DMS (eg, '' for WT DMS or ['', 'A34T', 'B45Y'] to DMSs starting from those three sequences).
        extra_seq_ids: additional sequence IDs to embed (eg, "G120W" to also try embedding that one).
        invokation_id: ID of the invokation object to update as we progress.
    """

    final_state = "failed"
    start_time = time.time()
    logs = []

    try:
        invokation = Invokation.get_by_id(invokation_id)

        def add_log(msg, tail_function=_live_update_tail, **kwargs):
            timestamp = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
            timestamped_msg = f"{timestamp} - {msg}"
            logs.append(timestamped_msg)
            print(timestamped_msg)
            invokation.update(
                log=tail_function("\n".join(logs)),
                timedelta=datetime.timedelta(seconds=time.time() - start_time),
                **kwargs,
            )

        add_log(
            "Starting embedding...",
            state="running",
            starttime=datetime.datetime.fromtimestamp(start_time),
            command="ESMC",
        )

        fold = Fold.get_by_id(fold_id)
        if not fold:
            raise KeyError(f"Fold ID {fold_id} not found!")
        if ":" in fold.sequence or ";" in fold.sequence:
            raise KeyError(
                f"Fold ID {fold_id} seems to be a multimer which is not supported for ESM embeddings yet."
            )
        wt_aa_seq = fold.sequence

        add_log(
            f"Getting all sequence IDs (dms_starting_seq_ids: {dms_starting_seq_ids}; extra_seq_ids: {extra_seq_ids})"
        )
        dms_seq_ids = get_seq_ids_for_deep_mutational_scan(
            wt_aa_seq, dms_starting_seq_ids, extra_seq_ids
        )
        add_log(f"Will be embedding {len(dms_seq_ids)} sequences")

        add_log(f"Importing ESM and creating client for {embedding_model}")
        import torch
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig, SamplingConfig

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        add_log(f"Using device for ESMC embeddings: {device}")
        client = ESMC.from_pretrained(embedding_model).to(device)  # or "cpu" or "cuda"

        def get_embedding(seq):
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor,
                LogitsConfig(
                    sequence=False, return_embeddings=True
                ),  # We don't need the sequence logits.
            )
            # # forward_and_sample is only available for ESM3, not ESMC
            # forward_and_sample_output = client.forward_and_sample(
            #     protein_tensor, SamplingConfig(return_mean_embeddings=True)
            # )
            # # print(logits_output.logits, logits_output.embeddings)
            # print(forward_and_sample_output)

            # logits_output.embeddings has dimension (N_residues, 960) (at least
            # for one of the models). I want the residue pooled average embedding,
            # if I am following "Rapid protein evolution by few-shot learning with
            # a  protein language model". So let's average the columns and save the
            # resulting list in the DF.
            # residue_pooled_avg_embedding = logits_output.embeddings.mean(
            #     dim=0
            # )  # Shape: (960,)
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
            if ii % 10 == 0:
                add_log(f"Finished embedding {ii}/{len(dms_seq_ids)}")

        embedding_df = pd.DataFrame(embedding_dicts)

        # Convert the DataFrame to a CSV string
        csv_buffer = StringIO()
        embedding_df.to_csv(
            csv_buffer, index=False
        )  # Use index=False to exclude the index
        embedding_csv_string = csv_buffer.getvalue()

        # Create a FoldStorageManager and store the embeddings.
        embedding_path = (
            f"esm/{padded_fold_id}_embeddings_{embedding_model}_{batch_name}.csv"
        )

        add_log(f"Saving output to {embedding_path}")
        fsm = FoldStorageManager()
        fsm.setup()
        fsm.storage_manager.write_file(fold_id, embedding_path, embedding_csv_string)

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
