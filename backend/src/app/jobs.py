import datetime
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
from app.util import FoldStorageManager
from app import email_to
from app.helpers.mutation_util import (
    get_seq_ids_for_deep_mutational_scan,
    seq_id_to_seq,
)


def add(x, y):
    return x + y


def _tail(stdout, max_char=5000):
    """Return just the last few lines of the stdout, a string."""
    if not stdout:
        return ""
    return stdout[-min(max_char, len(stdout)) :]


PSQL_CHAR_LIMIT = 100 * 1000 * 1000


def _psql_tail(stdout):
    return _tail(stdout, PSQL_CHAR_LIMIT)


def _live_update_tail(
    stdout,
):
    """Choose a tail size that is appropriate for streaming / live updates (not the whole logs)."""
    return _tail(stdout, 5000)


def start_generic_script(invokation_id, process_args):
    """Run some script, and track its results in the given PkModel."""
    final_state = "failed"
    stdout = []
    start_time = time.time()
    try:
        invokation = Invokation.get_by_id(invokation_id)

        invokation.update(
            state="running",
            log="Ongoing...",
            starttime=datetime.datetime.fromtimestamp(start_time),
            command=f"{process_args}",
        )

        def handle_sigterm(signum, frame):
            # This function will be called when SIGTERM is received.
            # You can perform any cleanup or termination logic here.
            # In this example, we simply exit the process.
            # Apparently stdout stops being a local variable when sigterm comes in?
            # stdout += [
            #     "\n\n\nRECEIVED SIGTERM\nRECEIVED SIGTERM\nRECEIVED SIGTERM\nRECEIVED SIGTERM\nRECEIVED SIGTERM"
            # ]
            raise RuntimeError(f"Got SIGTERM")

        # Set up the signal handler for SIGTERM
        signal.signal(signal.SIGTERM, handle_sigterm)
        # signal.signal(signal.SIGKILL, handle_sigterm)

        print(f"Running {process_args}")

        process = subprocess.Popen(
            process_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        for line in iter(lambda: process.stdout.readline(), b""):
            sys.stdout.buffer.write(line)
            stdout += line.decode("utf-8")

        process.wait(timeout=10)

        if process.returncode != 0:
            stdout += [f"Process returned with code {process.returncode}"]
            raise RuntimeError(
                f"Subprocess failed with error code {process.returncode} and stdout:\n{_tail(''.join(stdout))}"
            )

        # TODO: put this directly into the DB.
        # Don't return the whole job stdout. It's too big.
        final_state = "finished"
        return True
    except subprocess.SubprocessError as err:
        stdout += [f"\n\n\nInterrupted by SubprocessError: {str(err)}"]
        raise RuntimeError(f"Got error {err} and stdout:\n{_tail(''.join(stdout))}")
    except TimeoutError as err:
        stdout += [f"\n\n\nInterrupted by TimeoutError: {str(err)}"]
        raise RuntimeError("Somehow time ran out...")
    except KeyboardInterrupt as err:
        stdout += [f"\n\n\nInterrupted by KeyboardInterrupt: {str(err)}"]
        print("Received a KeyboardInterrupt.", flush=True)
        process.send_signal(signal.SIGINT)
        process.wait()
    except ValueError as err:
        stdout += [f"\n\n\nInterrupted by ValueError: {str(err)}"]
        raise RuntimeError(
            f"Called Popen invalidly, got error {err} and stdout:\n{_tail(''.join(stdout))}"
        )
    except subprocess.CalledProcessError as err:
        stdout += [f"\n\n\nInterrupted by CalledProcessError: {str(err)}"]
        raise RuntimeError(
            f"Got CalledProcessError, got error {err} and stdout:\n{_tail(''.join(stdout))}"
        )
    finally:
        print(f"Invokation ending with final state {final_state}", flush=True)
        # This will get executed regardless of the exceptions raised in try
        # or except statements.
        invokation.update(
            state=final_state,
            log=_psql_tail("".join(stdout)),
            timedelta=datetime.timedelta(seconds=time.time() - start_time),
        )
        assert (
            final_state == "finished"
        ), f'Job finished in state {final_state} with logs:\n\n{_tail("".join(stdout))}'


def run_features(
    fold_id,
    invokation_id,
):
    """Run alphafold feature generation and upload to cloud

    Throws exception if fold fails, returns stdout and sterr if successful.

    TODO: accept more parameters."""
    fold = Fold.get_by_id(fold_id)
    if not fold:
        raise KeyError(f"Fold ID {fold_id} not found!")

    models_to_relax = "NONE" if fold.disable_relaxation else "BEST"

    process_args = [
        current_app.config["RUN_AF2_PATH"],
        str(fold_id),
        "features",
        fold.af2_model_preset,
        models_to_relax,
        current_app.config["FOLDY_STORAGE_TYPE"],
    ]
    if current_app.config["FOLDY_STORAGE_TYPE"] == "Cloud":
        process_args.append(current_app.config["FOLDY_GSTORAGE_DIR"])

    start_generic_script(invokation_id, process_args)


def run_models(
    fold_id,
    invokation_id,
):
    """Run alphafold models pipeline and upload results to google cloud."""
    fold = Fold.get_by_id(fold_id)
    if not fold:
        raise KeyError(f"Fold ID {fold_id} not found!")

    models_to_relax = "NONE" if fold.disable_relaxation else "BEST"

    process_args = [
        current_app.config["RUN_AF2_PATH"],
        str(fold_id),
        "models",
        fold.af2_model_preset,
        models_to_relax,
        current_app.config["FOLDY_STORAGE_TYPE"],
    ]
    if current_app.config["FOLDY_STORAGE_TYPE"] == "Cloud":
        process_args.append(current_app.config["FOLDY_GSTORAGE_DIR"])

    start_generic_script(invokation_id, process_args)


def decompress_pkls(
    fold_id,
    invokation_id,
):
    process_args = [
        current_app.config["DECOMPRESS_PKLS_PATH"],
        str(fold_id),
        current_app.config["FOLDY_STORAGE_TYPE"],
    ]
    if current_app.config["FOLDY_STORAGE_TYPE"] == "Cloud":
        process_args.append(current_app.config["FOLDY_GSTORAGE_DIR"])

    start_generic_script(invokation_id, process_args)


def run_annotate(
    fold_id: int,
    invokation_id: int,
):
    fold = Fold.get_by_id(fold_id)
    if not fold:
        raise KeyError(f"Fold ID {fold_id} not found!")

    process_args = [
        current_app.config["RUN_ANNOTATE_PATH"],
        str(fold_id),
        current_app.config["FOLDY_STORAGE_TYPE"],
    ]
    if current_app.config["FOLDY_STORAGE_TYPE"] == "Cloud":
        process_args.append(current_app.config["FOLDY_GSTORAGE_DIR"])

    start_generic_script(invokation_id, process_args)


def send_email(fold_id, protein_name, recipient):
    if (
        not current_app.config["EMAIL_USERNAME"]
        or not current_app.config["EMAIL_PASSWORD"]
    ):
        raise KeyError("No email username / password provided: will not send email.")

    server = email_to.EmailServer(
        "smtp.gmail.com",
        587,
        current_app.config["EMAIL_USERNAME"],
        current_app.config["EMAIL_PASSWORD"],
    )

    link = f'{current_app.config["FRONTEND_URL"]}/fold/{fold_id}'

    header = f'### Finished folding <a href="{link}">{protein_name}</a>.'
    body = ""

    # Light blue: 28A5F5
    server.quick_email(
        recipient,
        f"Finished folding {protein_name}",
        [header, body],
        style="h3 {color: #333333}",
    )


def run_dock(dock_id, invokation_id):
    """Execute the docking run described by the provided Dock instance."""
    dock = Dock.get_by_id(dock_id)

    extra_args = []
    if dock.bounding_box_residue and dock.bounding_box_radius_angstrom:
        extra_args = [
            f"--bounding_box_residue={dock.bounding_box_residue}",
            f"--bounding_box_radius_angstrom={dock.bounding_box_radius_angstrom}",
        ]

    process_args = [
        current_app.config["RUN_DOCK"],
        str(dock.receptor_fold_id),
        dock.ligand_name,
        dock.ligand_smiles,
        dock.tool or "vina",
        current_app.config["FOLDY_STORAGE_TYPE"],
    ]
    if current_app.config["FOLDY_STORAGE_TYPE"] == "Cloud":
        process_args.append(current_app.config["FOLDY_GSTORAGE_DIR"])

    process_args += [*extra_args]

    successful = start_generic_script(invokation_id, process_args)

    if successful:
        fsm = FoldStorageManager()
        fsm.setup()

        if dock.tool == "vina":
            energy = fsm.storage_manager.get_binary(
                dock.receptor_fold_id, f"dock/{dock.ligand_name}/energy.txt"
            ).decode()

            dock.update(pose_energy=energy)
        elif dock.tool == "diffdock":
            confidence_str = fsm.get_diffdock_pose_confidences(
                dock.receptor_fold_id, dock.ligand_name
            )

            dock.update(pose_confidences=confidence_str)
        else:
            assert False, f"Invalid tool {dock.tool}"


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

        def add_log(msg, **kwargs):
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
                sep=" ", timespec="milliseconds"
            )
            timestamped_msg = f"{timestamp} - {msg}"
            logs.append(timestamped_msg)
            print(timestamped_msg)
            invokation.update(
                log=_live_update_tail("\n".join(logs)),
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
        padded_fold_id = "%06d" % fold_id
        embedding_path = f"{padded_fold_id}/esm/{padded_fold_id}_embeddings_{embedding_model}_{batch_name}.csv"

        add_log(f"Saving output to {embedding_path}")
        fsm = FoldStorageManager()
        fsm.setup()
        fsm.storage_manager.write_file(embedding_path, embedding_csv_string)

        final_state = "finished"
    except Exception as e:
        # Capture the full traceback
        full_traceback = traceback.format_exc()

        add_log(f"Job failed with exception:\n\n{e} {full_traceback}")
    finally:
        # This will get executed regardless of the exceptions raised in try
        # or except statements.
        add_log(f"Invokation ending with final state {final_state}", state=final_state)

        if final_state != "finished":
            print(
                f'Job finished in state {final_state} with logs:\n\n{_psql_tail("\n".join(logs))}',
                flush=True,
            )
            assert False, _psql_tail("\n".join(logs))
