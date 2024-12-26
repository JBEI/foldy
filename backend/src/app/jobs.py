import datetime
from flask import current_app
import signal
import subprocess
import sys
import time
from io import StringIO
import json

import docker
import pandas as pd

from app.database import db
from app.extensions import rq
from app.models import Fold, Invokation, Dock
from app.util import FoldStorageManager
from app import email_to


def add(x, y):
    return x + y


def _tail(stdout, max_char=5000):
    """Return just the last few lines of the stdout, a string."""
    if not stdout:
        return ""
    return stdout[-min(max_char, len(stdout)) : -1]


PSQL_CHAR_LIMIT = 100 * 1000 * 1000


def _psql_tail(stdout):
    return _tail(stdout, PSQL_CHAR_LIMIT)


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


def get_esm_embeddings(fold_id: int, embedding_model: str, invokation_id: int):
    """Compute the ESM embeddings and store them."""

    final_state = "failed"
    start_time = time.time()
    logs = []

    def add_log(msg):
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(
            sep=" ", timespec="milliseconds"
        )
        log = f"{timestamp} - {msg}"
        logs.append(log)
        print(log)

    try:
        invokation = Invokation.get_by_id(invokation_id)

        invokation.update(
            state="running",
            log="Ongoing...",
            starttime=datetime.datetime.fromtimestamp(start_time),
            command="",
        )

        fold = Fold.get_by_id(fold_id)
        if not fold:
            raise KeyError(f"Fold ID {fold_id} not found!")

        if ":" in fold.sequence or ";" in fold.sequence:
            raise KeyError(
                f"Fold ID {fold_id} seems to be a multimer which is not supported for ESM embeddings yet."
            )

        add_log(f"Importing ESM and creating client for {embedding_model}")
        invokation.update(
            state="running",
            log=_psql_tail("\n".join(logs)),
            timedelta=datetime.timedelta(seconds=time.time() - start_time),
        )

        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig, SamplingConfig

        VALID_AMINO_ACIDS = "ACDEFGHIKLMNOPQRSTUVWY"

        wt_aa_seq = fold.sequence

        client = ESMC.from_pretrained(embedding_model).to("cpu")  # or "cpu" or "cuda"

        def get_embedding(seq):
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            # # forward_and_sample is only available for ESM3, not ESMC
            # forward_and_sample_output = client.forward_and_sample(
            #     protein_tensor, SamplingConfig(return_mean_embeddings=True)
            # )
            # # print(logits_output.logits, logits_output.embeddings)
            # print(forward_and_sample_output)
            return json.dumps(logits_output.embeddings.tolist())

        def get_embedding_dict(seq_id, seq):
            return {"seq_id": seq_id, "seq": seq, "embedding": get_embedding(seq)}

        add_log(f"Starting with WT sequence")

        embedding_dicts = []
        wt_embedding_dict = get_embedding_dict("", wt_aa_seq)
        embedding_dicts.append(wt_embedding_dict)
        for aa_idx in range(len(wt_aa_seq)):
            wt_aa_prefix = f"{wt_aa_seq[aa_idx]}{aa_idx + 1}"
            for alternative_aa in VALID_AMINO_ACIDS:
                if wt_aa_seq[aa_idx] == alternative_aa:
                    continue
                mutant_seq_id = wt_aa_prefix + alternative_aa
                mutant_seq = (
                    wt_aa_seq[:aa_idx] + alternative_aa + wt_aa_seq[(aa_idx + 1) :]
                )
                assert len(mutant_seq) == len(wt_aa_seq)

                mutant_embedding_dict = get_embedding_dict(mutant_seq_id, mutant_seq)
                embedding_dicts.append(mutant_embedding_dict)
            add_log(f"Finished residue {aa_idx}/{len(wt_aa_seq)}")
            invokation.update(
                state="running",
                log=_psql_tail("\n".join(logs)),
                timedelta=datetime.timedelta(seconds=time.time() - start_time),
            )

        embedding_df = pd.DataFrame(embedding_dicts)

        # Convert the DataFrame to a CSV string
        csv_buffer = StringIO()
        embedding_df.to_csv(
            csv_buffer, index=False
        )  # Use index=False to exclude the index
        embedding_csv_string = csv_buffer.getvalue()

        # Create a FoldStorageManager and store the embeddings.
        padded_fold_id = "%06d" % fold_id
        embedding_path = f"{padded_fold_id}/esm/{padded_fold_id}_embeddings_esm3.csv"

        add_log(f"Saving output to {embedding_path}")
        fsm = FoldStorageManager()
        fsm.setup()
        fsm.storage_manager.write_file(embedding_path, embedding_csv_string)

        final_state = "finished"
    except Exception as e:
        add_log(f"Job failed with exception:\n\n{e}")
    finally:
        add_log(f"Invokation ending with final state {final_state}")
        # This will get executed regardless of the exceptions raised in try
        # or except statements.
        invokation.update(
            state=final_state,
            log=_psql_tail("\n".join(logs)),
            timedelta=datetime.timedelta(seconds=time.time() - start_time),
        )
        assert (
            final_state == "finished"
        ), f'Job finished in state {final_state} with logs:\n\n{_tail("\n".join(logs))}'
