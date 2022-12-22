import datetime
from flask import current_app
import signal
import subprocess
import sys
import time

import docker

from app.database import db
from app.extensions import rq
from app.models import Fold, Invokation, Dock
from app.util import FoldStorageUtil
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

        invokation.update(state="running", log="Ongoing...")

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
            raise RuntimeError(
                f"Subprocess failed with error code {process.returncode} and stdout:\n{_tail(''.join(stdout))}"
            )

        # TODO: put this directly into the DB.
        # Don't return the whole job stdout. It's too big.
        final_state = "finished"
        return True
    except subprocess.SubprocessError as err:
        raise RuntimeError(
            f"Got error {err} and stdout:\n{_tail(''.join(stdout))}"
        )
    except TimeoutError as err:
        raise RuntimeError("Somehow time ran out...")
    except KeyboardInterrupt as err:
        print("Received a KeyboardInterrupt.", flush=True)
        process.send_signal(signal.SIGINT)
        process.wait()
    except ValueError as err:
        raise RuntimeError(f"Called Popen invalidly, got error {err} and stdout:\n{_tail(''.join(stdout))}")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"Got CalledProcessError, got error {err} and stdout:\n{_tail(''.join(stdout))}")
    finally:
        print(f'Invokation ending with final state {final_state}', flush=True)
        # This will get executed regardless of the exceptions raised in try
        # or except statements.
        invokation.update(
            state=final_state,
            log=_psql_tail("".join(stdout)),
            timedelta=datetime.timedelta(seconds=time.time() - start_time),
        )
        assert final_state == "finished", f'Job finished in state {final_state} with logs:\n\n{_tail("".join(stdout))}'


def run_features(
    fold_id,
    invokation_id,
    fold_gcloud_bucket,
):
    """Run alphafold feature generation and upload to cloud

    Throws exception if fold fails, returns stdout and sterr if successful.

    TODO: accept more parameters."""
    fold = Fold.get_by_id(fold_id)
    if not fold:
        raise KeyError(f"Fold ID {fold_id} not found!")

    gs_out_folder = f"gs://{fold_gcloud_bucket}/out"
    process_args = [
        current_app.config["RUN_AF2_PATH"],
        str(fold_id),
        "features",
        fold.af2_model_preset,
        gs_out_folder,
        str(not fold.disable_relaxation),
    ]

    start_generic_script(invokation_id, process_args)


def run_models(
    fold_id,
    invokation_id,
    fold_gcloud_bucket,
):
    """Run alphafold models pipeline and upload results to google cloud."""
    fold = Fold.get_by_id(fold_id)
    if not fold:
        raise KeyError(f"Fold ID {fold_id} not found!")

    gs_out_folder = f"gs://{fold_gcloud_bucket}/out"
    process_args = [
        current_app.config["RUN_AF2_PATH"],
        str(fold_id),
        "models",
        fold.af2_model_preset,
        gs_out_folder,
        str(not fold.disable_relaxation),
    ]

    start_generic_script(invokation_id, process_args)


def decompress_pkls(
    fold_id,
    invokation_id,
    fold_gcloud_bucket,
):
    gs_out_folder = f"gs://{fold_gcloud_bucket}/out"
    process_args = [
        current_app.config["DECOMPRESS_PKLS_PATH"],
        str(fold_id),
        gs_out_folder
    ]

    start_generic_script(invokation_id, process_args)


def run_annotate(
    fold_id: int,
    invokation_id: int,
    fold_gcloud_bucket: str,
):
    gs_out_folder = f"gs://{fold_gcloud_bucket}/out"
    process_args = [
        current_app.config["RUN_ANNOTATE_PATH"],
        str(fold_id),
        gs_out_folder
    ]

    start_generic_script(invokation_id, process_args)


def send_email(fold_id, protein_name, recipient):
    if not current_app.config['EMAIL_USERNAME'] or not current_app.config['EMAIL_PASSWORD']:
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


def dock(dock_id, invokation_id, fold_gcloud_bucket):
    """Execute the docking run described by the provided Dock instance."""
    dock = Dock.get_by_id(dock_id)

    extra_args = []
    if dock.bounding_box_residue and dock.bounding_box_radius_angstrom:
        extra_args = [
            f'--bounding_box_residue={dock.bounding_box_residue}',
            f'--bounding_box_radius_angstrom={dock.bounding_box_radius_angstrom}'
        ]

    gs_out_folder = f"gs://{fold_gcloud_bucket}/out"
    process_args = [
        current_app.config["RUN_DOCK"],
        str(dock.receptor_fold_id),
        gs_out_folder,
        dock.ligand_name,
        dock.ligand_smiles,
        *extra_args,
    ]

    successful = start_generic_script(invokation_id, process_args)

    if successful:
        fsm = FoldStorageUtil()
        fsm.setup(fold_gcloud_bucket)

        energy = fsm.storage_manager.get_binary(dock.receptor_fold_id, f'dock/{dock.ligand_name}/energy.txt').decode()

        dock.update(pose_energy=energy)