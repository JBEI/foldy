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
from app.helpers.jobs_util import _live_update_tail, _psql_tail


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
