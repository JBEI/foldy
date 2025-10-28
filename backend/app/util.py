import datetime
import io
import json
import logging
import os
import re
import tempfile
import time
import zipfile
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from dnachisel import biotools
from flask import abort, current_app
from google.cloud.storage.client import Client
from redis import Redis
from rq import Callback
from rq.job import Job, Retry
from sqlalchemy.orm import joinedload
from sqlalchemy.sql.elements import or_
from werkzeug.exceptions import BadRequest

from app.extensions import compress, db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.rq_helpers import (
    add_meta_to_job,
    get_queue,
    send_failure_email,
    send_success_email,
)
from app.helpers.sequence_util import back_translate, validate_aa_sequence
from app.jobs import boltz_jobs, other_jobs
from app.models import Dock, Fold, Invokation, User


def get_job_type_replacement(fold: Fold, job_type: str) -> int:
    """Replace any existing job of the given type with a new one.

    Args:
        fold: The fold containing the jobs
        job_type: Type of job to replace

    Returns:
        ID of the newly created invokation
    """
    for job in fold.jobs:  # type: ignore[reportGeneralTypeIssues] # SQLAlchemy relationship properties are iterable at runtime
        if job.type == job_type:
            logging.info(f"Deleting existing job {job.id} of type {job_type} for fold {fold.id}")
            job.delete(commit=False)

    db.session.commit()

    new_invokation = Invokation(fold_id=fold.id, type=job_type, state="queued")
    new_invokation.save()

    logging.info(
        f"Created new invokation {new_invokation.id} of type {job_type} for fold {fold.id}"
    )
    return new_invokation.id


def start_stage(fold_id: int, stage: str, email_on_completion: bool) -> None:
    """Start the provided stage of processing.

    Args:
        fold_id: ID of the fold to process
        stage: Stage to start (email, both, write_fastas, annotate)
        email_on_completion: Whether to send an email when processing completes

    Raises:
        BadRequest: If fold not found or unsupported stage requested

    Returns:
        None
    """
    fold: Fold | None = Fold.get_by_id(fold_id)
    if not fold:
        raise BadRequest(f"Fold {fold_id} not found.")

    email_dependent_jobs: List[Job] = []

    if stage == "annotate":
        cpu_q = get_queue("cpu")
        annotate_job = cpu_q.enqueue(
            other_jobs.run_annotate,
            fold_id,
            get_job_type_replacement(fold, "annotate"),
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
        )
        logging.info(f"Queued annotate job {annotate_job.id} for fold {fold_id}")
        email_dependent_jobs = [annotate_job]

    elif stage == "write_fastas":
        fsu = FoldStorageManager()
        fsu.setup()
        fsu.write_fastas(fold_id, fold.yaml_config)
        logging.info(f"Wrote FASTA files for fold {fold_id}")
        email_dependent_jobs = []

    elif stage == "both":
        cpu_q = get_queue("cpu")
        gpu_q = get_queue("biggpu")
        boltz_q = get_queue("boltz")

        fold_jobs = []
        email_args = {}
        if email_on_completion:
            email_args = {
                "on_success": Callback(send_success_email, timeout="5s"),
                "on_failure": Callback(send_failure_email, timeout="5s"),
            }

        boltz_job = boltz_q.enqueue(
            boltz_jobs.run_boltz,
            fold_id,
            get_job_type_replacement(fold, "boltz"),
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
            **email_args,
        )
        add_meta_to_job(boltz_job, fold, "boltz", fold.id)

        fold_jobs.append(boltz_job)
        annotate_job = cpu_q.enqueue(
            other_jobs.run_annotate,
            fold_id,
            get_job_type_replacement(fold, "annotate"),
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
            # Note: no dependent other_jobs.
        )
        email_dependent_jobs = fold_jobs + [annotate_job]

    else:
        raise BadRequest(f"Unsupported stage {stage}")


def make_new_folds(
    fsm: FoldStorageManager,
    user_email: str,
    folds_data: List[Dict[str, Any]],
    start_fold_job: bool,
    email_on_completion: bool,
    skip_duplicate_entries: bool,
    is_dry_run: bool,
) -> bool:
    """Add a list of new folds to the database and optionally start jobs.

    Args:
        fsm: Storage manager for the folds
        user_email: Email of the user creating the folds
        folds_data: List of fold specifications
        start_fold_job: Whether to start processing jobs for new folds
        email_on_completion: Whether to send email when processing completes
        skip_duplicate_entries: Whether to skip entries that already exist

    Raises:
        BadRequest: If validation fails or user not found

    Returns:
        True if successful

    Note:
        Folds in folds_data should not have an ID or status.
    """

    # Validate some general params.
    user: Optional[User] = db.session.query(User).filter_by(email=user_email).first()
    if not user:
        logging.error(f"User with email {user_email} not found when creating folds")
        raise BadRequest("Somehow the user is empty...")

    # Validate all the folds and create models.
    try:
        new_fold_models: List[Fold] = []
        for fold_data in folds_data:
            if "id" in fold_data:
                raise BadRequest("New fold should not specify an ID.")
            if "state" in fold_data:
                raise BadRequest("New fold should not specify a state.")
            if "tags" in fold_data:
                tags: List[str] = fold_data["tags"]
                for tag in tags:
                    if not re.match(r"^[a-zA-Z0-9_-]+$", tag):
                        logging.warning(f"Invalid tag format: {tag}")
                        raise BadRequest(
                            f"Invalid tag: {tag} contains a character which is not a letter, number, hyphen, or underscore."
                        )

            tagstring: str = ",".join([t.strip() for t in fold_data.get("tags", [])])

            # New Boltz input.
            yaml_config: Optional[str] = fold_data.get("yaml_config")
            diffusion_samples: Optional[int] = fold_data.get("diffusion_samples", None)

            # Old AF2 handling.
            af2_model_preset: str = fold_data.get("af2_model_preset", None) or "boltz"
            # Commented out for now as it's not being used
            # validate_aa_sequence(
            #     fold_data["name"], fold_data["sequence"], af2_model_preset
            # )

            existing_entry: Optional[Fold] = (
                db.session.query(Fold).filter(Fold.name == fold_data["name"]).first()
            )
            if existing_entry:
                if not skip_duplicate_entries:
                    logging.warning(f"Attempted duplicate fold creation: {fold_data['name']}")
                    raise BadRequest(
                        f'Someone has already submitted a fold named {fold_data["name"]} ({existing_entry.id}).'
                    )

                if existing_entry.sequence == fold_data["sequence"]:
                    # Sweet, so the entry already exists, we can just skip it.
                    logging.info(f"Skipping duplicate fold entry: {fold_data['name']}")
                    continue
                else:
                    logging.warning(
                        f"Fold name collision with different sequence: {fold_data['name']}"
                    )
                    raise BadRequest(
                        f'Fold {fold_data["name"]} already exists ({existing_entry.id}) and its sequence does not match this new request.'
                    )

            if is_dry_run:
                continue
            new_fold_model = Fold(
                name=fold_data["name"],
                user_id=user.id,
                tagstring=tagstring,
                yaml_config=yaml_config,
                diffusion_samples=diffusion_samples,
                sequence=fold_data["sequence"],
                af2_model_preset=af2_model_preset,
                disable_relaxation=fold_data["disable_relaxation"],
            )
            new_fold_models.append(new_fold_model)

        if is_dry_run:
            return True

        # Bulk add!
        db.session.bulk_save_objects(new_fold_models, return_defaults=True, preserve_order=True)
        db.session.commit()
    except Exception as e:
        logging.error(f"Error creating fold models: {e}")
        raise BadRequest(str(e))

    for model in new_fold_models:
        try:
            fsm.write_fastas(model.id, model.yaml_config)
            logging.info(f"Successfully wrote FASTA files for fold {model.id}")
        except Exception as e:  # TODO: make this exception more specific.
            logging.error(
                f"Storage write failed for model {model.id}: {repr(e)}. "
                "Proceeding with fold creation despite error."
            )

    # Note that some of these may not have a fastq successfully written to
    # gcloud, but we forge ahead and debug when that occurs.
    fold_ids: List[int] = [m.id for m in new_fold_models]
    logging.info(f"Created {len(fold_ids)} new folds: {fold_ids}")

    if start_fold_job:
        for fold_id in fold_ids:
            try:
                start_stage(fold_id, "both", email_on_completion)
                logging.info(f"Started fold job for fold {fold_id}")
            except Exception as e:
                logging.error(f"Error starting fold job {fold_id}: {e}")
                raise BadRequest(str(e))

    db.session.commit()
    return True
