import datetime
from datetime import timezone
import io
import logging
import json
import time
import re
import tempfile
import zipfile
import os
from typing import List, Dict, Any, Optional, Union, Tuple, cast

from dnachisel import biotools
from flask import current_app
from flask import abort
from google.cloud.storage.client import Client
import numpy as np
from redis import Redis
from rq.job import Retry, Job
from sqlalchemy.sql.elements import or_
from sqlalchemy.orm import joinedload
from werkzeug.exceptions import BadRequest
from pathlib import Path

from app.jobs import other_jobs, boltz_jobs
from app.models import Dock, Fold, Invokation, User
from app.extensions import compress, db, rq
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.sequence_util import back_translate, validate_aa_sequence


def get_job_type_replacement(fold: Fold, job_type: str) -> int:
    """Replace any existing job of the given type with a new one.
    
    Args:
        fold: The fold containing the jobs
        job_type: Type of job to replace
        
    Returns:
        ID of the newly created invokation
    """
    for job in fold.jobs:
        if job.type == job_type:
            logging.info(f"Deleting existing job {job.id} of type {job_type} for fold {fold.id}")
            job.delete(commit=False)
            
    db.session.commit()
    
    new_invokation = Invokation(fold_id=fold.id, type=job_type, state="queued")
    new_invokation.save()
    
    logging.info(f"Created new invokation {new_invokation.id} of type {job_type} for fold {fold.id}")
    return new_invokation.id


def start_stage(fold_id: int, stage: str, email_on_completion: bool) -> None:
    """Start the provided stage of processing.
    
    Args:
        fold_id: ID of the fold to process
        stage: Stage to start (features, models, email, both, write_fastas, 
               decompress_pkls, annotate)
        email_on_completion: Whether to send an email when processing completes
        
    Raises:
        BadRequest: If fold not found or unsupported stage requested
        
    Returns:
        None
    """
    fold: Fold = Fold.get_by_id(fold_id)
    if not fold:
        raise BadRequest(f"Fold {fold_id} not found.")

    email_dependent_jobs: List[Job] = []

    if stage == "features":
        cpu_q = rq.get_queue("cpu")
        features_job = cpu_q.enqueue(
            other_jobs.run_features,
            fold_id,
            get_job_type_replacement(fold, "features"),
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
        )
        logging.info(f"Queued features job {features_job.id} for fold {fold_id}")
        email_dependent_jobs = [features_job]

    elif stage == "models":
        gpu_q = rq.get_queue("biggpu")
        models_job = gpu_q.enqueue(
            other_jobs.run_models,
            fold_id,
            get_job_type_replacement(fold, "models"),
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days,
            retry=Retry(max=3),
        )
        logging.info(f"Queued models job {models_job.id} for fold {fold_id}")
        email_dependent_jobs = [models_job]

    elif stage == "decompress_pkls":
        cpu_q = rq.get_queue("cpu")
        decompress_pkls_job = cpu_q.enqueue(
            other_jobs.decompress_pkls,
            fold_id,
            get_job_type_replacement(fold, "decompress_pkls"),
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
        )
        logging.info(f"Queued decompress_pkls job {decompress_pkls_job.id} for fold {fold_id}")
        email_dependent_jobs = [decompress_pkls_job]

    elif stage == "annotate":
        cpu_q = rq.get_queue("cpu")
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

    elif stage == "email":
        logging.info(f"Email-only stage requested for fold {fold_id}")
        email_dependent_jobs = []

    elif stage == "both":
        cpu_q = rq.get_queue("cpu")
        gpu_q = rq.get_queue("biggpu")
        boltz_q = rq.get_queue("boltz")

        fold_jobs = []
        if fold.af2_model_preset == "boltz":
            boltz_job = boltz_q.enqueue(
                boltz_jobs.run_boltz,
                fold_id,
                get_job_type_replacement(fold, "boltz"),
                job_timeout="12h",
                result_ttl=48 * 60 * 60,  # 2 days
            )
            fold_jobs.append(boltz_job)
        else:
            features_job = cpu_q.enqueue(
                other_jobs.run_features,
                fold_id,
                get_job_type_replacement(fold, "features"),
                job_timeout="12h",
                result_ttl=48 * 60 * 60,  # 2 days
            )
            models_job = gpu_q.enqueue(
                other_jobs.run_models,
                fold_id,
                get_job_type_replacement(fold, "models"),
                job_timeout="12h",
                result_ttl=48 * 60 * 60,  # 2 days
                depends_on=[features_job],
                retry=Retry(max=3),
            )
            decompress_pkls_job = cpu_q.enqueue(
                other_jobs.decompress_pkls,
                fold_id,
                get_job_type_replacement(fold, "decompress_pkls"),
                job_timeout="12h",
                result_ttl=48 * 60 * 60,  # 2 days
                depends_on=[models_job],
            )
            fold_jobs = [features_job, models_job, decompress_pkls_job]

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

    if email_on_completion:
        emailparrot_q = rq.get_queue("emailparrot")
        email_job = emailparrot_q.enqueue(
            other_jobs.send_email,
            fold_id,
            fold.name,
            fold.user.email,
            depends_on=email_dependent_jobs,
        )
        dependency_ids = [job.id for job in email_dependent_jobs]
        logging.info(
            f"Queued email job {email_job.id} for fold {fold_id}, dependent on jobs: {dependency_ids}"
        )


def make_new_folds(
    fsm: FoldStorageManager,
    user_email: str,
    folds_data: List[Dict[str, Any]],
    start_fold_job: bool,
    email_on_completion: bool,
    skip_duplicate_entries: bool,
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

        # Bulk add!
        db.session.bulk_save_objects(
            new_fold_models, return_defaults=True, preserve_order=True
        )
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
