import datetime
from datetime import timezone
import io
import logging
import json
import time
import re
from re import fullmatch
import tempfile
import zipfile
import os

from dnachisel import biotools
from flask import current_app
from flask import abort
from google.cloud.storage.client import Client
import numpy as np
from redis import Redis
from rq.job import Retry
from sqlalchemy.sql.elements import or_
from sqlalchemy.orm import joinedload
from werkzeug.exceptions import BadRequest
from pathlib import Path

from app.jobs import other_jobs, boltz_jobs
from app.models import Dock, Fold, Invokation, User
from app.extensions import compress, db, rq
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.sequence_util import back_translate, validate_aa_sequence


def get_gpu_queue_name(sequence: str) -> str:
    """Choose a GPU name.

    Inputs:
      sequence: string of amino acids

    Returns: tuple of (queue to use for fold, number of retries)
    """
    if len(sequence) > 900:
        return ("biggpu", 1)
    return ("gpu", 3)


def get_job_type_replacement(fold: Fold, job_type: str):
    for job in fold.jobs:
        if job.type == job_type:
            job.delete(commit=False)
    # db.session.commit()
    # Invokation.super_delete().where(
    #   (Invokation.parent_id == fold.id) &
    #   (Invokation.type == job_type)
    # )
    db.session.commit()
    new_invokation = Invokation(fold_id=fold.id, type=job_type, state="queued")
    new_invokation.save()
    return new_invokation.id


def start_stage(fold_id: int, stage, email_on_completion):
    """Start the provided stage of processing. Can be features, models, email, or both."""
    fold: Fold = Fold.get_by_id(fold_id)
    if not fold:
        raise BadRequest(f"Fold {fold_id} not found.")

    email_dependent_jobs = []

    if stage == "features":
        cpu_q = rq.get_queue("cpu")
        features_job = cpu_q.enqueue(
            other_jobs.run_features,
            fold_id,
            get_job_type_replacement(fold, "features"),
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
        )
        email_dependent_jobs = [features_job]

    elif stage == "models":
        (gpu_q_name, num_retries) = get_gpu_queue_name(fold.sequence)
        gpu_q = rq.get_queue(gpu_q_name)
        emailparrot_q = rq.get_queue("emailparrot")
        models_job = gpu_q.enqueue(
            other_jobs.run_models,
            fold_id,
            get_job_type_replacement(fold, "models"),
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days,
            retry=Retry(max=num_retries),
        )
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
        email_dependent_jobs = [annotate_job]

    elif stage == "write_fastas":
        fsu = FoldStorageManager()
        fsu.setup()
        fsu.write_fastas(fold_id, fold.sequence)
        email_dependent_jobs = []

    elif stage == "email":
        email_dependent_jobs = []

    elif stage == "both":
        cpu_q = rq.get_queue("cpu")
        (gpu_q_name, num_retries) = get_gpu_queue_name(fold.sequence)
        gpu_q = rq.get_queue(gpu_q_name)
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
                retry=Retry(max=num_retries),
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
        emailparrot_q.enqueue(
            other_jobs.send_email,
            fold_id,
            fold.name,
            fold.user.email,
            depends_on=email_dependent_jobs,
        )


def make_new_folds(
    fsm: FoldStorageManager,
    user_email: str,
    folds_data,
    start_fold_job: bool,
    email_on_completion: bool,
    skip_duplicate_entries: bool,
):
    """Adds a list of new folds to the DB and optionally starts the other_jobs.

    Note that the folds in folds_data should not have an ID or status."""

    # Validate some general params.
    user = db.session.query(User).filter_by(email=user_email).first()
    if not user:
        raise BadRequest("Somehow the user is empty...")

    # Validate all the folds and create models.
    try:
        new_fold_models = []
        for fold_data in folds_data:
            if "id" in fold_data:
                raise BadRequest("New fold should not specify an ID.")
            if "state" in fold_data:
                raise BadRequest("New fold should not specify a state.")
            if "tags" in fold_data:
                tags = fold_data["tags"]
                for tag in tags:
                    if not re.match(r"^[a-zA-Z0-9_-]+$", tag):
                        raise BadRequest(
                            f"Invalid tag: {tag} contains a character which is not a letter, number, hyphen, or underscore."
                        )

            tagstring = ",".join([t.strip() for t in fold_data.get("tags", [])])
            af2_model_preset = fold_data.get("af2_model_preset", None) or "monomer_ptm"

            validate_aa_sequence(
                fold_data["name"], fold_data["sequence"], af2_model_preset
            )

            existing_entry = (
                db.session.query(Fold).filter(Fold.name == fold_data["name"]).first()
            )
            if existing_entry:
                if not skip_duplicate_entries:
                    raise BadRequest(
                        f'Someone has already submitted a fold named {fold_data["name"]} ({existing_entry.id}).'
                    )

                if existing_entry.sequence == fold_data["sequence"]:
                    # Sweet, so the entry already exists, we can just skip it.
                    continue
                else:
                    raise BadRequest(
                        f'Fold {fold_data["name"]} already exists ({existing_entry.id}) and its sequence does not match this new request.'
                    )

            new_fold_model = Fold(
                name=fold_data["name"],
                user_id=user.id,
                tagstring=tagstring,
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
        print(e, flush=True)
        raise BadRequest(e)

    for model in new_fold_models:
        try:
            fsm.write_fastas(model.id, model.sequence)
        except Exception as e:  # TODO: make this exception more specific.
            print(
                f"Bad news, a gcloud write failed for model {model.id}. "
                + "This should be infrequent, so we will allow the feature "
                + f"computation to start and debug based on its logs when this happens. {repr(e)}",
                flush=True,
            )
            print(e, flush=True)

    # Note that some of these may not have a fastq successfully written to
    # gcloud, but we forge ahead and debug when that occurs.
    fold_ids = [m.id for m in new_fold_models]

    if start_fold_job:
        for fold_id in fold_ids:
            try:
                start_stage(fold_id, "both", email_on_completion)
            except Exception as e:
                print(e, flush=True)
                raise BadRequest(e)

    db.session.commit()
    return True
