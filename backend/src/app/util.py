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

from app.jobs import other_jobs
from app.models import Dock, Fold, Invokation, User
from app.extensions import compress, db, rq
from app.helpers.fold_storage_manager import FoldStorageManager


VALID_AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "Y",
]


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
        annotate_job = cpu_q.enqueue(
            other_jobs.run_annotate,
            fold_id,
            get_job_type_replacement(fold, "annotate"),
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
            # Note: no dependent other_jobs.
        )
        email_dependent_jobs = [
            features_job,
            models_job,
            decompress_pkls_job,
            annotate_job,
        ]

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


def back_translate(aa_seq):
    # Ignore selenocysteine...
    # https://www.frontiersin.org/articles/10.3389/fmolb.2020.00002/full
    aa_without_u = aa_seq.replace("U", "C")
    return biotools.reverse_translate(aa_without_u, table="Bacterial")
    # randomize_codons=True,
