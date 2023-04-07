import datetime
from datetime import timezone
import io
import logging
import json
import time
from re import fullmatch
import tempfile
import zipfile

from flask import current_app
from google.cloud.storage.client import Client
import numpy as np
from redis import Redis
from rq.job import Retry
from sqlalchemy.sql.elements import or_
from sqlalchemy.orm import joinedload
from werkzeug.exceptions import BadRequest
from pathlib import Path

from app import jobs
from app.models import Dock, Fold, Invokation, User
from app.extensions import compress, db, rq


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
            jobs.run_features,
            fold_id,
            get_job_type_replacement(fold, "features"),
            current_app.config["FOLDY_GCLOUD_BUCKET"],
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
        )
        email_dependent_jobs = [features_job]

    elif stage == "models":
        (gpu_q_name, num_retries) = get_gpu_queue_name(fold.sequence)
        gpu_q = rq.get_queue(gpu_q_name)
        emailparrot_q = rq.get_queue("emailparrot")
        models_job = gpu_q.enqueue(
            jobs.run_models,
            fold_id,
            get_job_type_replacement(fold, "models"),
            current_app.config["FOLDY_GCLOUD_BUCKET"],
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days,
            retry=Retry(max=num_retries),
        )
        email_dependent_jobs = [models_job]

    elif stage == "decompress_pkls":
        cpu_q = rq.get_queue("cpu")
        decompress_pkls_job = cpu_q.enqueue(
            jobs.decompress_pkls,
            fold_id,
            get_job_type_replacement(fold, "decompress_pkls"),
            current_app.config["FOLDY_GCLOUD_BUCKET"],
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
        )
        email_dependent_jobs = [decompress_pkls_job]

    elif stage == "annotate":
        cpu_q = rq.get_queue("cpu")
        annotate_job = cpu_q.enqueue(
            jobs.run_annotate,
            fold_id,
            get_job_type_replacement(fold, "annotate"),
            current_app.config["FOLDY_GCLOUD_BUCKET"],
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
        )
        email_dependent_jobs = [annotate_job]

    elif stage == "write_fastas":
        fsu = FoldStorageUtil()
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
            jobs.run_features,
            fold_id,
            get_job_type_replacement(fold, "features"),
            current_app.config["FOLDY_GCLOUD_BUCKET"],
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
        )
        models_job = gpu_q.enqueue(
            jobs.run_models,
            fold_id,
            get_job_type_replacement(fold, "models"),
            current_app.config["FOLDY_GCLOUD_BUCKET"],
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
            depends_on=[features_job],
            retry=Retry(max=num_retries),
        )
        decompress_pkls_job = cpu_q.enqueue(
            jobs.decompress_pkls,
            fold_id,
            get_job_type_replacement(fold, "decompress_pkls"),
            current_app.config["FOLDY_GCLOUD_BUCKET"],
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
            depends_on=[models_job],
        )
        annotate_job = cpu_q.enqueue(
            jobs.run_annotate,
            fold_id,
            get_job_type_replacement(fold, "annotate"),
            current_app.config["FOLDY_GCLOUD_BUCKET"],
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
            # Note: no dependent jobs.
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
            jobs.send_email,
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


class StorageManager:
    def list_files(self, fold_id):
        pass

    def write_file(self, file_path, file_contents_str):
        pass

    def get_binary(self, file_path):
        pass


class LocalStorageManager(StorageManager):
    local_directory = None

    def setup(self, local_directory):
        self.local_directory = Path(local_directory)

    def list_files(self, fold_id):
        padded_fold_id = "%06d" % fold_id
        dir = self.local_directory / padded_fold_id
        return [
            {
                "key": str(file.relative_to(dir)),
                "size": file.stat().st_size,
                "modified": file.stat().st_mtime * 1000,
            }
            for file in dir.glob("**/*")
            if not file.is_dir()
        ]

    def write_file(self, file_path, file_contents_str):
        with open(self.local_directory / file_path, "w") as f:
            f.write(file_contents_str)

    def get_binary(self, fold_id, file_path):
        download_start = time.time()
        padded_fold_id = "%06d" % fold_id
        fpath = self.local_directory / padded_fold_id / file_path
        try:
            with open(fpath, "rb") as f:
                blob_bytes = f.read()
        except FileNotFoundError as e:
            raise BadRequest(f"File {fpath} not found")
        print(
            f"Download to server took {time.time() - download_start} seconds for {fpath}.",
            flush=True,
        )
        return blob_bytes


class GcloudStorageManager(StorageManager):
    client = None

    def setup(self, fold_gcloud_bucket):
        self.client = Client(fold_gcloud_bucket)

    def list_files(self, fold_id):
        padded_fold_id = "%06d" % fold_id
        prefix = f"out/{padded_fold_id}/"

        bucket = self.client.get_bucket(current_app.config["FOLDY_GCLOUD_BUCKET"])
        blobs = list(bucket.list_blobs(max_results=10000, prefix=prefix))

        # https://dev.to/delta456/python-removeprefix-and-removesuffix-34jp
        def removeprefix(inputstr: str, prefix: str) -> str:
            if inputstr.startswith(prefix):
                return inputstr[len(prefix) :]
            else:
                return inputstr[:]

        return [
            {
                "key": removeprefix(blob.name, prefix),
                "size": blob.size,
                "modified": (
                    blob.updated - datetime.datetime.fromtimestamp(0, tz=timezone.utc)
                ).total_seconds()
                * 1000.0,
            }
            for blob in blobs
        ]

    def write_file(self, file_path, file_contents_str):
        bucket = self.client.get_bucket(current_app.config["FOLDY_GCLOUD_BUCKET"])
        aa_blob = bucket.blob(file_path)
        aa_blob.upload_from_string(file_contents_str)

    def get_binary(self, fold_id, relative_path):
        """Gets a file (as a binary string) from gcloud, with a relative path within the fold dir."""
        download_start = time.time()
        padded_fold_id = "%06d" % fold_id

        gcloud_path = f"out/{padded_fold_id}/{relative_path}"
        blobs = list(
            self.client.list_blobs(
                bucket_or_name=current_app.config["FOLDY_GCLOUD_BUCKET"],
                prefix=gcloud_path,
            )
        )
        if len(blobs) == 0:
            raise BadRequest(f"Found {len(blobs)} files at path {gcloud_path}.")

        blob_bytes = blobs[0].download_as_bytes()
        print(
            f"Download to server took {time.time() - download_start} seconds for {gcloud_path}.",
            flush=True,
        )
        return blob_bytes


class FoldStorageUtil:
    """Manages access to the backend storage for fold, for short term use only (see below).

    Note that this class aggressively caches query results to improve latency. There
    is minimal logic to refresh the cache entries. It it meant to be destroyed
    shortly after instantiation, eg in the lifetime of a single query."""

    storage_manager = None

    def setup(self, fold_gcloud_bucket=None):
        """Raises BadRequest if setup fails."""
        if current_app.config["ENV"] == "development":
            self.storage_manager = LocalStorageManager()
            self.storage_manager.setup("/app/integration_tests/testdata")
        else:
            if not fold_gcloud_bucket:
                fold_gcloud_bucket = current_app.config["FOLDY_GCLOUD_PROJECT"]
            self.storage_manager = GcloudStorageManager()
            self.storage_manager.setup(fold_gcloud_bucket)

    def get_fold_with_state(self, fold_id):
        fold = Fold.get_by_id(fold_id)
        if not fold:
            raise BadRequest(f"Fold {fold_id} not found.")
        return fold

    def get_folds_with_state(
        self,
        filter: str or None,
        tag: str or None,
        page: int or None,
        per_page: int or None,
    ):
        """Returns a list of folds with state populated."""

        def get_tag_regex(term):
            """Convert the tag into a regex for searching the tagstring CSV."""
            return "(^|,)" + term + "(,|$)"

        query = (
            db.session.query(Fold)
            .options(joinedload(Fold.jobs), joinedload(Fold.docks))
            .join(Fold.user)
        )

        if tag:
            query = query.filter(Fold.tagstring.op("~")(get_tag_regex(tag)))

        if filter:
            for term in filter.split(" "):
                if not term:
                    continue
                formatted_term = f"%{term}%"
                query = query.filter(
                    or_(
                        Fold.name.ilike(formatted_term),
                        Fold.sequence.ilike(formatted_term),
                        User.email.ilike(formatted_term),
                        Fold.tagstring.op("~")(get_tag_regex(term)),
                    )
                )

        query = query.order_by(Fold.id.desc())

        iterable = query
        if page and per_page:
            iterable = query.paginate(page, per_page).items

        folds = []
        for fold in iterable:
            if not fold:
                pass
            # if not include_logs:
            #   for job in fold.jobs:
            #     # Since the log field is deferred, this will keep the
            #     # response marshalling from accidentally triggering another
            #     # sql query.
            #     job.log = None
            folds.append(fold)
        return folds

    def validate_sequence(self, fold_name, sequence, af2_model_preset):
        """Raise BadRequest if the sequence contains invalid AAs."""
        if not fullmatch(r"[a-zA-Z0-9\-_ ]+", fold_name):
            raise BadRequest(f'Fold name has invalid characters: "{fold_name}"')

        if ":" in sequence or ";" in sequence:
            if af2_model_preset != "multimer":
                raise BadRequest(
                    f'This sequence looks like a multimer. Multimers are only supported when using "AF2" and using the AF2 model preset "multimer" (see Advanced Options).'
                )
            chains = []
            for chain in sequence.split(";"):
                if len(chain.split(":")) != 2:
                    raise BadRequest(
                        f'Each chain (separated by ";") must have a single ":".'
                    )
                chains.append(chain.split(":"))
        else:
            chains = [("1", sequence)]

        if len(chains) != len(set([c[0] for c in chains])):
            raise BadRequest("A chain name is duplicated.")

        for chain_name, chain_seq in chains:
            if not fullmatch(r"[a-zA-Z0-9_\-]+", chain_name):
                raise BadRequest(f'Invalid chain name "{chain_name}"')
            for ii, aa in enumerate(chain_seq):
                if aa not in VALID_AMINO_ACIDS:
                    raise BadRequest(
                        f'Invalid amino acid "{aa}" at index {ii+1} in chain {chain_name}. Valid amino acids are {"".join(VALID_AMINO_ACIDS)}.'
                    )

    def write_fastas(self, id, sequence):
        """Raises an exception if writing fails."""
        padded_fold_id = "%06d" % id
        aa_blob_path = f"out/{padded_fold_id}/{padded_fold_id}.fasta"
        dna_blob_path = f"out/{padded_fold_id}/{padded_fold_id}_dna.fasta"

        if ":" in sequence or ";" in sequence:
            monomers = [m.split(":") for m in sequence.split(";")]
            aa_fasta_entries = [f"> {m[0]}\n{m[1]}" for m in monomers]
            aa_contents = "\n\n".join(aa_fasta_entries)

            dna_contents = "\n\n".join(
                [f"> {m[0]}\n{back_translate(m[1])}" for m in monomers]
            )
        else:
            aa_contents = f"> {padded_fold_id}\n{sequence}"
            dna_contents = f"> {padded_fold_id}\n{back_translate(sequence)}"

        self.storage_manager.write_file(aa_blob_path, aa_contents)
        self.storage_manager.write_file(dna_blob_path, dna_contents)

    def make_new_folds(
        self,
        user_email,
        folds_data,
        start_fold_job,
        email_on_completion,
        skip_duplicate_entries,
    ):
        """Adds a list of new folds to the DB and optionally starts the jobs.

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
                        if not tag.isalnum():
                            raise BadRequest(
                                f"Bad group name, {tag} is not alphanumeric."
                            )

                tagstring = ",".join([t.strip() for t in fold_data.get("tags", [])])
                af2_model_preset = fold_data.get("af2_model_preset", "monomer_ptm")

                self.validate_sequence(
                    fold_data["name"], fold_data["sequence"], af2_model_preset
                )

                existing_entry = (
                    db.session.query(Fold)
                    .filter(Fold.name == fold_data["name"])
                    .first()
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
                self.write_fastas(model.id, model.sequence)
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

    def get_fold_pdb(self, fold_id, ranked_model_number):
        return self.storage_manager.get_binary(
            fold_id, f"ranked_{ranked_model_number}.pdb"
        ).decode()

    def get_fold_pdb_zip(self, fold_ids, dirname):
        """Returns an open file handle to a temporary file with PDBs zipped up."""
        tmp = tempfile.TemporaryFile()

        with zipfile.ZipFile(tmp, "w") as archive:
            for fold_id in fold_ids:
                fold = Fold.get_by_id(fold_id)
                if not fold:
                    raise BadRequest(f"Could not find fold {fold_id}")

                fold_pdb_binary = self.storage_manager.get_binary(
                    fold_id, "ranked_0.pdb"
                )
                archive.writestr(f"{dirname}/{fold.name}.pdb", fold_pdb_binary)
        tmp.seek(0)
        return tmp

    def get_fold_pkl(self, fold_id, ranked_model_number):
        """Returns a byte string."""
        return self.storage_manager.get_binary(
            fold_id, f"ranked_{ranked_model_number}.pkl"
        )

    def get_model_pae(self, fold_id, model_number):
        bytes_str = self.storage_manager.get_binary(
            fold_id, f"ranked_{model_number}/predicted_aligned_error.npy"
        )
        try:
            return np.load(io.BytesIO(bytes_str), allow_pickle=True)
        except Exception as e:
            print(e, flush=True)
            raise BadRequest(
                f"Failed to unpack file PAE for {fold_id} model {model_number} ({e})."
            )

    def get_contact_prob(self, fold_id, model_number, dist_thresh=12):
        bytes_str = self.storage_manager.get_binary(
            fold_id, f"ranked_{model_number}/contact_prob_{dist_thresh}A.npy"
        )
        try:
            return np.load(io.BytesIO(bytes_str), allow_pickle=True)
        except Exception as e:
            print(e, flush=True)
            raise BadRequest(
                f"Failed to unpack file contact prob for {fold_id} model {model_number} threshold {dist_thresh}A ({e})."
            )

    def get_pfam(self, fold_id):
        bytes_str = self.storage_manager.get_binary(fold_id, f"pfam/pfam.json")
        try:
            return json.load(io.BytesIO(bytes_str))
        except Exception as e:
            print(e, flush=True)
            raise BadRequest(f"Failed to unpack file pfam for {fold_id} ({e}).")

    def get_dock_sdf(self, fold_id, ligand_name):
        return self.storage_manager.get_binary(fold_id, f"dock/{ligand_name}/poses.sdf")
