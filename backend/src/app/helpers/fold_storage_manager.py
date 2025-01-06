import datetime
from datetime import timezone, UTC
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

from app.models import Dock, Fold, Invokation, User
from app.extensions import compress, db, rq


class StorageAccessor:
    def list_files(self, fold_id, subfolder=None):
        raise NotImplementedError

    def write_file(self, fold_id: int, file_path: str, contents, binary=False):
        """Write contents to a file under the specified fold directory."""
        raise NotImplementedError

    def get_binary(self, file_path):
        raise NotImplementedError

    def get_blob(self, fold_id, file_path):
        raise NotImplementedError


class LocalBlob:
    """Simulate the GCS blob, for local access."""

    def __init__(self, file_path):
        """
        Initializes the LocalBlob with the path to the local file.

        Args:
            file_path (Path or str): The path to the local file.
        """
        self.file_path = file_path

    def open(self, mode="rb"):
        """
        Opens the local file in the specified mode.

        Args:
            mode (str): The mode in which to open the file. Default is 'rb'.

        Returns:
            file object: A file object opened in the specified mode.
        """
        return open(self.file_path, mode)

    def exists(self):
        """
        Checks if the file exists.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return os.path.exists(self.file_path)


class LocalStorageAccessor(StorageAccessor):
    local_directory = None

    def setup(self, local_directory):
        self.local_directory = Path(local_directory)

    def list_files(self, fold_id, subfolder=None):
        """Returns a list of dicts describing the contents of this fold's folder.

        Arguments:
          fold_id: fold to query
          subfolder: prefix within the fold bucket to search, eg 'dock/nadh'.

        Output: List of dictionaries with keys:
          key: path to the file
          size: size of the file
          modified: last modification time
        """
        padded_fold_id = "%06d" % fold_id
        dir = self.local_directory / padded_fold_id
        if subfolder:
            dir = dir / subfolder
        return [
            {
                "key": str(file.relative_to(dir)),
                "size": file.stat().st_size,
                "modified": file.stat().st_mtime * 1000,
            }
            for file in dir.glob("**/*")
            if not file.is_dir()
        ]

    def write_file(self, fold_id: int, file_path: str, contents, binary=False):
        """Write contents to a local file under the fold directory."""
        padded_fold_id = "%06d" % fold_id
        target_path = self.local_directory / padded_fold_id / file_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        mode = "wb" if binary or isinstance(contents, (bytes, bytearray)) else "w"
        with open(target_path, mode) as f:
            f.write(contents)

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

    def get_blob(self, fold_id, file_path):
        """Gets a Blob object from local storage based on fold_id and relative_path."""
        """
        Retrieves a LocalBlob object for the specified file.

        Args:
            fold_id (int): The fold identifier.
            file_path (str): The relative path to the file within the fold.

        Returns:
            LocalBlob: An instance of LocalBlob representing the file.

        Raises:
            BadRequest: If the file does not exist.
        """
        padded_fold_id = f"{fold_id:06d}"
        fpath = self.local_directory / padded_fold_id / file_path

        blob = LocalBlob(fpath)

        if not blob.exists():
            raise BadRequest(f"File not found at path {fpath}.")

        return blob


class GcloudStorageAccessor(StorageAccessor):
    def __init__(self):
        """Initialize local variables."""
        self.project = None
        self.client = None
        self.bucket_name = None
        self.bucket_prefix = None

    def setup(self, fold_gcloud_project, foldy_gstorage_dir):
        self.project = fold_gcloud_project
        self.client = Client(fold_gcloud_project)

        match = re.match(r"gs://(.*)", foldy_gstorage_dir)
        assert match, "Invalid google storage directory: {foldy_gstorage_dir}"

        self.bucket_name = match.groups()[0].split("/")[0]
        self.bucket_prefix = "/".join(match.groups()[0].split("/")[1:])

    def list_files(self, fold_id, subfolder=None):
        """Returns a list of dicts describing the contents of this fold's folder.

        Arguments:
          fold_id: fold to query
          subfolder: prefix within the fold bucket to search, eg 'dock/nadh'.

        Output: List of dictionaries with keys:
          key: path to the file
          size: size of the file
          modified: last modification time
        """
        padded_fold_id = "%06d" % fold_id
        prefix = (
            f"{self.bucket_prefix}/{padded_fold_id}"
            if self.bucket_prefix
            else padded_fold_id
        )
        if subfolder:
            prefix = f"{prefix}/{subfolder}"

        bucket = self.client.get_bucket(self.bucket_name)
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

    def write_file(self, fold_id: int, file_path: str, contents, binary=False):
        """Write contents to GCloud Storage under the fold directory."""
        bucket = self.client.get_bucket(self.bucket_name)

        padded_fold_id = "%06d" % fold_id

        # Build the full path with fold prefix
        prefixed_path = f"{padded_fold_id}/{file_path}"
        if self.bucket_prefix:
            prefixed_path = f"{self.bucket_prefix}/{prefixed_path}"

        blob = bucket.blob(prefixed_path)

        if binary or isinstance(contents, (bytes, bytearray)):
            blob.upload_from_string(contents, content_type="application/octet-stream")
        else:
            blob.upload_from_string(contents)

    def get_binary(self, fold_id, relative_path):
        """Gets a file (as a binary string) from gcloud, with a relative path within the fold dir."""
        download_start = time.time()
        padded_fold_id = "%06d" % fold_id

        if self.bucket_prefix:
            gcloud_path = f"{self.bucket_prefix}/{padded_fold_id}/{relative_path}"
        else:
            gcloud_path = f"{padded_fold_id}/{relative_path}"
        blobs = list(
            self.client.list_blobs(
                bucket_or_name=self.bucket_name,
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

    def get_blob(self, fold_id, relative_path):
        """Gets a Blob object from GCS based on fold_id and relative_path."""
        padded_fold_id = f"{fold_id:06d}"
        if self.bucket_prefix:
            gcloud_path = f"{self.bucket_prefix}/{padded_fold_id}/{relative_path}"
        else:
            gcloud_path = f"{padded_fold_id}/{relative_path}"

        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(gcloud_path)

        if not blob.exists():
            raise BadRequest(f"File not found at path {gcloud_path}.")

        return blob


class FoldStorageManager:
    """Manages access to the backend storage for fold, for short term use only (see below).

    Note that this class aggressively caches query results to improve latency. There
    is minimal logic to refresh the cache entries. It it meant to be destroyed
    shortly after instantiation, eg in the lifetime of a single query."""

    storage_manager = None

    def setup(self):
        """Raises BadRequest if setup fails."""
        # print(current_app.config, flush=True)
        if current_app.config["FOLDY_STORAGE_TYPE"] == "Local":
            foldy_dir = current_app.config["FOLDY_LOCAL_STORAGE_DIR"]
            assert foldy_dir, "FOLDY_LOCAL_STORAGE_DIR is not set"
            self.storage_manager = LocalStorageAccessor()
            self.storage_manager.setup(foldy_dir)

        elif current_app.config["FOLDY_STORAGE_TYPE"] == "Cloud":
            project = current_app.config["FOLDY_GCLOUD_PROJECT"]
            bucket = current_app.config["FOLDY_GSTORAGE_DIR"]
            assert project, "FOLDY_GCLOUD_PROJECT is not set"
            assert bucket, "FOLDY_GSTORAGE_DIR is not set"
            self.storage_manager = GcloudStorageAccessor()
            self.storage_manager.setup(project, bucket)

    def get_fold_with_state(self, fold_id, only_public):
        fold = Fold.get_by_id(fold_id)

        if not fold:
            raise BadRequest(f"Fold {fold_id} not found.")

        if only_public and not fold.public:
            abort(403, description="You do not have access to this resource.")

        return fold

    def get_folds_with_state(
        self,
        filter: str or None,
        tag: str or None,
        only_public: bool,
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

        if only_public:
            query = query.filter(Fold.public)

        query = query.order_by(Fold.id.desc())

        iterable = query
        if page and per_page:
            iterable = query.paginate(page=page, per_page=per_page).items

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
        aa_blob_path = f"{padded_fold_id}.fasta"
        dna_blob_path = f"{padded_fold_id}_dna.fasta"

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

        self.storage_manager.write_file(id, aa_blob_path, aa_contents)
        self.storage_manager.write_file(id, dna_blob_path, dna_contents)

    def make_new_folds(
        self,
        user_email,
        folds_data,
        start_fold_job,
        email_on_completion,
        skip_duplicate_entries,
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
                af2_model_preset = (
                    fold_data.get("af2_model_preset", None) or "monomer_ptm"
                )

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

    def get_fold_file_zip(self, fold_ids, relative_fpath, output_dirname):
        """Returns an open file handle to a temporary file with a certain file zipped up."""
        tmp = tempfile.TemporaryFile()

        with zipfile.ZipFile(tmp, "w") as archive:
            for fold_id in fold_ids:
                fold = Fold.get_by_id(fold_id)
                if not fold:
                    raise BadRequest(f"Could not find fold {fold_id}")

                try:
                    fold_pdb_binary = self.storage_manager.get_binary(
                        fold_id, relative_fpath  # "ranked_0.pdb"
                    )
                    padded_fold_id = format(fold_id, "06")
                    archive.writestr(
                        f"{output_dirname}/{padded_fold_id}/{relative_fpath}",
                        fold_pdb_binary,
                    )
                except Exception as e:
                    logging.log(e)
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

    def get_dock_sdf(self, dock):
        """Returns a binary string with the contents of the SDF file."""
        # Prior to 10/31/23, we didn't extract DiffDock confidences ahead of time.
        # This code backfills for old runs.
        if dock.tool == "diffdock" and not dock.pose_confidences:
            print("START UPDATING CONFIDENCES", flush=True)
            confidence_str = self.get_diffdock_pose_confidences(
                dock.receptor_fold_id, dock.ligand_name
            )
            dock.update(pose_confidences=confidence_str)
            print("FINISHED UPDATING CONFIDENCES", flush=True)

        # All tools should have a poses.sdf file, but for old diffdock structures,
        # we'll also try the old output filename:
        try:
            return self.storage_manager.get_binary(
                dock.receptor_fold_id, f"dock/{dock.ligand_name}/poses.sdf"
            )
        except Exception as e:
            if dock.tool != "diffdock":
                raise e

            return self.storage_manager.get_binary(
                dock.receptor_fold_id, f"dock/{dock.ligand_name}/rank1.sdf"
            )

    def get_diffdock_pose_confidences(self, fold_id, ligand_name):
        confidence_map = {}

        ligand_files = self.storage_manager.list_files(
            fold_id, subfolder=f"dock/{ligand_name}"
        )
        for file_info in ligand_files:
            # Confidence can be negative or positive
            # https://github.com/gcorso/DiffDock/issues/152
            fname = file_info["key"].split("/")[-1]
            match = re.match(r"rank(\d+)_confidence(-?\d*\.?\d*).sdf", fname)
            if match:
                confidence_map[int(match.groups()[0])] = float(match.groups()[1])

        confidence_list = []
        max_rank = max(confidence_map.keys())
        for rank in range(1, max_rank + 1):
            if rank not in confidence_map:
                print(
                    f'ERROR: Missing rank {rank} among files {[l["key"] for l in ligand_files]}'
                )
            confidence_list.append(confidence_map[rank])

        return ",".join([str(c) for c in confidence_list])
