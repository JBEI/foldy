"""Flask views for admin usage (eg, upgrading DBs, killing jobs, etc)."""

import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

from flask import current_app, request
from flask_jwt_extended import jwt_required
from flask_migrate import stamp, upgrade
from flask_restx import Namespace, Resource, fields
from rq.command import send_shutdown_command
from rq.registry import FailedJobRegistry
from sqlalchemy.sql.elements import and_
from werkzeug.exceptions import BadRequest

from app.api_fields import (
    kill_worker_fields,
    queue_test_job_fields,
    remove_failed_jobs_fields,
    stamp_dbs_fields,
)
from app.authorization import verify_has_edit_access
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.rq_helpers import get_queue, get_redis_connection
from app.jobs import other_jobs
from app.models import Embedding, FewShot, Fold, Invokation, Naturalness
from app.util import start_stage

ns = Namespace("admin_views", decorators=[jwt_required(fresh=True), verify_has_edit_access])


@ns.route("/createdbs")
class CreateDbsResource(Resource):
    def post(self) -> None:
        """Create all database tables.

        Returns:
            None
        """
        db.create_all()


@ns.route("/upgradedbs")
class UpgradeDbsResource(Resource):
    def post(self) -> None:
        """Upgrade database to latest migration.

        Returns:
            None
        """
        upgrade()


@ns.route("/stampdbs")
class StampDbsResource(Resource):
    @ns.expect(stamp_dbs_fields)
    def post(self) -> None:
        """Stamp database with specified migration revision.

        Returns:
            None
        """
        data = request.get_json()
        revision = data["revision"]
        stamp(revision=revision)


@ns.route("/remove_failed_jobs")
class RemoveFailedJobsResource(Resource):
    @ns.expect(remove_failed_jobs_fields)
    def post(self) -> None:
        """Remove all failed jobs from specified queue.

        Returns:
            None

        Raises:
            BadRequest: If queue doesn't exist
        """
        data = request.get_json()
        queue_name = data["queue"]
        q = get_queue(queue_name)
        registry = FailedJobRegistry(queue=q)

        count = 0
        for job_id in registry.get_job_ids():
            try:
                registry.remove(job_id, delete_job=True)
                count += 1
            except Exception as e:
                logging.error(f"Error removing job {job_id}: {e}")

        logging.info(f"Removed {count} failed jobs from {queue_name} queue")


@ns.route("/kill_worker")
class KillWorkerResource(Resource):
    @ns.expect(kill_worker_fields)
    def post(self) -> None:
        """Send shutdown command to a specific worker.

        Returns:
            None
        """
        data = request.get_json()
        worker_id = data["worker_id"]
        logging.info(f"Sending shutdown command to worker {worker_id}")
        send_shutdown_command(get_redis_connection(), worker_id)


@ns.route("/set_all_unset_model_presets")
class SetAllUnsetModelPresetsResource(Resource):
    def post(self) -> bool:
        """Set default model preset for all folds without one.

        Returns:
            True if operation was successful
        """
        count = 0
        for (fold_id,) in db.session.query(Fold.id).all():
            fold = Fold.get_by_id(fold_id)

            if fold.af2_model_preset:
                continue

            fold.update(af2_model_preset="monomer_ptm")
            count += 1

        logging.info(f"Set model preset for {count} folds")
        return True


@ns.route("/killFolds/<string:folds_range>")
class KillFoldsResource(Resource):
    def post(self, folds_range: str) -> bool:
        """Delete invocations for folds within a specific ID range.

        Args:
            folds_range: Range of fold IDs in format "start-end"

        Returns:
            True if operation was successful

        Raises:
            BadRequest: If range format is invalid
        """
        range_parts = folds_range.split("-")
        if len(range_parts) != 2:
            raise BadRequest(f'Invalid fold range "{folds_range}" must look like "10-60".')

        try:
            fold_lower_bound = int(range_parts[0])
            fold_upper_bound = int(range_parts[1])
        except ValueError:
            raise BadRequest(f'Invalid fold range "{folds_range}", range must be integers.')

        folds_in_range = db.session.query(Fold.id).filter(
            and_(Fold.id >= fold_lower_bound, Fold.id < fold_upper_bound)
        )
        for (fold_id,) in folds_in_range.all():
            fold = Fold.get_by_id(fold_id)

            for invokation in fold.jobs:
                # Note, we can't cancel the jobs, since we don't store the RQ ids...
                # But if we delete the invokations, the jobs will die quickly.
                # job = Job.fetch(invokation.id, connection=rq.connection)
                # job.cancel()
                invokation.delete()

        return True


@ns.route("/bulkAddTag/<string:folds_range>/<string:new_tag>")
class BulkAddTagResource(Resource):
    def post(self, folds_range: str, new_tag: str) -> bool:
        """Add a tag to multiple folds within a specific ID range.

        Args:
            folds_range: Range of fold IDs in format "start-end"
            new_tag: Tag to add to folds

        Returns:
            True if operation was successful

        Raises:
            BadRequest: If range format is invalid or tag is not alphanumeric
        """
        range_parts = folds_range.split("-")
        if len(range_parts) != 2:
            raise BadRequest(f'Invalid fold range "{folds_range}" must look like "10-60".')

        try:
            fold_lower_bound = int(range_parts[0])
            fold_upper_bound = int(range_parts[1])
        except ValueError:
            raise BadRequest(f'Invalid fold range "{folds_range}", range must be integers.')

        if not new_tag.isalnum():
            raise BadRequest(f"Tags must be alphanumeric, got {new_tag}")

        folds_in_range = db.session.query(Fold.id).filter(
            and_(Fold.id >= fold_lower_bound, Fold.id < fold_upper_bound)
        )
        for (fold_id,) in folds_in_range.all():
            fold = Fold.get_by_id(fold_id)

            if fold.tagstring:
                tags = fold.tagstring.split(",")
            else:
                tags = []

            if new_tag not in tags:
                tags += [new_tag]
                new_tagstring = ",".join([tag for tag in tags if tag])
                fold.update(tagstring=new_tagstring)
            # fold.update(tagstring=new_tag)

        return True


@ns.route("/sendtestemail")
class SendTestEmailResource(Resource):
    @ns.expect(queue_test_job_fields)
    @verify_has_edit_access
    def post(self) -> bool:
        """Queue a test email job.

        Returns:
            True if email job was queued successfully
        """
        q = get_queue("emailparrot")
        job = q.enqueue(
            other_jobs.send_email,
            1,
            "test_prot",
            "jacoberts@gmail.com",
            job_timeout="6h",
            result_ttl=48 * 60 * 60,  # 2 days
        )
        logging.info(f"Queued test email job with ID {job.id}")
        return True


@ns.route("/addInvokationToAllJobs/<string:job_type>/<string:job_state>")
class AddInvokationToAllJobsResource(Resource):
    @verify_has_edit_access
    def post(self, job_type: str, job_state: str) -> bool:
        """Add invocation of specified type and state to all folds that don't have it.

        Args:
            job_type: Type of job/invocation to add
            job_state: Initial state for the invocation

        Returns:
            True if operation was successful
        """
        logging.info(f"Adding invocation type={job_type}, state={job_state} to all folds")
        count = 0
        for (fold_id,) in db.session.query(Fold.id).all():
            fold = Fold.get_by_id(fold_id)
            if any([j.type == job_type for j in fold.jobs]):
                continue
            new_invokation = Invokation(fold_id=fold.id, type=job_type, state=job_state)
            new_invokation.save()
            count += 1

        logging.info(f"Added {count} new invocations")
        return True


@ns.route("/runUnrunStages/<string:stage_name>")
class RunUnrunStagesResource(Resource):
    @verify_has_edit_access
    def post(self, stage_name: str) -> bool:
        """Run a specific stage for all folds that haven't run it successfully.

        Args:
            stage_name: Name of the stage to run

        Returns:
            True if operation was successful
        """
        logging.info(f"Running stage {stage_name} for folds that haven't run it")
        count = 0

        for (fold_id,) in db.session.query(Fold.id).all():
            if stage_name == "write_fastas":
                start_stage(fold_id, stage_name, False)
                count += 1
                continue

            fold = Fold.get_by_id(fold_id)

            is_already_run = False
            for j in fold.jobs:
                if j.type == stage_name and j.state != "failed":
                    is_already_run = True
                    break

            if is_already_run:
                continue

            start_stage(fold_id, stage_name, False)
            count += 1

        logging.info(f"Started {stage_name} stage for {count} folds")
        return True


@ns.route("/populate_output_fpath_fields")
class PopulateOutputFpathFieldsResource(Resource):
    @verify_has_edit_access
    def post(self) -> Dict[str, int]:
        """Populate output_fpath fields for all existing naturalness, embedding, and few shot records.

        This function sets the output_fpath field based on the computed path logic
        for records that don't already have this field populated.

        Returns:
            Dictionary with counts of updated records for each model type
        """
        logging.info("Starting population of output_fpath fields for existing records")

        naturalness_updated = 0
        embedding_updated = 0
        few_shot_updated = 0

        # Update Naturalness records
        naturalness_records = Naturalness.query.filter(
            (Naturalness.output_fpath.is_(None)) | (Naturalness.output_fpath == "")
        ).all()

        for naturalness in naturalness_records:
            if naturalness.fold_id is not None:
                continue
            padded_fold_id = str(naturalness.fold_id).zfill(6)
            # Use the same logic as the computed property
            computed_path = f"naturalness/logits_{naturalness.name}_melted.csv"
            naturalness.update(output_fpath=computed_path)
            naturalness_updated += 1
            logging.info(
                f"Updated naturalness {naturalness.id} ({naturalness.name}) with path: {computed_path}"
            )

        # Update Embedding records
        embedding_records = Embedding.query.filter(
            (Embedding.output_fpath.is_(None)) | (Embedding.output_fpath == "")
        ).all()

        for embedding in embedding_records:
            if embedding.fold_id is not None:
                continue
            padded_fold_id = str(embedding.fold_id).zfill(6)
            # Use the same logic as the computed property
            computed_path = f"embed/{padded_fold_id}_embeddings_{embedding.embedding_model}_{embedding.name}.csv"
            embedding.update(output_fpath=computed_path)
            embedding_updated += 1
            logging.info(
                f"Updated embedding {embedding.id} ({embedding.name}) with path: {computed_path}"
            )

        # Update FewShot records
        few_shot_records = FewShot.query.filter(
            (FewShot.output_fpath.is_(None)) | (FewShot.output_fpath == "")
        ).all()

        for few_shot in few_shot_records:
            if few_shot.fold_id is not None:
                continue
            padded_fold_id = str(few_shot.fold_id).zfill(6)
            # Use the same logic as the computed property
            computed_path = f"evolve/{few_shot.name}/predicted_activity.csv"
            few_shot.update(output_fpath=computed_path)
            few_shot_updated += 1
            logging.info(
                f"Updated few_shot {few_shot.id} ({few_shot.name}) with path: {computed_path}"
            )

        total_updated = naturalness_updated + embedding_updated + few_shot_updated

        result = {
            "naturalness_updated": naturalness_updated,
            "embedding_updated": embedding_updated,
            "few_shot_updated": few_shot_updated,
            "total_updated": total_updated,
        }

        logging.info(
            f"Population complete: {total_updated} total records updated - "
            f"Naturalness: {naturalness_updated}, Embedding: {embedding_updated}, FewShot: {few_shot_updated}"
        )

        return result


@ns.route("/backfill_date_created")
class BackfillDateCreatedResource(Resource):
    @verify_has_edit_access
    def post(self) -> Dict[str, int]:
        """Backfill date_created fields for all existing naturalness, embedding, and few shot records.

        This function sets the date_created field based on the modification time of the output file
        using FoldStorageManager.storage_manager.list_files, or 1/1/2024 if file doesn't exist.

        Returns:
            Dictionary with counts of updated records for each model type
        """
        logging.info("Starting backfill of date_created fields for existing records")

        naturalness_updated = 0
        embedding_updated = 0
        few_shot_updated = 0

        # Fallback date if file doesn't exist or can't get mtime
        fallback_date = datetime(2024, 1, 1, tzinfo=UTC)

        # Initialize storage manager
        storage_manager = FoldStorageManager()
        storage_manager.setup()

        if not storage_manager.storage_manager:
            logging.error("Storage manager not initialized, using fallback date for all records")

        # Update Naturalness records
        naturalness_records = Naturalness.query.filter(Naturalness.date_created.is_(None)).all()

        for naturalness in naturalness_records:
            date_to_set = fallback_date

            try:
                if (
                    naturalness.fold_id
                    and naturalness.output_fpath_computed
                    and storage_manager.storage_manager
                ):
                    # Try to get file modification time from storage
                    files = storage_manager.storage_manager.list_files(naturalness.fold_id)
                    target_file = next(
                        (f for f in files if f["key"] == naturalness.output_fpath_computed), None
                    )

                    if target_file and "modified" in target_file:
                        # Convert from milliseconds to datetime
                        date_to_set = datetime.fromtimestamp(
                            target_file["modified"] / 1000.0, tz=UTC
                        )
                        logging.info(
                            f"Found file modification time for naturalness {naturalness.id}: {date_to_set}"
                        )
                    else:
                        logging.info(
                            f"File not found for naturalness {naturalness.id}, using fallback date"
                        )
            except Exception as e:
                logging.warning(
                    f"Error getting file mtime for naturalness {naturalness.id}: {e}, using fallback date"
                )

            naturalness.update(date_created=date_to_set)
            naturalness_updated += 1

        # Update Embedding records
        embedding_records = Embedding.query.filter(Embedding.date_created.is_(None)).all()

        for embedding in embedding_records:
            date_to_set = fallback_date

            try:
                if (
                    embedding.fold_id
                    and embedding.output_fpath_computed
                    and storage_manager.storage_manager
                ):
                    # Try to get file modification time from storage
                    files = storage_manager.storage_manager.list_files(embedding.fold_id)
                    target_file = next(
                        (f for f in files if f["key"] == embedding.output_fpath_computed), None
                    )

                    if target_file and "modified" in target_file:
                        # Convert from milliseconds to datetime
                        date_to_set = datetime.fromtimestamp(
                            target_file["modified"] / 1000.0, tz=UTC
                        )
                        logging.info(
                            f"Found file modification time for embedding {embedding.id}: {date_to_set}"
                        )
                    else:
                        logging.info(
                            f"File not found for embedding {embedding.id}, using fallback date"
                        )
            except Exception as e:
                logging.warning(
                    f"Error getting file mtime for embedding {embedding.id}: {e}, using fallback date"
                )

            embedding.update(date_created=date_to_set)
            embedding_updated += 1

        # Update FewShot records
        few_shot_records = FewShot.query.filter(FewShot.date_created.is_(None)).all()

        for few_shot in few_shot_records:
            date_to_set = fallback_date

            try:
                if (
                    few_shot.fold_id
                    and few_shot.output_fpath_computed
                    and storage_manager.storage_manager
                ):
                    # Try to get file modification time from storage
                    files = storage_manager.storage_manager.list_files(few_shot.fold_id)
                    target_file = next(
                        (f for f in files if f["key"] == few_shot.output_fpath_computed), None
                    )

                    if target_file and "modified" in target_file:
                        # Convert from milliseconds to datetime
                        date_to_set = datetime.fromtimestamp(
                            target_file["modified"] / 1000.0, tz=UTC
                        )
                        logging.info(
                            f"Found file modification time for few_shot {few_shot.id}: {date_to_set}"
                        )
                    else:
                        logging.info(
                            f"File not found for few_shot {few_shot.id}, using fallback date"
                        )
            except Exception as e:
                logging.warning(
                    f"Error getting file mtime for few_shot {few_shot.id}: {e}, using fallback date"
                )

            few_shot.update(date_created=date_to_set)
            few_shot_updated += 1

        total_updated = naturalness_updated + embedding_updated + few_shot_updated

        result = {
            "naturalness_updated": naturalness_updated,
            "embedding_updated": embedding_updated,
            "few_shot_updated": few_shot_updated,
            "total_updated": total_updated,
        }

        logging.info(
            f"Backfill complete: {total_updated} total records updated - "
            f"Naturalness: {naturalness_updated}, Embedding: {embedding_updated}, FewShot: {few_shot_updated}"
        )

        return result


@ns.route("/backfill_input_activity_fpath")
class BackfillInputActivityFpathResource(Resource):
    @verify_has_edit_access
    def post(self) -> Dict[str, int]:
        """Backfill input_activity_fpath fields for all existing FewShot records.

        This function sets the input_activity_fpath field using the following logic:
        (A) Skip if input_activity_fpath is already set
        (B) If output_fpath is set, use that directory but change filename to activity.xlsx
        (C) Otherwise, set to few_shots/<few_shot_name>/activity.xlsx

        Returns:
            Dictionary with count of updated records
        """
        logging.info("Starting backfill of input_activity_fpath fields for FewShot records")

        few_shot_updated = 0

        # Get all FewShot records that don't have input_activity_fpath set
        few_shot_records = FewShot.query.filter(
            (FewShot.input_activity_fpath.is_(None)) | (FewShot.input_activity_fpath == "")
        ).all()

        for few_shot in few_shot_records:
            input_activity_fpath = None

            if few_shot.output_fpath and few_shot.output_fpath.strip():
                # Case B: Use output_fpath directory but change filename to activity.xlsx
                output_dir = "/".join(few_shot.output_fpath.split("/")[:-1])
                input_activity_fpath = f"{output_dir}/activity.xlsx"
                logging.info(
                    f"Using output_fpath directory for few_shot {few_shot.id}: {input_activity_fpath}"
                )
            else:
                # Case C: Default to few_shots/<name>/activity.xlsx
                input_activity_fpath = f"few_shots/{few_shot.name}/activity.xlsx"
                logging.info(
                    f"Using default path for few_shot {few_shot.id}: {input_activity_fpath}"
                )

            few_shot.update(input_activity_fpath=input_activity_fpath)
            few_shot_updated += 1
            logging.info(
                f"Updated few_shot {few_shot.id} ({few_shot.name}) with input_activity_fpath: {input_activity_fpath}"
            )

        result = {"few_shot_updated": few_shot_updated, "total_updated": few_shot_updated}

        logging.info(
            f"Completed backfill of input_activity_fpath fields. Updated {few_shot_updated} FewShot records"
        )

        return result


# Create a separate namespace for messages endpoint without authentication
messages_ns = Namespace("messages", description="System messages")


@messages_ns.route("/messages")
class MessagesResource(Resource):
    def get(self) -> List[Dict[str, str]]:
        """Get system messages to display to users.

        Returns:
            List of message dictionaries with 'message' and 'type' keys
        """
        messages: List[Dict[str, str]] = []

        # Check for warning message from environment
        warning_message = current_app.config.get("WARNING_MESSAGE")
        if warning_message:
            messages.append({"message": warning_message, "type": "warning"})

        return messages
