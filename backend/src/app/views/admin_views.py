"""Flask views for admin usage (eg, upgrading DBs, killing jobs, etc)."""

import logging
from typing import Dict, Any, List, Tuple, Union, Optional, Type, cast

from flask import request
from flask_migrate import stamp, upgrade
from flask_restx import Namespace
from flask_jwt_extended import jwt_required
from flask_restx import Resource
from flask_restx import fields
from rq.command import send_shutdown_command
from rq.registry import FailedJobRegistry
from sqlalchemy.sql.elements import and_
from werkzeug.exceptions import BadRequest

from app.jobs import other_jobs
from app.models import Fold, Invokation
from app.extensions import db, rq
from app.authorization import verify_has_edit_access
from app.util import start_stage


ns = Namespace(
    "admin_views", decorators=[jwt_required(fresh=True), verify_has_edit_access]
)


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


stamp_dbs_fields = ns.model(
    "StampDbsFields",
    {
        "revision": fields.String(),
    },
)


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


remove_failed_jobs_fields = ns.model("RemoveFailedJobs", {"queue": fields.String()})


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
        q = rq.get_queue(queue_name)
        registry = FailedJobRegistry(queue=q)
        
        count = 0
        for job_id in registry.get_job_ids():
            try:
                registry.remove(job_id, delete_job=True)
                count += 1
            except Exception as e:
                logging.error(f"Error removing job {job_id}: {e}")
        
        logging.info(f"Removed {count} failed jobs from {queue_name} queue")


kill_worker_fields = ns.model("KillWorkerFields", {"worker_id": fields.String()})


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
        send_shutdown_command(rq.connection, worker_id)


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
            raise BadRequest(
                f'Invalid fold range "{folds_range}" must look like "10-60".'
            )

        try:
            fold_lower_bound = int(range_parts[0])
            fold_upper_bound = int(range_parts[1])
        except ValueError:
            raise BadRequest(
                f'Invalid fold range "{folds_range}", range must be integers.'
            )

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
            raise BadRequest(
                f'Invalid fold range "{folds_range}" must look like "10-60".'
            )

        try:
            fold_lower_bound = int(range_parts[0])
            fold_upper_bound = int(range_parts[1])
        except ValueError:
            raise BadRequest(
                f'Invalid fold range "{folds_range}", range must be integers.'
            )

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


queue_test_job_fields = ns.model(
    "QueueTestJobField",
    {"queue": fields.String(), "command": fields.String(required=False)},
)


@ns.route("/sendtestemail")
class SendTestEmailResource(Resource):
    @ns.expect(queue_test_job_fields)
    @verify_has_edit_access
    def post(self) -> bool:
        """Queue a test email job.
        
        Returns:
            True if email job was queued successfully
        """
        q = rq.get_queue("emailparrot")
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
