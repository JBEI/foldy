"""Flask views for admin usage (eg, upgrading DBs, killing jobs, etc)."""

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

from app import jobs
from app.models import Fold, Invokation
from app.extensions import db, rq
from app.authorization import verify_has_edit_access
from app.util import start_stage


ns = Namespace(
    "admin_views", decorators=[jwt_required(fresh=True), verify_has_edit_access]
)


@ns.route("/createdbs")
class CreateDbsResource(Resource):
    def post(self):
        db.create_all()


@ns.route("/upgradedbs")
class UpgradeDbsResource(Resource):
    def post(self):
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
    def post(self):
        stamp(revision=request.get_json()["revision"])


remove_failed_jobs_fields = ns.model("RemoveFailedJobs", {"queue": fields.String()})


@ns.route("/remove_failed_jobs")
class RemoveFailedJobsResource(Resource):
    @ns.expect(remove_failed_jobs_fields)
    def post(self):
        q = rq.get_queue(request.get_json()["queue"])
        registry = FailedJobRegistry(queue=q)
        for job_id in registry.get_job_ids():
            try:
                registry.remove(job_id, delete_job=True)
            except Exception as e:
                print(e)


kill_worker_fields = ns.model("KillWorkerFields", {"worker_id": fields.String()})


@ns.route("/kill_worker")
class KillWorkerResource(Resource):
    @ns.expect(kill_worker_fields)
    def post(self):
        worker_id = request.get_json()["worker_id"]
        send_shutdown_command(rq.connection, worker_id)


@ns.route("/set_all_unset_model_presets")
class SetAllUnsetModelPresets(Resource):
    def post(self):
        for (fold_id,) in db.session.query(Fold.id).all():
            fold = Fold.get_by_id(fold_id)

            if fold.af2_model_preset:
                continue

            fold.update(af2_model_preset="monomer_ptm")

        return True


@ns.route("/killFolds/<string:folds_range>")
class SetAllUnsetModelPresets(Resource):
    def post(self, folds_range):
        range = folds_range.split("-")
        if len(range) != 2:
            raise BadRequest(
                f'Invalid fold range "{folds_range}" must look like "10-60".'
            )

        try:
            fold_lower_bound = int(range[0])
            fold_upper_bound = int(range[1])
        except:
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
    def post(self, folds_range, new_tag):
        range = folds_range.split("-")
        if len(range) != 2:
            raise BadRequest(
                f'Invalid fold range "{folds_range}" must look like "10-60".'
            )

        try:
            fold_lower_bound = int(range[0])
            fold_upper_bound = int(range[1])
        except:
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

            tags = fold.tagstring.split(",")

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


@ns.route("/queuetestjob")
class QueueTestJobResource(Resource):
    @ns.expect(queue_test_job_fields)
    @verify_has_edit_access
    def post(self):
        q = rq.get_queue(request.get_json()["queue"])
        q.enqueue(
            jobs.add,
            1,
            3,
            job_timeout="6h",
            result_ttl=48 * 60 * 60,  # 2 days
        )
        return True


@ns.route("/sendtestemail")
class SendTestEmailResource(Resource):
    @ns.expect(queue_test_job_fields)
    @verify_has_edit_access
    def post(self):
        q = rq.get_queue("emailparrot")
        q.enqueue(
            jobs.send_email,
            1,
            "test_prot",
            "jacoberts@gmail.com",
            job_timeout="6h",
            result_ttl=48 * 60 * 60,  # 2 days
        )
        return True


@ns.route("/addInvokationToAllJobs/<string:job_type>/<string:job_state>")
class SendTestEmailResource(Resource):
    @verify_has_edit_access
    def post(self, job_type, job_state):
        for (fold_id,) in db.session.query(Fold.id).all():
            fold = Fold.get_by_id(fold_id)
            if any([j.type == job_type for j in fold.jobs]):
                continue
            new_invokation = Invokation(fold_id=fold.id, type=job_type, state=job_state)
            new_invokation.save()

        return True


@ns.route("/runUnrunStages/<string:stage_name>")
class RunUnrunStagesResource(Resource):
    @verify_has_edit_access
    def post(self, stage_name):
        for (fold_id,) in db.session.query(Fold.id).all():
            if stage_name == "write_fastas":
                start_stage(fold_id, stage_name, False)
                continue

            fold = Fold.get_by_id(fold_id)

            is_already_run = False
            for j in fold.jobs:
                if j.type == stage_name and j.state != "failed":
                    is_already_run = True
            if is_already_run:
                continue

            start_stage(fold_id, stage_name, False)

        return True
