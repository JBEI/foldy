import datetime
import io
import json
import sys

from dnachisel import biotools
from flask import current_app, request, send_file, Response, make_response, jsonify
from flask_jwt_extended import get_raw_jwt
from flask_jwt_extended.utils import get_jwt_identity
from flask_migrate import stamp, upgrade
from flask_restplus import Namespace
from flask_jwt_extended import fresh_jwt_required
from flask_restplus import Resource
from flask_restplus import fields
from flask_restplus import reqparse
from flask_restplus.inputs import email
from google.cloud.storage.client import Client
import numpy as np
from redis import Redis
from rq.command import send_shutdown_command
from rq.job import Job, Retry
from rq.registry import FailedJobRegistry
from sqlalchemy.sql.elements import and_
from sqlalchemy import delete, func
from sqlalchemy.orm import joinedload
from werkzeug.exceptions import BadRequest

from app import jobs
from app.models import Dock, Fold, Invokation, User
from app.extensions import compress, db, rq
from app.util import start_stage, FoldStorageUtil, get_job_type_replacement

ns = Namespace("most_views", decorators=[fresh_jwt_required])


@ns.route("/test")
class TestResource(Resource):
    def get(self):
        return "Healthy"


simple_invokation_fields = ns.model(
    "SimpleInvokationFields",
    {
        "id": fields.Integer(required=True),
        "type": fields.String(required=False),
        "state": fields.String(required=False),
    },
)

full_invokation_fields = ns.clone(
    "InvokationFields",
    simple_invokation_fields,
    {
        "job_id": fields.String(required=False),
        "log": fields.String(required=False),
        "timedelta_sec": fields.Float(
            readonly=True,
            required=False,
            attribute=lambda r: r.timedelta.total_seconds() if r.timedelta else None,
        ),
    },
)


dock_fields = ns.model(
    "DockFields",
    {
        "id": fields.Integer(readonly=True),
        "fold_id": fields.Integer(required=True),
        "ligand_name": fields.String(required=True),
        "ligand_smiles": fields.String(required=True),
        "bounding_box_residue": fields.String(
            required=False,
            nullable=True,
            help="Residue to center bounding box on, like Y74.",
        ),
        "bounding_box_radius_angstrom": fields.Float(
            required=False, nullable=True, help="Radius of bounding box in angstroms."
        ),
        "invokation_id": fields.Integer(required=False),
        "pose_energy": fields.Float(required=False),
    },
)

get_folds_fields = ns.model("GetFolds", {"filter": fields.String(required=False)})

fold_fields = ns.model(
    "Fold",
    {
        "id": fields.Integer(readonly=True, required=False),
        "name": fields.String(),
        "owner": fields.String(attribute="user.email", required=False),
        "tags": fields.List(fields.String()),
        "create_date": fields.DateTime(readonly=True, required=False),
        "sequence": fields.String(),
        "af2_model_preset": fields.String(required=False),
        "disable_relaxation": fields.String(required=False),
        "jobs": fields.List(fields.Nested(simple_invokation_fields)),
        "docks": fields.List(fields.Nested(dock_fields)),
    },
)

new_folds_fields = ns.model(
    "NewFolds",
    {
        "folds_data": fields.List(fields.Nested(fold_fields, skip_none=True)),
        "start_fold_job": fields.Boolean(required=False),
        "email_on_completion": fields.Boolean(required=False),
        "skip_duplicate_entries": fields.Boolean(required=False),
    },
)


get_folds_parser = reqparse.RequestParser()
get_folds_parser.add_argument(
    "filter",
    type=str,
    help="Optional filter parameter.",
    required=False,
)
get_folds_parser.add_argument(
    "tag",
    type=str,
    help="Optionally filter to this tag.",
    required=False,
)
get_folds_parser.add_argument(
    "page",
    type=int,
    help="Page number to retrieve.",
    required=False,
)
get_folds_parser.add_argument(
    "per_page",
    type=int,
    help="Number of items per page.",
    required=False,
)


@ns.route("/fold")
class FoldsResource(Resource):
    @ns.expect(get_folds_parser)
    @ns.marshal_list_with(fold_fields, skip_none=True)
    def get(self):
        args = get_folds_parser.parse_args()

        filter = args.get("filter", None)
        tag = args.get("tag", None)
        page = args.get("page", None)
        per_page = args.get("per_page", None)

        manager = FoldStorageUtil()
        manager.setup()

        folds = manager.get_folds_with_state(filter, tag, page, per_page)
        return folds

    # TODO(jbr): Figure out what is causing this call to fail and add validation.
    @ns.expect(new_folds_fields, validate=False)
    def post(self):
        """Returns True if queueing is successful."""
        manager = FoldStorageUtil()
        manager.setup()

        folds_data = request.get_json()["folds_data"]
        start_fold_job = request.get_json()["start_fold_job"]
        email_on_completion = request.get_json().get("email_on_completion", False)
        skip_duplicate_entries = request.get_json().get("skip_duplicate_entries", False)

        return manager.make_new_folds(
            get_jwt_identity(),
            folds_data,
            start_fold_job,
            email_on_completion,
            skip_duplicate_entries,
        )


@ns.route("/fold/<int:fold_id>")
class FoldResource(Resource):
    @ns.marshal_with(fold_fields)
    def get(self, fold_id):
        manager = FoldStorageUtil()
        manager.setup()
        return manager.get_fold_with_state(fold_id)

    @ns.expect(fold_fields, validate=False)
    def post(self, fold_id):
        try:
            fields_to_update = request.get_json()
            if "tags" in fields_to_update:
                for tag in fields_to_update["tags"]:
                    if not tag.isalnum():
                        raise BadRequest(f"Bad group name, {tag} is not alphanumeric.")
                fields_to_update["tagstring"] = ",".join(fields_to_update["tags"])
                del fields_to_update["tags"]
            Fold.get_by_id(fold_id).update(**fields_to_update)
            return True
        except Exception as e:
            raise BadRequest(f"Update operation failed {e}")


@ns.route("/invokation/<int:invokation_id>")
class InvokationLogsResource(Resource):
    @ns.marshal_with(full_invokation_fields)
    def get(self, invokation_id):
        return Invokation.get_by_id(invokation_id)


fold_pdb_fields = ns.model(
    "FoldPdb",
    {
        "pdb_string": fields.String(readonly=True),
    },
)

# @compress.compressed()
@ns.route("/fold_pdb/<int:fold_id>/<int:model_number>")
class FoldResource(Resource):
    @ns.marshal_with(fold_pdb_fields)
    def get(self, fold_id, model_number):
        manager = FoldStorageUtil()
        manager.setup()
        return {"pdb_string": manager.get_fold_pdb(fold_id, model_number)}


fold_pdb_zip_fields = ns.model(
    "FoldPdbZip",
    {
        "fold_ids": fields.List(fields.Integer(required=True)),
        "dirname": fields.String(),
    },
)


# @compress.compressed()
@ns.route("/fold_pdb_zip")
class FoldPdbZipResource(Resource):
    @ns.expect(fold_pdb_zip_fields)
    def post(self):
        manager = FoldStorageUtil()
        manager.setup()

        return send_file(
            manager.get_fold_pdb_zip(
                request.get_json()["fold_ids"], request.get_json()["dirname"]
            ),
            mimetype="application/octet-stream",
            attachment_filename="fold_pdbs.zip",
            as_attachment=True,
        )


@ns.route("/fold_pkl/<int:fold_id>/<int:model_number>")
class FoldPklResource(Resource):
    def post(self, fold_id, model_number):
        manager = FoldStorageUtil()
        manager.setup()
        pkl_byte_str = manager.get_fold_pkl(fold_id, model_number)
        # TODO: Stream!
        # https://flask.palletsprojects.com/en/1.1.x/patterns/streaming/
        # https://www.reddit.com/r/Python/comments/dha9kw/flask_send_large_file/
        return send_file(
            io.BytesIO(pkl_byte_str),
            mimetype="application/octet-stream",
            attachment_filename="model.pkl",
            as_attachment=True,
        )


@ns.route("/dock_sdf/<int:fold_id>/<string:ligand_name>")
class FoldPklResource(Resource):
    def post(self, fold_id, ligand_name):
        manager = FoldStorageUtil()
        manager.setup()
        sdf_str = manager.get_dock_sdf(fold_id, ligand_name)
        return send_file(
            io.BytesIO(sdf_str),
            mimetype="application/octet-stream",
            attachment_filename=f"{ligand_name}.sdf",
            as_attachment=True,
        )


@ns.route("/file/list/<int:fold_id>")
class FoldFileResource(Resource):
    def get(self, fold_id):
        manager = FoldStorageUtil()
        manager.setup()
        return manager.storage_manager.list_files(fold_id)


@ns.route("/file/download/<int:fold_id>/<path:subpath>")
class FileDownloadResource(Resource):
    def post(self, fold_id, subpath):
        # TODO: test this.
        manager = FoldStorageUtil()
        manager.setup()
        sdf_str = manager.storage_manager.get_binary(fold_id, subpath)
        fname = subpath.split("/")[-1]
        return send_file(
            io.BytesIO(sdf_str),
            mimetype="application/octet-stream",
            attachment_filename=fname,
            as_attachment=True,
        )


def convert_array_to_json_string(table):
    """Convert a numpy array of floats to json compatible table.

    Useful when array.tolist would take up too much space
    (which is the case for big contact maps and PAE)."""
    json_table = "["
    for ii, row in enumerate(table):
        if ii > 0:
            json_table += ",\n"
        row_str = "[" + ", ".join(["%.4g" % e for e in row]) + "]"
        json_table += row_str
    json_table += " ]"
    return json_table


pae_fields = ns.model(
    "PAE",
    {
        "pae": fields.List(fields.List(fields.Float(readonly=True))),
    },
)


@ns.route("/pae/<int:fold_id>/<int:model_number>")
class FoldResource(Resource):
    # We can't marshal, since we're not returning json.
    # @ns.marshal_with(pae_fields)
    def get(self, fold_id, model_number):
        manager = FoldStorageUtil()
        manager.setup()
        pae = manager.get_model_pae(fold_id, model_number)

        # Note that using "tolist" and then json.dumps (or jsonify) would take
        # up too much memory, so we convert to json manually.
        json_resp = '{ "pae": ' + convert_array_to_json_string(pae) + " }"
        resp = make_response(
            json_resp,
        )
        resp.headers["Content-Type"] = "application/json"
        resp.status_code = 200

        return resp


contact_prob_fields = ns.model(
    "ContactProb",
    {
        "contact_prob": fields.List(fields.List(fields.Float(readonly=True))),
    },
)


@ns.route("/contact_prob/<int:fold_id>/<int:model_number>")
class ContactProbResource(Resource):
    # We can't marshal, since we're not returning json.
    # @ns.marshal_with(contact_prob_fields)
    def get(self, fold_id, model_number):
        manager = FoldStorageUtil()
        manager.setup()
        contact_prob = manager.get_contact_prob(fold_id, model_number)

        json_resp = (
            '{ "contact_prob": ' + convert_array_to_json_string(contact_prob) + " }"
        )

        # with np.printoptions(threshold=np.inf):
        #   json_resp = ('{ "contact_prob": ' + np.array2string(
        #     contact_prob,
        #     precision=3,
        #     separator=',',
        #     suppress_small=True,
        #   ) + ' }')
        resp = make_response(
            json_resp,
        )
        resp.headers["Content-Type"] = "application/json"
        resp.status_code = 200

        return resp


@ns.route("/pfam/<int:fold_id>")
class PfamResource(Resource):
    def get(self, fold_id):
        manager = FoldStorageUtil()
        manager.setup()
        pfam_annotations = manager.get_pfam(fold_id)

        return pfam_annotations


@ns.route("/dock")
class DockResource(Resource):
    # TODO(jbr): Figure out what is causing this call to fail and add validation.
    # @ns.expect(dock_fields)
    def post(self):
        req = request.get_json()

        ligand_name = req["ligand_name"]
        ligand_smiles = req["ligand_smiles"]
        fold_id = req["fold_id"]

        if not ligand_name.isalnum():
            raise BadRequest(
                f"Ligand names must be alphanumeric, {ligand_name} is invalid."
            )

        fold = Fold.get_by_id(fold_id)

        new_invokation_id = get_job_type_replacement(fold, f"dock_{ligand_name}")

        new_dock = Dock(
            ligand_name=ligand_name,
            ligand_smiles=ligand_smiles,
            receptor_fold_id=fold_id,
            invokation_id=new_invokation_id,
            bounding_box_residue=req.get("bounding_box_residue", None),
            bounding_box_radius_angstrom=req.get("bounding_box_radius_angstrom", None),
        )
        new_dock.save()

        cpu_q = rq.get_queue("cpu")
        cpu_q.enqueue(
            jobs.dock,
            new_dock.id,
            new_invokation_id,
            current_app.config["FOLDY_GCLOUD_BUCKET"],
            job_timeout="2h",
            result_ttl=48 * 60 * 60,  # 2 days
        )

        return True


@ns.route("/dock/<int:dock_id>")
class DockResource(Resource):
    def delete(self, dock_id):
        dock = Dock.get_by_id(dock_id)

        dock.delete()

        return True


queue_test_job_fields = ns.model(
    "QueueTestJobField",
    {"queue": fields.String(), "command": fields.String(required=False)},
)


@ns.route("/queuetestjob")
class QueueTestJobResource(Resource):
    @ns.expect(queue_test_job_fields)
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


queue_job_fields = ns.model(
    "QueueFold",
    {
        "fold_id": fields.String(),
        "stage": fields.String(),
        "email_on_completion": fields.Boolean(),
    },
)


@ns.route("/queuejob")
class QueueJobResource(Resource):
    @ns.expect()
    def post(self):
        start_stage(
            request.get_json()["fold_id"],
            request.get_json()["stage"],
            request.get_json()["email_on_completion"],
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
