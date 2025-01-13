import pandas as pd
import json

import numpy as np
from flask import request
from flask_restx import Resource, fields
from flask_jwt_extended import jwt_required
from flask_restx import Namespace
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

from app.views.other_views import evolution_fields
from app.authorization import verify_has_edit_access
from app.extensions import db, rq
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.sequence_util import (
    maybe_get_seq_id_error_message,
    get_measured_and_unmeasured_mutant_seq_ids,
)
from app.models import Fold, Evolution
from app.util import get_job_type_replacement
from app.jobs import evolve_jobs

ns = Namespace("evolve_views", decorators=[jwt_required(fresh=True)])

upload_parser = ns.parser()
upload_parser.add_argument("name", type=str, location="form", required=True)
upload_parser.add_argument("fold_id", type=str, location="form", required=True)
upload_parser.add_argument(
    "activity_file", type=FileStorage, location="files", required=True
)
upload_parser.add_argument("embedding_paths", type=str, location="form", required=True)


@ns.route("/evolve")
class EvolveResource(Resource):
    @ns.marshal_with(evolution_fields)
    def get(self, evolution_id: int):
        evolution = Evolution.query.get(evolution_id)
        return evolution

    @verify_has_edit_access
    @ns.expect(upload_parser)
    @ns.marshal_with(evolution_fields)
    # @ns.consumes('multipart/form-data')
    def post(self):
        args = upload_parser.parse_args()

        # Get form data
        name = args["name"]
        fold_id = int(args["fold_id"])
        embedding_paths = json.loads(args["embedding_paths"])
        activity_file = args["activity_file"]
        evolve_directory = Path("evolve") / name

        # 0. Check if an evolve job with this name already exists.
        fold = Fold.query.get(fold_id)
        if not fold:
            raise BadRequest(f"Fold not found {fold_id}")
        existing_evolve = Evolution.query.filter(
            Evolution.name == name, Evolution.fold_id == fold_id
        ).first()
        if existing_evolve:
            # Delete existing evolve job.
            existing_evolve.delete()

        # 1. Upload the activity file to the storage manager.
        # Reset file pointer to start
        activity_file.seek(0)
        fsm = FoldStorageManager()
        fsm.setup()
        fsm.storage_manager.write_file(
            fold_id=fold_id,
            file_path=str(
                evolve_directory / "activity.xlsx"
            ),  # or whatever path/extension you wan_t
            contents=activity_file.read(),
            binary=True,
        )

        # 2. Create an invokation record for the evolve job.
        new_invokation_id = get_job_type_replacement(fold, f"evolve_{name}")

        # 3. Create a new Evolution record.
        evolve_record = Evolution.create(
            name=name,
            fold_id=fold_id,
            embedding_files=",".join(embedding_paths),
            invokation_id=new_invokation_id,
        )

        # 4. Start the job.
        job = rq.get_queue("cpu").enqueue(
            evolve_jobs.run_evolvepro,
            evolve_record.id,
        )
        return evolve_record
