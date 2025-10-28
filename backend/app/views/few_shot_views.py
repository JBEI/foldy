import base64
import json
import logging
from pathlib import Path
from typing import IO, Any, BinaryIO, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource, fields
from google.cloud.storage import Blob
from google.cloud.storage.blob import BlobReader
from rq import Callback
from rq.job import Job
from sklearn.ensemble import RandomForestRegressor
from werkzeug.exceptions import BadRequest

from app.api_fields import few_shot_fields, few_shot_input_fields
from app.authorization import verify_has_edit_access
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager, LocalBlob
from app.helpers.rq_helpers import (
    add_meta_to_job,
    get_queue,
    send_failure_email,
    send_success_email,
)
from app.helpers.sequence_util import (
    maybe_get_seq_id_error_message,
)
from app.jobs import esm_jobs, evolve_jobs
from app.models import CampaignRound, FewShot, Fold, Invokation
from app.util import get_job_type_replacement
from folde.few_shot_models import is_valid_few_shot_model_name

ns = Namespace("few_shot_views", decorators=[jwt_required(fresh=True)])


@ns.route("/few_shot")
class FewShotResource(Resource):
    # @ns.consumes('multipart/form-data')
    @verify_has_edit_access
    @ns.expect(few_shot_input_fields)
    @ns.marshal_with(few_shot_fields)
    def post(self) -> FewShot:
        """Create a new slate build job with activity data file.

        Returns:
            Newly created FewShot record

        Raises:
            BadRequest: If required fields are missing or if fold is not found
        """
        req = request.get_json()

        print(f"request.data: {request.data}", flush=True)
        print(f"request.is_json: {request.is_json}", flush=True)

        # Get form data
        name: str = req["name"]
        fold_id: int = int(req["fold_id"])
        few_shot_directory: Path = Path("few_shots") / name

        # Set up storage manager
        fsm = FoldStorageManager()
        fsm.setup()
        assert fsm.storage_manager is not None

        # Handle activity file from one of three sources
        activity_file: bytes
        if req.get("activity_file_bytes"):
            # Decode base64 encoded file
            activity_file = base64.b64decode(req["activity_file_bytes"])
        elif req.get("activity_file_from_few_shot_id"):
            few_shot_for_activity_file = FewShot.query.get(req["activity_file_from_few_shot_id"])
            if not few_shot_for_activity_file:
                raise BadRequest(f"FewShot not found {req['activity_file_from_few_shot_id']}")

            try:
                assert (
                    few_shot_for_activity_file.input_activity_fpath is not None
                ), f"FewShot {req['activity_file_from_few_shot_id']} has no input_activity_fpath"
                activity_file_blob = fsm.storage_manager.get_blob(
                    fold_id, few_shot_for_activity_file.input_activity_fpath
                )
                activity_file = activity_file_blob.open("rb").read()  # type: ignore[reportAssignmentType]
            except Exception as e:
                raise BadRequest(
                    f"Failed to get activity file from few shot {req['activity_file_from_few_shot_id']}: {e}"
                )
        elif req.get("activity_file_from_round_id"):
            campaign_round = CampaignRound.query.get(req["activity_file_from_round_id"])
            if not campaign_round:
                raise BadRequest(f"CampaignRound not found {req['activity_file_from_round_id']}")

            if not campaign_round.result_activity_fpath:
                raise BadRequest(
                    f"CampaignRound {req['activity_file_from_round_id']} has no activity file"
                )

            try:
                activity_file_blob = fsm.storage_manager.get_blob(
                    fold_id, campaign_round.result_activity_fpath
                )
                activity_file = activity_file_blob.open("rb").read()  # type: ignore[reportAssignmentType]
            except Exception as e:
                raise BadRequest(
                    f"Failed to get activity file from campaign round {req['activity_file_from_round_id']}: {e}"
                )
        else:
            raise BadRequest(
                "activity_file_bytes, activity_file_from_few_shot_id, or activity_file_from_round_id is required"
            )

        # Delete existing directory and write new activity file
        fsm.storage_manager.delete_folder(fold_id, str(few_shot_directory))
        fsm.storage_manager.write_file(
            fold_id=fold_id,
            file_path=str(few_shot_directory / "activity.xlsx"),
            contents=activity_file,
            binary=True,
        )
        input_activity_fpath = str(few_shot_directory / "activity.xlsx")

        mode: str = req["mode"]
        try:
            embedding_files: Optional[List[str]] = (
                req["embedding_files"].split(",") if "embedding_files" in req else None
            )
        except Exception as e:
            raise BadRequest(f"Failed loading embedding_files {e}")

        try:
            naturalness_files: Optional[List[str]] = (
                req["naturalness_files"].split(",") if "naturalness_files" in req else None
            )
        except Exception as e:
            raise BadRequest(f"Failed loading naturalness_files {e}")

        finetuning_model_checkpoint: Optional[str] = req.get("finetuning_model_checkpoint", None)
        few_shot_params: Optional[str] = req.get("few_shot_params", None)
        num_mutants: int = req["num_mutants"]

        if mode == "randomforest" or mode == "mlp":
            if not embedding_files:
                raise BadRequest("embedding_files are required for randomforest mode")
        elif mode == "finetuning":
            if not finetuning_model_checkpoint:
                raise BadRequest("finetuning_model_checkpoint is required for finetuning mode")
        elif is_valid_few_shot_model_name(mode):
            if not few_shot_params:
                raise BadRequest("few_shot_params are required for few shot models")
            if not embedding_files or not naturalness_files:
                raise BadRequest(
                    "embedding_files and naturalness_files are required for few shot models"
                )
        else:
            raise BadRequest(f"Invalid mode: {mode}")

        # 0. Check if a few shot job with this name already exists.
        fold = Fold.query.get(fold_id)
        if not fold:
            raise BadRequest(f"Fold not found {fold_id}")

        # Make sure the folder and existing few shot have been cleared.
        existing_few_shot = FewShot.query.filter(
            FewShot.name == name, FewShot.fold_id == fold_id
        ).first()
        if existing_few_shot:
            # Delete existing few shot job.
            logging.info(f"Deleting existing few shot job {existing_few_shot.id} for {name}")
            existing_few_shot.delete()

        # 2. Create an invokation record for the slate build job.
        new_invokation_id: int = get_job_type_replacement(fold, f"few_shot_{name}")

        # 3. Create a new FewShot record.
        few_shot_record: FewShot = FewShot.create(
            name=name,
            fold_id=fold_id,
            mode=mode,
            embedding_files=",".join(embedding_files) if embedding_files else None,
            naturalness_files=",".join(naturalness_files) if naturalness_files else None,
            finetuning_model_checkpoint=finetuning_model_checkpoint,
            invokation_id=new_invokation_id,
            few_shot_params=few_shot_params,
            num_mutants=num_mutants,
            input_activity_fpath=input_activity_fpath,
        )

        # 4. Start the job based on mode
        enqueued_job: Job

        if mode == "finetuning":
            enqueued_job = get_queue("esm").enqueue(
                esm_jobs.finetune_esm_model,
                few_shot_record.id,
                job_timeout="12h",
                result_ttl=48 * 60 * 60,  # 2 days
                on_success=Callback(send_success_email, timeout="10s"),
                on_failure=Callback(send_failure_email, timeout="10s"),
            )
            add_meta_to_job(enqueued_job, fold, "few_shot", few_shot_record.id)

            logging.info(
                f"Queued finetuning job {enqueued_job.id} for slate build {few_shot_record.id}"
            )
        else:
            enqueued_job = get_queue("cpu").enqueue(
                evolve_jobs.run_few_shot_prediction,
                few_shot_record.id,
                job_timeout="6h",
                on_success=Callback(send_success_email, timeout="10s"),
                on_failure=Callback(send_failure_email, timeout="10s"),
            )
            add_meta_to_job(enqueued_job, fold, "few_shot", few_shot_record.id)

            logging.info(
                f"Queued {mode} job {enqueued_job.id} for slate build {few_shot_record.id}"
            )

        return few_shot_record


@ns.route("/few_shot_predicted_slate/<int:few_shot_id>")
class FewShotPredictedSlateResource(Resource):
    def get(self, few_shot_id: int):
        """Get predicted slate data for a FewShot run.

        Args:
            few_shot_id: ID of the FewShot run

        Query Parameters:
            selected_only: boolean (default: true) - filter rows where selected=True
            limit: integer (optional) - limit number of returned rows

        Returns:
            List of slate data objects with predictions and metadata
        """
        few_shot = FewShot.query.get(few_shot_id)
        if not few_shot:
            raise BadRequest(f"FewShot not found {few_shot_id}")

        if not few_shot.output_fpath:
            raise BadRequest(f"FewShot {few_shot_id} has no output file")

        # Parse query parameters
        selected_only = request.args.get("selected_only", "true").lower() == "true"
        limit = request.args.get("limit", type=int)

        # Load CSV from storage
        fsm = FoldStorageManager()
        fsm.setup()
        assert fsm.storage_manager is not None

        try:
            csv_blob = fsm.storage_manager.get_blob(few_shot.fold_id, few_shot.output_fpath)

            # Read CSV with pandas
            import io

            csv_content = csv_blob.open("rb").read()
            csv_df = pd.read_csv(io.BytesIO(csv_content), index_col=0)

            # Apply filtering
            if selected_only and "selected" in csv_df.columns:
                csv_df = csv_df[csv_df["selected"] == True]

            # Apply limit if specified
            if limit:
                csv_df = csv_df.head(limit)

            # Transform to slate data format
            slate_data = []

            for seq_id, row in csv_df.iterrows():
                # Extract model predictions (model_0, model_1, etc.)
                model_predictions = []
                model_cols = [col for col in csv_df.columns if col.startswith("model_")]
                model_cols.sort(key=lambda x: int(x.split("_")[1]))  # Sort by model number

                for col in model_cols:
                    if not pd.isna(row[col]):
                        model_predictions.append(float(row[col]))

                # Calculate mean and stddev
                if model_predictions:
                    prediction_mean = float(np.mean(model_predictions))
                    prediction_stddev = float(np.std(model_predictions))
                else:
                    prediction_mean = 0.0
                    prediction_stddev = 0.0

                slate_item = {
                    "seqId": str(seq_id),
                    "selected": bool(row.get("selected", False)),
                    "order": None,  # Will be set by frontend based on sort options
                    "relevantMeasuredMutants": str(row.get("relevant_measured_mutants", "")),
                    "predictionMean": prediction_mean,
                    "predictionStddev": prediction_stddev,
                    "score": prediction_mean,  # Same as predictionMean
                    "modelPredictions": model_predictions,
                }
                slate_data.append(slate_item)

            return {"data": slate_data, "total_count": len(csv_df), "few_shot_id": few_shot_id}

        except Exception as e:
            raise BadRequest(f"Failed to load predicted slate data: {str(e)}")


@ns.route("/few_shots/<int:few_shot_id>")
class SingleFewShotResource(Resource):
    @ns.marshal_with(few_shot_fields)
    def get(self, few_shot_id: int) -> FewShot:
        """Get slate build record by ID.

        Args:
            few_shot_id: ID of the slate build to retrieve

        Returns:
            FewShot record
        """
        few_shot = FewShot.query.get(few_shot_id)
        if not few_shot:
            raise BadRequest(f"FewShot not found {few_shot_id}")
        return few_shot

    @verify_has_edit_access
    def delete(self, few_shot_id: int) -> None:
        """Delete a slate build record by ID.

        Args:
            few_shot_id: ID of the slate build to delete
        """
        few_shot = FewShot.query.get(few_shot_id)
        if not few_shot:
            raise BadRequest(f"FewShot not found {few_shot_id}")

        manager = FoldStorageManager()
        manager.setup()

        assert manager.storage_manager is not None

        manager.storage_manager.delete_folder(few_shot.fold_id, f"few_shots/{few_shot.name}")

        if few_shot.invokation_id:
            invokation = Invokation.query.get(few_shot.invokation_id)
            if invokation:
                invokation.delete()

        few_shot.delete()

        return None
