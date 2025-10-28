"""Campaign management API endpoints."""

import logging
from datetime import UTC, datetime
from typing import Dict, List, Optional

from flask import abort, request
from flask_jwt_extended import jwt_required
from flask_jwt_extended.utils import get_jwt
from flask_restx import Namespace, Resource, fields
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from app.authorization import user_jwt_grants_edit_access, verify_has_edit_access
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.models import Campaign, CampaignRound, Fold

ns = Namespace("campaign_views", decorators=[jwt_required(fresh=True)])

# Define the fields for API serialization
from app.api_fields import (
    NullableInteger,
    campaign_fields,
    campaign_input_fields,
    campaign_round_fields,
    campaign_round_input_fields,
    few_shot_fields,
    naturalness_fields,
    paginated_campaigns_fields,
)

# Parser for pagination
get_campaigns_parser = ns.parser()
get_campaigns_parser.add_argument("page", type=int, default=1, help="Page number")
get_campaigns_parser.add_argument("per_page", type=int, default=20, help="Items per page")
get_campaigns_parser.add_argument("fold_id", type=int, help="Filter by fold ID")


@ns.route("/campaigns")
class CampaignsResource(Resource):
    @ns.expect(get_campaigns_parser)
    @ns.marshal_with(paginated_campaigns_fields)
    def get(self):
        """Get paginated list of campaigns.

        Returns:
            Paginated campaigns with metadata
        """
        args = get_campaigns_parser.parse_args()
        page = args.get("page", 1)
        per_page = min(args.get("per_page", 20), 100)  # Limit to 100 items per page
        fold_id = args.get("fold_id")

        # Add permission filtering based on fold visibility
        only_public = not user_jwt_grants_edit_access(get_jwt()["user_claims"])

        query = Campaign.query.join(Fold)

        # Filter campaigns based on fold visibility permissions
        if only_public:
            query = query.filter(Fold.public == True)

        if fold_id:
            query = query.filter(Campaign.fold_id == fold_id)

        # Order by most recent first
        query = query.order_by(Campaign.created_at.desc())

        pagination = query.paginate(page=page, per_page=per_page, error_out=False)

        # Debug: Check each campaign for problematic data
        for campaign in pagination.items:
            for round_obj in campaign.rounds:
                try:

                    # Check input_templates field
                    if hasattr(round_obj, "input_templates"):
                        logging.info(
                            f"CampaignRound {round_obj.id} input_templates type: {type(round_obj.input_templates)}, value: {round_obj.input_templates}"
                        )

                except Exception as e:
                    logging.error(f"Error checking CampaignRound {round_obj.id}: {e}")

        return {
            "campaigns": pagination.items,
            "total": pagination.total,
            "page": pagination.page,
            "per_page": pagination.per_page,
            "pages": pagination.pages,
        }

    @ns.expect(campaign_input_fields)
    @ns.marshal_with(campaign_fields)
    @verify_has_edit_access
    def post(self):
        """Create a new campaign.

        Returns:
            Created campaign
        """
        data = request.get_json()

        # Validate required fields
        if not data.get("name"):
            raise BadRequest("Campaign name is required")
        if not data.get("fold_id"):
            raise BadRequest("Fold ID is required")

        # Check if fold exists
        fold = Fold.query.get(data["fold_id"])
        if not fold:
            raise BadRequest(f"Fold with ID {data['fold_id']} not found")

        # Check if campaign name is unique for this fold
        existing_campaign = Campaign.query.filter_by(
            name=data["name"], fold_id=data["fold_id"]
        ).first()
        if existing_campaign:
            raise BadRequest(f"Campaign with name '{data['name']}' already exists for this fold")

        # Create new campaign
        campaign = Campaign(
            name=data["name"],
            fold_id=data["fold_id"],
            description=data.get("description"),
            naturalness_model=data.get("naturalness_model"),
            embedding_model=data.get("embedding_model"),
            domain_boundaries=data.get("domain_boundaries"),
        )

        db.session.add(campaign)
        db.session.commit()

        logging.info(f"Created new campaign: {campaign.name} (ID: {campaign.id})")
        return campaign


@ns.route("/campaigns/<int:campaign_id>")
class CampaignResource(Resource):
    @ns.marshal_with(campaign_fields)
    def get(self, campaign_id: int):
        """Get a specific campaign by ID.

        Args:
            campaign_id: ID of the campaign to retrieve

        Returns:
            Campaign with rounds
        """
        # Add permission filtering based on fold visibility
        only_public = not user_jwt_grants_edit_access(get_jwt()["user_claims"])

        campaign = Campaign.query.join(Fold).filter(Campaign.id == campaign_id).first()
        if not campaign:
            raise BadRequest(f"Campaign with ID {campaign_id} not found")

        # Check fold permissions - deny access to private fold campaigns for view-only users
        if only_public and not campaign.fold.public:
            abort(403, description="You do not have access to this resource.")

        return campaign

    @ns.expect(campaign_fields, validate=False)
    @ns.marshal_with(campaign_fields)
    @verify_has_edit_access
    def put(self, campaign_id: int):
        """Update a campaign.

        Args:
            campaign_id: ID of the campaign to update

        Returns:
            Updated campaign
        """
        campaign = Campaign.query.get(campaign_id)
        if not campaign:
            raise BadRequest(f"Campaign with ID {campaign_id} not found")

        data = request.get_json()

        # Update fields if provided
        if "name" in data:
            # Check if new name conflicts with existing campaigns for this fold
            existing_campaign = (
                Campaign.query.filter_by(name=data["name"], fold_id=campaign.fold_id)
                .filter(Campaign.id != campaign_id)
                .first()
            )
            if existing_campaign:
                raise BadRequest(
                    f"Campaign with name '{data['name']}' already exists for this fold"
                )
            campaign.name = data["name"]

        if "description" in data:
            campaign.description = data["description"]

        if "naturalness_model" in data:
            campaign.naturalness_model = data["naturalness_model"]

        if "embedding_model" in data:
            campaign.embedding_model = data["embedding_model"]

        if "domain_boundaries" in data:
            campaign.domain_boundaries = data["domain_boundaries"]

        db.session.commit()

        logging.info(f"Updated campaign: {campaign.name} (ID: {campaign.id})")
        return campaign

    @verify_has_edit_access
    def delete(self, campaign_id: int):
        """Delete a campaign and all its rounds.

        Args:
            campaign_id: ID of the campaign to delete
        """
        campaign = Campaign.query.get(campaign_id)
        if not campaign:
            raise BadRequest(f"Campaign with ID {campaign_id} not found")

        campaign_name = campaign.name
        db.session.delete(campaign)
        db.session.commit()

        logging.info(f"Deleted campaign: {campaign_name} (ID: {campaign_id})")
        return {"message": f"Campaign '{campaign_name}' deleted successfully"}


@ns.route("/campaigns/<int:campaign_id>/rounds")
class CampaignRoundsResource(Resource):
    @ns.marshal_list_with(campaign_round_fields)
    def get(self, campaign_id: int):
        """Get all rounds for a campaign.

        Args:
            campaign_id: ID of the campaign

        Returns:
            List of campaign rounds
        """
        # Add permission filtering based on fold visibility
        only_public = not user_jwt_grants_edit_access(get_jwt()["user_claims"])

        campaign = Campaign.query.join(Fold).filter(Campaign.id == campaign_id).first()
        if not campaign:
            raise BadRequest(f"Campaign with ID {campaign_id} not found")

        # Check fold permissions - deny access to private fold campaigns for view-only users
        if only_public and not campaign.fold.public:
            abort(403, description="You do not have access to this resource.")

        return campaign.rounds

    @ns.expect(campaign_round_input_fields)
    @ns.marshal_with(campaign_round_fields)
    @verify_has_edit_access
    def post(self, campaign_id: int):
        """Create a new round for a campaign.

        Args:
            campaign_id: ID of the campaign

        Returns:
            Created campaign round
        """
        campaign = Campaign.query.get(campaign_id)
        if not campaign:
            raise BadRequest(f"Campaign with ID {campaign_id} not found")

        data = request.get_json()

        # Auto-increment round number if not provided
        if "round_number" not in data or data["round_number"] is None:
            max_round = (
                db.session.query(db.func.max(CampaignRound.round_number))
                .filter_by(campaign_id=campaign_id)
                .scalar()
            )
            round_number = (max_round or 0) + 1
        else:
            round_number = data["round_number"]
            # Check if round number already exists
            existing_round = CampaignRound.query.filter_by(
                campaign_id=campaign_id, round_number=round_number
            ).first()
            if existing_round:
                raise BadRequest(f"Round {round_number} already exists for this campaign")

        # Create new round
        campaign_round = CampaignRound(
            campaign_id=campaign_id,
            round_number=round_number,
            date_started=datetime.now(UTC) if "date_started" not in data else data["date_started"],
        )

        db.session.add(campaign_round)
        db.session.commit()

        logging.info(
            f"Created new round {round_number} for campaign {campaign.name} (ID: {campaign.id})"
        )
        return campaign_round


@ns.route("/campaigns/<int:campaign_id>/rounds/<int:round_id>")
class CampaignRoundResource(Resource):
    @ns.marshal_with(campaign_round_fields)
    def get(self, campaign_id: int, round_id: int):
        """Get a specific campaign round.

        Args:
            campaign_id: ID of the campaign
            round_id: ID of the round

        Returns:
            Campaign round
        """
        # Add permission filtering based on fold visibility
        only_public = not user_jwt_grants_edit_access(get_jwt()["user_claims"])

        campaign_round = (
            CampaignRound.query.join(Campaign)
            .join(Fold)
            .filter(CampaignRound.id == round_id, CampaignRound.campaign_id == campaign_id)
            .first()
        )

        if not campaign_round:
            raise BadRequest(f"Round with ID {round_id} not found for campaign {campaign_id}")

        # Check fold permissions - deny access to private fold campaigns for view-only users
        if only_public and not campaign_round.campaign.fold.public:
            abort(403, description="You do not have access to this resource.")

        return campaign_round

    @ns.expect(campaign_round_input_fields)
    @ns.marshal_with(campaign_round_fields)
    @verify_has_edit_access
    def put(self, campaign_id: int, round_id: int):
        """Update a campaign round.

        Args:
            campaign_id: ID of the campaign
            round_id: ID of the round to update

        Returns:
            Updated campaign round
        """
        campaign_round = CampaignRound.query.filter_by(id=round_id, campaign_id=campaign_id).first()
        if not campaign_round:
            raise BadRequest(f"Round with ID {round_id} not found for campaign {campaign_id}")

        data = request.get_json()

        # Update fields if provided
        if "mode" in data:
            campaign_round.mode = data["mode"]

        if "naturalness_run_id" in data:
            campaign_round.naturalness_run_id = data["naturalness_run_id"]

        if "few_shot_run_id" in data:
            campaign_round.few_shot_run_id = data["few_shot_run_id"]

        if "slate_seq_ids" in data:
            campaign_round.slate_seq_ids = data["slate_seq_ids"]

        if "result_activity_fpath" in data:
            campaign_round.result_activity_fpath = data["result_activity_fpath"]

        if "input_templates" in data:
            campaign_round.input_templates = data["input_templates"]

        db.session.commit()

        logging.info(f"Updated round {campaign_round.round_number} for campaign {campaign_id}")
        return campaign_round

    @verify_has_edit_access
    def delete(self, campaign_id: int, round_id: int):
        """Delete a campaign round.

        Args:
            campaign_id: ID of the campaign
            round_id: ID of the round to delete
        """
        campaign_round = CampaignRound.query.filter_by(id=round_id, campaign_id=campaign_id).first()
        if not campaign_round:
            raise BadRequest(f"Round with ID {round_id} not found for campaign {campaign_id}")

        round_number = campaign_round.round_number
        db.session.delete(campaign_round)
        db.session.commit()

        logging.info(f"Deleted round {round_number} from campaign {campaign_id}")
        return {"message": f"Round {round_number} deleted successfully"}


@ns.route("/campaigns/<int:campaign_id>/<int:round_number>/activity_file")
class CampaignRoundActivityFileResource(Resource):
    @verify_has_edit_access
    def post(self, campaign_id: int, round_number: int):
        """Upload activity file for a campaign round using FoldStorageManager.

        Args:
            campaign_id: ID of the campaign
            round_number: Round number

        Returns:
            Success message with file path
        """
        # Verify campaign exists
        campaign = Campaign.query.get(campaign_id)
        if not campaign:
            raise BadRequest(f"Campaign with ID {campaign_id} not found")

        # Find the campaign round by round number - ensure uniqueness
        campaign_rounds = CampaignRound.query.filter_by(
            campaign_id=campaign_id, round_number=round_number
        ).all()

        if len(campaign_rounds) == 0:
            raise BadRequest(f"Round {round_number} not found for campaign {campaign_id}")
        elif len(campaign_rounds) > 1:
            raise BadRequest(
                f"Multiple rounds found with number {round_number} for campaign {campaign_id}"
            )

        campaign_round = campaign_rounds[0]

        # Check if file was uploaded
        if "activity_file" not in request.files:
            raise BadRequest("No activity file provided")

        file = request.files["activity_file"]
        if file.filename == "":
            raise BadRequest("No file selected")

        # Validate file extension
        if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls")):
            raise BadRequest("File must be an Excel file (.xlsx or .xls)")

        # Use FoldStorageManager to store file
        fsm = FoldStorageManager()
        fsm.setup()
        assert fsm.storage_manager is not None

        # Read file contents
        file.seek(0)
        file_contents = file.read()

        # Validate file contents before storing
        try:
            from io import BytesIO

            import pandas as pd

            # Read Excel file with pandas to validate structure
            file_buffer = BytesIO(file_contents)
            df = pd.read_excel(file_buffer)

            # Validate required columns
            required_columns = ["seq_id", "activity"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise BadRequest(
                    f"Activity file missing required columns: {', '.join(missing_columns)}. Found columns: {', '.join(df.columns.tolist())}"
                )

            # Check if file has any data rows
            if df.empty:
                raise BadRequest("Activity file is empty (no data rows)")

            # Check for rows with valid data
            valid_rows = 0
            for _, row in df.iterrows():
                if not (pd.isna(row["seq_id"]) or pd.isna(row["activity"])):
                    valid_rows += 1

            if valid_rows == 0:
                raise BadRequest(
                    "Activity file contains no valid data rows (all seq_id or activity values are missing)"
                )

            logging.info(
                f"Validated activity file: {len(df)} total rows, {valid_rows} valid data rows"
            )

        except pd.errors.EmptyDataError:
            raise BadRequest("Activity file is empty")
        except pd.errors.ParserError as e:
            raise BadRequest(f"Failed to parse activity file: {str(e)}")
        except Exception as e:
            # Re-raise BadRequest exceptions as-is, wrap other exceptions
            if isinstance(e, BadRequest):
                raise
            logging.error(f"Error validating activity file: {e}")
            raise BadRequest(f"Failed to validate activity file: {str(e)}")

        # Store file with path: campaign_{campaign_id}/round_{round_number}/activity.xlsx
        relative_path = f"campaigns/{campaign_id}/round_{round_number}/activity.xlsx"

        fsm.storage_manager.write_file(
            fold_id=campaign.fold_id,
            file_path=relative_path,
            contents=file_contents,
            binary=True,
        )

        # Update campaign round with relative path
        campaign_round.result_activity_fpath = relative_path

        db.session.commit()

        logging.info(
            f"Uploaded activity file for campaign {campaign_id}, round {round_number}: {relative_path}"
        )

        return {"message": "Activity file uploaded successfully", "file_path": relative_path}


@ns.route("/campaigns/<int:campaign_id>/<int:round_number>/activity_data")
class CampaignRoundActivityDataResource(Resource):
    def get(self, campaign_id: int, round_number: int):
        """Get activity data from campaign round file using FoldStorageManager.

        Args:
            campaign_id: ID of the campaign
            round_number: Round number

        Returns:
            List of (seq_id, activity) tuples
        """
        # Add permission filtering based on fold visibility
        only_public = not user_jwt_grants_edit_access(get_jwt()["user_claims"])

        # Verify campaign exists and check permissions
        campaign = Campaign.query.join(Fold).filter(Campaign.id == campaign_id).first()
        if not campaign:
            raise BadRequest(f"Campaign with ID {campaign_id} not found")

        # Check fold permissions - deny access to private fold campaigns for view-only users
        if only_public and not campaign.fold.public:
            abort(403, description="You do not have access to this resource.")

        # Find the campaign round by round number - ensure uniqueness
        campaign_rounds = CampaignRound.query.filter_by(
            campaign_id=campaign_id, round_number=round_number
        ).all()

        if len(campaign_rounds) == 0:
            raise BadRequest(f"Round {round_number} not found for campaign {campaign_id}")
        elif len(campaign_rounds) > 1:
            raise BadRequest(
                f"Multiple rounds found with number {round_number} for campaign {campaign_id}"
            )

        campaign_round = campaign_rounds[0]

        if not campaign_round.result_activity_fpath:
            raise BadRequest(f"Round {round_number} has no activity file")

        try:
            import pandas as pd

            # Use FoldStorageManager to read the file
            fsm = FoldStorageManager()
            fsm.setup()
            assert fsm.storage_manager is not None

            # Get the file blob and read it
            activity_file_blob = fsm.storage_manager.get_blob(
                campaign.fold_id, campaign_round.result_activity_fpath
            )

            # Read Excel file with pandas from binary data
            with activity_file_blob.open("rb") as f:
                df = pd.read_excel(f)

            # Validate required columns
            required_columns = ["seq_id", "activity"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise BadRequest(
                    f"Activity file missing required columns: {', '.join(missing_columns)}"
                )

            # Convert to list of tuples, filtering out rows with NaN values
            activity_data = []
            for _, row in df.iterrows():
                seq_id = row["seq_id"]
                activity = row["activity"]

                # Skip rows with missing data
                if pd.isna(seq_id) or pd.isna(activity):
                    continue

                activity_data.append({"seq_id": str(seq_id), "activity": float(activity)})

            logging.info(
                f"Retrieved {len(activity_data)} activity records for campaign {campaign_id}, round {round_number}"
            )

            return {"data": activity_data, "count": len(activity_data)}

        except pd.errors.EmptyDataError:
            raise BadRequest("Activity file is empty")
        except pd.errors.ParserError as e:
            raise BadRequest(f"Failed to parse activity file: {str(e)}")
        except Exception as e:
            logging.error(
                f"Error reading activity file for campaign {campaign_id}, round {round_number}: {e}"
            )
            raise BadRequest(f"Failed to read activity file: {str(e)}")
