"""Views for DNA build and Teselagen integration."""

import logging

from flask import current_app, request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource
from werkzeug.exceptions import BadRequest

from app.api_fields import dna_build_response_fields
from app.helpers.boltz_yaml_helper import BoltzYamlHelper
from app.helpers.dna_build_util import create_dna_build
from app.models import Fold

ns = Namespace(
    "dna-build",
    description="DNA build and Teselagen integration",
    decorators=[jwt_required(fresh=True)],
)


@ns.route("/dna-build")
class DnaBuildResource(Resource):
    @ns.marshal_with(dna_build_response_fields)
    def post(self):
        """
        Create a DNA build design and optionally post to Teselagen.

        Expected JSON payload:
        {
            "design_id": "MyDesign_R2",
            "fold_id": "fold-uuid",
            "genbank_files": {
                "template1.gb": "genbank content...",
                "template2.gb": "genbank content..."
            },
            "seq_ids": ["D104G", "G429R", "D104G_G429R"],
            "number_of_mutations": 1,
            "dry_run": true,
            "username": "user@example.com",
            "otp": "one-time-password",
            "project_id": "teselagen-project-uuid"
        }

        Returns:
        {
            "design_name": "MyDesign_R2",
            "successful_builds": ["D104G", "G429R"],
            "failed_builds": [{"seq_id": "D104G_G429R", "error": "..."}],
            "template_map": {"WT": "template1.gb", "G429R": "template2.gb"},
            "dry_run": true,
            "teselagen_id": "design-uuid"
        }
        """
        try:
            data = request.get_json()
            if not data:
                raise BadRequest("JSON payload required")

            # Validate required fields
            required_fields = [
                "design_id",
                "fold_id",
                "genbank_files",
                "seq_ids",
                "number_of_mutations",
            ]
            for field in required_fields:
                if field not in data:
                    raise BadRequest(f"Missing required field: {field}")

            design_id = data["design_id"]
            fold_id = data["fold_id"]
            genbank_files = data["genbank_files"]
            seq_ids = data["seq_ids"]
            number_of_mutations = data["number_of_mutations"]
            dry_run = data.get("dry_run", True)

            # Optional Teselagen fields
            username = data.get("username")
            otp = data.get("otp")
            project_id = data.get("project_id")

            # Validate types
            if not isinstance(design_id, str):
                raise BadRequest("design_id must be a string")
            if not isinstance(fold_id, str):
                raise BadRequest("fold_id must be a string")
            if not isinstance(genbank_files, dict):
                raise BadRequest("genbank_files must be a dictionary")
            if not isinstance(seq_ids, list):
                raise BadRequest("seq_ids must be a list")
            if not isinstance(number_of_mutations, int) or number_of_mutations < 1:
                raise BadRequest("number_of_mutations must be a positive integer")
            if not isinstance(dry_run, bool):
                raise BadRequest("dry_run must be a boolean")

            # Get fold and amino acid sequence
            try:
                fold = Fold.query.get(fold_id)
            except Exception as e:
                logging.error(f"Database error retrieving fold {fold_id}: {e}")
                raise BadRequest(f"Error accessing fold data: {e}")

            if not fold:
                raise BadRequest(f"Fold not found: {fold_id}")

            if not fold.yaml_config:
                raise BadRequest("Fold does not have a YAML config!")

            try:
                boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)
                if len(boltz_yaml_helper.get_protein_sequences()) > 1:
                    raise BadRequest(
                        "Fold has multiple protein sequences, which is not supported yet."
                    )

                wt_aa_sequence = boltz_yaml_helper.get_protein_sequences()[0][1]
            except Exception as e:
                logging.error(f"Error parsing fold YAML config: {e}")
                raise BadRequest(f"Invalid fold configuration: {e}")

            # Validate genbank files
            if not genbank_files:
                raise BadRequest("At least one genbank file must be provided")

            for filename, content in genbank_files.items():
                if not isinstance(filename, str) or not isinstance(content, str):
                    raise BadRequest("genbank_files must map strings to strings")
                if not content.strip():
                    raise BadRequest(f"Genbank file {filename} is empty")

            # Validate seq_ids
            if not seq_ids:
                raise BadRequest("At least one seq_id must be provided")

            for seq_id in seq_ids:
                if not isinstance(seq_id, str):
                    raise BadRequest("All seq_ids must be strings")

            # Validate Teselagen credentials if not dry run
            if not dry_run:
                if not username or not isinstance(username, str):
                    raise BadRequest("username required and must be string when dry_run=false")
                if not otp or not isinstance(otp, str):
                    raise BadRequest("otp required and must be string when dry_run=false")
                if not project_id or not isinstance(project_id, str):
                    raise BadRequest("project_id required and must be string when dry_run=false")

            logging.info(
                f"Creating DNA build: design_id={design_id}, fold_id={fold_id}, "
                f"seq_ids={len(seq_ids)}, mutations={number_of_mutations}, dry_run={dry_run}"
            )

            # Get Teselagen base URL from app config (optional)
            teselagen_base_url = current_app.config.get("TESELAGEN_BACKEND")

            # Create the DNA build
            design_name, teselagen_id, seq_id_results = create_dna_build(
                design_id=design_id,
                genbank_files=genbank_files,
                wt_aa_sequence=wt_aa_sequence,
                seq_ids=seq_ids,
                number_of_mutations=number_of_mutations,
                username=username,
                otp=otp,
                project_id=project_id,
                dry_run=dry_run,
                teselagen_base_url=teselagen_base_url,
            )

            # Transform to expected response format
            response = {
                "design_name": design_name,
                "teselagen_id": teselagen_id,
                "seq_id_results": seq_id_results,
            }

            return response, 200

        except BadRequest:
            # Re-raise BadRequest exceptions
            raise
        except ValueError as e:
            # Convert ValueError to BadRequest for better error handling
            logging.error(f"DNA build validation error: {e}")
            raise BadRequest(str(e))
        except Exception as e:
            # Log unexpected errors and return generic message
            logging.error(f"Unexpected error in DNA build: {e}", exc_info=True)
            raise BadRequest("An unexpected error occurred while creating the DNA build")
