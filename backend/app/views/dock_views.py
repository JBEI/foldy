import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from app.api_fields import dock_fields
from app.authorization import verify_has_edit_access
from app.helpers.rq_helpers import get_queue
from app.jobs import other_jobs
from app.models import Dock, Fold
from app.util import get_job_type_replacement
from flask import (
    request,
)
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource
from werkzeug.exceptions import BadRequest

ns = Namespace("dock_views", decorators=[jwt_required(fresh=True)])


@ns.route("/dock")
class DockCreateResource(Resource):
    # TODO(jbr): Figure out what is causing this call to fail and add validation.
    # @ns.expect(dock_fields)
    @verify_has_edit_access
    def post(self) -> bool:
        """Create a new docking run for a fold with specified ligand.

        Returns:
            True if docking job was queued successfully

        Raises:
            BadRequest: If ligand name is not alphanumeric or tool is not supported
        """
        req = request.get_json()

        ligand_name = req["ligand_name"]
        ligand_smiles = req["ligand_smiles"]
        tool = req["tool"]
        fold_id = req["fold_id"]

        if not ligand_name.isalnum():
            raise BadRequest(f"Ligand names must be alphanumeric, {ligand_name} is invalid.")

        ALLOWED_DOCKING_TOOLS = ["vina", "diffdock"]
        if not tool in ALLOWED_DOCKING_TOOLS:
            raise BadRequest(f"Invalid docking tool {tool}: must be one of {ALLOWED_DOCKING_TOOLS}")

        fold = Fold.get_by_id(fold_id)
        if not fold:
            raise BadRequest("Fold not found")

        new_invokation_id = get_job_type_replacement(fold, f"dock_{ligand_name}")

        new_dock: Dock = Dock(
            ligand_name=ligand_name,
            ligand_smiles=ligand_smiles,
            tool=tool,
            receptor_fold_id=fold_id,
            invokation_id=new_invokation_id,
            bounding_box_residue=req.get("bounding_box_residue", None),
            bounding_box_radius_angstrom=req.get("bounding_box_radius_angstrom", None),
        )
        new_dock.save()

        cpu_q = get_queue("cpu")
        job = cpu_q.enqueue(
            other_jobs.run_dock,
            new_dock.id,
            new_invokation_id,
            job_timeout="2h",
            result_ttl=48 * 60 * 60,  # 2 days
        )

        logging.info(f"Queued docking job {job.id} for fold {fold_id} with ligand {ligand_name}")
        return True


@ns.route("/dock/<int:dock_id>")
class DockResource(Resource):
    def delete(self, dock_id: int) -> bool:
        """Delete a docking run by ID.

        Args:
            dock_id: ID of the dock to delete

        Returns:
            True if deletion was successful
        """
        dock = Dock.get_by_id(dock_id)

        if not dock:
            raise BadRequest(f"Dock with ID {dock_id} not found")

        logging.info(f"Deleting dock {dock_id} (ligand: {dock.ligand_name})")
        dock.delete()

        return True
