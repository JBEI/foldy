"""
Centralized Flask-RESTx field definitions for all API endpoints.

This module contains all field models and schemas used by the REST API endpoints.
Organized by domain for better maintainability.
"""

from flask_restx import Namespace, fields


def log_getattr(a, field, default, debuginfo):
    """Helper function for attribute access with optional logging."""
    returnval = getattr(a, field, default)
    # logging.error(f"Got {returnval} for {field} ({debuginfo})")
    return returnval


class NullableInteger(fields.Integer):
    """Integer field that can be null."""

    __schema_type__ = ["integer", "null"]


# Create a temporary namespace for field definitions
# These will be used across multiple view namespaces
type_ns = Namespace("types")


# =============================================================================
# JOB AND INVOCATION FIELDS
# =============================================================================

simple_invokation_fields = type_ns.model(
    "SimpleInvokationFields",
    {
        "id": fields.Integer(required=True),
        "type": fields.String(required=False),
        "state": fields.String(required=False),
        "command": fields.String(required=False),
    },
)

full_invokation_fields = type_ns.clone(
    "InvokationFields",
    simple_invokation_fields,
    {
        "job_id": fields.String(required=False),
        "log": fields.String(required=False),
        "timedelta_sec": fields.Float(
            readonly=True,
            required=False,
            attribute=lambda r: (r.timedelta.total_seconds() if r and r.timedelta else None),
        ),
        "starttime": fields.DateTime(
            format="iso8601Z", dt_format="iso8601", readonly=True, required=False
        ),
    },
)

queue_job_fields = type_ns.model(
    "QueueFold",
    {
        "fold_id": fields.String(),
        "stage": fields.String(),
        "email_on_completion": fields.Boolean(),
    },
)


# =============================================================================
# DOCKING FIELDS (defined early since fold_fields references it)
# =============================================================================

dock_fields = type_ns.model(
    "DockFields",
    {
        "id": fields.Integer(readonly=True),
        "fold_id": fields.Integer(required=True),
        "ligand_name": fields.String(required=True),
        "ligand_smiles": fields.String(required=True),
        "tool": fields.String(required=False),
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
        "pose_confidences": fields.String(required=False),
    },
)


# =============================================================================
# ESM / EMBEDDING FIELDS (defined early since fold_fields references them)
# =============================================================================

naturalness_fields = type_ns.model(
    "NaturalnessFields",
    {
        "id": fields.Integer(required=False),
        "name": fields.String(required=True),
        "fold_id": fields.Integer(required=False),
        "logit_model": fields.String(required=True),
        "use_structure": fields.Boolean(required=False),
        "get_depth_two_logits": fields.Boolean(required=False),
        "output_fpath": fields.String(required=False),
        "output_fpath_computed": fields.String(required=False),
        "invokation_id": fields.Integer(required=False),
        "date_created": fields.DateTime(required=False),
    },
)

embedding_fields = type_ns.model(
    "EmbeddingFields",
    {
        "name": fields.String(required=True),
        "fold_id": fields.Integer(required=True),
        "embedding_model": fields.String(required=True),
        "id": fields.Integer(required=False),
        "extra_seq_ids": fields.String(required=False),
        "dms_starting_seq_ids": fields.String(required=False),
        "homolog_fasta": fields.String(required=False),
        "extra_layers": fields.String(required=False),
        "domain_boundaries": fields.String(required=False),
        "output_fpath": fields.String(required=False),
        "output_fpath_computed": fields.String(required=False),
        "invokation_id": fields.Integer(required=False),
        "date_created": fields.DateTime(required=False),
    },
)


# =============================================================================
# FEW-SHOT FIELDS (defined early since fold_fields references it)
# =============================================================================

few_shot_fields = type_ns.model(
    "FewShotFields",
    {
        "id": fields.Integer(required=False),
        "name": fields.String(required=False),
        "fold_id": fields.Integer(required=False),
        "mode": fields.String(required=False),
        "embedding_files": fields.String(nullable=True),
        "naturalness_files": fields.String(nullable=True),
        "finetuning_model_checkpoint": fields.String(nullable=True),
        "invokation_id": fields.Integer(required=False),
        "few_shot_params": fields.String(nullable=True),
        "num_mutants": fields.Integer(required=False),
        "input_activity_fpath": fields.String(required=False, nullable=True),
        "output_fpath": fields.String(required=False, nullable=True),
        "output_fpath_computed": fields.String(required=False, nullable=True),
        "date_created": fields.DateTime(required=False, nullable=True),
    },
)

few_shot_input_fields = type_ns.inherit(
    "FewShotInputFields",
    few_shot_fields,
    {
        "activity_file_bytes": fields.String(
            required=False, description="Base64 encoded activity file"
        ),
        "activity_file_from_few_shot_id": fields.Integer(required=False),
        "activity_file_from_round_id": fields.Integer(required=False),
    },
)


# =============================================================================
# FOLD RELATED FIELDS
# =============================================================================

get_folds_fields = type_ns.model("GetFolds", {"filter": fields.String(required=False)})

fold_fields = type_ns.model(
    "Fold",
    {
        "id": fields.Integer(readonly=True, required=False),
        "name": fields.String(),
        "owner": fields.String(attribute="user.email", required=False),
        "tags": fields.List(fields.String()),
        "create_date": fields.DateTime(
            format="iso8601Z", dt_format="iso8601", readonly=True, required=False
        ),
        "public": fields.Boolean(required=False),
        "yaml_config": fields.String(required=False),
        "diffusion_samples": fields.Integer(required=False),
        "disable_relaxation": fields.Boolean(required=False),
        "jobs": fields.List(fields.Nested(simple_invokation_fields)),
        "docks": fields.List(
            fields.Nested(dock_fields),
            attribute=lambda x: (
                [] if log_getattr(x, "_skip_embedded_fields", False, "docks") else x.docks
            ),
        ),
        "naturalness_runs": fields.List(
            fields.Nested(naturalness_fields),
            attribute=lambda x: (
                []
                if log_getattr(x, "_skip_embedded_fields", False, "naturalness_runs")
                else x.naturalness_runs
            ),
        ),
        "embeddings": fields.List(
            fields.Nested(embedding_fields),
            attribute=lambda x: (
                [] if log_getattr(x, "_skip_embedded_fields", False, "embeddings") else x.embeddings
            ),
        ),
        "few_shots": fields.List(
            fields.Nested(few_shot_fields),
            attribute=lambda x: (
                [] if log_getattr(x, "_skip_embedded_fields", False, "few_shots") else x.few_shots
            ),
        ),
        # Old AF2 inputs.
        "sequence": fields.String(required=False),
        "af2_model_preset": fields.String(required=False),
    },
)

new_folds_fields = type_ns.model(
    "NewFolds",
    {
        "folds_data": fields.List(fields.Nested(fold_fields, skip_none=True)),
        "start_fold_job": fields.Boolean(required=False),
        "email_on_completion": fields.Boolean(required=False),
        "skip_duplicate_entries": fields.Boolean(required=False),
        "is_dry_run": fields.Boolean(required=False),
    },
)


# =============================================================================
# PAGINATION FIELDS
# =============================================================================

pagination_fields = type_ns.model(
    "Pagination",
    {
        "page": fields.Integer(),
        "per_page": fields.Integer(),
        "total": fields.Integer(required=False),
        "pages": fields.Integer(required=False),
        "has_prev": fields.Boolean(required=False),
        "has_next": fields.Boolean(required=False),
    },
)

paginated_folds_fields = type_ns.model(
    "PaginatedFolds",
    {
        "data": fields.List(fields.Nested(fold_fields, skip_none=True)),
        "pagination": fields.Nested(pagination_fields),
    },
)


# =============================================================================
# TAG FIELDS
# =============================================================================

tag_info_fields = type_ns.model(
    "TagInfo",
    {
        "tag": fields.String(required=True, description="The tag name"),
        "fold_count": fields.Integer(required=True, description="Number of folds with this tag"),
        "contributors": fields.List(
            fields.String, description="Users who have folds with this tag"
        ),
        "recent_folds": fields.List(fields.String, description="Recent fold names with this tag"),
    },
)

tags_response_fields = type_ns.model(
    "TagsResponse", {"tags": fields.List(fields.Nested(tag_info_fields))}
)


# =============================================================================
# Note: Docking, ESM/Embedding, and Few-Shot fields are defined above
# before fold_fields to satisfy dependency requirements
# =============================================================================


# =============================================================================
# ANALYSIS FIELDS (PAE, CONTACT PROBABILITY, ETC.)
# =============================================================================

pae_fields = type_ns.model(
    "PAE",
    {
        "pae": fields.List(fields.List(fields.Float(readonly=True))),
    },
)

contact_prob_fields = type_ns.model(
    "ContactProb",
    {
        "contact_prob": fields.List(fields.List(fields.Float(readonly=True))),
    },
)


# =============================================================================
# ADMIN FIELDS
# =============================================================================

stamp_dbs_fields = type_ns.model(
    "StampDbsFields",
    {
        "revision": fields.String(),
    },
)

remove_failed_jobs_fields = type_ns.model("RemoveFailedJobs", {"queue": fields.String()})

kill_worker_fields = type_ns.model("KillWorkerFields", {"worker_id": fields.String()})

queue_test_job_fields = type_ns.model(
    "QueueTestJobField",
    {"queue": fields.String(), "command": fields.String(required=False)},
)


# =============================================================================
# CAMPAIGN FIELDS
# =============================================================================

campaign_round_input_fields = type_ns.model(
    "CampaignRoundInputFields",
    {
        "round_number": fields.Integer(required=False),
        "date_started": fields.DateTime(format="iso8601Z", dt_format="iso8601", required=False),
        "mode": fields.String(required=False),
        "naturalness_run_id": fields.Integer(required=False),
        "few_shot_run_id": NullableInteger(required=False),
        "slate_seq_ids": fields.String(required=False),
        "result_activity_fpath": fields.String(required=False),
        "input_templates": fields.String(required=False),
    },
)

campaign_round_fields = type_ns.inherit(
    "CampaignRoundFields",
    campaign_round_input_fields,
    {
        "id": fields.Integer(readonly=True),
        "campaign_id": fields.Integer(required=True),
        "naturalness_run": fields.Nested(naturalness_fields, required=False, allow_null=True),
        "few_shot_run": fields.Nested(
            few_shot_fields, required=False, allow_null=True, skip_none=True
        ),
    },
)

campaign_input_fields = type_ns.model(
    "CampaignInputFields",
    {
        "name": fields.String(required=True),
        "description": fields.String(required=False),
        "fold_id": fields.Integer(required=True),
        "naturalness_model": fields.String(required=False),
        "embedding_model": fields.String(required=False),
        "domain_boundaries": fields.String(required=False),
    },
)

campaign_fields = type_ns.inherit(
    "CampaignFields",
    campaign_input_fields,
    {
        "id": fields.Integer(readonly=True),
        "created_at": fields.DateTime(format="iso8601Z", dt_format="iso8601", readonly=True),
        "rounds": fields.List(fields.Nested(campaign_round_fields), readonly=True),
        "fold_name": fields.String(
            attribute=lambda x: x.fold.name if x.fold else None, readonly=True
        ),
    },
)

paginated_campaigns_fields = type_ns.model(
    "PaginatedCampaignsFields",
    {
        "campaigns": fields.List(fields.Nested(campaign_fields)),
        "total": fields.Integer(),
        "page": fields.Integer(),
        "per_page": fields.Integer(),
        "pages": fields.Integer(),
    },
)


# =============================================================================
# FILE FIELDS
# =============================================================================

fold_file_zip_fields = type_ns.model(
    "FoldPdbZip",
    {
        "fold_ids": fields.List(fields.Integer(required=True)),
        "relative_fpath": fields.String(),
        "output_dirname": fields.String(),
        "flatten_filepath": fields.Boolean(required=False, default=False),
        "use_fold_name": fields.Boolean(required=False, default=False),
    },
)


# =============================================================================
# DNA BUILD FIELDS
# =============================================================================

dna_build_seq_result_fields = type_ns.model(
    "DnaBuildSeqResult",
    {
        "success": fields.Boolean(required=True),
        "error_msg": fields.String(required=False, nullable=True),
        "template_used": fields.String(required=False, nullable=True),
        "teselagen_seq_id": fields.String(required=False, nullable=True),
    },
)

dna_build_response_fields = type_ns.model(
    "DnaBuildResponse",
    {
        "design_name": fields.String(required=True),
        "teselagen_id": fields.String(required=False, nullable=True),
        "seq_id_results": fields.Raw(required=True, description="Dict[str, DnaBuildSeqResult]"),
    },
)


# =============================================================================
# FIELD REGISTRATION FUNCTIONS
# =============================================================================


def register_fields_with_namespace(ns: Namespace):
    """
    Register all field definitions with the given namespace.
    Call this from each view module to use the centralized field definitions.

    Returns a dictionary of field names to field objects for easy access.
    """
    # Copy all field definitions to the target namespace
    field_definitions = {}

    # Get all field definitions from this module
    for name, value in globals().items():
        if name.endswith("_fields") and hasattr(value, "_schema"):
            # Re-create the field with the target namespace
            field_definitions[name] = value

    return field_definitions


def get_field_by_name(field_name: str):
    """Get a field definition by name."""
    return globals().get(field_name)


# No fixup needed - dependencies are properly ordered in DAG fashion
