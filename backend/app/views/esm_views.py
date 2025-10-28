"""Defines API endpoints related to protein language models."""

import logging
from typing import List

from flask import (
    request,
)
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource
from rq import Callback
from sqlalchemy.sql.elements import and_
from werkzeug.exceptions import BadRequest

from app.api_fields import embedding_fields, naturalness_fields
from app.authorization import verify_has_edit_access
from app.helpers.boltz_yaml_helper import BoltzYamlHelper
from app.helpers.rq_helpers import (
    add_meta_to_job,
    get_queue,
    send_failure_email,
    send_success_email,
)
from app.helpers.sequence_util import VALID_AMINO_ACIDS, maybe_get_seq_id_error_message
from app.jobs import esm_jobs
from app.models import Embedding, Fold, Naturalness
from app.util import get_job_type_replacement

ns = Namespace("esm_views", decorators=[jwt_required(fresh=True)])

ALLOWED_ESM_MODELS: List[str] = [
    "esmc_600m",
    "esmc_300m",
    "esm3-open",
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
    "esm1v_t33_650M_UR90S_1",
    "esm1v_t33_650M_UR90S_2",
    "esm1v_t33_650M_UR90S_3",
    "esm1v_t33_650M_UR90S_4",
    "esm1v_t33_650M_UR90S_5",
]

ALLOWED_NATURALNESS_MODELS: List[str] = ALLOWED_ESM_MODELS + ["esm1v_t33_650M_UR90S_ensemble"]


@ns.route("/embeddings")
class CalculateEmbeddingsResource(Resource):
    @verify_has_edit_access
    @ns.expect(embedding_fields)
    def post(self) -> bool:
        """Create a new embedding calculation job for a fold.

        Args:
            fold_id: ID of the fold to create embeddings for

        Returns:
            True if the embedding job was successfully created

        Raises:
            BadRequest: If embedding model is not allowed or fold doesn't exist
        """
        req = request.get_json()

        fold_id: int = req["fold_id"]
        embedding_name: str = req["name"]
        embedding_model: str = req["embedding_model"]
        extra_seq_ids_str: str = req.get("extra_seq_ids", "")
        dms_starting_seq_ids_str: str = req.get("dms_starting_seq_ids", "")
        extra_layers_str: str = req.get("extra_layers", "")
        domain_boundaries_str: str = req.get("domain_boundaries", "")
        homolog_fasta: str = req.get("homolog_fasta", None)

        extra_seq_ids: list[str] = [
            seq_id.strip() for seq_id in extra_seq_ids_str.split(",") if seq_id.strip()
        ]
        dms_starting_seq_ids: list[str] = [
            seq_id.strip() for seq_id in dms_starting_seq_ids_str.split(",") if seq_id.strip()
        ]
        extra_layers: list[str] = [
            layer.strip() for layer in extra_layers_str.split(",") if layer.strip()
        ]
        domain_boundaries: list[str] = [
            boundary.strip() for boundary in domain_boundaries_str.split(",") if boundary.strip()
        ]

        if embedding_model not in ALLOWED_ESM_MODELS:
            raise BadRequest(
                f"Invalid embedding model {embedding_model}: must be one of {ALLOWED_ESM_MODELS}"
            )

        fold = Fold.get_by_id(fold_id)
        if not fold:
            raise BadRequest(f"Could not find fold {fold_id}")
        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)
        if len(boltz_yaml_helper.get_protein_sequences()) > 1:
            raise ValueError(
                "Fold has multiple protein sequences, which is not supported for ESM embeddings yet."
            )
        wt_aa_seq = boltz_yaml_helper.get_protein_sequences()[0][1]

        homolog_id_to_seq_map = esm_jobs.load_fasta_to_dict(homolog_fasta)
        esm_jobs.validate_embedding_inputs(
            wt_aa_seq, extra_seq_ids, dms_starting_seq_ids, homolog_id_to_seq_map
        )

        new_invokation_id = get_job_type_replacement(fold, f"embed_{embedding_name}")

        embed_record = Embedding.create(
            name=embedding_name,
            fold_id=fold_id,
            embedding_model=embedding_model,
            extra_seq_ids=",".join(extra_seq_ids),
            dms_starting_seq_ids=",".join(dms_starting_seq_ids),
            extra_layers=",".join(extra_layers),
            domain_boundaries=",".join(domain_boundaries) if domain_boundaries else None,
            homolog_fasta=homolog_fasta,
            invokation_id=new_invokation_id,
        )

        esm_q = get_queue("esm")
        enqueued_job = esm_q.enqueue(
            esm_jobs.get_esm_embeddings,
            embed_record.id,
            job_timeout="24h",
            result_ttl=48 * 60 * 60,  # 2 days
            on_success=Callback(send_success_email, timeout="5s"),
            on_failure=Callback(send_failure_email, timeout="5s"),
        )
        add_meta_to_job(enqueued_job, fold, "embed", embed_record.id)

        logging.info(
            f"Queued embedding job {enqueued_job.id} for fold {fold_id}, model {embedding_model}"
        )
        return True


@ns.route("/startnaturalness/<int:fold_id>")
class StartNaturalnessResource(Resource):
    @verify_has_edit_access
    @ns.expect(naturalness_fields)
    @ns.marshal_with(naturalness_fields)
    def post(self, fold_id: int) -> Naturalness:
        """Create a new naturalness calculation job for a fold.

        Args:
            fold_id: ID of the fold to create naturalness for

        Returns:
            The created Naturalness record

        Raises:
            BadRequest: If naturalness model is not allowed or fold doesn't exist
        """
        req = request.get_json()

        name: str = req["name"]
        logit_model: str = req["logit_model"]
        use_structure: bool = req.get("use_structure", False)
        get_depth_two_logits: bool = req.get("get_depth_two_logits", False)

        if logit_model not in ALLOWED_NATURALNESS_MODELS:
            raise BadRequest(
                f"Invalid naturalness model {logit_model}: must be one of {ALLOWED_NATURALNESS_MODELS}"
            )

        fold = Fold.get_by_id(fold_id)

        if not fold:
            raise BadRequest(f"Fold with ID {fold_id} not found")

        existing_naturalness = Naturalness.query.filter(
            Naturalness.name == name, Naturalness.fold_id == fold_id
        ).first()
        if existing_naturalness:
            logging.info(f"Deleting existing naturalness job {existing_naturalness.id} for {name}")
            existing_naturalness.delete()

        new_invokation_id: int = get_job_type_replacement(fold, f"naturalness_{name}")

        naturalness_record: Naturalness = Naturalness.create(
            name=name,
            fold_id=fold_id,
            logit_model=logit_model,
            use_structure=use_structure,
            get_depth_two_logits=get_depth_two_logits,
            invokation_id=new_invokation_id,
        )

        esm_q = get_queue("esm")
        enqueued_job = esm_q.enqueue(
            esm_jobs.get_esm_naturalness,
            naturalness_record.id,
            job_timeout="24h",
            result_ttl=48 * 60 * 60,  # 2 days
            on_success=Callback(send_success_email, timeout="5s"),
            on_failure=Callback(send_failure_email, timeout="5s"),
        )
        add_meta_to_job(enqueued_job, fold, "naturalness", naturalness_record.id)

        logging.info(
            f"Queued naturalness job {enqueued_job.id} for fold {fold_id}, model {logit_model}, "
            f"use_structure={use_structure}, get_depth_two_logits={get_depth_two_logits}"
        )

        return naturalness_record


@ns.route("/naturalness/<int:naturalness_id>")
class NaturalnessResource(Resource):
    @verify_has_edit_access
    def delete(self, naturalness_id: int) -> None:
        """Delete a naturalness run by ID.

        Args:
            naturalness_id: ID of the naturalness run to delete
        """
        naturalness = Naturalness.query.get(naturalness_id)
        if not naturalness:
            raise BadRequest(f"Naturalness run not found {naturalness_id}")

        logging.info(f"Deleting naturalness run {naturalness_id} ({naturalness.name})")
        naturalness.delete()


@ns.route("/embedding/<int:embedding_id>")
class EmbeddingResource(Resource):
    @verify_has_edit_access
    def delete(self, embedding_id: int) -> None:
        """Delete an embedding run by ID.

        Args:
            embedding_id: ID of the embedding run to delete
        """
        embedding = Embedding.query.get(embedding_id)
        if not embedding:
            raise BadRequest(f"Embedding run not found {embedding_id}")

        logging.info(f"Deleting embedding run {embedding_id} ({embedding.name})")
        embedding.delete()
