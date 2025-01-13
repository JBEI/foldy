import io
import re

from flask import Response, stream_with_context
from flask import current_app, request, send_file, make_response
from flask_jwt_extended.utils import get_jwt_identity, get_jwt
from flask_restx import Namespace
from flask_jwt_extended import jwt_required
from flask_restx import Resource
from flask_restx import fields
from flask_restx import reqparse
from sqlalchemy.sql.elements import and_
from werkzeug.exceptions import BadRequest

from app.jobs import other_jobs
from app.jobs import embed_jobs
from app.models import Dock, Fold, Invokation, Embedding
from app.extensions import db, rq
from app.util import get_job_type_replacement, make_new_folds
from app.helpers.fold_storage_manager import FoldStorageManager
from app.authorization import (
    user_jwt_grants_edit_access,
    verify_has_edit_access,
)

ns = Namespace("embed_views", decorators=[jwt_required(fresh=True)])


embeddings_fields = ns.model(
    "Embeddings",
    {
        "batch_name": fields.String(required=True),
        "embedding_model": fields.String(required=True),
        "extra_seq_ids": fields.List(fields.String(), required=False),
        "dms_starting_seq_ids": fields.List(fields.String(), required=False),
    },
)


@ns.route("/embeddings/<int:fold_id>")
class CalculateEmbeddingsResource(Resource):
    @verify_has_edit_access
    @ns.expect(embeddings_fields)
    def post(self, fold_id):
        req = request.get_json()

        batch_name = req["batch_name"]
        embedding_model = req["embedding_model"]
        extra_seq_ids = req.get("extra_seq_ids", [])
        dms_starting_seq_ids = req.get("dms_starting_seq_ids", [])

        extra_seq_ids = [seq_id.strip() for seq_id in extra_seq_ids if seq_id.strip()]
        dms_starting_seq_ids = [
            seq_id.strip() for seq_id in dms_starting_seq_ids if seq_id.strip()
        ]

        ALLOWED_EMBEDDING_MODELS = ["esmc_300m", "esmc_600m"]
        if embedding_model not in ALLOWED_EMBEDDING_MODELS:
            raise BadRequest(
                f"Invalid embedding model {embedding_model}: must be one of {ALLOWED_EMBEDDING_MODELS}"
            )

        fold = Fold.get_by_id(fold_id)

        new_invokation_id = get_job_type_replacement(fold, f"embed_{batch_name}")

        embed_record = Embedding.create(
            name=batch_name,
            fold_id=fold_id,
            embedding_model=embedding_model,
            extra_seq_ids=",".join(extra_seq_ids),
            dms_starting_seq_ids=",".join(dms_starting_seq_ids),
            invokation_id=new_invokation_id,
        )

        esm_q = rq.get_queue("esm")
        esm_q.enqueue(
            embed_jobs.get_esm_embeddings,
            embed_record.id,
            job_timeout="12h",
            result_ttl=48 * 60 * 60,  # 2 days
        )

        # name = Column(db.String, nullable=False)

        # fold_id = Column(
        #     db.Integer, db.ForeignKey("roles.id", ondelete="CASCADE", onupdate="CASCADE")
        # )
        # fold = relationship("Fold", back_populates="embeddings")

        # embedding_model = Column(db.String, nullable=False)
        # extra_seq_ids = Column(db.String)
        # dms_starting_seq_ids = Column(db.String)

        # # State tracking.
        # invokation_id = Column(
        #     db.Integer,
        #     db.ForeignKey("invokation.id", ondelete="CASCADE", onupdate="CASCADE"),
        # )

        return True
