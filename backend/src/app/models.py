""" Database models.

Copied from https://github.com/cookiecutter-flask/cookiecutter-flask
"""

from datetime import datetime, UTC
from typing import List, Dict, Any, Optional, Union

from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy import func
from sqlalchemy.orm import deferred
from sqlalchemy import Index

from app.database import Column, PkModel, db, reference_col, relationship


class Invokation(PkModel):
    """A single invokation of a command and its results."""

    __tablename__ = "invokation"

    fold_id = Column(
        db.Integer, db.ForeignKey("roles.id", ondelete="CASCADE", onupdate="CASCADE")
    )
    fold = relationship("Fold", back_populates="jobs")

    type = Column(db.String(80), nullable=False)
    state = Column(db.String(80), nullable=True)
    starttime = Column(db.DateTime(timezone=True), nullable=True)
    timedelta = Column(db.Interval, nullable=True)

    command = Column(db.Text, nullable=True)
    log = deferred(Column(db.Text, nullable=True))

    def __init__(self, fold_id: int, type: str, state: str) -> None:
        super().__init__(fold_id=fold_id, type=type, state=state)


class User(PkModel):
    """A user of the app."""

    __tablename__ = "users"

    # Id is created automatically.

    email = Column(db.String(80), unique=True, nullable=False)
    created_at = Column(db.DateTime, nullable=False, default=datetime.now(UTC))
    access_type = Column(db.String(80), nullable=True)
    attributes = Column(db.JSON, nullable=True, default=dict)  # Add this line

    folds = relationship("Fold", back_populates="user")

    def __init__(self, email: str, access_type: str) -> None:
        """Create a new user."""
        super().__init__(email=email, access_type=access_type)

    def __repr__(self) -> str:
        return f"{self.email}"

    @hybrid_property
    def num_folds(self) -> int:
        return self.folds.count(self)


class Fold(PkModel):
    """A protein fold."""

    __tablename__ = "roles"

    # 2/13/25: Dashboard queries are taking >10 seconds at times, so we are
    # adding indices to try to speed them up. Unclear at this time if it will
    # help.
    __table_args__ = (
        Index("ix_roles_user_id", "user_id"),
        Index("ix_roles_public", "public"),
        Index("ix_roles_yaml_config", "yaml_config"),
        # Indexing a large text column (like 'sequence') depends on your DB engine;
        # if it supports text indexing, you can enable it, but for large fields you may
        # want more specialized search solutions. If you still want a simple index:
        # Index("ix_roles_sequence", "sequence"),
    )

    # Id is created automatically.

    name = Column(db.String(80), unique=True, nullable=False)
    user_id = reference_col("users", nullable=True)
    user = relationship("User", back_populates="folds")
    tagstring = Column(db.String(80), nullable=True)
    create_date = Column("create_date", db.DateTime, default=func.now())
    public = Column(db.Boolean, nullable=True)

    jobs = relationship(
        "Invokation",
        back_populates="fold",
        passive_deletes=True,
        cascade="all,delete-orphan",
    )

    docks = relationship(
        "Dock",
        back_populates="receptor_fold",
        passive_deletes=True,
        cascade="all,delete-orphan",
    )

    logits = relationship(
        "Logit",
        back_populates="fold",
        passive_deletes=True,
        cascade="all,delete-orphan",
    )

    evolutions = relationship(
        "Evolution",
        back_populates="fold",
        passive_deletes=True,
        cascade="all,delete-orphan",
    )

    embeddings = relationship(
        "Embedding",
        back_populates="fold",
        passive_deletes=True,
        cascade="all,delete-orphan",
    )

    # New, good, Boltz input.
    yaml_config = Column(db.String, nullable=True)
    diffusion_samples = Column(db.Integer, nullable=True)
    # Old AF2 inputs.
    sequence = Column(db.Text)
    af2_model_preset = Column(db.String, nullable=True)
    disable_relaxation = Column(db.Boolean, nullable=True)

    @hybrid_property
    def tags(self) -> List[str]:
        if not self.tagstring:
            return []
        return self.tagstring.split(",")


class Dock(PkModel):
    """A docking run."""

    __tablename__ = "docking"

    # Id is created automatically.

    ligand_name = Column(db.String, nullable=False)
    ligand_smiles = Column(db.String, nullable=False)
    tool = Column(db.String, nullable=True)  # Either vina or diffdock.

    receptor_fold_id = Column(
        db.Integer, db.ForeignKey("roles.id", ondelete="CASCADE", onupdate="CASCADE")
    )
    receptor_fold = relationship("Fold", back_populates="docks")

    bounding_box_residue = Column(db.String, nullable=True)
    bounding_box_radius_angstrom = Column(db.Float, nullable=True)

    # State tracking.
    invokation_id = Column(
        db.Integer,
        db.ForeignKey("invokation.id", ondelete="CASCADE", onupdate="CASCADE"),
    )

    # Vina output.
    pose_energy = Column(db.String, nullable=True)

    # Diffdock output - a CSV of pose confidences.
    pose_confidences = Column(db.String, nullable=True)


class Logit(PkModel):
    """A logit run."""

    __tablename__ = "logits"

    name = Column(db.String, nullable=False)

    fold_id = Column(
        db.Integer, db.ForeignKey("roles.id", ondelete="CASCADE", onupdate="CASCADE")
    )
    fold = relationship("Fold", back_populates="logits")

    logit_model = Column(db.String, nullable=False)
    use_structure = Column(db.Boolean, nullable=True)
    get_depth_two_logits = Column(db.Boolean, nullable=True)

    invokation_id = Column(
        db.Integer,
        db.ForeignKey("invokation.id", ondelete="CASCADE", onupdate="CASCADE"),
    )


class Embedding(PkModel):
    """An embedding run."""

    __tablename__ = "embeddings"

    name = Column(db.String, nullable=False)

    fold_id = Column(
        db.Integer, db.ForeignKey("roles.id", ondelete="CASCADE", onupdate="CASCADE")
    )
    fold = relationship("Fold", back_populates="embeddings")

    embedding_model = Column(db.String, nullable=False)
    extra_seq_ids = Column(db.String)
    dms_starting_seq_ids = Column(db.String)

    # State tracking.
    invokation_id = Column(
        db.Integer,
        db.ForeignKey("invokation.id", ondelete="CASCADE", onupdate="CASCADE"),
    )


class Evolution(PkModel):
    """A single evolution of a fold."""

    __tablename__ = "fold_evolution"

    # Id is created automatically.

    name = Column(db.String, nullable=False)

    fold_id = Column(
        db.Integer, db.ForeignKey("roles.id", ondelete="CASCADE", onupdate="CASCADE")
    )
    fold = relationship("Fold", back_populates="evolutions")

    # Two options: "finetuning" on rank or "randomforest" on logits.
    mode = Column(db.String, nullable=True)

    # If mode == randomforest, then this is the fixed embeddings to use.
    embedding_files = Column(
        db.String, nullable=True
    )  # A list of embedding file paths.

    # If mode == finetuning, then this is the model checkpoint to use.
    finetuning_model_checkpoint = Column(db.String, nullable=True)

    # State tracking.
    invokation_id = Column(
        db.Integer,
        db.ForeignKey("invokation.id", ondelete="CASCADE", onupdate="CASCADE"),
    )
