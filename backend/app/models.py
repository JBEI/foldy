"""Database models.

Copied from https://github.com/cookiecutter-flask/cookiecutter-flask
"""

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Union

from app.database import Column, PkModel, db, reference_col, relationship
from sqlalchemy import Index, func
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.orm import deferred


class Invokation(PkModel):
    """A single invokation of a command and its results."""

    __tablename__ = "invokation"

    fold_id = Column(db.Integer, db.ForeignKey("roles.id", ondelete="CASCADE", onupdate="CASCADE"))
    fold = relationship("Fold", back_populates="jobs")

    type = Column(db.String(80), nullable=False)
    state = Column(db.String(80), nullable=True)
    starttime = Column(db.DateTime(timezone=True), nullable=True)
    timedelta = Column(db.Interval, nullable=True)

    command = Column(db.Text, nullable=True)
    log = deferred(Column(db.Text, nullable=True))

    def __init__(self, fold_id: int, type: str, state: str) -> None:
        super().__init__()
        self.fold_id = fold_id
        self.type = type
        self.state = state


class User(PkModel):
    """A user of the app."""

    __tablename__ = "users"

    # Id is created automatically.

    email = Column(db.String(80), unique=True, nullable=False)
    name = Column(db.String(80), nullable=True)
    created_at = Column(db.DateTime, nullable=False, default=datetime.now(UTC))
    access_type = Column(db.String(80), nullable=True)
    attributes = Column(db.JSON, nullable=True, default=dict)  # Add this line

    folds = relationship("Fold", back_populates="user")

    teselagen_api_key = Column(db.String(80), nullable=True)

    def __init__(self, email: str, access_type: str) -> None:
        """Create a new user."""
        super().__init__()
        self.email = email
        self.access_type = access_type

    def __repr__(self) -> str:
        return f"{self.email}"

    @hybrid_property
    def num_folds(self) -> int:
        count: int = self.folds.count(self)
        return count


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

    naturalness_runs = relationship(
        "Naturalness",
        back_populates="fold",
        passive_deletes=True,
        cascade="all,delete-orphan",
    )

    few_shots = relationship(
        "FewShot",
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
            empty_list: List[str] = []
            return empty_list
        result: List[str] = self.tagstring.split(",")
        return result


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

    def __init__(
        self,
        ligand_name: str,
        ligand_smiles: str,
        tool: str,
        receptor_fold_id: int,
        invokation_id: int,
        bounding_box_residue: Optional[str] = None,
        bounding_box_radius_angstrom: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.ligand_name = ligand_name
        self.ligand_smiles = ligand_smiles
        self.tool = tool
        self.receptor_fold_id = receptor_fold_id
        self.invokation_id = invokation_id
        self.bounding_box_residue = bounding_box_residue
        self.bounding_box_radius_angstrom = bounding_box_radius_angstrom


class Naturalness(PkModel):
    """A naturalness run."""

    __tablename__ = "logits"

    name = Column(db.String, nullable=False)

    fold_id = Column(db.Integer, db.ForeignKey("roles.id", ondelete="CASCADE", onupdate="CASCADE"))
    fold = relationship("Fold", back_populates="naturalness_runs")

    logit_model = Column(db.String, nullable=False)
    use_structure = Column(db.Boolean, nullable=True)
    get_depth_two_logits = Column(db.Boolean, nullable=True)
    output_fpath = Column(db.String, nullable=True)
    date_created = Column(db.DateTime(timezone=True), nullable=True, default=datetime.now(UTC))

    invokation_id = Column(
        db.Integer,
        db.ForeignKey("invokation.id", ondelete="CASCADE", onupdate="CASCADE"),
    )

    @hybrid_property
    def output_fpath_computed(self) -> str:
        """Get the output CSV path, falling back to computed path if not set."""
        if self.output_fpath:
            return self.output_fpath
        return f"naturalness/naturalness_{self.name}_melted.csv"


class Embedding(PkModel):
    """An embedding run."""

    __tablename__ = "embeddings"

    name = Column(db.String, nullable=False)

    fold_id = Column(db.Integer, db.ForeignKey("roles.id", ondelete="CASCADE", onupdate="CASCADE"))
    fold = relationship("Fold", back_populates="embeddings")

    embedding_model = Column(db.String, nullable=False)
    extra_seq_ids = Column(db.String)
    dms_starting_seq_ids = Column(db.String)
    homolog_fasta = Column(db.String, nullable=True)
    extra_layers = Column(db.String, nullable=True)
    domain_boundaries = Column(db.String, nullable=True)
    output_fpath = Column(db.String, nullable=True)
    date_created = Column(db.DateTime(timezone=True), nullable=True, default=datetime.now(UTC))

    # State tracking.
    invokation_id = Column(
        db.Integer,
        db.ForeignKey("invokation.id", ondelete="CASCADE", onupdate="CASCADE"),
    )

    @hybrid_property
    def output_fpath_computed(self) -> str:
        """Get the output CSV path, falling back to computed path if not set."""
        if self.output_fpath:
            return self.output_fpath
        return f"embeddings/embeddings_{self.name}_processed.csv"


class FewShot(PkModel):
    """A single slate build for a fold."""

    __tablename__ = "fold_evolution"

    # Id is created automatically.

    name = Column(db.String, nullable=False)

    fold_id = Column(db.Integer, db.ForeignKey("roles.id", ondelete="CASCADE", onupdate="CASCADE"))
    fold = relationship("Fold", back_populates="few_shots")

    mode = Column(db.String, nullable=True)
    embedding_files = Column(db.String, nullable=True)  # A list of embedding file paths.
    naturalness_files = Column(db.String, nullable=True)  # A list of embedding file paths.
    few_shot_params = Column(db.String, nullable=True)
    num_mutants = Column(db.Integer, nullable=True)
    input_activity_fpath = Column(db.String, nullable=True)
    date_created = Column(db.DateTime(timezone=True), nullable=True, default=datetime.now(UTC))

    output_fpath = Column(db.String, nullable=True)

    # State tracking.
    invokation_id = Column(
        db.Integer,
        db.ForeignKey("invokation.id", ondelete="CASCADE", onupdate="CASCADE"),
    )

    # NO LONGER USED.
    finetuning_model_checkpoint = Column(db.String, nullable=True)

    @hybrid_property
    def output_fpath_computed(self) -> str:
        """Get the output CSV path, falling back to computed path if not set."""
        if self.output_fpath:
            return self.output_fpath
        return f"few_shots/{self.name}/predicted_activity.csv"


class Campaign(PkModel):
    """A directed evolution campaign."""

    __tablename__ = "campaigns"

    # Id is created automatically.

    name = Column(db.String(80), nullable=False)
    description = Column(db.Text, nullable=True)
    created_at = Column(db.DateTime, nullable=False, default=datetime.now(UTC))

    fold_id = Column(db.Integer, db.ForeignKey("roles.id", ondelete="CASCADE", onupdate="CASCADE"))
    fold = relationship("Fold", backref="campaigns")

    naturalness_model = Column(db.String(80), nullable=True, default="esm2_t33_650M_UR50D")
    embedding_model = Column(db.String(80), nullable=True, default="esm2_t33_650M_UR50D")
    domain_boundaries = Column(db.String, nullable=True)

    rounds = relationship(
        "CampaignRound",
        back_populates="campaign",
        passive_deletes=True,
        cascade="all,delete-orphan",
        order_by="CampaignRound.date_started",
    )

    def __init__(
        self,
        name: str,
        fold_id: int,
        description: Optional[str] = None,
        naturalness_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        domain_boundaries: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.fold_id = fold_id
        self.description = description
        self.naturalness_model = naturalness_model or "esm2_t33_650M_UR50D"
        self.embedding_model = embedding_model or "esm2_t33_650M_UR50D"
        self.domain_boundaries = domain_boundaries


class CampaignRound(PkModel):
    """A round within a directed evolution campaign."""

    __tablename__ = "campaign_rounds"

    # Id is created automatically.

    campaign_id = Column(
        db.Integer, db.ForeignKey("campaigns.id", ondelete="CASCADE", onupdate="CASCADE")
    )
    campaign = relationship("Campaign", back_populates="rounds")

    date_started = Column(db.DateTime, nullable=False, default=datetime.now(UTC))
    round_number = Column(db.Integer, nullable=False)
    mode = Column(db.String(20), nullable=True)  # 'zero-shot' or 'few-shot'
    naturalness_run_id = Column(
        db.Integer,
        db.ForeignKey("logits.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
    )
    naturalness_run = relationship("Naturalness")
    few_shot_run_id = Column(
        db.Integer,
        db.ForeignKey("fold_evolution.id", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
    )
    few_shot_run = relationship("FewShot")
    slate_seq_ids = Column(db.Text, nullable=True)  # Comma-separated sequence IDs
    result_activity_fpath = Column(db.Text, nullable=True)  # Relative path to activity file
    input_templates = Column(
        db.Text, nullable=True
    )  # Comma-separated list of selected template IDs

    def __init__(
        self,
        campaign_id: int,
        round_number: int,
        date_started: Optional[datetime] = None,
        mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.campaign_id = campaign_id
        self.round_number = round_number
        self.mode = mode
        if date_started:
            self.date_started = date_started
