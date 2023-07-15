""" Database models.

Copied from https://github.com/cookiecutter-flask/cookiecutter-flask
"""
import datetime

from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy import func
from sqlalchemy.orm import deferred

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
    starttime = Column(db.DateTime, nullable=True)
    timedelta = Column(db.Interval, nullable=True)

    log = deferred(Column(db.Text, nullable=True))

    def __init__(self, fold_id, type, state):
        super().__init__(fold_id=fold_id, type=type, state=state)


class User(PkModel):
    """A user of the app."""

    __tablename__ = "users"

    # Id is created automatically.

    email = Column(db.String(80), unique=True, nullable=False)
    created_at = Column(db.DateTime, nullable=False, default=datetime.datetime.utcnow)
    # fold = relationship("Fold", backref="user",lazy='dynamic')

    def __init__(self, email):
        """Create a new user."""
        super().__init__(email=email)

    def __repr__(self):
        return f"{self.email}"

    @hybrid_property
    def num_folds(self):
        return self.folds.count(self)


class Fold(PkModel):
    """A protein fold."""

    __tablename__ = "roles"

    # Id is created automatically.

    name = Column(db.String(80), unique=True, nullable=False)
    user_id = reference_col("users", nullable=True)
    user = relationship("User", backref="folds")
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

    sequence = Column(db.Text)
    af2_model_preset = Column(db.String, nullable=True)
    disable_relaxation = Column(db.Boolean, nullable=True)

    @hybrid_property
    def tags(self):
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

    # Outputs.
    pose_energy = Column(db.String, nullable=True)
