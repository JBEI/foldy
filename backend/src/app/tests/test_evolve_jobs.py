import pytest
from unittest.mock import MagicMock, patch
from flask import Flask
from werkzeug.exceptions import BadRequest
import pandas as pd
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.models import User, Fold, Evolution, Invokation
from app.jobs.evolve_jobs import run_evolvepro


@pytest.fixture
def app():
    """Create and configure a test Flask application instance."""
    app = Flask(__name__)
    app.config.update(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "SQLALCHEMY_TRACK_MODIFICATIONS": False,
            "FOLDY_STORAGE_TYPE": "Local",
            "FOLDY_LOCAL_STORAGE_DIR": "/tmp/test_storage",
        }
    )

    # Initialize extensions
    db.init_app(app)

    # Create all tables in the test database
    with app.app_context():
        db.create_all()

    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture
def mock_storage_manager():
    """Create a mock storage manager."""
    with patch("app.util.FoldStorageManager") as mock:
        storage_manager = MagicMock()
        mock.return_value = storage_manager
        yield storage_manager


@pytest.fixture
def test_user(app):
    """Create a test user."""
    with app.app_context():
        user = User(email="test@example.com", access_type="admin")
        db.session.add(user)
        db.session.commit()
        # Refresh the user to ensure it's attached to the session
        db.session.refresh(user)
        yield user
        # Clean up
        db.session.delete(user)
        db.session.commit()


@pytest.fixture
def test_fold(app, test_user):
    """Create a test fold."""
    with app.app_context():
        # Get a fresh user object from the database
        user = db.session.get(User, test_user.id)
        fold = Fold(
            name="test_fold",
            user_id=test_user.id,  # Now using the refreshed user
            sequence="ACD",
            tagstring="test",
            af2_model_preset="monomer_ptm",
        )
        db.session.add(fold)
        db.session.commit()
        yield fold
        # Clean up
        db.session.delete(fold)
        db.session.commit()


@pytest.fixture
def test_invokation(app, test_fold):
    """Create a test invokation."""
    with app.app_context():
        invokation = Invokation(
            fold_id=test_fold.id, type="evolve_test1", state="running"
        )
        db.session.add(invokation)
        db.session.commit()
        yield invokation
        # Clean up
        db.session.delete(invokation)
        db.session.commit()


@pytest.fixture
def test_fold_evolution(app, test_fold, test_invokation):
    """Create a test fold evolution."""
    with app.app_context():
        evolution = Evolution.create(
            name="test_evolve",
            fold_id=test_fold.id,
            invokation_id=test_invokation.id,
            embedding_files="embed/test_embedding_file.csv",
        )
        db.session.add(evolution)
        db.session.commit()
        yield evolution
        # Clean up
        db.session.delete(evolution)
        db.session.commit()


@pytest.fixture
def mock_foldy_storage(app, tmp_path, test_fold, test_fold_evolution):
    """Setup a temporary storage directory with test files."""
    # Create the base storage structure
    storage_dir = tmp_path / "folds"
    storage_dir.mkdir()

    # Update app config to use this temporary directory
    app.config["FOLDY_LOCAL_STORAGE_DIR"] = str(storage_dir)

    # Create example folder structure
    fold_dir = storage_dir / ("%06d" % test_fold.id)

    evolve_dir = fold_dir / "evolve" / test_fold_evolution.name
    evolve_dir.mkdir(parents=True)
    embed_dir = fold_dir / "embed"
    embed_dir.mkdir(parents=True)

    # Create the necessary files.
    pd.DataFrame({"seq_id": ["", "A1G"], "activity": [1, 2]}).to_excel(
        evolve_dir / "activity.xlsx"
    )
    pd.DataFrame(
        {
            "seq_id": ["", "A1G", "A1G_C2Y"],
            "embedding": ["[0, 1, 0]", "[0, 2, 0]", "[0, 3, 0]"],
        }
    ).to_csv(embed_dir / "test_embedding_file.csv")
    yield storage_dir


def test_run_evolvepro_fails_nofile(
    app, client, mock_storage_manager, test_fold_evolution
):
    """Basic test for run_evolvepro function."""
    with app.app_context():
        with pytest.raises(AssertionError, match="xlsx not found"):
            run_evolvepro(evolve_id=test_fold_evolution.id)


def test_run_evolvepro_succeeds(
    app, client, tmp_path, mock_storage_manager, test_fold_evolution, mock_foldy_storage
):
    """Basic test for run_evolvepro function."""
    with app.app_context():
        run_evolvepro(evolve_id=test_fold_evolution.id)

        fold_dir = mock_foldy_storage / ("%06d" % test_fold_evolution.fold_id)
        evolve_dir = fold_dir / "evolve" / test_fold_evolution.name
        assert (evolve_dir / "model.joblib").exists()
        assert (evolve_dir / "predicted_activity_df.csv").exists()

        predicted_activity_df = pd.read_csv(
            evolve_dir / "predicted_activity_df.csv", keep_default_na=False
        )
        assert predicted_activity_df.shape[0] == 3
        predicted_activity_df.index = predicted_activity_df.seq_id

        # The tool should know that both mutants are more active than WT.
        # If this were a linear model, it would know that the second mutant should
        # be more active than the first, based on the constructed embedding. But the
        # random forest model can't extrapolate.
        assert predicted_activity_df.loc["A1G"].predicted_activity > 1.5
        assert predicted_activity_df.loc["A1G_C2Y"].predicted_activity > 1.5
