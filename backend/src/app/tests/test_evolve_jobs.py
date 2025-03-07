import pytest
from unittest.mock import MagicMock, patch
from flask import Flask
from werkzeug.exceptions import BadRequest
import pandas as pd
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.models import User, Fold, Evolution, Invokation
from app.jobs.evolve_jobs import run_evolvepro
import random
import numpy as np


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
    pd.DataFrame({"seq_id": ["WT", "A1G"], "activity": [1, 2]}).to_excel(
        evolve_dir / "activity.xlsx"
    )
    pd.DataFrame(
        {
            "seq_id": ["WT", "A1G", "A1G_C2Y"],
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
        assert (evolve_dir / "predicted_activity.csv").exists()

        predicted_activity_df = pd.read_csv(
            evolve_dir / "predicted_activity.csv", keep_default_na=False
        )
        assert predicted_activity_df.shape[0] == 3
        predicted_activity_df.index = predicted_activity_df.seq_id

        # The tool should know that both mutants are more active than WT.
        # If this were a linear model, it would know that the second mutant should
        # be more active than the first, based on the constructed embedding. But the
        # random forest model can't extrapolate.
        assert predicted_activity_df.loc["A1G"].predicted_activity > 1.5
        assert predicted_activity_df.loc["A1G_C2Y"].predicted_activity > 1.5
