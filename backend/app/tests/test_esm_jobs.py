import random
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.jobs.esm_jobs import get_esm_embeddings
from app.models import Embedding, Evolution, Fold, Invokation, User
from flask import Flask
from werkzeug.exceptions import BadRequest


@pytest.fixture
def test_invokation(app, test_fold):
    """Create a test invokation."""
    with app.app_context():
        invokation = Invokation(fold_id=test_fold.id, type="embed_test1", state="queued")
        db.session.add(invokation)
        db.session.commit()
        yield invokation
        # Clean up
        db.session.delete(invokation)
        db.session.commit()


@pytest.fixture
def test_fold_embedding(app, test_fold, test_invokation):
    """Create a test fold embedding."""
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esm2_t6_8M_UR50D",
            extra_seq_ids="WT",
            dms_starting_seq_ids="WT",
            invokation_id=test_invokation.id,
        )
        db.session.add(embedding)
        db.session.commit()
        yield embedding
        # Clean up
        db.session.delete(embedding)
        db.session.commit()


@pytest.fixture
def mock_foldy_storage(app, tmp_path, test_fold):
    """Setup a temporary storage directory with test files."""
    # Create the base storage structure
    storage_dir = tmp_path / "folds"
    storage_dir.mkdir()

    # Update app config to use this temporary directory
    app.config["FOLDY_LOCAL_STORAGE_DIR"] = str(storage_dir)

    # Create example folder structure
    fold_dir = storage_dir / ("%06d" % test_fold.id)

    yield storage_dir


# We don't run torch in tests, so this is as far as we can go for a test.
ERROR_MESSAGE_IF_EVERYTHING_GOES_WELL = "No module named 'torch'"


def test_run_embed_tolerates_no_extra_seq_ids(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation, mock_foldy_storage
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esm2_t6_8M_UR50D",
            extra_seq_ids="",
            dms_starting_seq_ids="WT",
            invokation_id=test_invokation.id,
        )
        get_esm_embeddings(embed_id=embedding.id)
        padded_fold_id = f"{test_fold.id:06d}"
        embed_dir = mock_foldy_storage / padded_fold_id / "embed"
        assert (
            embed_dir / f"{padded_fold_id}_embeddings_esm2_t6_8M_UR50D_name1.csv"
        ).exists(), f"File does not exist, found {list(embed_dir.glob('*'))}"


def test_run_embed_tolerates_no_dms_seq_ids(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation, mock_foldy_storage
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esm2_t6_8M_UR50D",
            extra_seq_ids="WT",
            dms_starting_seq_ids="",
            invokation_id=test_invokation.id,
        )
        get_esm_embeddings(embed_id=embedding.id)
        padded_fold_id = f"{test_fold.id:06d}"
        embed_dir = mock_foldy_storage / padded_fold_id / "embed"
        assert (
            embed_dir / f"{padded_fold_id}_embeddings_esm2_t6_8M_UR50D_name1.csv"
        ).exists(), f"File does not exist, found {list(embed_dir.glob('*'))}"


def test_run_embed_tolerates_no_inputs_at_all(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation, mock_foldy_storage
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esm2_t6_8M_UR50D",
            extra_seq_ids="",
            dms_starting_seq_ids="",
            invokation_id=test_invokation.id,
        )
        get_esm_embeddings(embed_id=embedding.id)
        padded_fold_id = f"{test_fold.id:06d}"
        embed_dir = mock_foldy_storage / padded_fold_id / "embed"
        assert (
            embed_dir / f"{padded_fold_id}_embeddings_esm2_t6_8M_UR50D_name1.csv"
        ).exists(), f"File does not exist, found {list(embed_dir.glob('*'))}"


def test_run_embed_invalid_amino_acid_with_custom_message(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation, mock_foldy_storage
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esm2_t6_8M_UR50D",
            extra_seq_ids="A16Z",
            dms_starting_seq_ids="WT",
            invokation_id=test_invokation.id,
        )
        with pytest.raises(
            AssertionError, match='third character must be a valid amino acid, got "Z"'
        ):
            get_esm_embeddings(embed_id=embedding.id)


def test_run_embed_invalid_amino_acid_third_case(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation, mock_foldy_storage
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esm2_t6_8M_UR50D",
            extra_seq_ids="D3G",
            dms_starting_seq_ids="WT",
            invokation_id=test_invokation.id,
        )
        get_esm_embeddings(embed_id=embedding.id)
        padded_fold_id = f"{test_fold.id:06d}"
        embed_dir = mock_foldy_storage / padded_fold_id / "embed"
        assert (
            embed_dir / f"{padded_fold_id}_embeddings_esm2_t6_8M_UR50D_name1.csv"
        ).exists(), f"File does not exist, found {list(embed_dir.glob('*'))}"


def test_run_embed_with_extra_layers(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation, mock_foldy_storage
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esm2_t6_8M_UR50D",
            extra_seq_ids=None,
            dms_starting_seq_ids="WT",
            extra_layers="1,2,3",
            invokation_id=test_invokation.id,
        )
        with pytest.raises(Exception, match="ESM1 and 2 do not support extra layers"):
            get_esm_embeddings(embed_id=embedding.id)
