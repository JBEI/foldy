import pytest
from unittest.mock import MagicMock, patch
from flask import Flask
from werkzeug.exceptions import BadRequest
import pandas as pd
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.models import User, Fold, Evolution, Invokation, Embedding
from app.jobs.esm_jobs import get_esm_embeddings
import random
import numpy as np


@pytest.fixture
def test_invokation(app, test_fold):
    """Create a test invokation."""
    with app.app_context():
        invokation = Invokation(
            fold_id=test_fold.id, type="embed_test1", state="queued"
        )
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
            embedding_model="esmc_600m",
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


# We don't run torch in tests, so this is as far as we can go for a test.
ERROR_MESSAGE_IF_EVERYTHING_GOES_WELL = "No module named 'torch'"


def test_run_embed_tolerates_no_extra_seq_ids(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esmc_600m",
            extra_seq_ids="",
            dms_starting_seq_ids="WT",
            invokation_id=test_invokation.id,
        )
        # Make sure it "succeeds" (gets past validation phase, it should tolerate an empty string).
        with pytest.raises(AssertionError, match=ERROR_MESSAGE_IF_EVERYTHING_GOES_WELL):
            get_esm_embeddings(embed_id=embedding.id)


def test_run_embed_tolerates_no_dms_seq_ids(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esmc_600m",
            extra_seq_ids="WT",
            dms_starting_seq_ids="",
            invokation_id=test_invokation.id,
        )
        # Make sure it "succeeds" (gets past validation phase, it should tolerate an empty string).
        # with pytest.raises(AssertionError, match="Invalid seq_id in extra seq_ids: ''"):
        with pytest.raises(AssertionError, match=ERROR_MESSAGE_IF_EVERYTHING_GOES_WELL):
            get_esm_embeddings(embed_id=embedding.id)


def test_run_embed_tolerates_no_inputs_at_all(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esmc_600m",
            extra_seq_ids="",
            dms_starting_seq_ids="",
            invokation_id=test_invokation.id,
        )
        # Make sure it "succeeds" (gets past validation phase, it should tolerate an empty string).
        # with pytest.raises(AssertionError, match="Invalid seq_id in extra seq_ids: ''"):
        with pytest.raises(AssertionError, match=ERROR_MESSAGE_IF_EVERYTHING_GOES_WELL):
            get_esm_embeddings(embed_id=embedding.id)


def test_run_embed_invalid_amino_acid(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esmc_600m",
            extra_seq_ids="A16Z",
            dms_starting_seq_ids="WT",
            invokation_id=test_invokation.id,
        )
        # Make sure it "succeeds" (gets past validation phase, it should tolerate an empty string).
        # with pytest.raises(AssertionError, match="Invalid seq_id in extra seq_ids: ''"):
        with pytest.raises(AssertionError, match=ERROR_MESSAGE_IF_EVERYTHING_GOES_WELL):
            get_esm_embeddings(embed_id=embedding.id)


def test_run_embed_invalid_amino_acid(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esmc_600m",
            extra_seq_ids="A16Z",
            dms_starting_seq_ids="WT",
            invokation_id=test_invokation.id,
        )
        with pytest.raises(
            AssertionError, match='third character must be a valid amino acid, got "Z"'
        ):
            get_esm_embeddings(embed_id=embedding.id)


def test_run_embed_invalid_amino_acid(
    app, client, tmp_path, mock_storage_manager, test_fold, test_invokation
):
    with app.app_context():
        embedding = Embedding.create(
            name="name1",
            fold_id=test_fold.id,
            embedding_model="esmc_600m",
            extra_seq_ids="D3G",
            dms_starting_seq_ids="WT",
            invokation_id=test_invokation.id,
        )
        with pytest.raises(AssertionError, match=ERROR_MESSAGE_IF_EVERYTHING_GOES_WELL):
            get_esm_embeddings(embed_id=embedding.id)


def test_run_embed_fails_no_torch(
    app, client, tmp_path, mock_storage_manager, test_fold_embedding
):
    """We don't run torch in tests, so this is as far as we can go for a test."""
    with app.app_context():
        with pytest.raises(AssertionError, match=ERROR_MESSAGE_IF_EVERYTHING_GOES_WELL):
            get_esm_embeddings(embed_id=test_fold_embedding.id)
