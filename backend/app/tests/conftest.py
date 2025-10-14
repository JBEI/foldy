import random
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.models import Evolution, Fold, Invokation, User
from app.util import FoldStorageManager
from flask import Flask
from werkzeug.exceptions import BadRequest


@pytest.fixture(autouse=True)
def set_random_seed():
    """Fixture to set random seeds before each test"""
    random.seed(42)
    np.random.seed(42)
    yield


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
        FOLD_YAML_CONFIG = """version: 1
sequences:
  - protein:
      id: A
      sequence: ACD
"""
        # Get a fresh user object from the database
        user = db.session.get(User, test_user.id)
        fold = Fold()
        fold.name = "test_fold"
        fold.user_id = test_user.id
        fold.yaml_config = FOLD_YAML_CONFIG
        fold.tagstring = "test"
        fold.af2_model_preset = "monomer_ptm"
        fold.sequence = "ACD"
        db.session.add(fold)
        db.session.commit()
        yield fold
        # Clean up
        db.session.delete(fold)
        db.session.commit()
