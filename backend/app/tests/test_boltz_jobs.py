import random
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from flask import Flask
from werkzeug.exceptions import BadRequest

from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.jobs.boltz_jobs import run_boltz
from app.models import Evolution, Fold, Invokation, User


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


# We don't run torch in tests, so this is as far as we can go for a test.
ERROR_MESSAGE_IF_EVERYTHING_GOES_WELL = "No module named 'torch'"


def test_run_boltz_get_decently_far(app, client, tmp_path, test_fold, test_invokation):
    """Basic test for run_few_shot_prediction function."""
    with app.app_context():
        with pytest.raises(AssertionError, match=ERROR_MESSAGE_IF_EVERYTHING_GOES_WELL):
            run_boltz(test_fold.id, test_invokation.id)
