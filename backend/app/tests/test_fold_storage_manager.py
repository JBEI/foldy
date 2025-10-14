import random
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from app.extensions import db
from app.helpers.fold_storage_manager import FoldStorageManager
from app.jobs.evolve_jobs import run_few_shot_prediction
from app.models import Evolution, Fold, Invokation, User
from flask import Flask
from werkzeug.exceptions import BadRequest


def test_write_fastas(app, client, tmp_path, test_fold):
    fsm = FoldStorageManager()
    fsm.setup()
    fsm.write_fastas(test_fold.id, test_fold.yaml_config)
