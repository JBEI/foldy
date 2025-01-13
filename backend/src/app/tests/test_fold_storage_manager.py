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


def test_write_fastas(app, client, tmp_path, test_fold):
    fsm = FoldStorageManager()
    fsm.setup()
    fsm.write_fastas(test_fold.id, "MAG")
