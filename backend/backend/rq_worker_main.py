import os

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app

from app.app import create_app
from app.extensions import db


app = create_app("rq_worker_settings")
