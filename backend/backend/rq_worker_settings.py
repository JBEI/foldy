"""A version of settings.py for the RQ workers."""
from environs import Env

env = Env()
env.read_env()

# Either "Local" or "Cloud". Determines which of below flags are necessary.
FOLDY_STORAGE_TYPE = env.str("FOLDY_STORAGE_TYPE")

FOLDY_LOCAL_STORAGE_DIR = env.str("FOLDY_LOCAL_STORAGE_DIR", "")

FOLDY_GCLOUD_PROJECT = env.str("FOLDY_GCLOUD_PROJECT", "")
FOLDY_GCLOUD_BUCKET = env.str("FOLDY_GCLOUD_BUCKET", "")

SQLALCHEMY_DATABASE_URI = env.str("DATABASE_URL")
RQ_REDIS_URL = env.str("RQ_REDIS_URL")
FRONTEND_URL = env.str("FRONTEND_URL")
EMAIL_USERNAME = env.str("EMAIL_USERNAME", "")
EMAIL_PASSWORD = env.str("EMAIL_PASSWORD", "")

RUN_AF2_PATH = env.str("RUN_AF2_PATH", "not_provided")
DECOMPRESS_PKLS_PATH = env.str("DECOMPRESS_PKLS_PATH", "not_provided")
RUN_ANNOTATE_PATH = env.str("RUN_ANNOTATE_PATH", "not_provided")
RUN_DOCK = env.str("RUN_DOCK", "not_provided")

RQ_WORKER_CLASS = "rq_util.worker.ColdWorker"
