"""Application configuration.

Most configuration is set via environment variables.

For local development, use a .env file to set
environment variables.

Copied from https://github.com/cookiecutter-flask/cookiecutter-flask
"""
import datetime

from environs import Env

env = Env()
env.read_env()


# TODO(jacob): Consider using refresh tokens, instead of access tokens
# with a long timeout (#41).
JWT_ACCESS_TOKEN_EXPIRES = datetime.timedelta(hours=24 * 7)
JWT_TOKEN_LOCATION = ["headers", "cookies"]
JWT_COOKIE_CSRF_PROTECT = False
RESTPLUS_JSON = {"indent": 0}
COMPRESS_MIMETYPES = [
    "text/html",
    "text/css",
    "text/xml",
    "application/json",
    "application/javascript" "application/octet-stream",
]
COMPRESS_REGISTER = True

SECRET_KEY = bytes(env.str("SECRET_KEY"), "utf8")
GOOGLE_CLIENT_ID = env.str("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = env.str("GOOGLE_CLIENT_SECRET")
OAUTH_REDIRECT_URI = env.str("OAUTH_REDIRECT_URI")
FRONTEND_URL = env.str("FRONTEND_URL")
FOLDY_USER_EMAIL_DOMAIN = env.str("FOLDY_USER_EMAIL_DOMAIN")
FOLDY_USERS = env.list("FOLDY_USERS", [])
FOLDY_GCLOUD_PROJECT = env.str("FOLDY_GCLOUD_PROJECT", "not_provided")
FOLDY_GCLOUD_BUCKET = env.str("FOLDY_GCLOUD_BUCKET", "not_provided")
SQLALCHEMY_DATABASE_URI = env.str("DATABASE_URL")
RQ_REDIS_URL = env.str("RQ_REDIS_URL")
RQ_DASHBOARD_REDIS_URL = env.str("RQ_REDIS_URL")
RQ_DASHBOARD_BIND = env.str("BACKEND_URL")
SQLALCHEMY_TRACK_MODIFICATIONS = False

STORE_IN_LOCAL_DIRECTORY = env.str("STORE_IN_LOCAL_DIRECTORY", "not_provided")
