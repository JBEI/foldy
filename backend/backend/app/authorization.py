"""Utilities around authorization."""

from functools import wraps

from flask import current_app
from flask import abort
from flask_jwt_extended import verify_jwt_in_request
from flask_jwt_extended.utils import get_jwt_identity, get_current_user, get_jwt_claims


def email_should_get_edit_permission_by_default(current_user):
    if not current_user:
        return False
    if current_app.config.get(
        "FOLDY_USER_EMAIL_DOMAIN", None
    ) and current_user.endswith("@" + current_app.config["FOLDY_USER_EMAIL_DOMAIN"]):
        return True
    if (
        current_app.config.get("FOLDY_USERS", None)
        and current_user in current_app.config["FOLDY_USERS"]
    ):
        return True
    return False


def user_jwt_grants_edit_access(jwt_claims):
    return jwt_claims["type"] == "editor"


def verify_has_edit_access(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()

        if user_jwt_grants_edit_access(get_jwt_claims()):
            return fn(*args, **kwargs)
        else:
            abort(403, description="You do not have access to this resource.")

    return wrapper
