from functools import wraps

from flask import current_app
from flask import abort
from flask_jwt_extended import verify_jwt_in_request
from flask_jwt_extended.utils import get_jwt_identity


def has_full_authorization(current_user):
    if not current_user:
        return False
    if current_app.config.get(
        "FOLDY_USER_EMAIL_DOMAIN", None
    ) and current_user.endswith("@" + current_app.config["FOLDY_USER_EMAIL_DOMAIN"]):
        return True
    # if (
    #     current_app.config.get("FOLDY_USERS", None)
    #     and current_user in current_app.config["FOLDY_USERS"]
    # ):
    #     return True
    return False


def verify_fully_authorized(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        if has_full_authorization(get_jwt_identity()):
            return fn(*args, **kwargs)
        else:
            abort(403, description="You do not have access to this resource.")

    return wrapper
