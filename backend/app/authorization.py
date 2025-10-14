"""Utilities around authorization."""

from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

from flask import abort, current_app
from flask_jwt_extended import verify_jwt_in_request
from flask_jwt_extended.utils import get_current_user, get_jwt, get_jwt_identity

F = TypeVar("F", bound=Callable[..., Any])


def email_should_get_edit_permission_by_default(current_user: Optional[str]) -> bool:
    """Determine if a user email should get edit permission by default.

    Args:
        current_user: The email of the current user.

    Returns:
        True if the user should have edit permission, False otherwise.
    """
    if not current_user:
        return False
    if current_app.config.get("FOLDY_USER_EMAIL_DOMAIN", None) and current_user.endswith(
        "@" + current_app.config["FOLDY_USER_EMAIL_DOMAIN"]
    ):
        return True
    if current_user.lower() in [
        v.lower() for v in current_app.config.get("FOLDY_ADMIN_UPGRADE_LIST", [])
    ]:
        return True
    return False


def email_should_get_upgraded_to_admin(current_user: str) -> bool:
    """Determine if a user email should be upgraded to admin status.

    Args:
        current_user: The email of the current user.

    Returns:
        True if the user should be an admin, False otherwise.
    """
    return current_user.lower() in [
        v.lower() for v in current_app.config.get("FOLDY_ADMIN_UPGRADE_LIST", [])
    ]


def user_jwt_grants_edit_access(jwt_claims: Dict[str, Any]) -> bool:
    """Check if the JWT claims grant edit access.

    Args:
        jwt_claims: The JWT claims dictionary.

    Returns:
        True if the claims grant edit access, False otherwise.
    """
    result: bool = jwt_claims["type"] == "editor" or jwt_claims["type"] == "admin"
    return result


def verify_has_edit_access(fn: F) -> F:
    """Decorator to verify that the current user has edit access.

    Args:
        fn: The function to wrap.

    Returns:
        The wrapped function that verifies edit access before execution.

    Raises:
        403: If the user does not have edit access.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        verify_jwt_in_request()

        if user_jwt_grants_edit_access(get_jwt()["user_claims"]):
            return fn(*args, **kwargs)
        else:
            print(f"Rejecting user for not having access {get_jwt()}", flush=True)
            abort(403, description="You do not have access to this resource.")

    return cast(F, wrapper)
