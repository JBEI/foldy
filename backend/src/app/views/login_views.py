import urllib
import logging
from typing import Dict, Any, List, Tuple, Union, Optional, cast

from authlib.integrations.flask_client import OAuth
from flask import redirect, current_app, request, jsonify, url_for, Response
from flask_restx import Namespace
from flask_restx import fields
from flask_restx import Resource
from flask_jwt_extended import create_access_token
from flask_jwt_extended import set_access_cookies, unset_jwt_cookies

from app.authorization import (
    email_should_get_edit_permission_by_default,
    email_should_get_upgraded_to_admin,
)
from app.models import User
from app.extensions import db


ns = Namespace("login_views")

oauth = OAuth()
CONF_URL: str = "https://accounts.google.com/.well-known/openid-configuration"
oauth.register(
    name="google",
    server_metadata_url=CONF_URL,
    client_kwargs={"scope": "openid email profile"},
)


@ns.route("/login")
class LoginResource(Resource):
    def get(self) -> Response:
        """Initiate the OAuth login flow.
        
        Returns:
            Flask redirect response to either OAuth provider or directly to authorize endpoint
        """
        if "frontend_url" in request.args:
            state: str = request.args["frontend_url"]
        else:
            state = current_app.config["FRONTEND_URL"]  # + '/channel/4'

        redirect_uri: str = current_app.config[
            "OAUTH_REDIRECT_URI"
        ]  # url_for('login_views_authorize_resource', _external=True)

        if current_app.config["DISABLE_OAUTH_AUTHENTICATION"]:
            assert (
                current_app.config["ENV"] == "development"
            ), "It would be a grave mistake to disable OAuth authentication in production."
            logging.info("OAuth authentication disabled, redirecting directly to authorize endpoint")
            return redirect(
                url_for("login_views_authorize_resource", state=state, _external=True)
            )
        else:
            assert current_app.config["FOLDY_USER_EMAIL_DOMAIN"]
            assert current_app.config["GOOGLE_CLIENT_ID"]
            assert current_app.config["GOOGLE_CLIENT_SECRET"]
            logging.info("Redirecting to Google OAuth for authentication")
            return oauth.google.authorize_redirect(redirect_uri, state=state)


def make_error_redirect(message: str) -> Response:
    """Create a redirect response to frontend with error message in query params.
    
    Args:
        message: Error message to add to the redirect URL
        
    Returns:
        Flask redirect response with error message in query parameters
    """
    frontend_parsed = urllib.parse.urlparse(current_app.config["FRONTEND_URL"])
    frontend_queries: Dict[str, str] = dict(urllib.parse.parse_qsl(frontend_parsed.query))
    frontend_queries["error_message"] = message
    frontend_parsed = frontend_parsed._replace(
        query=urllib.parse.urlencode(frontend_queries)
    )
    rd_url = urllib.parse.urlunparse(frontend_parsed)
    logging.warning(f"Redirecting with error: {message}")
    return redirect(location=rd_url)


@ns.route("/authorize")
class AuthorizeResource(Resource):
    def get(self) -> Response:
        """Handle OAuth callback and create user session.
        
        Returns:
            Flask redirect response to frontend with access token
            
        Raises:
            AssertionError: If OAuth is disabled outside of development environment
        """
        frontend_url: str = current_app.config["FRONTEND_URL"]
        if "state" in request.args:
            frontend_url = request.args["state"]

        # If in development, we don't need to do any authentication.
        if current_app.config["DISABLE_OAUTH_AUTHENTICATION"]:
            assert (
                current_app.config["ENV"] == "development"
            ), "It would be a grave mistake to disable OAuth authentication in production."
            name: str = "Testy Mcgoo"
            email: str = "tester@test.edu"
            logging.info(f"OAuth disabled, using test user: {email}")
        else:
            _ = oauth.google.authorize_access_token()
            userinfo = oauth.google.userinfo()
            name = userinfo["name"]
            email = userinfo["email"]
            logging.info(f"Authenticated user via OAuth: {email}")

        # Check if it's a new user. If it is, add them to the db.
        matching_users = list(db.session.query(User).filter_by(email=email))
        user: Optional[User] = matching_users[0] if matching_users else None
        user_was_registered: bool = user is not None
        
        if not user_was_registered:
            new_user_type: str = (
                "editor"
                if email_should_get_edit_permission_by_default(email)
                else "viewer"
            )
            logging.info(f"Creating new user {email} with access type: {new_user_type}")
            user = User.create(email=email, access_type=new_user_type)
            if not user:
                return make_error_redirect(
                    "Unfortunately, new user creation failed! Please contact the admins for help."
                )

        # Users pre 8/4/23 don't have access_type set. So we set it here.
        if not user.access_type:
            user_type: str = (
                "editor"
                if email_should_get_edit_permission_by_default(email)
                else "viewer"
            )
            logging.info(f"Updating user {email} with missing access type: {user_type}")
            user = user.update(access_type=user_type)

        # If user is listed in FOLDY_ADMIN_UPGRADE_LIST, then they'll be upgraded
        # to admin.
        if email_should_get_upgraded_to_admin(email) and user.access_type != "admin":
            logging.info(f"Upgrading user {email} to admin")
            user = user.update(access_type="admin")

        access_token = create_access_token(
            identity=email,
            fresh=True,
            additional_claims={
                "user_claims": {
                    "name": name,
                    "email": email,
                    "type": user.access_type,
                    "attributes": user.attributes or {},
                }
            },
        )

        frontend_parsed = urllib.parse.urlparse(frontend_url)
        frontend_queries: Dict[str, str] = dict(urllib.parse.parse_qsl(frontend_parsed.query))
        frontend_queries["access_token"] = access_token
        if not user_was_registered:
            frontend_queries["new_user"] = "true"
        frontend_parsed = frontend_parsed._replace(
            query=urllib.parse.urlencode(frontend_queries)
        )
        rd_url = urllib.parse.urlunparse(frontend_parsed)

        response = redirect(location=rd_url)
        set_access_cookies(response, access_token)
        logging.info(f"Setting access cookies for user {email}")
        return response


@ns.route("/logout")
class LogoutResource(Resource):
    def get(self) -> Response:
        """Log user out by unsetting JWT cookies.
        
        Returns:
            Flask redirect response to frontend with cleared cookies
        """
        logging.info("User logout requested")
        response = redirect(location=current_app.config["FRONTEND_URL"])
        unset_jwt_cookies(response)
        logging.info("JWT cookies cleared")
        return response
