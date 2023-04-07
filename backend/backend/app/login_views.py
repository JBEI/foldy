from re import DEBUG
import urllib

from authlib.integrations.flask_client import OAuth
from flask import redirect, current_app, request, jsonify
from flask_restplus import Namespace
from flask import current_app, url_for
from flask_restplus import Resource
from flask_jwt_extended import create_access_token
from flask_jwt_extended import set_access_cookies, unset_jwt_cookies

from app.models import User
from app.extensions import db


ns = Namespace("login_views")

oauth = OAuth()
CONF_URL = "https://accounts.google.com/.well-known/openid-configuration"
oauth.register(
    name="google",
    server_metadata_url=CONF_URL,
    client_kwargs={"scope": "openid email profile"},
)


@ns.route("/login")
class LoginResource(Resource):
    def get(self):
        if "frontend_url" in request.args:
            state = request.args["frontend_url"]
        else:
            state = current_app.config["FRONTEND_URL"]  # + '/channel/4'

        redirect_uri = current_app.config[
            "OAUTH_REDIRECT_URI"
        ]  # url_for('login_views_authorize_resource', _external=True)

        if current_app.config["DISABLE_OAUTH_AUTHENTICATION"]:
            assert (
                current_app.config["ENV"] == "development"
            ), "It would be a grave mistake to disable OAuth authentication in production."
            return redirect(url_for('login_views_authorize_resource', state=state, _external=True))
        else:
            assert current_app.config['FOLDY_USER_EMAIL_DOMAIN']
            assert current_app.config["GOOGLE_CLIENT_ID"]
            assert current_app.config["GOOGLE_CLIENT_SECRET"]
            return oauth.google.authorize_redirect(redirect_uri, state=state)


def make_error_redirect(message):
    frontend_parsed = urllib.parse.urlparse(current_app.config["FRONTEND_URL"])
    frontend_queries = dict(urllib.parse.parse_qsl(frontend_parsed.query))
    frontend_queries["error_message"] = message
    frontend_parsed = frontend_parsed._replace(
        query=urllib.parse.urlencode(frontend_queries)
    )
    rd_url = urllib.parse.urlunparse(frontend_parsed)
    return redirect(location=rd_url)


@ns.route("/authorize")
class AuthorizeResource(Resource):
    def get(self):
        frontend_url = current_app.config["FRONTEND_URL"]
        if "state" in request.args:
            frontend_url = request.args["state"]

        # If in development, we don't need to do any authentication.
        if current_app.config["DISABLE_OAUTH_AUTHENTICATION"]:
            assert (
                current_app.config["ENV"] == "development"
            ), "It would be a grave mistake to disable OAuth authentication in production."
            name = "Testy Mcgoo"
            email = "tester@test.edu"
        else:
            _ = oauth.google.authorize_access_token()
            name = oauth.google.userinfo()["name"]
            email = oauth.google.userinfo()["email"]

            # Make sure they're authorized.
            if (
                not email.endswith("@" + current_app.config["FOLDY_USER_EMAIL_DOMAIN"])
                and email not in current_app.config["FOLDY_USERS"]
            ):
                return make_error_redirect(
                    f'Authorization unsuccessful. Only users with @{current_app.config["FOLDY_USER_EMAIL_DOMAIN"]} addresses can access this resource.'
                )

        # Check if it's a new user. If it is, add them to the db.
        user_was_registered = (
            db.session.query(User.email).filter_by(email=email).first() is not None
        )
        if not user_was_registered:
            creation_successful = User.create(email=email)
            if not creation_successful:
                return make_error_redirect(
                    "Unfortunately, new user creation failed! Please contact the admins for help."
                )

        access_token = create_access_token(
            identity=email, fresh=True, user_claims={"name": name, "email": email}
        )

        frontend_parsed = urllib.parse.urlparse(frontend_url)
        frontend_queries = dict(urllib.parse.parse_qsl(frontend_parsed.query))
        frontend_queries["access_token"] = access_token
        if not user_was_registered:
            frontend_queries["new_user"] = True
        frontend_parsed = frontend_parsed._replace(
            query=urllib.parse.urlencode(frontend_queries)
        )
        rd_url = urllib.parse.urlunparse(frontend_parsed)

        response = redirect(location=rd_url)
        set_access_cookies(response, access_token)
        return response


@ns.route("/logout")
class LogoutResource(Resource):
    def get(self):
        response = redirect(location=current_app.config["FRONTEND_URL"])
        unset_jwt_cookies(response)
        return response


# @ns.route('/setupdbsBECAREFUL')
# class SetupDbsResource(Resource):
#   def get(self):
#     db.create_all()
