import json
import os

from flask import Flask, jsonify
from flask.helpers import make_response
from flask_admin.contrib.sqla import ModelView
from flask_jwt_extended.view_decorators import (
    fresh_jwt_required,
    verify_fresh_jwt_in_request,
)
from flask_restplus import Api
from flask_restplus import Resource
from flask_cors import CORS
from flask_jwt_extended.exceptions import JWTExtendedException
from jwt.exceptions import ExpiredSignatureError
from flask_jwt_extended import JWTManager
from markupsafe import Markup
import werkzeug
from werkzeug.exceptions import BadRequest, HTTPException
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app

from app import models
from app.authorization import verify_fully_authorized
from app.extensions import admin, db, migrate, rq, compress
import rq_dashboard


app = Flask(__name__)


def createRestplusApi():
    api = Api(doc="/doc/", validate=True)

    # Handle authentication errors.
    @api.errorhandler(JWTExtendedException)
    @api.errorhandler(ExpiredSignatureError)
    def handle_expired_signature(error):
        return {"message": "Login failed: %s" % str(error)}, 401

    return api


def register_extensions(app):
    """Registers Flask extensions."""

    class VerifiedModelView(ModelView):
        def is_accessible(self):
            verify_fresh_jwt_in_request()
            verify_fully_authorized()
            return True

    class UserModelView(VerifiedModelView):
        column_list = ["email", "created_at", "num_folds"]
        column_sortable_list = ["email", "created_at"]
        can_export = True
        page_size = 50

    class FoldModelView(VerifiedModelView):
        can_export = True
        page_size = 50
        column_display_pk = True
        column_default_sort = "id"
        can_set_page_size = True
        can_export = True
        column_editable_list = [
            "name",
            "user",
            "tagstring",
            "create_date",
            "af2_model_preset",
            "disable_relaxation",
        ]
        column_searchable_list = ["id", "name", "sequence", "tagstring"]
        column_exclude_list = [
            "features_log",
            "models_log",
        ]
        column_default_sort = ("id", True)

        def _sequence_formatter(view, context, model, name):
            return Markup(
                f"<div style='overflow-x: auto; width: 100px'>{model.sequence}</div>"
            )

        # def _features_log_formatter(view, context, model, name):
        #   return Markup(f"<div style='overflow-y: auto'>{model.features_log}</div>")
        # def _models_log_formatter(view, context, model, name):
        #   return Markup(f"<div style='overflow-y: auto'>{model.models_log}</div>")
        column_formatters = {
            "sequence": _sequence_formatter,
            # 'features_log': _features_log_formatter,
            # 'models_log': _models_log_formatter,
        }

    admin.add_view(UserModelView(models.User, db.session))
    admin.add_view(FoldModelView(models.Fold, db.session))
    admin.add_view(VerifiedModelView(models.Invokation, db.session))
    admin.add_view(VerifiedModelView(models.Dock, db.session))

    admin.init_app(app)
    db.init_app(app)
    migrate.init_app(app, db)
    rq.init_app(app)
    compress.init_app(app)


def create_app(config_object="settings"):
    """Creates the app.

    args:
    test_config: optional dict of overrides to the defaults of flask config.
    """
    app = Flask(__name__.split(".")[0])

    app.config.from_object(config_object)

    if app.config.get("ENV") == "development":
        print("ALLOWING CORS (security risk - do not enable when serving PHI!")
        CORS(app)
    else:
        print("NOT ALLOWING CORS: environment is %s" % app.config.get("ENV"))

    jwt = JWTManager(app)

    from app.views import ns as views_ns
    from app.login_views import ns as login_views_ns, oauth
    from app.admin_views import ns as admin_views_ns

    api = createRestplusApi()
    register_extensions(app)

    @api.route("/healthz", strict_slashes=False)
    class HealthCheckResource(Resource):
        def get(self):
            return True

    api.add_namespace(views_ns, "/api")
    api.add_namespace(login_views_ns, "/api")
    api.add_namespace(admin_views_ns, "/api")

    api.init_app(app)

    oauth.init_app(app)

    @rq_dashboard.blueprint.before_request
    @fresh_jwt_required
    def before_request():
        """Protect RQ pages."""
        pass

    app.register_blueprint(
        rq_dashboard.blueprint,
        url_prefix="/rq",
    )

    @app.errorhandler(werkzeug.exceptions.BadRequest)
    def handle_unexpected_error(error):
        # Found here: https://newbedev.com/python-flask-json-error-message-format-code-example
        response = {"message": str(error.description)}
        return jsonify(response), 400

    # Add prometheus wsgi middleware to route /metrics requests
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": make_wsgi_app()})

    from app import metrics

    return app


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
