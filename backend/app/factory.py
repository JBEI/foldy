import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import rq_dashboard
import werkzeug
from app.authorization import user_jwt_grants_edit_access
from app.extensions import admin, compress, db, migrate  # , rq
from flask import Flask, jsonify, request
from flask.helpers import make_response
from flask_admin.contrib.sqla import ModelView
from flask_admin.form.fields import JSONField
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_jwt_extended.exceptions import JWTExtendedException
from flask_jwt_extended.utils import get_jwt
from flask_jwt_extended.view_decorators import jwt_required, verify_jwt_in_request
from flask_restx import Api, Resource
from jwt.exceptions import ExpiredSignatureError
from markupsafe import Markup
from prometheus_client import make_wsgi_app
from werkzeug.exceptions import BadRequest, HTTPException
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from wtforms import TextAreaField
from wtforms.widgets import TextArea

from app import models

app = Flask(__name__)


def createRestxApi() -> Api:
    """Creates and configures a Flask-RestX API instance with error handlers.

    Returns:
        Api: Configured Flask-RestX API instance
    """
    api = Api(doc="/doc/", validate=True)

    # Handle authentication errors.
    @api.errorhandler(JWTExtendedException)
    @api.errorhandler(ExpiredSignatureError)
    def handle_expired_signature(
        error: Union[JWTExtendedException, ExpiredSignatureError],
    ) -> Tuple[Dict[str, str], int]:
        return {"message": f"Login failed: {str(error)}"}, 401

    return api


def register_extensions(app: Flask) -> None:
    """Registers Flask extensions and configures admin views.

    Args:
        app: Flask application instance
    """

    class VerifiedModelView(ModelView):
        def is_accessible(self):  # pyright: ignore[reportIncompatibleMethodOverride]
            """Checks if the current user has access to this admin view.

            Returns:
                bool: True if user has edit access, False otherwise
            """
            # TODO(jbr): Base class expects Literal[True] but we need conditional access
            verify_jwt_in_request()
            result: bool = user_jwt_grants_edit_access(get_jwt()["user_claims"])
            return result

        # Add these defaults for all model views
        column_display_pk = True
        column_default_sort = "id"

    class UserModelView(VerifiedModelView):
        column_list = ["email", "name", "created_at", "access_type", "num_folds", "attributes"]
        column_editable_list = ["created_at", "name", "access_type"]
        column_sortable_list = ["email", "name", "created_at", "access_type"]
        column_searchable_list = ["email", "name", "access_type"]
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
            # "name",
            "user",
            "tagstring",
            "create_date",
            "af2_model_preset",
            "disable_relaxation",
            "sequence",
        ]
        column_searchable_list = ["id", "name", "sequence", "tagstring"]
        column_exclude_list = [
            "features_log",
            "models_log",
        ]
        column_default_sort = "id"  # Use consistent type (string, not tuple)
        form_overrides = {
            "yaml_config": TextAreaField,
        }
        form_widget_args = {
            "yaml_config": {
                "rows": 10,
                "style": "width: 100%;",
            }
        }

        # Ensure user field is properly configured for editing
        form_columns = [
            "name",
            "user",
            "tagstring",
            "create_date",
            "af2_model_preset",
            "disable_relaxation",
            "sequence",
            "yaml_config",
            "diffusion_samples",
            "public",
        ]

        # @staticmethod
        # def _sequence_formatter(view: Any, context: Any, model: models.Fold, name: str) -> Markup:
        #     """Format sequence field for display in admin view.

        #     Args:
        #         view: Admin view instance
        #         context: Rendering context
        #         model: Fold model instance
        #         name: Field name

        #     Returns:
        #         Markup: HTML-safe content for rendering
        #     """
        #     return Markup(f"<div style='overflow-x: auto; width: 100px'>{model.sequence}</div>")

        # # def _features_log_formatter(view, context, model, name):
        # #   return Markup(f"<div style='overflow-y: auto'>{model.features_log}</div>")
        # # def _models_log_formatter(view, context, model, name):
        # #   return Markup(f"<div style='overflow-y: auto'>{model.models_log}</div>")
        # column_formatters = {
        #     "sequence": _sequence_formatter,
        #     # 'features_log': _features_log_formatter,
        #     # 'models_log': _models_log_formatter,
        # }

    class InvokationModelView(VerifiedModelView):
        column_searchable_list = ["id", "fold_id", "type", "state", "command"]
        column_editable_list = ["state"]
        column_list = [
            "id",
            "fold_id",
            "type",
            "state",
            "starttime",
            "timedelta",
            "command",
        ]

    class DockModelView(VerifiedModelView):
        can_export = True
        page_size = 50
        column_display_pk = True
        column_default_sort = "id"
        can_set_page_size = True
        column_editable_list = [
            # "name",
            "ligand_name",
            "ligand_smiles",
            "tool",
            "receptor_fold_id",
            "receptor_fold",
            "bounding_box_residue",
            "bounding_box_radius_angstrom",
        ]

    class NaturalnessModelView(VerifiedModelView):
        column_list = [
            "id",
            "name",
            "fold",
            "fold.user",
            "logit_model",
            "use_structure",
            "date_created",
        ]
        column_sortable_list = ["id", "name", "logit_model", "use_structure", "date_created"]
        column_searchable_list = ["name", "logit_model"]
        column_editable_list = ["date_created"]

    class EmbeddingModelView(VerifiedModelView):
        column_list = [
            "id",
            "name",
            "fold",
            "fold.user",
            "embedding_model",
            "extra_seq_ids",
            "dms_starting_seq_ids",
            "extra_layers",
            "date_created",
        ]
        column_sortable_list = ["id", "name", "embedding_model", "date_created"]
        column_searchable_list = ["name", "embedding_model"]
        column_editable_list = ["date_created"]

        # # Add custom CSS to truncate/scroll long text in extra_seq_ids column
        # column_formatters = {
        #     "extra_seq_ids": lambda v, c, m, p: (
        #         Markup(
        #             f'<div style="max-width:200px; overflow-x:auto; white-space:nowrap;">{m.extra_seq_ids}</div>'
        #         )
        #         if m.extra_seq_ids
        #         else ""
        #     )
        # }

    class FewShotModelView(VerifiedModelView):
        column_list = [
            "id",
            "name",
            "fold",
            "fold.user",
            "mode",
            "embedding_files",
            "input_activity_fpath",
            "finetuning_model_checkpoint",
            "date_created",
        ]
        column_sortable_list = ["id", "name", "date_created"]
        column_searchable_list = ["name"]
        column_editable_list = ["date_created", "input_activity_fpath"]

    class CampaignModelView(VerifiedModelView):
        column_list = [
            "id",
            "name",
            "fold",
            "fold.user",
            "description",
            "created_at",
        ]
        column_sortable_list = ["id", "name", "created_at"]
        column_searchable_list = ["name", "description"]
        column_editable_list = ["name", "description"]

    class CampaignRoundModelView(VerifiedModelView):
        column_list = [
            "id",
            "campaign",
            "campaign.name",
            "round_number",
            "date_started",
        ]
        column_sortable_list = ["id", "round_number", "date_started"]
        column_searchable_list = ["round_number"]
        column_editable_list = ["round_number", "date_started"]

    admin.add_view(UserModelView(models.User, db.session))
    admin.add_view(FoldModelView(models.Fold, db.session))
    admin.add_view(InvokationModelView(models.Invokation, db.session))
    admin.add_view(DockModelView(models.Dock, db.session))
    admin.add_view(NaturalnessModelView(models.Naturalness, db.session))
    admin.add_view(EmbeddingModelView(models.Embedding, db.session))
    admin.add_view(FewShotModelView(models.FewShot, db.session))
    admin.add_view(CampaignModelView(models.Campaign, db.session))
    admin.add_view(CampaignRoundModelView(models.CampaignRound, db.session))
    admin.init_app(app)
    db.init_app(app)
    migrate.init_app(app, db)
    # rq.init_app(app)
    compress.init_app(app)

    # @app.before_request
    # def dbg():
    #     raw = request.get_data(cache=True, as_text=True)
    #     print("LEN", len(raw), "START", raw[:80], flush=True)


def create_app(config_object: str = "settings") -> Flask:
    """Creates and configures a Flask application instance.

    Args:
        config_object: Python module with configuration variables

    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__.split(".")[0])

    app.config.from_object(config_object)

    if app.config.get("ENV") == "development":
        print("ALLOWING CORS (security risk - do not enable when serving PHI!")
        CORS(app)
    else:
        print("NOT ALLOWING CORS: environment is %s" % app.config.get("ENV"))

    jwt = JWTManager(app)

    from app.views.admin_views import messages_ns
    from app.views.admin_views import ns as admin_views_ns
    from app.views.campaign_views import ns as campaign_views_ns
    from app.views.dna_build_views import ns as dna_build_views_ns
    from app.views.dock_views import ns as dock_views_ns
    from app.views.esm_views import ns as esm_views_ns
    from app.views.few_shot_views import ns as few_shot_views_ns
    from app.views.file_views import ns as file_views_ns
    from app.views.fold_views import ns as fold_views_ns
    from app.views.login_views import ns as login_views_ns
    from app.views.login_views import oauth
    from app.views.other_views import ns as other_views_ns

    from app import api_fields

    api = createRestxApi()
    register_extensions(app)

    @api.route("/healthz", strict_slashes=False)
    class HealthCheckResource(Resource):
        def get(self) -> bool:
            """Simple health check endpoint.

            Returns:
                bool: Always returns True if the service is running
            """
            return True

    @app.errorhandler(ValueError)
    @app.errorhandler(BadRequest)
    @api.errorhandler(BadRequest)
    @api.errorhandler(ValueError)
    def handle_unexpected_error(
        error: Union[ValueError, BadRequest],
    ) -> Tuple[Dict[str, str], int]:
        """Handle ValueError and BadRequest exceptions.

        Args:
            error: The exception that was raised

        Returns:
            Tuple containing error response dictionary and HTTP status code
        """
        # Found here: https://newbedev.com/python-flask-json-error-message-format-code-example
        if isinstance(error, BadRequest):
            message = str(error.description)
        else:
            message = str(error)

        return {"message": message}, 400

    api.add_namespace(api_fields.type_ns, "/api")
    api.add_namespace(login_views_ns, "/api")
    api.add_namespace(admin_views_ns, "/api")
    api.add_namespace(messages_ns, "/api")
    api.add_namespace(campaign_views_ns, "/api")
    api.add_namespace(dna_build_views_ns, "/api")
    api.add_namespace(dock_views_ns, "/api")
    api.add_namespace(file_views_ns, "/api")
    api.add_namespace(fold_views_ns, "/api")
    api.add_namespace(esm_views_ns, "/api")
    api.add_namespace(few_shot_views_ns, "/api")
    api.add_namespace(other_views_ns, "/api")

    api.init_app(app)

    oauth.init_app(app)

    # https://github.com/Parallels/rq-dashboard/issues/464
    rq_dashboard.web.setup_rq_connection(app)

    # @rq_dashboard.blueprint.before_request
    @jwt_required(fresh=True)
    def before_request() -> None:
        """Protect RQ dashboard pages with JWT authentication.

        Returns:
            None: Function doesn't return anything, but has side effect of verifying JWT
        """
        pass

    app.register_blueprint(
        rq_dashboard.blueprint,
        url_prefix="/rq",
    )

    # Add prometheus wsgi middleware to route /metrics requests
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {"/metrics": make_wsgi_app()})

    from app import metrics

    return app


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
