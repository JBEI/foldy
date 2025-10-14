from app.api_fields import queue_job_fields
from app.authorization import verify_has_edit_access
from app.util import start_stage
from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource

ns = Namespace("other_views", decorators=[jwt_required(fresh=True)])


@ns.route("/test")
class TestResource(Resource):
    def get(self):
        return "Healthy"


# Queue job fields imported from api_fields.py


@ns.route("/queuejob")
class QueueJobResource(Resource):
    @ns.expect()
    @verify_has_edit_access
    def post(self):
        start_stage(
            request.get_json()["fold_id"],
            request.get_json()["stage"],
            request.get_json()["email_on_completion"],
        )
