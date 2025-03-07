from re import DEBUG
import urllib

from authlib.integrations.flask_client import OAuth
from flask import redirect, current_app, request, jsonify
from flask_restx import Namespace
from flask import current_app, url_for
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


ns = Namespace("open_views")


@ns.route("/check_for_dead_jobs")
class CheckForDeadJobsResource(Resource):
    def get(self):
        """Identify and handle dead jobs.

        Search for invokations with 'pending', 'running', or 'queued' status -
        any that are not found in flask-rq2 will be marked as failed.
        """
        # Get all invokations with 'pending', 'running', or 'queued' status
        invokations = Invokation.query.filter(
            Invokation.status.in_(["pending", "running", "queued"])
        ).all()
        # Check if each invokation is in flask-rq2
        for invokation in invokations:
            if invokation.id not in rq.get_queue().get_job_ids():
                invokation.status = "failed"
                db.session.commit()
        return True
