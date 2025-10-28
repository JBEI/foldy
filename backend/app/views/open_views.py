from flask_restx import Namespace, Resource

from app.extensions import db
from app.helpers.rq_helpers import get_queue
from app.models import Invokation

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
            if invokation.id not in get_queue().get_job_ids():
                invokation.status = "failed"
                db.session.commit()
        return True
