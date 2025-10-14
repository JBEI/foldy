import logging
from typing import Optional

from app.email_to import EmailServer
from app.models import Fold
from flask import current_app
from redis import Redis
from rq import Queue
from rq.job import Job


def get_redis_connection(redis_url: Optional[str] = None):
    """Get a Redis connection from app config"""
    if not redis_url:
        redis_url = current_app.config.get("RQ_REDIS_URL", "redis://redis:6379/0")
    return Redis.from_url(redis_url)  # type: ignore[reportArgumentType] # flask config value typing uncertain


def get_queue(queue_name: Optional[str] = None, redis_url: Optional[str] = None):
    """Get an RQ queue with a new Redis connection"""
    if queue_name:
        return Queue(queue_name, connection=get_redis_connection(redis_url))
    else:
        return Queue(connection=get_redis_connection(redis_url))


def add_meta_to_job(job: Job, fold: Fold, job_type: str, related_object_id: Optional[int] = None):
    """Add metadata to a job"""
    job.meta["fold_id"] = fold.id
    job.meta["fold_name"] = fold.name
    job.meta["job_type"] = job_type
    job.meta["related_object_id"] = related_object_id
    job.save_meta()


def send_status_update_email(job: Job, status: str, extra_body: Optional[str] = None) -> None:
    """Send a status update email"""
    if not current_app.config["EMAIL_USERNAME"] or not current_app.config["EMAIL_PASSWORD"]:
        raise KeyError("No email username / password provided: will not send email.")

    if any(
        key not in job.meta for key in ["fold_id", "fold_name", "job_type", "related_object_id"]
    ):
        logging.error(
            f"Job {job.id} has no fold_id, fold_name, job_type, or related_object_id in meta"
        )
        return

    fold_id = job.meta["fold_id"]
    fold_name = job.meta["fold_name"]
    job_type = job.meta["job_type"]
    # related_object_id = job.meta['related_object_id']

    fold: Fold | None = Fold.get_by_id(fold_id)
    if not fold:
        logging.error(f"No fold found for id {fold_id}")
        return
    recipient = fold.user.email

    server = EmailServer(
        "smtp.gmail.com",
        587,
        current_app.config["EMAIL_USERNAME"],
        current_app.config["EMAIL_PASSWORD"],
    )

    link = f'{current_app.config["FRONTEND_URL"]}/fold/{fold_id}'

    header = f'### {job_type.capitalize()} job for <a href="{link}">{fold_name}</a>.'
    body = f"Status: {status}"
    if extra_body:
        body += f"\n\n{extra_body}"

    # Light blue: 28A5F5
    server.quick_email(
        recipient,
        f"{job_type.capitalize()} job for {fold_name} status: {status}",
        [header, body],
        style="h3 {color: #333333}",
    )


def send_success_email(job, connection, result, *args, **kwargs) -> None:
    """Sends a success email."""
    send_status_update_email(job, "success", extra_body=f"Result: {result}")


def send_failure_email(job, connection, type, value, traceback, *args, **kwargs) -> None:
    """Sends a failure email."""
    send_status_update_email(job, "failure", extra_body=f"Error: {type} {value} {traceback}")
