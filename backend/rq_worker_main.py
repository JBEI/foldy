#!/usr/bin/env python
import argparse
import logging
import os
import signal
import sys

import redis
from app.factory import create_app
from app.helpers.rq_helpers import get_redis_connection
from rq import Worker
from rq.utils import now
from rq.worker import Worker, signal_name


class GracefulWorker(Worker):
    """
    Fast but *graceful* shutdown:
      • on SIGTERM/SIGINT → mark stop-requested
      • forward SIGTERM to the horse
      • let RQ’s own loop clean up & mark the job failed
    """

    def request_stop(self, signum, frame):
        self.log.warning("%s received – requesting warm shutdown", signal_name(signum))

        # Record when we asked so RQ's debounce still works
        self._shutdown_requested_date = now()

        # Tell RQ main loop to break after current job
        self._stop_requested = True

        # Forward TERM to the child so its finally-blocks can run
        if self.horse_pid:
            os.killpg(os.getpgid(self.horse_pid), signal.SIGTERM)

        # Do *not* wait() or raise SystemExit here.
        # monitor_work_horse() will reap exactly once and
        # handle_job_failure() will fire if the job dies.


# Parse command line arguments
parser = argparse.ArgumentParser(description="Run RQ worker")
parser.add_argument("queues", nargs="+", help="Queues to listen on")
parser.add_argument("--burst", action="store_true", help="Run in burst mode")
parser.add_argument(
    "--max-jobs", type=int, help="Maximum number of jobs to process before quitting"
)
args = parser.parse_args()


def handle_worker_death(job, *exc_info):
    # Custom handler for worker deaths (e.g., OOM)
    print(f"Worker for job {job.id} was killed: {exc_info}")
    # Add logging or notification logic here


def main():
    # Initialize Flask app
    app = create_app("rq_worker_settings")

    with app.app_context():
        # Get Redis connection from Flask app config
        redis_conn = get_redis_connection()

        # Create and run worker with the specific queues
        worker = GracefulWorker(
            args.queues,
            connection=redis_conn,
            exception_handlers=[handle_worker_death],
            work_horse_killed_handler=handle_worker_death,
            # You can add other worker config here:
            # job_timeout=app.config.get('RQ_DEFAULT_TIMEOUT')
        )

        print(f"Worker listening on queues: {', '.join(args.queues)}")
        worker.work(burst=args.burst, max_jobs=args.max_jobs)


if __name__ == "__main__":
    main()
