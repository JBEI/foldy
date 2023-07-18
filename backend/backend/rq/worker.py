"""Define a not-gentle worker, that propagates SIGTERM immediately."""

import signal

from rq.worker import Worker, signal_name


class ColdWorker(Worker):
    """
    Modified version of rq worker which kills the horse cold, not warm.
    """

    def request_stop(self, signum, frame):
        """Shut down immediately. Cold, not warm.

        Args:
            signum (Any): Signum
            frame (Any): Frame
        """
        self.log.debug("Got signal %s and killing immediately", signal_name(signum))

        # Don't set shutdown_requested_date, so our kill signal is not ignored.
        # self._shutdown_requested_date = utcnow()

        self.log.info("Worker %s [PID %d]: doing a cold shutdown!", self.name, self.pid)

        # The following logic is mimicking "request_force_stop" which normally gets called
        # after a CTRL-C double tap. But we are bypassing that function, because it has
        # some debouncing logic. So, skip the debouncing logic, and kill the horse directly.
        # Take down the horse with the worker
        if self.horse_pid:
            self.log.debug("Taking down horse %s with me", self.horse_pid)
            self.kill_horse()
            self.wait_for_horse()
        self.stop_scheduler()
        raise SystemExit()
