# """Define a not-gentle worker, that propagates SIGTERM immediately."""

# import signal

# from rq.worker import Worker, signal_name


# class ColdWorker(Worker):
#     """
#     Modified version of rq worker which kills the horse cold, not warm.
#     """

#     def request_stop(self, signum, frame):
#         """Shut down immediately. Cold, not warm.

#         Args:
#             signum (Any): Signum
#             frame (Any): Frame
#         """
#         self.log.warning("SIG%s received – initiating shutdown", signal_name(signum))
#         super().request_force_stop(signum, frame)   # preserves debounce + metrics

#         # Don't set shutdown_requested_date, so our kill signal is not ignored.
#         # self._shutdown_requested_date = utcnow()

#         self.log.info("Worker %s [PID %d]: doing a cold shutdown!", self.name, self.pid)

#         # The following logic is mimicking "request_force_stop" which normally gets called
#         # after a CTRL-C double tap. But we are bypassing that function, because it has
#         # some debouncing logic. So, skip the debouncing logic, and kill the horse directly.
#         # Take down the horse with the worker
#         GRACE = 30      # seconds you’re willing to wait

#         if self.horse_pid:
#             self.log.warning("Asking horse %s to exit cleanly (SIGTERM)", self.horse_pid)
#             self.kill_horse(signal.SIGTERM)

#             if not self.wait_for_horse(timeout=GRACE):
#                 self.log.error("Horse didn’t exit in %s s – sending SIGKILL", GRACE)
#                 self.kill_horse(signal.SIGKILL)
#         else:
#             self.log.warning(f'No horse_pid found...')
#         self.stop_scheduler()

#         code_map = {signal.SIGTERM: 143,  # 128+15
#                     signal.SIGINT: 130,   # 128+2
#                     signal.SIGUSR1: 138}  # pick your own
#         exit_code = code_map.get(signum, 1)

#         raise SystemExit(exit_code)
