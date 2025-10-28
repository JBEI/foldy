import logging
import os
import signal
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytz

from app.models import Invokation

PSQL_CHAR_LIMIT: int = 100 * 1000 * 1000


def _tail(stdout: str, max_char: int = 5000) -> str:
    """
    Return just the last few lines of the stdout string.

    Args:
        stdout: The complete stdout string
        max_char: Maximum number of characters to include

    Returns:
        A truncated string containing the last max_char characters
    """
    if not stdout:
        return ""
    return stdout[-min(max_char, len(stdout)) :]


def _psql_tail(stdout: str) -> str:
    """
    Truncate stdout to fit within PostgreSQL character limits.

    Args:
        stdout: The complete stdout string

    Returns:
        A truncated string that fits within PostgreSQL limits
    """
    return _tail(stdout, PSQL_CHAR_LIMIT)


def _live_update_tail(stdout: str) -> str:
    """
    Choose a tail size appropriate for streaming/live updates.

    Args:
        stdout: The complete stdout string

    Returns:
        A truncated string suitable for live updates
    """
    return _tail(stdout, 30000)


def get_torch_cuda_is_available_and_add_logs(add_log: Callable[[str], Any]) -> bool:
    """
    Check CUDA availability and log GPU diagnostics.

    Args:
        add_log: Function to add log messages

    Returns:
        True if CUDA is available, False otherwise
    """
    import torch

    add_log("=== GPU Diagnostics ===")
    add_log(f"PyTorch version: {torch.__version__}")
    add_log(f"CUDA is{' not' if not torch.cuda.is_available() else ''} available")

    # Check if PyTorch was built with CUDA
    add_log(f"PyTorch CUDA built: {torch.version.cuda is not None}")  # type: ignore[reportAttributeAccessIssue] # torch.version module incomplete typing

    try:
        # Try to get NVIDIA driver version
        import subprocess

        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
        add_log(f"nvidia-smi output:\n{nvidia_smi.decode()}")
    except Exception as e:
        add_log(f"Failed to run nvidia-smi: {str(e)}")

    # Check environment variables
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    add_log(f"CUDA_VISIBLE_DEVICES env var: {cuda_visible_devices}")

    return torch.cuda.is_available()


class LoggingRecorder(logging.Handler):
    """A logging handler that records logs to an Invokation."""

    def __init__(self, invokation: Invokation, level: int = logging.INFO) -> None:
        """
        Initialize the LoggingRecorder.

        Args:
            invokation: The Invokation model instance to record logs to
            level: The minimum logging level to record (default: INFO)
        """
        super().__init__(level)
        self.invokation = invokation
        self.logs: list[str] = []
        self.starttime: float = time.time()
        self.final_state: str = "failed"  # Default state, will be set to "finished" on success
        self._previous_level: int = logging.INFO
        self._previous_handlers: list[logging.Handler] = []
        self._sigterm_stack: list[str] = []
        # ---------- install graceful-term handler ----------
        self._old_sigterm = signal.getsignal(signal.SIGTERM)

        def _graceful_term(signum, frame):
            # Capture *current* stack so we know where we were killed
            self._sigterm_stack = traceback.format_stack(frame, limit=10)
            raise SystemExit("terminated by SIGTERM")

        signal.signal(signal.SIGTERM, _graceful_term)

        try:
            self.invokation.update(
                state="running",
                starttime=datetime.fromtimestamp(self.starttime, timezone.utc),
            )
        except Exception as e:
            logging.error(f"Failed to update invokation: {e}")

    def emit(self, record: logging.LogRecord) -> None:
        """
        Process a log record by adding it to the invokation.

        Args:
            record: The logging record to process
        """
        # Convert the log record to a string
        msg = self.format(record)

        # Add timestamp and severity
        pt_tz = pytz.timezone("America/Los_Angeles")
        timestamp = datetime.now(pt_tz).isoformat(sep=" ", timespec="milliseconds")
        severity = record.levelname
        formatted_msg = f"{timestamp} [{severity}] - {msg}"

        # Add the log to the invokation
        self.logs.append(formatted_msg)
        self.invokation.update(
            log=_live_update_tail("\n".join(self.logs)),
            timedelta=timedelta(seconds=time.time() - self.starttime),
        )

    def __enter__(self) -> "LoggingRecorder":
        """
        Set up logging when entering the context.

        Returns:
            The LoggingRecorder instance
        """
        # Get the root logger
        logger = logging.getLogger()

        # Store the previous level and handlers
        self._previous_level = logger.level
        self._previous_handlers = logger.handlers[:]

        # Don't remove existing handlers, just add ours and maybe adjust level
        logger.setLevel(min(self.level, logger.level))  # Use the more verbose level

        # Add ourselves as a handler alongside existing ones
        logger.addHandler(self)

        log_node_info(logger)

        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """
        Restore previous logging state when exiting the context.

        Args:
            exc_type: The type of exception raised, if any
            exc_val: The exception instance raised, if any
            exc_tb: The traceback information, if an exception was raised
            False to propagate exceptions
        """
        # Restore previous SIGTERM handler first
        signal.signal(signal.SIGTERM, self._old_sigterm)

        logger = logging.getLogger()

        try:
            if exc_type is not None:
                if exc_type is SystemExit and str(exc_val) != "0":
                    # Treat non-zero SystemExit (SIGTERM path) as failure
                    self.final_state = "failed"
                    # Attach captured stack (if any)
                    if getattr(self, "_sigterm_stack", None):
                        self.logs.append(
                            "----- stack @ SIGTERM -----\n" + "".join(self._sigterm_stack)
                        )
                    logger.info(
                        """Check for preemption with 'gcloud compute operations list --filter="compute.instances.preempted"'"""
                    )
                else:
                    self.final_state = "failed"
                    full_tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
                    self.logs.append("----- exception traceback -----\n" + full_tb)
            else:
                self.final_state = "finished"
        finally:
            log_node_info(logger)

            # Log the final state
            self.logs.append(f"Invokation ending with final state {self.final_state}")

            # Final update with complete logs
            self.invokation.update(
                log=_psql_tail("\n".join(self.logs)),
                timedelta=timedelta(seconds=time.time() - self.starttime),
                state=self.final_state,
            )

            # Restore previous logging state
            logger.removeHandler(self)
            logger.level = self._previous_level
            logger.handlers = self._previous_handlers

            # If the job failed, print logs and raise assertion
            if self.final_state != "finished":
                logs_tail = _psql_tail("\n".join(self.logs))
                print(
                    f"Job finished in state {self.final_state} with logs:\n\n{logs_tail}",
                    flush=True,
                )
        return None


def log_node_info(logger: logging.Logger) -> None:
    """
    Log node information.
    """
    logger.info(f"Node name: {os.environ.get('NODE_NAME', 'unknown')}")
    logger.info(f"Node IP: {os.environ.get('NODE_IP', 'unknown')}")
    logger.info(f"Pod name: {os.environ.get('POD_NAME', 'unknown')}")
    logger.info(f"Pod namespace: {os.environ.get('POD_NAMESPACE', 'unknown')}")
