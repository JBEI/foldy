import time
from datetime import datetime, timedelta, UTC, timezone
import traceback
import os
import logging
from contextlib import contextmanager
from typing import Optional, Callable, Any, Dict, List, Union, TypeVar, cast
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


# Define type for the add_log function
AddLogFn = Callable[..., None]

def try_run_job_with_logging(
    f: Callable[[AddLogFn], None], 
    invokation: Invokation
) -> None:
    """
    Execute a job function with logging and exception handling.
    
    Args:
        f: The job function to execute, which takes an add_log function
        invokation: The Invokation model instance to update with logs and state
    """
    final_state: str = "failed"
    start_time: float = time.time()
    logs: List[str] = []

    def sanitize_log(log_str: str) -> str:
        """
        Remove or replace problematic characters from log strings.
        
        Args:
            log_str: The log string to sanitize
            
        Returns:
            A sanitized string with problematic characters removed
        """
        # Remove NUL characters
        sanitized = log_str.replace("\x00", "")

        # Optionally replace other problematic characters
        sanitized = "".join(char if ord(char) >= 32 else " " for char in sanitized)

        return sanitized

    def add_log(
        msg: str, 
        tail_function: Callable[[str], str] = _live_update_tail, 
        **kwargs: Any
    ) -> None:
        """
        Add a log message and update the invokation record.
        
        Args:
            msg: The log message to add
            tail_function: Function to truncate logs for display
            **kwargs: Additional fields to update in the invokation
        """
        timestamp = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
        timestamped_msg = f"{timestamp} - {sanitize_log(msg)}"
        logs.append(timestamped_msg)
        print(timestamped_msg, flush=True)

        # Ensure starttime is in UTC with timezone info
        if "starttime" in kwargs:
            kwargs["starttime"] = datetime.fromtimestamp(start_time, timezone.utc)

        invokation.update(
            log=tail_function("\n".join(logs)),
            timedelta=timedelta(seconds=time.time() - start_time),
            **kwargs,
        )

    try:
        add_log(
            "Starting...",
            state="running",
            starttime=datetime.fromtimestamp(start_time),
        )
        f(add_log)
        # 7. Updated Evolution record with model and visualizations.
        final_state = "finished"
    except Exception as e:
        # Capture the full traceback
        full_traceback = traceback.format_exc()

        add_log(f"Job failed with exception:\n\n{e} {full_traceback}")
    finally:
        # This will get executed regardless of the exceptions raised in try
        # or except statements.
        add_log(
            f"Invokation ending with final state {final_state}",
            tail_function=_psql_tail,
            state=final_state,
        )

        if final_state != "finished":
            print(
                f'Job finished in state {final_state} with logs:\n\n{_psql_tail("\n".join(logs))}',
                flush=True,
            )
            assert False, _psql_tail("\n".join(logs))


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
    add_log(f"CUDA is{'not' if not torch.cuda.is_available() else ''} available")

    # Check if PyTorch was built with CUDA
    add_log(f"PyTorch CUDA built: {torch.version.cuda is not None}")

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
        self.logs: List[str] = []
        self.starttime: float = time.time()
        self.final_state: str = "failed"  # Default state, will be set to "finished" on success
        self._previous_level: int = logging.INFO
        self._previous_handlers: List[logging.Handler] = []
        
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
        timestamp = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
        severity = record.levelname
        formatted_msg = f"{timestamp} [{severity}] - {msg}"

        # Add the log to the invokation
        self.logs.append(formatted_msg)
        self.invokation.update(
            log=_live_update_tail("\n".join(self.logs)),
            timedelta=timedelta(seconds=time.time() - self.starttime),
        )

    def __enter__(self) -> 'LoggingRecorder':
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

        # Remove existing handlers and set new level
        logger.handlers = []
        logger.setLevel(self.level)

        # Add ourselves as a handler
        logger.addHandler(self)
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Any) -> bool:
        """
        Restore previous logging state when exiting the context.
        
        Args:
            exc_type: The type of exception raised, if any
            exc_val: The exception instance raised, if any
            exc_tb: The traceback information, if an exception was raised
            
        Returns:
            False to propagate exceptions
        """
        logger = logging.getLogger()

        try:
            if exc_type is not None:
                # Capture the full traceback
                full_traceback = traceback.format_exc()
                logging.error(
                    f"Job failed with exception:\n\n{exc_val}\n{full_traceback}"
                )
            else:
                self.final_state = "finished"
        finally:
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
                print(
                    f'Job finished in state {self.final_state} with logs:\n\n{_psql_tail("\n".join(self.logs))}',
                    flush=True,
                )
                assert False, _psql_tail("\n".join(self.logs))

            return False  # Re-raise any exceptions
