import time
from datetime import datetime, timedelta, UTC
import traceback
import os


PSQL_CHAR_LIMIT = 100 * 1000 * 1000


def _tail(stdout, max_char=5000):
    """Return just the last few lines of the stdout, a string."""
    if not stdout:
        return ""
    return stdout[-min(max_char, len(stdout)) :]


def _psql_tail(stdout):
    return _tail(stdout, PSQL_CHAR_LIMIT)


def _live_update_tail(
    stdout,
):
    """Choose a tail size that is appropriate for streaming / live updates (not the whole logs)."""
    return _tail(stdout, 5000)


def try_run_job_with_logging(f, invokation):
    final_state = "failed"
    start_time = time.time()
    logs = []

    def add_log(msg, tail_function=_live_update_tail, **kwargs):
        timestamp = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
        timestamped_msg = f"{timestamp} - {msg}"
        logs.append(timestamped_msg)
        print(timestamped_msg)
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


def get_torch_cuda_is_available_and_add_logs(add_log):
    """Return True if cuda is availableand add logs about the GPU."""
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
