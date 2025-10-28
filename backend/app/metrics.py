import os

from prometheus_client import Gauge

from app.helpers.rq_helpers import get_queue


def get_queue_size(queue_name: str) -> int:
    redis_url = os.environ.get("RQ_REDIS_URL")
    if not redis_url:
        raise ValueError("RQ_REDIS_URL is not set...")
    return len(get_queue(queue_name, redis_url=redis_url))


def get_size_gauge(queue_name):
    g = Gauge(f"size_{queue_name}_queue", f"Number of jobs on the {queue_name} queue")
    g.set_function(lambda: get_queue_size(queue_name=queue_name))
    return g


def get_normsize_gauge(queue_name):
    g = Gauge(
        f"normsize_{queue_name}_queue",
        f"Number of jobs per worker on the {queue_name} queue",
    )
    g.set_function(lambda: get_queue_size(queue_name))
    return g


size_cpu_g = get_size_gauge("cpu")
size_esm_g = get_size_gauge("esm")
size_boltz_g = get_size_gauge("boltz")
size_gpu_g = get_size_gauge("gpu")
size_biggpu_g = get_size_gauge("biggpu")
size_emailparrot_g = get_size_gauge("emailparrot")
size_failed_g = get_size_gauge("failed")

normsize_cpu_g = get_normsize_gauge("cpu")
normsize_esm_g = get_normsize_gauge("esm")
normsize_boltz_g = get_normsize_gauge("boltz")
normsize_gpu_g = get_normsize_gauge("gpu")
normsize_biggpu_g = get_normsize_gauge("biggpu")
normsize_emailparrot_g = get_normsize_gauge("emailparrot")
normsize_failed_g = get_normsize_gauge("failed")
