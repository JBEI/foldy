from prometheus_client import Gauge

from app.extensions import rq


def get_queue_size(queue_name: str):
    return len(rq.get_queue(queue_name))


def get_size_gauge(queue_name):
    g = Gauge(f"size_{queue_name}_queue", f"Number of jobs on the {queue_name} queue")
    g.set_function(lambda: get_queue_size(queue_name))
    return g


def get_normsize_gauge(queue_name):
    g = Gauge(
        f"normsize_{queue_name}_queue",
        f"Number of jobs per worker on the {queue_name} queue",
    )
    g.set_function(lambda: get_queue_size(queue_name))
    return g


size_cpu_g = get_size_gauge("cpu")
size_gpu_g = get_size_gauge("gpu")
size_biggpu_g = get_size_gauge("biggpu")
size_emailparrot_g = get_size_gauge("emailparrot")
size_failed_g = get_size_gauge("failed")

normsize_cpu_g = get_normsize_gauge("cpu")
normsize_gpu_g = get_normsize_gauge("gpu")
normsize_biggpu_g = get_normsize_gauge("biggpu")
normsize_emailparrot_g = get_normsize_gauge("emailparrot")
normsize_failed_g = get_normsize_gauge("failed")
