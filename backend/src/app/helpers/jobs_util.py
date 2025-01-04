

def _tail(stdout, max_char=5000):
    """Return just the last few lines of the stdout, a string."""
    if not stdout:
        return ""
    return stdout[-min(max_char, len(stdout)) :]


PSQL_CHAR_LIMIT = 100 * 1000 * 1000


def _psql_tail(stdout):
    return _tail(stdout, PSQL_CHAR_LIMIT)


def _live_update_tail(
    stdout,
):
    """Choose a tail size that is appropriate for streaming / live updates (not the whole logs)."""
    return _tail(stdout, 5000)