import time
import functools
import logging

_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def timing(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        _log.info(f'Function executed in {time.perf_counter() - start:.1f} s')
        return result

    return wrapper
