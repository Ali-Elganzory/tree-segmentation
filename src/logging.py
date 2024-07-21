from enum import Enum
from time import strftime

from src.distributed import is_enabled, get_global_rank, get_local_rank


class LogLevel(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2


LOG_LEVEL = LogLevel.INFO

_builtin_print = None
_logging_print = None


def set_log_level(log_level: LogLevel):
    global LOG_LEVEL
    LOG_LEVEL = log_level

    enable(LOG_LEVEL)


def log_prefix():
    global LOG_LEVEL
    return f"{f'[{get_global_rank()}|{get_local_rank()}] ' if is_enabled() else ''}[{LOG_LEVEL.name}][{strftime('%Y-%m-%d %H:%M:%S')}]"


def enable(log_level: LogLevel = LogLevel.INFO):
    """
    This function disables printing when not in the main process
    """
    # Built-in print
    global _builtin_print, _logging_print, LOG_LEVEL

    if log_level != LOG_LEVEL:
        LOG_LEVEL = log_level

    import builtins as __builtin__

    builtin_print = __builtin__.print

    if builtin_print is _logging_print:
        builtin_print = _builtin_print
    else:
        _builtin_print = builtin_print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        log_level = kwargs.pop("log_level", LogLevel.INFO)
        if log_level.value >= LOG_LEVEL.value or force:
            _builtin_print(
                log_prefix(),
                flush=True,
                *args,
                **kwargs,
            )

    __builtin__.print = print

    _logging_print = print

    # TQDM
    from functools import partial
    import tqdm

    __init__ = tqdm.tqdm.__init__
    tqdm.tqdm.__init__ = lambda self, *args, **kwargs: __init__(
        self, position=get_global_rank() if is_enabled() else 0, *args, **kwargs
    )
