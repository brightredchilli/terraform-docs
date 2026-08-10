from enum import StrEnum
import functools
import os
from signal import SIGPIPE
import sys


def handle_broken_pipe(fn):
    """
    Decorator that handles broken pipe error. This can happen if the output of the script
    is piped to head, and is therefore closed early.

    """

    @functools.wraps(fn)
    def decorator(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except BrokenPipeError:
            # Point stdout at devnull so the interpreter's final flush cannot
            # raise again, then report the conventional SIGPIPE status.
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())

            # 128 + error is a way to indicate to unix shell that the script failed for that specific reason
            return 128 + SIGPIPE

    return decorator


def all_values(e: type[StrEnum]) -> str:
    """
    Concatenate all members of the StrEnum and return it as a string
    """
    return ", ".join((str(p) for p in e))
