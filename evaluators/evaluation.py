"""
Reference: Evalplus
"""
# Packages
import sys
import contextlib
import signal


class TimeoutException(Exception):
    """Custom exception to handle timeout."""
    pass


@contextlib.contextmanager
def time_limit(seconds):
    """Limits the time a code block can execute."""
    def signal_handler(signum, frame):
        raise TimeoutException("Execution timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


@contextlib.contextmanager
def swallow_io():
    """Suppress stdout and stderr during execution."""
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr
    try:
        sys.stdout = open('/dev/null', 'w')
        sys.stderr = open('/dev/null', 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr


def execute_function_call(function_call: str, code: str, timeout: int = 20):
    """
    Executes a function call from a string and returns the result.
    If the execution takes longer than the timeout, it raises a RuntimeError.
    """
    exec_globals = {}
    try:
        with swallow_io():
            with time_limit(int(timeout)):
                exec(code, exec_globals)
                return True, eval(function_call, exec_globals)
    except TimeoutException:
        return False, "timed out"
    except AssertionError as e:
        return False, f"{e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def verify_correctness(expression: str, code: str, timeout: int = 20) -> bool:
    external_modules = ["math", "numpy", "pandas", "collections", "itertools", "functools", "re", "random", "bisect"]
    exec_globals = {module: __import__(module) for module in external_modules}
    
    exec("from math import *", exec_globals)
    exec("from typing import *", exec_globals)
    exec("from collections import *", exec_globals)
    exec("from functools import *", exec_globals)
    exec("from itertools import *", exec_globals)
    exec("from builtins import *", exec_globals)
    exec("from heapq import *", exec_globals)
    
    try:
        if not expression.startswith("assert"):
            expression = f"assert {expression}"
        with swallow_io():
            with time_limit(int(timeout)):
                exec(code, exec_globals)
                exec(f"{expression}", exec_globals)
                return True, None
    except TimeoutException as e:
        return False, e
    except AssertionError as e:
        return False, e
    except Exception as e:
        return False, e
