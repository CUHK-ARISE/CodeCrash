import os
import io
import sys
import ast
import traceback
import contextlib
import multiprocessing as mp
from typing import Any, Dict, Tuple, List


@contextlib.contextmanager
def swallow_io_child():
    """ Suppress stdout and stderr in the child process only. """
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        try:
            sys.stdout.close()
            sys.stderr.close()
        except Exception:
            pass
        sys.stdout, sys.stderr = saved_out, saved_err


DEFAULT_ALLOWED_MODULES: List[str] = [
    "math", "numpy", "pandas",
    "collections", "itertools", "functools",
    "re", "random", "bisect", "heapq",
    "typing", "builtins"
]


def _prepare_exec_globals(import_modules: List[str]) -> Dict[str, Any]:
    """
    Prepare execution globals with a controlled import set.
    We import modules into the global dict and also `from X import *` for convenience.
    Missing heavy libs (e.g., numpy/pandas) won't break: they are optional.
    """
    g: Dict[str, Any] = {"__name__": "__main__"}
    for mod in import_modules:
        try:
            g[mod] = __import__(mod)
        except Exception:
            pass

    star_imports = [
        "from math import *",
        "from typing import *",
        "from collections import *",
        "from functools import *",
        "from itertools import *",
        "from builtins import *",
        "from heapq import *",
    ]
    for stmt in star_imports:
        try:
            exec(stmt, g)
        except Exception:
            pass
    return g



def _child_verify(code: str, test: str, import_modules: List[str], out_q: mp.Queue) -> None:
    """Child process: run `exec(code)` then `exec(test)` silently."""
    g = _prepare_exec_globals(import_modules)
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f), swallow_io_child():
        try:
            exec(code, g)
            exec(test, g)
            out_q.put({"status": "success"})
        except AssertionError as e:
            out_q.put({"status": "fail", "etype": "AssertionError", "msg": str(e)})
        except Exception as e:
            tb = traceback.format_exc()
            out_q.put({"status": "error", "etype": type(e).__name__, "msg": str(e), "tb": tb})


def _child_eval_call(code: str, function_call: str, import_modules: List[str], out_q: mp.Queue) -> None:
    """Child process: run `exec(code)` then `eval(function_call)` silently."""
    g = _prepare_exec_globals(import_modules)
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f), swallow_io_child():
        try:
            exec(code, g)
            result = eval(function_call, g)
            out_q.put({"status": "success", "result": result})
        except AssertionError as e:
            out_q.put({"status": "fail", "etype": "AssertionError", "msg": str(e)})
        except Exception as e:
            tb = traceback.format_exc()
            out_q.put({"status": "error", "etype": type(e).__name__, "msg": str(e), "tb": tb})


def _run_in_subprocess(target, args: Tuple[Any, ...], timeout: int) -> Dict[str, Any]:
    """
    Launch `target` in a new process, enforce timeout. On timeout, terminate.
    """
    out_q: mp.Queue = mp.Queue(maxsize=1)
    p = mp.Process(target=target, args=(*args, out_q), daemon=True)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return {"status": "timeout"}

    if not out_q.empty():
        return out_q.get()
    return {"status": "error", "etype": "RuntimeError", "msg": "Child process exited without result.", "tb": ""}


def execute_function_call(function_call: str, code: str, timeout: int = 20) -> Tuple[bool, Any]:
    """
    Executes `code` then evaluates `function_call` inside an isolated process.

    Returns:
        (True, result) on success
        (False, "timed out" | "AssertionError: ..." | "SomeError: ...") on failure
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError in code: {e}"
    try:
        # Note: function_call is an expression, parsing helps catch early errors.
        ast.parse(function_call)
    except SyntaxError as e:
        return False, f"SyntaxError in function_call: {e}"

    res = _run_in_subprocess(
        _child_eval_call,
        (code, function_call, DEFAULT_ALLOWED_MODULES),
        timeout=timeout,
    )

    status = res.get("status")
    if status == "success":
        return True, res.get("result", None)
    if status == "timeout":
        return False, "timed out"
    if status == "fail":
        return False, f"{res.get('etype','AssertionError')}: {res.get('msg','')}"

    et = res.get("etype", "Error")
    msg = res.get("msg", "")
    return False, f"{et}: {msg}"


def verify_correctness(code: str, test: str, timeout: int = 20) -> Dict[str, str]:
    """
    Verifies correctness of `code` against `test` (typically an 'assert ...' script).
    Runs in a separate process with hard timeout.

    Return dict schema:
        - status: "passed" | "failed" | "error"
        - traceback: str (empty on success)
    """
    try:
        ast.parse(code)
        ast.parse(test)
    except SyntaxError as e:
        return {"status": "error", "traceback": f"SyntaxError: {e}"}

    res = _run_in_subprocess(
        _child_verify,
        (code, test, DEFAULT_ALLOWED_MODULES),
        timeout=timeout,
    )

    status = res.get("status")
    if status == "success":
        return {"status": "passed", "traceback": ""}

    if status == "timeout":
        return {"status": "failed", "traceback": "Timed out"}

    if status == "fail":
        et = res.get("etype", "AssertionError")
        msg = res.get("msg", "")
        return {"status": "failed", "traceback": f"{et}: {msg}"}

    et = res.get("etype", "Error")
    msg = res.get("msg", "")
    tb = res.get("tb", "")
    tb_line = (tb or msg).strip()
    return {"status": "error", "traceback": f"{et}: {tb_line}"}


def execute_function_call_unsafe(function_call: str, code: str) -> Tuple[bool, Any]:
    """
    UNSAFE version: directly exec & eval in the current process (no isolation).
    - Shares globals() and imports with caller.
    - No timeout, no sandbox, no stdout suppression (unless you add redirect).

    Returns:
        (True, result) on success
        (False, "AssertionError: ..." | "SomeError: ...") on failure
    """
    exec_globals: Dict[str, Any] = _prepare_exec_globals(DEFAULT_ALLOWED_MODULES)
    try:
        ast.parse(code)
        ast.parse(function_call)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            exec(code, exec_globals)
            result = eval(function_call, exec_globals)
        return True, result
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"{type(e).__name__}: {e}\n{tb}"


def verify_correctness_unsafe(code: str, test: str) -> Dict[str, str]:
    """
    UNSAFE version: directly executes code + test (assertions).
    - No isolation, no timeout, no sandbox.
    - Shares imports and environment with current process.
    """
    exec_globals: Dict[str, Any] = _prepare_exec_globals(DEFAULT_ALLOWED_MODULES)
    try:
        ast.parse(code)
        ast.parse(test)
    except SyntaxError as e:
        return {"status": "error", "traceback": f"SyntaxError: {e}"}

    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            exec(code, exec_globals)
            exec(test, exec_globals)
        return {"status": "passed", "traceback": ""}
    except AssertionError as e:
        tb = traceback.format_exc()
        return {"status": "failed", "traceback": f"AssertionError: {e}\n{tb}"}
    except Exception as e:
        tb = traceback.format_exc()
        return {"status": "error", "traceback": f"{type(e).__name__}: {e}\n{tb}"}