"""Math-Verify scoring for the paper's evaluation profiles.

The reported response pools used MaxRL's nested SIGALRM behavior. Math-Verify
can replace that timer internally, so historical mode is not a wall deadline.
The submitted code's external deadline is available as an explicit option.
"""

import logging
import multiprocessing as mp
import signal
import sys
import time

MATH_TIMEOUT_SECONDS = 1
_metric = None
_task = None
_timeout_policy = "historical"


class ItemTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise ItemTimeout("MaxRL item timeout")


def initialize(task, timeout_policy="historical"):
    global _metric, _task, _timeout_policy
    if task not in ("maze", "smollm", "qwen25", "qwen3") or timeout_policy not in ("historical", "external"):
        raise ValueError("Unknown evaluation task or timeout policy")
    _task = task
    _timeout_policy = timeout_policy
    if task == "maze":
        from verl.utils.reward_score.maze import judge_maze
        _metric = judge_maze
    else:
        # Fork only from CPU grading workers, never from the generation process.
        if timeout_policy == "external":
            mp.get_context("fork")
        if hasattr(sys, "set_int_max_str_digits"):
            sys.set_int_max_str_digits(4300)
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
        logging.getLogger("math_verify").setLevel(logging.ERROR)
        logging.getLogger("math_verify.metric").setLevel(logging.ERROR)
        _metric = math_metric(gold_extraction_target=(LatexExtractionConfig(),),
                              pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()))


def _result(score=0.0, status="ok", error=None, evaluated=True):
    if score not in (0.0, 1.0):
        raise ValueError(f"Nonbinary verifier score: {score}")
    return dict(verifier_correct=bool(score), correct=bool(score), score=score,
                verifier_evaluated=evaluated, grading_status=status, grading_error=error,
                verifier_extractions=None, verifier_metadata_error=None, grading_seconds=0.0)


def _grade_math(item):
    from math_verify.errors import TimeoutException

    text, gold = item
    start = time.monotonic()
    status, error, detail, score = "ok", None, None, 0.0
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(MATH_TIMEOUT_SECONDS)
    try:
        value, detail = _metric(["\\boxed{" + gold + "}"], [text])
        score = float(value)
    except (ItemTimeout, TimeoutException) as exc:
        status, error = "timeout", type(exc).__name__
    except MemoryError:
        raise
    except Exception as exc:
        status, error = "verifier_error", f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
    result = _result(score, status, error)
    try:
        result["verifier_extractions"] = repr(detail)
    except Exception as exc:
        result["verifier_metadata_error"] = f"{type(exc).__name__}: {exc}"
    result["grading_seconds"] = time.monotonic() - start
    return result


def _child_grade(connection, item):
    try:
        connection.send((True, _grade_math(item)))
    except BaseException as exc:
        connection.send((False, (type(exc).__name__, str(exc))))
    finally:
        connection.close()


def _bounded_grade(item):
    # Math-Verify can replace SIGALRM internally; the parent deadline survives.
    context = mp.get_context("fork")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_child_grade, args=(send, item))
    start = time.monotonic()
    try:
        process.start()
        send.close()
        if not receive.poll(MATH_TIMEOUT_SECONDS):
            result = _result(status="timeout", error="ExternalItemDeadline")
        else:
            try:
                success, result = receive.recv()
            except EOFError as exc:
                raise RuntimeError("Grading subprocess exited without a result") from exc
            if not success:
                raise RuntimeError("Grading infrastructure failure: " + repr(result))
        result["grading_seconds"] = time.monotonic() - start
        return result
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()


def grade(item):
    text, gold, capped = item
    if _metric is None or _task is None:
        raise RuntimeError("Grader must be initialized before grading")
    if not isinstance(text, str) or not isinstance(gold, str) or type(capped) is not bool:
        raise ValueError("Invalid grading input")
    if _task == "maze":
        start = time.monotonic()
        result = _result(float(_metric(text, gold)))
        result["grading_seconds"] = time.monotonic() - start
        return result
    if capped:
        return _result(status="length_cap", evaluated=False)
    return (_bounded_grade if _timeout_policy == "external" else _grade_math)((text, gold))
