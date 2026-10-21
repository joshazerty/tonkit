"""Shared fixtures. Every closed-form comparison goes through `check(label, value, reference)`,
which asserts |value/reference - 1| < tol (default 1 %, the plan's criterion) and records the
deviation; the session writes tests/deviation_report.csv (one row per comparison)."""
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TOL = 1e-2
_ROWS = []


def _check(label, value, reference, tol=TOL, mode="rel"):
    value, reference = float(value), float(reference)
    dev = abs(value - reference) if mode == "abs" else abs(value / reference - 1.0)
    _ROWS.append((os.environ.get("PYTEST_CURRENT_TEST", "").split(" ")[0], label, value, reference, mode, dev, tol))
    assert dev < tol, f"{label}: {value:.8g} vs {reference:.8g} ({mode} dev {dev:.2e} > {tol:g})"
    return dev


@pytest.fixture
def check():
    return _check


def pytest_sessionfinish(session, exitstatus):
    if not _ROWS:
        return
    path = os.path.join(os.path.dirname(__file__), "deviation_report.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test", "quantity", "engine", "closed_form_or_exact", "mode", "deviation", "tolerance"])
        for r in _ROWS:
            w.writerow([r[0], r[1], f"{r[2]:.10g}", f"{r[3]:.10g}", r[4], f"{r[5]:.3e}", f"{r[6]:g}"])
