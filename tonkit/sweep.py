"""Small helpers for parameter sweeps: grids, parallel map, CSV output."""
from __future__ import annotations

import csv
import itertools
import os
from multiprocessing import get_context
from typing import Callable, Iterable


def grid(**axes) -> list:
    """Cartesian product of named axes -> list of dicts (first axis varies slowest)."""
    keys = list(axes)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[axes[k] for k in keys])]


def pmap(fn: Callable, items: Iterable, processes: int | None = None, chunksize: int = 1) -> list:
    """Order-preserving parallel map (fork pool). fn must be a module-level function."""
    items = list(items)
    n = processes if processes is not None else max(1, (os.cpu_count() or 1))
    if n <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with get_context("fork").Pool(n) as pool:
        return pool.map(fn, items, chunksize=chunksize)


def write_csv(path: str, rows: list, fields: list | None = None) -> None:
    """Write a list of dicts to CSV (floats in %.10g)."""
    if not rows:
        return
    fields = fields or list(rows[0])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
