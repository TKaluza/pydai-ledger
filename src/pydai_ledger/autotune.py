"""Find the concurrency with the highest throughput for one task on one target."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from .casebook import Casebook
from .execute import execute
from .journal import Journal
from .ledger import Ledger
from .plan import RunPlan
from .target import Target

LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class Autotune:
    """Outcome of :func:`autotune`. ``levels`` lists ``{"concurrency", "rate"}`` in the
    order measured; ``rate`` is in ``metric`` (output tokens or attempts per second).
    """

    best: int
    metric: str
    levels: list[dict[str, Any]]
    ledger: Ledger


async def autotune(
    casebook: Casebook,
    agent: Agent[Any, Any],
    target: Target,
    destination: str | os.PathLike[str],
    *,
    sample: int = 64,
    levels: Sequence[int] = LADDER,
    min_gain: float = 0.05,
    warmup: int = 1,
    reset_prefix_cache: bool = True,
    parent: str | os.PathLike[str] = "tmp",
    **execute_options: Any,
) -> Autotune:
    """Measure the first ``sample`` cases at rising concurrency and stop when a level is
    not at least ``min_gain`` faster than the best before it.

    Every level runs the same cases, so the server's prefix cache is reset before each
    one (vLLM dev mode); where that is not possible a warning is recorded and later
    levels look better than they are. Levels above ``sample`` are left out because they
    could not be filled. ``best`` is the lowest level within ``min_gain`` of the highest
    rate, so a tie goes to the smaller concurrency. The measurements are sealed to ``destination`` like any run,
    so they can be reported and compared.
    """
    count = min(sample, len(casebook))
    ladder = [level for level in levels if level <= count] or [1]
    plan = RunPlan(casebook, agent, [target], concurrency=ladder, limit=count)
    with Journal.open(plan, parent=parent) as journal:
        await execute(
            plan,
            journal,
            autotune=True,
            min_gain=min_gain,
            warmup=warmup,
            reset_prefix_cache=reset_prefix_cache,
            **execute_options,
        )
        ledger = journal.seal(Path(destination))
    result = ledger.autotune()[target.name]
    return Autotune(best=result["best"], metric=result["metric"], levels=result["levels"], ledger=ledger)
