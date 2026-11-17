"""Benchmark one target: find the concurrency with the highest throughput.

This is a step of its own, apart from a run. It measures, writes its own ledger (report
it with ``write_report`` or ``write_pdf`` for the throughput figures) and hands on a
:class:`Tuning` that a :class:`RunPlan` can take as its concurrency.
"""

from __future__ import annotations

import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from ._util import now_iso
from .casebook import Casebook
from .execute import execute
from .journal import Journal
from .ledger import Ledger
from .plan import RunPlan, Stage, agent_fingerprint
from .target import Target
from .tuning import Tuning

MAX_CONCURRENCY = 128
MIN_CASES = 32  # a level runs at least this many cases, so low levels are not judged on a handful


@dataclass(frozen=True)
class BenchPlan(RunPlan):
    """The plan of a benchmark: a ladder of stages, each with its own random cases."""

    ladder: tuple[Stage, ...] = ()
    drawn: Mapping[str, Any] = field(default_factory=dict)
    gain: float = 0.05

    @property
    def min_gain(self) -> float:
        """A stage must be this much faster than the best before it, or the search stops."""
        return self.gain

    def stages(self, target: str) -> tuple[Stage, ...]:
        """The ladder, lowest concurrency first."""
        return self.ladder

    @property
    def case_count(self) -> int:
        """Number of distinct cases drawn."""
        return len({index for stage in self.ladder for index in stage.cases or ()})

    @cached_property
    def spec(self) -> dict[str, Any]:
        """The plan as plain JSON, with the drawn cases of every stage."""
        return {
            "kind": "benchmark",
            "casebook_id": self.casebook.id,
            "agent": agent_fingerprint(self.agent),
            "targets": [t.describe() for t in self.targets],
            "seeds": None,
            "repeats": 1,
            "attempts_per_case": 1,
            "concurrency": [stage.concurrency for stage in self.ladder],
            "case_count": self.case_count,
            "sample": dict(self.drawn),
            "min_gain": self.gain,
        }


@dataclass(frozen=True)
class Benchmark:
    """Outcome of :func:`benchmark`. ``levels`` lists ``{"concurrency", "rate", "cases"}`` in
    the order measured; ``rate`` is in ``metric`` (output tokens or attempts per second).
    """

    best: int
    metric: str
    levels: list[dict[str, Any]]
    ledger: Ledger

    @property
    def tuning(self) -> Tuning:
        """The recommendation alone, to pass to a :class:`RunPlan`."""
        return self.ledger.tuning()

    def close(self) -> None:
        """Close the benchmark ledger."""
        self.ledger.close()

    def __enter__(self) -> Benchmark:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def ladder(
    case_count: int, *, max_concurrency: int = MAX_CONCURRENCY, rounds: int = 10, levels: Sequence[int] | None = None
) -> list[tuple[int, int]]:
    """``(concurrency, number of cases)`` per level: 1, 2, 4, ... up to ``max_concurrency``.

    A level runs ``rounds`` times its concurrency in cases (at least 32), so every slot is
    filled about ``rounds`` times. Levels that would need more cases than the casebook
    has are left out: beyond a tenth of the available cases a level could not be
    measured properly.
    """
    if levels is None:
        levels = [2**n for n in range(max_concurrency.bit_length()) if 2**n <= max_concurrency]
    usable = [level for level in sorted(set(levels)) if level * rounds <= case_count] or [min(levels)]
    return [(level, min(case_count, max(level * rounds, MIN_CASES))) for level in usable]


async def benchmark(
    casebook: Casebook,
    agent: Agent[Any, Any],
    target: Target,
    destination: str | os.PathLike[str],
    *,
    max_concurrency: int = MAX_CONCURRENCY,
    rounds: int = 10,
    seed: int = 0,
    levels: Sequence[int] | None = None,
    min_gain: float = 0.05,
    warmup: int = 1,
    reset_prefix_cache: bool = True,
    parent: str | os.PathLike[str] = "tmp",
    signing_key: Any = None,
    **execute_options: Any,
) -> Benchmark:
    """Measure rising concurrency on randomly drawn cases and stop when a level is not at
    least ``min_gain`` faster than the best before it.

    The cases are drawn from the casebook with ``seed`` and recorded, so the benchmark
    can be repeated. How many is decided here, see :func:`ladder`: level ``c`` runs the
    first ``rounds * c`` drawn cases, and the search ends at ``max_concurrency`` (128)
    or where a level would need more than the casebook holds, whichever comes first.

    Lower levels run a subset of the cases of higher ones, so the server's prefix cache
    is reset before each level (vLLM dev mode); where that is not possible a warning is
    recorded and later levels look better than they are. ``best`` is the lowest level
    within ``min_gain`` of the highest rate, so a tie goes to the smaller concurrency.

    The measurements are sealed to ``destination`` as a benchmark ledger; its report
    shows throughput and latency. ``result.tuning`` is the recommendation alone.
    Single failed attempts do not stop a benchmark; they are counted in its report.
    ``signing_key`` signs the ledger; other keyword arguments go to :func:`execute`.
    """
    steps = ladder(len(casebook), max_concurrency=max_concurrency, rounds=rounds, levels=levels)
    order = random.Random(seed).sample(range(len(casebook)), max(count for _, count in steps))
    plan = BenchPlan(
        casebook,
        agent,
        [target],
        ladder=tuple(Stage(level, tuple(order[:count])) for level, count in steps),
        drawn={"seed": seed, "rounds": rounds, "order": order},
        gain=min_gain,
    )
    _ = plan._picked  # read the cases now, not inside the first measured stage
    with Journal.open(plan, parent=parent) as journal:
        started = now_iso()
        await execute(plan, journal, warmup=warmup, reset_prefix_cache=reset_prefix_cache, **execute_options)
        # Only a recommendation of this call counts: one left by an earlier call may come from
        # a server that is gone. A call that finds every level measured writes it again.
        if not any(record.get("kind") == "tuning" and record["at"] >= started for record in journal.records()):
            raise RuntimeError(
                f"The benchmark of {target.name!r} has no level with a successful attempt, so there is "
                f"nothing to recommend. The journal is kept at {journal.directory}."
            )
        ledger = journal.seal(Path(destination), signing_key=signing_key, require_complete=False)
    result = ledger.tunings()[target.name]
    return Benchmark(best=result["best"], metric=result["metric"], levels=result["levels"], ledger=ledger)
