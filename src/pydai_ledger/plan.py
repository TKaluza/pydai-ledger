"""RunPlan: which cases run with which agent on which targets, and how often."""

from __future__ import annotations

import random
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from typing import Any

import pydantic_ai
from pydantic_ai import Agent
from pydantic_core import to_jsonable_python

from ._util import canonical_json, sha256_hex
from .casebook import Casebook
from .target import Target
from .tuning import Tuning


def agent_fingerprint(agent: Agent[Any, Any]) -> dict[str, Any]:
    """Everything about the agent that shapes a request, as plain JSON.

    Pydantic AI has no public way to list instructions, system prompts and tools, so
    private attributes are read. If a future version renames them, the affected parts
    are listed under ``"unreadable"`` and a warning is emitted; the requests themselves
    are still recorded exactly as sent by ``execute()``.
    """
    unreadable = [
        name for name in ("_instructions", "_system_prompts", "_function_toolset") if not hasattr(agent, name)
    ]
    if unreadable:
        warnings.warn(
            f"agent_fingerprint() cannot read {', '.join(unreadable)} in pydantic-ai {pydantic_ai.__version__}; "
            "the fingerprint is incomplete.",
            stacklevel=2,
        )

    def text(value: Any) -> str:
        return value if isinstance(value, str) else f"<dynamic:{getattr(value, '__qualname__', value)!r}>"

    instructions = [text(getattr(i, "instruction", i)) for i in getattr(agent, "_instructions", [])]
    system_prompts = [text(p) for p in getattr(agent, "_system_prompts", ())]
    tools: dict[str, Any] = {}
    for name, tool in getattr(getattr(agent, "_function_toolset", None), "tools", {}).items():
        tool_def = getattr(tool, "tool_def", None)
        tools[name] = {
            "description": getattr(tool_def, "description", None),
            "parameters": getattr(tool_def, "parameters_json_schema", None),
        }
    try:
        output_schema = agent.output_json_schema()
    except Exception as exc:  # schema generation is best effort
        output_schema = {"error": repr(exc)}
    return {
        "name": agent.name,
        "instructions": instructions,
        "system_prompts": system_prompts,
        "output_schema": output_schema,
        "model_settings": to_jsonable_python(agent.model_settings, fallback=repr),
        "tools": tools,
        "pydantic_ai_version": pydantic_ai.__version__,
        **({"unreadable": unreadable} if unreadable else {}),
    }


def attempt_id(target: str, concurrency: int, case_index: int, rep: int) -> str:
    """Stable id of one attempt within a run."""
    return f"{target}|c{concurrency}|{case_index}|{rep}"


@dataclass(frozen=True)
class Stage:
    """One measured stretch of a target: a concurrency and the cases that run at it.
    ``cases`` is None for all planned cases.
    """

    concurrency: int
    cases: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RunPlan:
    """A run is ``cases x targets x seeds x repeats``.

    ``seeds`` and ``repeats``
        The two ways to run a case more than once on a target. ``repeats=3`` runs it
        three times under the same conditions: that measures repeatability.
        ``seeds=[1, 2]`` runs it once per seed: that measures whether the answer depends
        on the seed. Together they give ``len(seeds) * repeats`` attempts per case. A
        seed overrides the one in the agent's model settings; every request records the
        seed it was sent with. Without ``seeds`` the agent's settings are used unchanged.
    ``sample``
        Run only this many cases, drawn at random with ``sample_seed`` and recorded in
        the plan. Validating a setup on a sample with several seeds and repeats costs
        far less than doing so on every case. Without ``sample`` all cases run, which
        with seeds and repeats validates the whole casebook.
    ``concurrency``
        How many attempts are in flight at once per target: one number for all targets,
        a mapping from target name to number, or a :class:`Tuning` from
        :func:`pydai_ledger.benchmark`. Left out, every target uses its own
        ``Target.concurrency``, or 8. It shapes how fast the run is, not what it records
        as result; measuring throughput is the job of ``benchmark``.
    ``sequential_targets``
        Targets are independent and run side by side. Set this when they share
        hardware and must run one after another.
    ``limit``
        Restricts the run (and the pool ``sample`` draws from) to the first N cases.
    """

    casebook: Casebook
    agent: Agent[Any, Any]
    targets: Sequence[Target]
    seeds: Sequence[int] | None = None
    repeats: int = 1
    sample: int | None = None
    sample_seed: int = 0
    concurrency: int | Mapping[str, int] | Tuning | None = None
    sequential_targets: bool = False
    limit: int | None = None
    tuning: Tuning | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        names = [t.name for t in targets]
        if not targets or len(set(names)) != len(names):
            raise ValueError("RunPlan needs at least one target and unique target names.")
        if self.seeds is not None:
            seeds = tuple(self.seeds)
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("seeds needs at least one seed, each only once; use repeats to run a seed again.")
            object.__setattr__(self, "seeds", seeds)
        if self.repeats < 1 or (self.sample is not None and self.sample < 1):
            raise ValueError("repeats and sample must be >= 1.")
        given = self.concurrency
        if isinstance(given, Tuning):
            object.__setattr__(self, "tuning", given)
            given = given.concurrency
        if given is None:
            levels = {t.name: t.concurrency or 8 for t in targets}
        elif isinstance(given, int):
            levels = dict.fromkeys(names, given)
        elif isinstance(given, Mapping):
            levels = dict(given)
            if set(levels) != set(names):
                raise ValueError("A concurrency mapping needs exactly one entry per target name.")
        else:
            raise TypeError(
                "concurrency is one number, a mapping per target or a Tuning. "
                "To compare several levels, use benchmark()."
            )
        if any(not isinstance(c, int) or c < 1 for c in levels.values()):
            raise ValueError("concurrency must be >= 1.")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "concurrency", levels)

    @property
    def min_gain(self) -> float | None:
        """Set by a benchmark plan only: stop when a stage gains less than this."""
        return None

    @cached_property
    def attempts(self) -> tuple[tuple[int | None, int], ...]:
        """``(seed, repeat)`` of every attempt of a case, in the order of their number."""
        seeds: Sequence[int | None] = self.seeds if self.seeds is not None else (None,)
        return tuple((seed, repeat) for seed in seeds for repeat in range(self.repeats))

    @property
    def attempts_per_case(self) -> int:
        """Seeds times repeats."""
        return len(self.attempts)

    @cached_property
    def cases(self) -> tuple[int, ...] | None:
        """Indices of the drawn cases, or None when all cases run."""
        if self.sample is None:
            return None
        pool = len(self.casebook) if self.limit is None else min(len(self.casebook), self.limit)
        return tuple(sorted(random.Random(self.sample_seed).sample(range(pool), min(self.sample, pool))))

    def stages(self, target: str) -> tuple[Stage, ...]:
        """The stages of one target; a run has exactly one."""
        return (Stage(self.concurrency[target], self.cases),)  # ty: ignore[not-subscriptable]

    @cached_property
    def _picked(self) -> dict[int, dict[str, Any]]:
        """Raw records of all drawn cases, read from the casebook in one pass."""
        wanted = {i for target in self.targets for stage in self.stages(target.name) for i in stage.cases or ()}
        return {r["index"]: r for r in self.casebook.records() if r["index"] in wanted}

    def records(self, stage: Stage) -> Iterator[dict[str, Any]]:
        """Raw case records of a stage, in the order they run."""
        if stage.cases is None:
            return islice(self.casebook.records(), self.case_count)
        return (self._picked[index] for index in stage.cases)

    @property
    def case_count(self) -> int:
        """Number of cases the plan runs, after ``limit`` and ``sample``."""
        if self.cases is not None:
            return len(self.cases)
        total = len(self.casebook)
        return total if self.limit is None else min(total, self.limit)

    @property
    def total_attempts(self) -> int:
        """Number of attempts over all targets, stages and attempts per case."""
        return self.attempts_per_case * sum(
            self.case_count if stage.cases is None else len(stage.cases)
            for target in self.targets
            for stage in self.stages(target.name)
        )

    @cached_property
    def spec(self) -> dict[str, Any]:
        """The plan as plain JSON; this is what the ledger stores."""
        return {
            "kind": "run",
            "casebook_id": self.casebook.id,
            "agent": agent_fingerprint(self.agent),
            "targets": [t.describe() for t in self.targets],
            "seeds": list(self.seeds) if self.seeds is not None else None,
            "repeats": self.repeats,
            "attempts_per_case": self.attempts_per_case,
            "concurrency": dict(self.concurrency),  # ty: ignore[no-matching-overload]
            "sequential_targets": self.sequential_targets,
            "case_count": self.case_count,
            "cases": list(self.cases) if self.cases is not None else None,
            "sample_seed": self.sample_seed if self.cases is not None else None,
            **({"tuning": self.tuning.describe()} if self.tuning else {}),
        }

    @cached_property
    def id(self) -> str:
        """Hash of the spec. The same plan always maps to the same journal, which is
        what makes an interrupted run resumable.
        """
        return sha256_hex(canonical_json(self.spec))

    def attempt_ids(self, shard: tuple[int, int] | None = None) -> Iterator[str]:
        """All planned attempt ids, or only those of ``shard = (i, n)``: cases with
        ``index % n == i``.
        """
        for target in self.targets:
            for stage in self.stages(target.name):
                for index in range(self.case_count) if stage.cases is None else stage.cases:
                    if shard is not None and index % shard[1] != shard[0]:
                        continue
                    for rep in range(self.attempts_per_case):
                        yield attempt_id(target.name, stage.concurrency, index, rep)
