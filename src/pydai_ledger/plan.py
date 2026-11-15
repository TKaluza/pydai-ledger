"""RunPlan: which cases run with which agent on which targets, how often, how parallel."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import pydantic_ai
from pydantic_ai import Agent
from pydantic_core import to_jsonable_python

from ._util import canonical_json, sha256_hex
from .casebook import Casebook
from .target import Target


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
class RunPlan:
    """A run is ``cases x targets x concurrency levels x attempts_per_case``.

    Several ``concurrency`` values form a sweep: every level runs the full case set so
    throughput can be compared. ``limit`` restricts the run to the first N cases.
    """

    casebook: Casebook
    agent: Agent[Any, Any]
    targets: Sequence[Target]
    attempts_per_case: int = 1
    concurrency: int | Sequence[int] = 8
    limit: int | None = None

    def __post_init__(self) -> None:
        targets = tuple(self.targets)
        names = [t.name for t in targets]
        if not targets or len(set(names)) != len(names):
            raise ValueError("RunPlan needs at least one target and unique target names.")
        levels = (self.concurrency,) if isinstance(self.concurrency, int) else tuple(self.concurrency)
        if not levels or any(c < 1 for c in levels) or self.attempts_per_case < 1:
            raise ValueError("concurrency levels and attempts_per_case must be >= 1.")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "concurrency", levels)

    @property
    def levels(self) -> tuple[int, ...]:
        """The concurrency levels as a tuple, also when a single number was given."""
        return (self.concurrency,) if isinstance(self.concurrency, int) else tuple(self.concurrency)

    @property
    def case_count(self) -> int:
        """Number of cases the plan runs, after ``limit``."""
        total = len(self.casebook)
        return total if self.limit is None else min(total, self.limit)

    @property
    def total_attempts(self) -> int:
        """Number of attempts over all targets, levels and repetitions."""
        return self.case_count * self.attempts_per_case * len(self.targets) * len(self.levels)

    @cached_property
    def spec(self) -> dict[str, Any]:
        """The plan as plain JSON; this is what the ledger stores."""
        return {
            "casebook_id": self.casebook.id,
            "agent": agent_fingerprint(self.agent),
            "targets": [t.describe() for t in self.targets],
            "attempts_per_case": self.attempts_per_case,
            "concurrency": list(self.levels),
            "case_count": self.case_count,
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
            for level in self.levels:
                for index in range(self.case_count):
                    if shard is not None and index % shard[1] != shard[0]:
                        continue
                    for rep in range(self.attempts_per_case):
                        yield attempt_id(target.name, level, index, rep)
