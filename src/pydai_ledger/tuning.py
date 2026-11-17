"""Tuning: the concurrency a benchmark recommends for a target."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Tuning:
    """The recommended concurrency and where it was measured.

    It is a recommendation, nothing more: a :class:`~pydai_ledger.RunPlan` takes it for
    any target, also for another server or provider than the one it was measured on,
    without a warning. ``model_name`` and ``system`` (the provider) say where it comes
    from. It lives in the benchmark ledger (:meth:`Ledger.tuning`); to keep the number,
    write it as ``concurrency`` into your targets file.
    """

    concurrency: int
    metric: str = "output_tokens_per_s"
    rate: float | None = None
    attempts_per_min: float | None = None
    levels: list[dict[str, Any]] = field(default_factory=list)
    target: str | None = None
    model_name: str | None = None
    system: str | None = None
    casebook_id: str | None = None
    run_id: str | None = None
    measured_at: str | None = None

    def describe(self) -> dict[str, Any]:
        """Where the recommendation comes from, without the measured levels."""
        return {k: v for k, v in asdict(self).items() if k != "levels"}
