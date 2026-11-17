"""ReportHeader: who wrote a report, its title, number and date."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReportHeader:
    """What heads a report. The same header fits the run report and the benchmark
    report, in HTML and PDF::

        header = ReportHeader("Colour recognition", authors=["Tim", "Kim"], number="R-2026-007")
        write_pdf(ledger, "run.pdf", header=header)

    ``number`` is your own report or run number. ``date`` defaults to the time the
    ledger was sealed, ``title`` to the kind of report.
    """

    title: str | None = None
    authors: Sequence[str] | str = ()
    number: str | None = None
    date: str | None = None
    organisation: str | None = None

    @property
    def names(self) -> list[str]:
        """The authors as a list, also when a single name was given."""
        return [self.authors] if isinstance(self.authors, str) else list(self.authors)
