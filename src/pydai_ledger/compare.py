"""Read several ledgers of one casebook as a single run, for a report that compares them.

    with Comparison([Ledger.open("a.ledger.zip"), Ledger.open("b.ledger.zip")]) as both:
        write_report(both, "compare.html", casebook=book)

The runs may come from different days and machines. They must use the same casebook and
distinct target names. Nothing is written: the comparison is a view on sealed ledgers.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ._util import canonical_json, sha256_hex
from .ledger import Ledger


class Comparison(Ledger):
    """A read-only view that joins the targets, attempts and events of several ledgers."""

    def __init__(self, ledgers: Sequence[Ledger]) -> None:
        if len(ledgers) < 2:
            raise ValueError("A comparison needs at least two ledgers.")
        plans = [ledger.manifest["plan"] for ledger in ledgers]
        if len({plan["casebook_id"] for plan in plans}) != 1:
            raise ValueError("The ledgers were run on different casebooks and cannot be compared.")
        names = [target["name"] for plan in plans for target in plan["targets"]]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Target names must differ between the ledgers; repeated: {', '.join(repeated)}.")
        if len({canonical_json(plan["agent"]) for plan in plans}) != 1:
            warnings.warn(
                "The ledgers were run with different agents (instructions, settings or output "
                "schema); the report shows the agent of the first one.",
                stacklevel=2,
            )
        self.ledgers = list(ledgers)
        self.path = Path("+".join(ledger.path.name for ledger in ledgers))
        first = ledgers[0].manifest
        manifest: dict[str, Any] = {
            **first,
            "run_id": sha256_hex("+".join(ledger.manifest["run_id"] for ledger in ledgers).encode()),
            "plan": {
                **plans[0],
                "targets": [target for plan in plans for target in plan["targets"]],
                "concurrency": sorted({level for plan in plans for level in plan["concurrency"]}),
                "case_count": max(plan["case_count"] for plan in plans),
                "attempts_per_case": max(plan["attempts_per_case"] for plan in plans),
            },
            "events": [event for ledger in ledgers for event in ledger.manifest["events"]],
            "attempt_count": sum(ledger.manifest["attempt_count"] for ledger in ledgers),
            "complete": all(ledger.manifest["complete"] for ledger in ledgers),
            "sealed_at": max(ledger.manifest["sealed_at"] for ledger in ledgers),
            "members": {},
            "runs": [
                {
                    "run_id": ledger.manifest["run_id"],
                    "sealed_at": ledger.manifest["sealed_at"],
                    "targets": [target["name"] for target in ledger.manifest["plan"]["targets"]],
                    "stamp": ledger.stamp(),
                }
                for ledger in ledgers
            ],
        }
        self.manifest = manifest

    def attempts(self) -> Iterator[dict[str, Any]]:
        """All attempt records of all ledgers, one ledger after the other."""
        for ledger in self.ledgers:
            yield from ledger.attempts()

    def blob(self, sha: str) -> bytes:
        """A binary produced during one of the runs, by SHA-256."""
        for ledger in self.ledgers:
            try:
                return ledger.blob(sha)
            except KeyError:
                continue
        raise KeyError(sha)

    @property
    def signer(self) -> str | None:
        """The public key when every ledger carries a valid signature of the same key, else None."""
        signers = {ledger.signer for ledger in self.ledgers}
        return signers.pop() if len(signers) == 1 else None

    def stamp(self) -> str:
        """Identifier of the comparison: its id and a hash over the stamps of its ledgers.

        It carries no signature of its own; check each ledger by its own stamp, listed in
        ``manifest["runs"]`` and in the run details of a report.
        """
        joined = sha256_hex("\n".join(ledger.stamp() for ledger in self.ledgers).encode())
        return f"PYDAI-LEDGER:1:{self.manifest['run_id'][:16]}:{joined}".upper()

    def verify(self, public_key: str | None = None) -> bool:
        """Check every ledger, see :meth:`Ledger.verify`."""
        return all(ledger.verify(public_key) for ledger in self.ledgers)

    def close(self) -> None:
        """Close all ledgers."""
        for ledger in self.ledgers:
            ledger.close()

    def __repr__(self) -> str:
        """Name the joined ledgers."""
        return f"Comparison({', '.join(ledger.path.name for ledger in self.ledgers)})"
