"""Validate a setup step by step: benchmark, validation run on a sample, a report for each.

    uv run python examples/validate.py examples/targets.toml

The targets come from a TOML file, so no server address stands in this script.

1. Benchmark every target (throughput per concurrency level, a recommendation each).
2. Validation run: a random sample of the cases, with two seeds and two repeats per
   seed, on all targets side by side. That measures repeatability (same seed), seed
   robustness (other seed), reproducibility (same model on another target) and
   trueness (against the expected outputs).
3. A benchmark report per target and the run report with precision, trueness and accuracy.

For the full run afterwards, drop ``sample``, ``seeds`` and ``repeats`` from the plan:
every case then runs once, at the recommended concurrency. Keep ``seeds`` and
``repeats`` without ``sample`` to validate every case instead of a sample.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from pydantic_ai import Agent

from pydai_ledger import (
    Case,
    Casebook,
    Journal,
    Ledger,
    ReportHeader,
    RunPlan,
    benchmark,
    casefold,
    execute,
    load_targets,
    write_pdf,
    write_report,
)

CAPITALS = {"France": "Paris", "Italy": "Rome", "Spain": "Madrid", "Austria": "Vienna", "Norway": "Oslo"}

agent = Agent(
    instructions="Answer with the single word asked for.",
    model_settings={"temperature": 0, "seed": 1, "max_tokens": 16},
)


def cases() -> list[Case]:
    """Small questions with a known answer, so trueness can be measured."""
    return [
        Case(turns=[f"What is the capital of {country}? ({n})"], expected=[capital], key=f"{country}-{n}")
        for n in range(40)
        for country, capital in CAPITALS.items()
    ]


async def main(targets_file: Path, work: Path = Path("runs"), out: Path = Path("reports")) -> None:
    work.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    targets = load_targets(targets_file)
    book_path = work / "capitals.casebook.zip"
    with Casebook.open(book_path) if book_path.exists() else Casebook.write(book_path, cases()) as book:
        # 1. Benchmark: one ledger and one report per target; an existing ledger is reused.
        recommended = {}
        for target in targets:
            path = work / f"{target.name}.benchmark.zip"
            if not path.exists():
                with await benchmark(book, agent, target, path, parent=work / "journal"):
                    pass
            with Ledger.open(path) as measured:
                recommended[target.name] = measured.tuning().concurrency
                header = ReportHeader(f"Benchmark of {target.name}", authors=["Your name"], number="B-0001")
                write_pdf(measured, out / f"benchmark-{target.name}.pdf", header=header)
        for name, concurrency in recommended.items():
            print(f"{name}: recommended concurrency {concurrency}  (write it into {targets_file} to keep it)")

        # 2. Validation run on a sample. The same plan resumes if it is interrupted.
        plan = RunPlan(book, agent, targets, seeds=[1, 2], repeats=2, sample=50, concurrency=recommended)
        with Journal.open(plan, parent=work / "journal") as journal:
            await execute(plan, journal)
            ledger = journal.seal(work / "validation.ledger.zip", require_complete=False)

        # 3. The run report: targets, validation and the cases.
        with ledger:
            header = ReportHeader("Validation of the capitals setup", authors=["Your name"], number="V-0001")
            write_report(ledger, out / "validation.html", casebook=book, header=header, normalize=casefold)
            write_pdf(ledger, out / "validation.pdf", casebook=book, header=header, normalize=casefold)
    print(f"reports written to {out}")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1] if len(sys.argv) > 1 else "examples/targets.toml")))
