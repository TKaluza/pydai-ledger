"""Write sample reports from a synthetic run; needs no server.

    uv run python examples/synthetic_report.py reports

Three pretend targets with invented models and museum GPUs go through the whole procedure. Each is benchmarked, then all
240 cases run with two seeds and two repeats per seed. Two targets serve the same
model, the third another one; one is sometimes wrong, one sometimes fails, one answers
differently for one seed and one is not quite repeatable. A few cases carry very long
texts and an image. So every part of the reports has something to show: the run report
(targets, validation, cases) and the benchmark report of each target.
"""

from __future__ import annotations

import asyncio
import struct
import sys
import tempfile
import warnings
import zlib
from pathlib import Path

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydai_ledger import (
    Case,
    Casebook,
    Journal,
    ReportHeader,
    RunPlan,
    Target,
    Turn,
    benchmark,
    casefold,
    execute,
    write_pdf,
    write_report,
)

LONG = " ".join(
    f"Clause {n}: the supplier delivers the goods described in annex {n % 7 + 1} within {n % 30 + 5} days."
    for n in range(1, 90)
)


def png(rgb: tuple[int, int, int], size: int = 48) -> bytes:
    """A plain coloured PNG."""
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def model(
    name: str, delay: float, *, wrong_every: int = 0, fail_every: int = 0, seed_quirk: int = 0, flaky_every: int = 0
) -> FunctionModel:
    """A pretend server for model ``name``: echoes in capitals, slower with ``delay``.

    It can be wrong for some cases (``wrong_every``), fail (``fail_every``), answer
    differently for seed 2 (``seed_quirk``) and differently every ``flaky_every``-th call.
    """
    limit = asyncio.Semaphore(8)
    calls = 0

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        async with limit:
            await asyncio.sleep(delay)
        part = messages[-1].parts[-1]
        content = part.content if isinstance(part, UserPromptPart) else ""
        text = content if isinstance(content, str) else str(content[0])
        number = int(text.split("#")[1].split()[0]) if "#" in text else 0
        if fail_every and number % fail_every == 5:
            raise RuntimeError("simulated: 503 server overloaded")
        if "Summarise" in text:
            reply = f"Summary of #{number}: " + " ".join(LONG.split()[: 120 + number % 40])
            return ModelResponse(parts=[TextPart(reply + (" (draft)" if wrong_every and number % 80 == 47 else ""))])
        suffix = "!" if wrong_every and number % wrong_every == 3 else ""
        if seed_quirk and (info.model_settings or {}).get("seed") == 2 and number % seed_quirk == 11:
            suffix += " (seed 2)"
        if flaky_every and calls % flaky_every == 0:
            suffix += " ~"
        return ModelResponse(parts=[TextPart(text.upper() + suffix)])

    return FunctionModel(answer, model_name=name)


def cases(count: int) -> list[Case]:
    """Short two-turn cases, with a long document every 40th and an image every 25th case."""
    out = []
    for i in range(count):
        if i % 40 == 7:
            turns: list[Turn | str] = [f"Summarise contract #{i} in one paragraph. {LONG}", f"shorter #{i}"]
            out.append(Case(turns=turns, key=f"contract-{i:03d}"))
        elif i % 25 == 3:
            image = BinaryContent(png((30 + i % 200, 90, 200 - i % 150)), media_type="image/png")
            out.append(
                Case(
                    turns=[Turn([f"describe scan #{i}", image])], expected=[f"DESCRIBE SCAN #{i}"], key=f"scan-{i:03d}"
                )
            )
        else:
            out.append(
                Case(turns=[f"question #{i}", f"follow-up #{i}"], expected=[f"QUESTION #{i}", None], key=f"row-{i:03d}")
            )
    return out


def main(out: Path) -> None:
    """Run the synthetic plan and write HTML and PDF reports into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("ignore")
    with tempfile.TemporaryDirectory() as work:
        book = Casebook.write(Path(work) / "book.zip", cases(240))
        targets = [
            Target("rig-a-lumen", model("lumen-7b", 0.004), {"gpu": "2x Voodoo2 (SLI)"}),
            Target("rig-b-lumen", model("lumen-7b", 0.006, wrong_every=17, seed_quirk=37), {"gpu": "1x GeForce 256"}),
            Target("rig-c-basalt", model("basalt-12b", 0.009, fail_every=60, flaky_every=211), {"gpu": "1x Riva TNT2"}),
        ]
        agent = Agent(instructions="Answer briefly.", model_settings={"temperature": 0, "seed": 1})

        async def measure_and_run():
            # Step 1, on its own: benchmark every target. Each has its own ledger and recommendation.
            tunes = [
                await benchmark(
                    book, agent, target, Path(work) / f"{target.name}.zip", parent=Path(work) / "bench", progress=False
                )
                for target in targets
            ]
            # Step 2: the validation run. Two seeds, two repeats each, on every case. With
            # sample=60 it would draw 60 cases instead. Concurrency comes from the benchmarks.
            plan = RunPlan(
                book,
                agent,
                targets,
                seeds=[1, 2],
                repeats=2,
                concurrency={tune.tuning.target or "": tune.best for tune in tunes},
            )
            with Journal.open(plan, parent=Path(work) / "journal") as journal:
                await execute(plan, journal, progress=False, max_consecutive_failures=None)
                return tunes, journal.seal(Path(work) / "run.zip", require_complete=False)

        tunes, ledger = asyncio.run(measure_and_run())
        authors = ["pydai-ledger", "examples/synthetic_report.py"]
        # The run report: targets, validation and cases.
        titles = (("en", "Synthetic validation on three targets"), ("de", "Künstliche Validierung auf drei Targets"))
        for lang, title in titles:
            header = ReportHeader(title, authors=authors, number="R-0001")
            write_report(ledger, out / f"synthetic.{lang}.html", casebook=book, lang=lang, header=header)
            write_pdf(
                ledger, out / f"synthetic.{lang}.pdf", casebook=book, lang=lang, header=header, normalize=casefold
            )
        # Every benchmark has its own report: throughput, latency and the recommendation.
        for number, tune in enumerate(tunes, start=1):
            name = tune.tuning.target
            for lang, title in (("en", f"Synthetic benchmark of {name}"), ("de", f"Künstlicher Benchmark von {name}")):
                header = ReportHeader(title, authors=authors, number=f"B-{number:04d}")
                write_report(tune.ledger, out / f"synthetic-benchmark-{name}.{lang}.html", lang=lang, header=header)
                write_pdf(tune.ledger, out / f"synthetic-benchmark-{name}.{lang}.pdf", lang=lang, header=header)
        for tune in tunes:
            tune.close()
        ledger.close()
        book.close()
    print(f"reports written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "reports"))
