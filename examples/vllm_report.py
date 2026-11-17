"""Run a small image benchmark against a live vLLM server and write signed reports.

    uv run python examples/vllm_report.py vllm-local http://127.0.0.1:8000/v1 gemma4-26b-qat R-0002

Each case shows a plain coloured square and asks for its base colour, then for the
German word. First the server is benchmarked; then the run uses the recommended
concurrency, repeats every case once to check repeatability, is signed with a fresh key and ends in an HTML and a
PDF report per language. The benchmark has its own report. A fifth argument limits the
run to the first N cases, which helps on slow servers.
"""

from __future__ import annotations

import asyncio
import colorsys
import struct
import sys
import zlib
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from pydai_ledger import (
    Case,
    Casebook,
    Journal,
    Ledger,
    ReportHeader,
    RunPlan,
    Target,
    Turn,
    benchmark,
    casefold,
    execute,
    generate_key,
    write_pdf,
    write_report,
)

# hue in degrees -> (English, German)
HUES = {0: ("red", "rot"), 55: ("yellow", "gelb"), 120: ("green", "grün"), 225: ("blue", "blau")}


class Colour(BaseModel):
    """One colour word."""

    name: str


def png(rgb: tuple[int, int, int], size: int = 32) -> bytes:
    """A plain coloured PNG."""
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def cases(count: int) -> list[Case]:
    """Two-turn cases over shades of four hues; brightness and saturation vary with the index."""
    out = []
    hues = list(HUES)
    for i in range(count):
        hue = hues[i % len(hues)]
        english, german = HUES[hue]
        shade = colorsys.hsv_to_rgb(((hue + (i * 7) % 21 - 10) % 360) / 360, 0.55 + (i % 9) * 0.05, 0.5 + (i % 6) * 0.1)
        r, g, b = (round(v * 255) for v in shade)
        image = BinaryContent(png((r, g, b)), media_type="image/png")
        out.append(
            Case(
                turns=[
                    Turn(["Which base colour is this square: red, yellow, green or blue?", image]),
                    "And the German word for it, lower case?",
                ],
                expected=[Colour(name=english), Colour(name=german)],
                key=f"{english}-{i:03d}",
            )
        )
    return out


def main(
    name: str,
    url: str,
    model: str,
    number: str,
    limit: int | None = None,
    out: Path = Path("reports"),
) -> None:
    """Benchmark and run on one server (or reuse the stored ledgers) and write the reports into ``out``."""
    runs = Path("runs")  # casebook, key and ledgers stay here, so reports can be rebuilt without the server
    runs.mkdir(exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    key = runs / "report.key"
    if not key.exists():
        generate_key(key)
    book_path = runs / "colours.casebook.zip"
    book = Casebook.open(book_path) if book_path.exists() else Casebook.write(book_path, cases(120), signing_key=key)
    ledger_path = runs / f"{name}.ledger.zip"
    bench_path = runs / f"{name}.benchmark.zip"  # the benchmark ledger holds the recommendation
    if ledger_path.exists():
        ledger = Ledger.open(ledger_path)
    else:
        target = Target(name, OpenAIChatModel(model, provider=OpenAIProvider(base_url=url, api_key="EMPTY")))
        agent = Agent(
            instructions="Answer with the single colour word asked for.",
            model_settings={"temperature": 0, "seed": 7, "max_tokens": 64},
            output_type=NativeOutput(Colour),  # guided JSON; tool output drifts to plain text in later turns
        )
        if not bench_path.exists():
            with asyncio.run(
                benchmark(book, agent, target, bench_path, parent=runs / "journal", signing_key=key)
            ) as tune:
                for lang in ("en", "de"):
                    header = ReportHeader(authors=["pydai-ledger"], number=number)
                    write_report(tune.ledger, out / f"{name}-benchmark.{lang}.html", lang=lang, header=header)
                    write_pdf(tune.ledger, out / f"{name}-benchmark.{lang}.pdf", lang=lang, header=header)
        with Ledger.open(bench_path) as measured:
            tuning = measured.tuning()
        plan = RunPlan(book, agent, [target], repeats=2, concurrency=tuning, limit=limit)
        with Journal.open(plan, parent=runs / "journal") as journal:
            asyncio.run(execute(plan, journal))  # interrupted? run again, it resumes
            ledger = journal.seal(ledger_path, require_complete=False, signing_key=key, cleanup=True)
    print(ledger)
    titles = {"en": f"Colour naming on {name}", "de": f"Farbbenennung auf {name}"}
    for lang, title in titles.items():
        header = ReportHeader(title, authors=["pydai-ledger"], number=number)
        write_report(ledger, out / f"{name}.{lang}.html", casebook=book, lang=lang, header=header, normalize=casefold)
        write_pdf(ledger, out / f"{name}.{lang}.pdf", casebook=book, lang=lang, header=header, normalize=casefold)
    ledger.close()
    book.close()
    print(f"reports written to {out}")


if __name__ == "__main__":
    defaults = ["vllm-local", "http://127.0.0.1:8000/v1", "gemma4-26b-qat", "R-0002", "0"]
    args = [*sys.argv[1:], *defaults[len(sys.argv) - 1 :]]
    main(args[0], args[1], args[2], args[3], int(args[4]) or None)
