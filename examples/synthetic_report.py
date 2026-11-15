"""Write sample reports from a synthetic run; needs no server.

    uv run python examples/synthetic_report.py reports

Three pretend targets answer 240 cases at three concurrency levels. One target is
sometimes wrong, one sometimes fails, a few cases carry very long texts and an image,
so every part of the HTML and PDF reports has something to show.
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

from pydai_ledger import Case, Casebook, Journal, RunPlan, Target, Turn, casefold, execute, write_pdf, write_report

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


def model(delay: float, *, wrong_every: int = 0, fail_every: int = 0) -> FunctionModel:
    """A pretend server: echoes in capitals, slower with ``delay``, sometimes wrong or failing."""
    limit = asyncio.Semaphore(8)

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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
        return ModelResponse(
            parts=[TextPart(text.upper() + ("!" if wrong_every and number % wrong_every == 3 else ""))]
        )

    return FunctionModel(answer)


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
            Target("gpu-a-qwen3", model(0.004)),
            Target("gpu-b-qwen3", model(0.006, wrong_every=17)),
            Target("gpu-c-gemma", model(0.009, fail_every=60)),
        ]
        agent = Agent(instructions="Answer briefly.", model_settings={"temperature": 0, "seed": 1})
        plan = RunPlan(book, agent, targets, concurrency=[1, 2, 4, 8, 16])
        with Journal.open(plan, parent=Path(work) / "journal") as journal:
            asyncio.run(execute(plan, journal, progress=False, max_consecutive_failures=None))
            ledger = journal.seal(Path(work) / "run.zip", require_complete=False)
        write_report(
            ledger,
            out / "synthetic.en.html",
            casebook=book,
            title="Synthetic three-target sweep",
            author="pydai-ledger",
            number="R-0001",
        )
        write_report(
            ledger,
            out / "synthetic.de.html",
            casebook=book,
            lang="de",
            title="Künstlicher Lauf mit drei Targets",
            author="pydai-ledger",
            number="R-0001",
        )
        for style in ("plain", "ruled", "banded"):
            write_pdf(
                ledger,
                out / f"synthetic-{style}.de.pdf",
                casebook=book,
                lang="de",
                style=style,
                title="Künstlicher Lauf mit drei Targets",
                normalize=casefold,
                author="pydai-ledger",
                number="R-0001",
            )
        write_pdf(
            ledger,
            out / "synthetic-banded.en.pdf",
            casebook=book,
            style="banded",
            title="Synthetic three-target sweep",
            author="pydai-ledger",
            number="R-0001",
        )
        ledger.close()
        book.close()
    print(f"reports written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "reports"))
