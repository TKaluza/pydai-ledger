import warnings

import pytest
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from support import make_cases, run

from pydai_ledger import (
    Case,
    Casebook,
    Journal,
    RunPlan,
    Target,
    Turn,
    execute,
    write_report,
)


def test_report(tmp_path):
    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1].parts[-1].content
        return ModelResponse(parts=[TextPart(last if isinstance(last, str) else "<image>")])

    def shout(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1].parts[-1].content
        if last == "b":
            raise RuntimeError("no <b> today")
        return ModelResponse(parts=[TextPart(last.upper() if isinstance(last, str) else "<image>")])

    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    real_png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(b"\x00" + b"\xff\x00\x00" * 8 for _ in range(8))))
        + chunk(b"IEND", b"")
    )
    cases = [
        Case(turns=["a <script>"], expected=["a <script>"], key="first"),
        Case(turns=["b"]),
        Case(turns=[Turn(["look", BinaryContent(real_png, media_type="image/png")])]),
    ]
    book = Casebook.write(tmp_path / "book.zip", cases)
    plan = RunPlan(
        book,
        Agent(),
        [Target("plain", FunctionModel(answer)), Target("loud", FunctionModel(shout))],
        concurrency=[1, 2],
    )
    with Journal.open(plan, parent=tmp_path / "staging") as journal:
        run(execute(plan, journal, progress=False))
        ledger = journal.seal(tmp_path / "run.zip", require_complete=False)

    import base64
    import gzip
    import json

    def cases_of(html: str) -> dict:
        packed = html.split('<script id="cases-data" type="application/octet-stream">')[1].split("</script>")[0]
        return json.loads(gzip.decompress(base64.b64decode(packed)))

    html = write_report(ledger, tmp_path / "report.html", casebook=book, title="T").read_text(encoding="utf-8")
    assert "<script>" not in html.split("<main>")[1].split("</main>")[0]  # case content is never markup
    assert "no_seed" in html and "✖ incomplete" in html
    assert "http://" not in html and "https://" not in html  # self-contained
    payload = cases_of(html)
    by_index = {c["i"]: c for c in payload["cases"]}
    assert [c["i"] for c in payload["cases"]] == [1, 0, 2]  # failed first, then differing
    assert by_index[1]["f"] & 1 and "no <b> today" in by_index[1]["x"][0][1]
    assert by_index[0]["k"] == "first" and by_index[0]["f"] == 2 | 4  # differing and not as expected
    assert by_index[0]["t"][0]["p"] == ["a <script>"] and by_index[0]["t"][0]["e"] == '"a <script>"'
    assert payload["thumbs"][by_index[2]["t"][0]["m"][0]["u"]].startswith("data:image/jpeg;base64,")

    limited = cases_of(
        write_report(ledger, tmp_path / "few.html", casebook=book, max_cases=1, max_images=0).read_text(
            encoding="utf-8"
        )
    )
    assert [c["i"] for c in limited["cases"]] == [1] and "limited" in limited["note"]

    without = write_report(ledger, tmp_path / "plain.html").read_text(encoding="utf-8")
    assert "Pass the casebook" in without
    other = Casebook.write(tmp_path / "other.zip", [Case(turns=["x"])])
    with pytest.raises(ValueError):
        write_report(ledger, tmp_path / "bad.html", casebook=other)


def test_reports_in_german_and_pdf(tmp_path):
    pytest.importorskip("typst")
    book = Casebook.write(tmp_path / "c.zip", make_cases())
    plan = RunPlan(book, Agent(), [Target("t", TestModel())], attempts_per_case=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Journal.open(plan, parent=tmp_path) as journal:
            run(execute(plan, journal, progress=False, warmup=1))
            ledger = journal.seal(tmp_path / "l.zip")
    assert ledger.events("execution")[0]["options"]["warmup"] == 1

    german = write_report(ledger, tmp_path / "de.html", casebook=book, lang="de").read_text(encoding="utf-8")
    english = write_report(ledger, tmp_path / "en.html", casebook=book, embed_fonts=False).read_text(encoding="utf-8")
    assert 'lang="de"' in german and "Übereinstimmung" in german and "Laufbericht" in german
    assert "Agreement" in english and "@font-face" not in english and "@font-face" in german
    assert "<svg" in german.split("</header>")[0]  # the ledger stamp as QR code
    with pytest.raises(ValueError):
        write_report(ledger, tmp_path / "x.html", lang="fr")

    from pydai_ledger import write_pdf

    for lang in ("en", "de"):
        pdf = write_pdf(ledger, tmp_path / f"{lang}.pdf", casebook=book, lang=lang)
        assert pdf.read_bytes().startswith(b"%PDF") and pdf.stat().st_size > 20_000


def test_normalised_comparison(tmp_path):
    from pydai_ledger import casefold

    def model(text: str) -> FunctionModel:
        return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(text)]))

    book = Casebook.write(tmp_path / "c.zip", [Case(turns=["colour?"], expected=["Blue"])])
    plan = RunPlan(book, Agent(), [Target("a", model("Blue")), Target("b", model(" blue"))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Journal.open(plan, parent=tmp_path) as journal:
            run(execute(plan, journal, progress=False))
            ledger = journal.seal(tmp_path / "l.zip")
    assert ledger.agreement().agreeing == 0 and ledger.agreement(casefold).agreeing == 1

    from pydai_ledger.summary import build

    strict, relaxed = build(ledger, book), build(ledger, book, normalize=casefold)
    assert strict["cases"]["cases"][0]["f"] == 2 | 4 and relaxed["cases"]["cases"][0]["f"] == 0
    assert strict["facts"][3][1] == "50.0 %" and relaxed["facts"][3][1] == "100.0 %"
