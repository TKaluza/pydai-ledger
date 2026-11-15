import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from support import PNG, Answer, make_cases, run

from pydai_ledger import (
    Case,
    Casebook,
    Journal,
    RunPlan,
    Target,
    Turn,
    execute,
    open_bundle,
    pack,
)


def test_casebook_roundtrip(tmp_path):
    book = Casebook.write(tmp_path / "book.zip", make_cases(), shard_size=3)
    assert len(book) == 4
    assert book.manifest["blob_count"] == 1  # the image is stored once
    assert len(book.manifest["shards"]) == 2
    stored = list(book)
    assert stored[1].case.expected == ({"text": "a"}, None, {"text": "c"})
    assert stored[1].case.key == "multi"
    assert stored[2].case.turns[0].content[1].data == PNG
    assert stored[3].case.metadata == {"page": 2}
    # ids depend on content only and are stable across writes
    again = Casebook.write(tmp_path / "again.zip", make_cases())
    assert again.id == book.id
    assert stored[2].id != stored[3].id


def test_rejects_urls():
    from pydantic_ai import ImageUrl

    with pytest.raises(TypeError):
        Turn(["x", ImageUrl("https://example.com/a.png")])
    with pytest.raises(ValueError):
        Case(turns=["a"], expected=[1, 2])


def test_pack_and_open_bundle(tmp_path):
    book = Casebook.write(tmp_path / "book.zip", make_cases())
    ledgers = []
    for reps in (1, 2):  # two different plans = two runs
        plan = RunPlan(book, Agent(output_type=Answer), [Target("t", TestModel())], attempts_per_case=reps)
        with Journal.open(plan, parent=tmp_path / "staging") as journal:
            run(execute(plan, journal, progress=False))
            ledgers.append(journal.seal(tmp_path / f"run{reps}.zip"))

    bundle = pack(book, ledgers, tmp_path / "all.bundle.zip")
    packed_book, packed_ledgers = open_bundle(bundle)
    assert packed_book.id == book.id
    assert list(packed_book)[2].case.turns[0].content[1].data == PNG
    assert [ledger_.manifest["attempt_count"] for ledger_ in packed_ledgers] == [4, 8]
    assert all(ledger_.verify() for ledger_ in packed_ledgers)

    other = Casebook.write(tmp_path / "other.zip", [Case(turns=["different"])])
    with pytest.raises(ValueError):
        pack(other, ledgers, tmp_path / "bad.zip")
