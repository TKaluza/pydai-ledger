import asyncio
import warnings

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from support import PNG, Answer, _always, make_cases, run

from pydai_ledger import (
    Case,
    Casebook,
    IncompleteRunError,
    Journal,
    Ledger,
    RunPlan,
    Target,
    execute,
)


def test_run_seal_and_analyse(tmp_path):
    book = Casebook.write(tmp_path / "book.zip", make_cases())
    agent = Agent(instructions="be brief", output_type=Answer)
    plan = RunPlan(
        casebook=book,
        agent=agent,
        targets=[Target("t1", TestModel()), Target("t2", TestModel())],
        attempts_per_case=2,
        concurrency=[1, 3],
    )
    assert plan.total_attempts == 4 * 2 * 2 * 2

    with Journal.open(plan, parent=tmp_path / "staging") as journal:
        run(execute(plan, journal, progress=False))
        ledger = journal.seal(tmp_path / "run.ledger.zip")
    assert not (tmp_path / "staging").exists() or not any((tmp_path / "staging").iterdir())

    assert ledger.manifest["complete"] and ledger.manifest["attempt_count"] == 32
    assert ledger.verify()
    assert ledger.manifest["plan"]["agent"]["instructions"] == ["be brief"]
    assert ledger.manifest["plan"]["casebook_id"] == book.id
    snapshots = ledger.events("target")
    assert {s["target"] for s in snapshots} == {"t1", "t2"}
    assert all(s["snapshot"]["provenance"] == "provider_only" for s in snapshots)

    attempts = list(ledger.attempts())
    multi = next(a for a in attempts if a["case_index"] == 1)
    assert [t["index"] for t in multi["turns"]] == [0, 1, 2]
    # input image is referenced, not copied into the ledger
    image_attempt = next(a for a in attempts if a["case_index"] == 2)
    part = image_attempt["turns"][0]["messages"][0]["parts"][0]["content"][1]
    assert "data" not in part and book.has_blob(part["blob"])
    assert ledger.manifest["blob_count"] == 0

    agreement = ledger.agreement()
    assert agreement.compared == 6 and agreement.rate == 1.0
    stages = ledger.throughput()
    assert len(stages) == 4 and all(s["attempts_ok"] == 8 and s["attempts_per_s"] for s in stages)

    reopened = Ledger.open(tmp_path / "run.ledger.zip")
    assert sum(1 for _ in reopened.rows()) == 2 * 2 * 2 * 6


def test_failure_resume_and_disagreement(tmp_path):
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=["x", "y"]), Case(turns=["z"])])
    state = {"fail": True, "calls": 0}

    def flaky(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        state["calls"] += 1
        last = messages[-1].parts[-1].content
        if state["fail"] and last == "y":
            raise RuntimeError("boom")
        return ModelResponse(parts=[TextPart(f"{last}-{state['calls'] if last == 'z' else 0}")])

    plan = RunPlan(
        casebook=book,
        agent=Agent(),
        targets=[Target("fn", FunctionModel(flaky))],
        attempts_per_case=2,
        concurrency=1,
    )
    journal = Journal.open(plan, parent=tmp_path / "staging")
    run(execute(plan, journal, progress=False))
    assert len(journal.done()) == 2  # only the "z" case succeeded
    with pytest.raises(IncompleteRunError):
        journal.seal(tmp_path / "run.zip")
    assert journal.directory.exists()

    # resume in a fresh journal object: only the failed attempts run again
    journal.close()
    state["fail"] = False
    calls_before = state["calls"]
    journal = Journal.open(plan, parent=tmp_path / "staging")
    run(execute(plan, journal, progress=False))
    assert state["calls"] - calls_before == 4  # 2 attempts x 2 turns
    ledger = journal.seal(tmp_path / "run.zip")
    assert ledger.manifest["attempt_count"] == 4
    assert all(a["status"] == "ok" for a in ledger.attempts())

    agreement = ledger.agreement()
    assert agreement.compared == 3 and agreement.agreeing == 2
    assert agreement.disagreements[0]["case_index"] == 1


def test_requests_warnings_and_environment(tmp_path):
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=["x"])])
    agent = Agent(output_type=Answer)

    @agent.tool_plain
    def add(x: int) -> int:
        return x

    targets = [
        Target("warm", TestModel(settings={"temperature": 0.7})),
        Target("fixed", TestModel(settings={"seed": 1, "temperature": 0})),
    ]
    plan = RunPlan(book, agent, targets)
    with Journal.open(plan, parent=tmp_path / "staging") as journal:
        with pytest.warns(UserWarning):
            asyncio.run(execute(plan, journal, progress=False))
        ledger = journal.seal(tmp_path / "run.zip")

    codes = {(w["target"], w["code"]) for w in ledger.warnings()}
    assert ("warm", "no_seed") in codes and ("warm", "temperature_nonzero") in codes
    assert ("fixed", "no_seed") not in codes and ("fixed", "temperature_nonzero") not in codes
    assert ("fixed", "provider_only") in codes

    turn = next(a for a in ledger.attempts() if a["target"] == "fixed")["turns"][0]
    assert len(turn["requests"]) == 2  # tool call, then final answer
    assert turn["requests"][0]["settings"] == {"seed": 1, "temperature": 0}
    parameters = {e["sha"]: e["parameters"] for e in ledger.events("request_parameters")}
    tools = parameters[turn["requests"][0]["parameters_sha"]]["function_tools"]
    assert [t["name"] for t in tools] == ["add"]

    env = ledger.events("environment")[0]
    assert env["python"] and env["packages"]["pydantic-ai-slim"] and env["packages"]["pydai-ledger"]
    assert next(ledger.rows())["model_request_s"] > 0


def test_abort_after_consecutive_failures(tmp_path):
    from pydai_ledger import RunAbortedError

    calls = {"n": 0}

    def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        raise RuntimeError("always broken")

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(50)])
    plan = RunPlan(
        book,
        Agent(),
        [Target("bad", FunctionModel(broken)), Target("good", FunctionModel(_always("ok")))],
        concurrency=1,
    )
    journal = Journal.open(plan, parent=tmp_path / "staging")
    with pytest.raises(RunAbortedError, match="3 attempts failed in a row"):
        run(execute(plan, journal, progress=False, max_consecutive_failures=3))
    assert calls["n"] == 3  # stopped early instead of failing all 50
    records = list(journal.records())
    assert sum(1 for r in records if r["kind"] == "attempt" and r["target"] == "good" and r["status"] == "ok") == 50
    assert [r["target"] for r in records if r["kind"] == "aborted"] == ["bad"]
    journal.close()


def test_abort_when_server_unreachable(tmp_path):
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from pydai_ledger import RunAbortedError

    model = OpenAIChatModel("none", provider=OpenAIProvider(base_url="http://127.0.0.1:9/v1", api_key="EMPTY"))
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(30)])
    plan = RunPlan(book, Agent(), [Target("dead", model)], concurrency=8)
    journal = Journal.open(plan, parent=tmp_path / "staging")
    with pytest.raises(RunAbortedError, match="first request failed"):
        run(execute(plan, journal, progress=False))
    attempts = [r for r in journal.records() if r["kind"] == "attempt"]
    assert len(attempts) == 1 and attempts[0]["error_kind"] == "api"
    journal.close()


def test_timeout_is_recorded(tmp_path):
    async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await asyncio.sleep(5)
        return ModelResponse(parts=[TextPart("late")])

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=["q"])])
    plan = RunPlan(book, Agent(), [Target("slow", FunctionModel(slow))])
    journal = Journal.open(plan, parent=tmp_path / "staging")
    run(execute(plan, journal, progress=False, timeout=0.1))
    attempt = next(r for r in journal.records() if r["kind"] == "attempt")
    assert attempt["status"] == "error" and attempt["error_kind"] == "timeout"
    assert attempt["duration_s"] < 2
    journal.close()


def test_shards_merge_into_one_ledger(tmp_path):
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(7)])
    plan = RunPlan(book, Agent(), [Target("t", FunctionModel(_always("ok")))], concurrency=2)
    for shard in range(3):
        with Journal.open(plan, parent=tmp_path / "staging", worker_id=f"w{shard}") as journal:
            run(execute(plan, journal, progress=False, shard=(shard, 3)))
            if shard < 2:
                with pytest.raises(IncompleteRunError):
                    journal.seal(tmp_path / "early.zip")
    with Journal.open(plan, parent=tmp_path / "staging", worker_id="sealer") as journal:
        ledger = journal.seal(tmp_path / "run.zip")
    attempts = list(ledger.attempts())
    assert sorted(a["case_index"] for a in attempts) == list(range(7))
    assert {a["worker_id"] for a in attempts} == {"w0", "w1", "w2"}
    assert ledger.throughput()[0]["attempts_ok"] == 7
    with pytest.raises(ValueError):
        run(execute(plan, Journal.open(plan, parent=tmp_path / "s2"), shard=(3, 3)))


def test_inputs_are_not_hashed_again(tmp_path, monkeypatch):
    import pydai_ledger._util as util

    book = Casebook.write(tmp_path / "book.zip", make_cases())
    hashed = []
    original = util.sha256_hex
    monkeypatch.setattr(util, "sha256_hex", lambda data: (hashed.append(len(data)), original(data))[1])
    plan = RunPlan(book, Agent(output_type=Answer), [Target("t", TestModel())])
    with Journal.open(plan, parent=tmp_path / "staging") as journal:
        run(execute(plan, journal, progress=False))
        ledger = journal.seal(tmp_path / "run.zip")
    assert len(PNG) not in hashed
    part = next(a for a in ledger.attempts() if a["case_index"] == 2)["turns"][0]["messages"][0]["parts"][0]["content"][
        1
    ]
    assert book.blob(part["blob"]) == PNG and part["media_type"] == "image/png"


def test_prefix_cache_warnings(tmp_path):
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=["q"])])
    model = TestModel(settings={"seed": 1, "temperature": 0})

    plan = RunPlan(book, Agent(), [Target("t", model)], attempts_per_case=2)
    with Journal.open(plan, parent=tmp_path / "staging") as journal:
        run(execute(plan, journal, progress=False))
        codes = {w["code"] for w in journal.seal(tmp_path / "a.zip").warnings()}
    assert "prefix_cache_not_reset" in codes

    plan = RunPlan(book, Agent(), [Target("t", model)])
    with Journal.open(plan, parent=tmp_path / "staging") as journal:
        run(execute(plan, journal, progress=False, reset_prefix_cache=True))
        ledger = journal.seal(tmp_path / "b.zip")
    assert {w["code"] for w in ledger.warnings()} == {"provider_only", "prefix_cache_reset_failed"}
    assert ledger.events("stage")[0]["prefix_cache_reset"] is False


def test_warmup_is_not_counted(tmp_path):
    calls = {"n": 0}

    def count(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        return ModelResponse(parts=[TextPart("ok")])

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(3)])
    plan = RunPlan(book, Agent(), [Target("t", FunctionModel(count))])
    with Journal.open(plan, parent=tmp_path / "staging") as journal:
        run(execute(plan, journal, progress=False, warmup=2))
        ledger = journal.seal(tmp_path / "run.zip")
    assert calls["n"] == 5
    assert ledger.manifest["attempt_count"] == 3
    assert ledger.events("warmup")[0]["attempts_ok"] == 2


def test_autotune(tmp_path):
    from pydai_ledger import autotune

    limit = 4  # the pretend server handles four requests at a time
    slots = asyncio.Semaphore(limit)

    async def serve(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        async with slots:
            await asyncio.sleep(0.05)
        return ModelResponse(parts=[TextPart("ok")])

    book = Casebook.write(tmp_path / "c.zip", [Case(turns=[f"q{i}"]) for i in range(32)])

    async def go():
        nonlocal slots
        slots = asyncio.Semaphore(limit)
        return await autotune(
            book,
            Agent(),
            Target("t", FunctionModel(serve)),
            tmp_path / "tune.zip",
            sample=32,
            parent=tmp_path,
            reset_prefix_cache=False,
            progress=False,
            min_gain=0.3,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run(go())
    measured = [level["concurrency"] for level in result.levels]
    assert result.best in (4, 8) and measured[-1] <= 16 and 32 not in measured
    assert result.ledger.manifest["complete"]  # skipped levels are not missing
    assert result.ledger.autotune()["t"]["skipped"]


def test_parquet_export(tmp_path):
    polars = pytest.importorskip("polars")
    book = Casebook.write(tmp_path / "c.zip", make_cases())
    plan = RunPlan(book, Agent(), [Target("t", TestModel())])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Journal.open(plan, parent=tmp_path) as journal:
            run(execute(plan, journal, progress=False))
            ledger = journal.seal(tmp_path / "l.zip")
    frame = polars.read_parquet(ledger.to_parquet(tmp_path / "rows.parquet"))
    assert frame.height == 6 and {"target", "turn", "output", "duration_s"} <= set(frame.columns)
