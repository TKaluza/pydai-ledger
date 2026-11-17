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
        repeats=2,
        concurrency={"t1": 1, "t2": 3},
    )
    assert plan.total_attempts == 4 * 2 * 2
    with pytest.raises(TypeError, match="benchmark"):
        RunPlan(book, agent, [Target("t", TestModel())], concurrency=[1, 3])

    with Journal.open(plan, parent=tmp_path / "staging") as journal:
        run(execute(plan, journal, progress=False))
        ledger = journal.seal(tmp_path / "run.ledger.zip")
    assert not (tmp_path / "staging").exists() or not any((tmp_path / "staging").iterdir())

    assert ledger.manifest["complete"] and ledger.manifest["attempt_count"] == 16
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
    stages = ledger.throughput()  # one stage per target, each at its own concurrency
    assert [(s["target"], s["concurrency"], s["attempts_ok"]) for s in stages] == [("t1", 1, 8), ("t2", 3, 8)]
    assert ledger.kind == "run" and ledger.manifest["plan"]["concurrency"] == {"t1": 1, "t2": 3}

    reopened = Ledger.open(tmp_path / "run.ledger.zip")
    assert sum(1 for _ in reopened.rows()) == 2 * 2 * 6


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
        repeats=2,
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

    plan = RunPlan(book, Agent(), [Target("t", model)], repeats=2)
    with Journal.open(plan, parent=tmp_path / "staging") as journal:
        run(execute(plan, journal, progress=False))
        codes = {w["code"] for w in journal.seal(tmp_path / "a.zip").warnings()}
    assert "prefix_cache_not_reset" not in codes  # a run measures no throughput, so this is no concern

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


def test_attempts_run_with_their_seeds(tmp_path):
    def seeded(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seed = (info.model_settings or {}).get("seed")
        return ModelResponse(parts=[TextPart("same" if seed != 3 else "other")])

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=["q"]), Case(turns=["r"])])
    agent = Agent(model_settings={"temperature": 0, "seed": 99})
    plan = RunPlan(book, agent, [Target("t", FunctionModel(seeded))], seeds=[1, 2, 3])
    assert plan.attempts_per_case == 3 and plan.spec["seeds"] == [1, 2, 3]
    with pytest.raises(ValueError, match="each only once"):
        RunPlan(book, agent, [Target("t", FunctionModel(seeded))], seeds=[1, 1])
    with Journal.open(plan, parent=tmp_path / "staging") as journal:
        run(execute(plan, journal, progress=False))
        ledger = journal.seal(tmp_path / "run.zip")
    sent = {(a["rep"], a["seed"], a["turns"][0]["requests"][0]["settings"]["seed"]) for a in ledger.attempts()}
    assert sent == {(0, 1, 1), (1, 2, 2), (2, 3, 3)}  # the plan's seed replaces the agent's
    assert ledger.agreement().agreeing == 0

    from pydai_ledger.summary import build

    summary = build(ledger, book)
    attempts = next(b["table"] for b in summary["validation"] if (b.get("cap") or "").startswith("Attempts per case"))
    assert [row[1] for row in attempts["rows"]] == ["1", "2", "3"]
    assert [row[5] for row in attempts["rows"]] == ["–", "100.0 %", "0.0 %"]
    assert not summary["benchmark"] and not summary["show"]["benchmark"]
    assert summary["facts"][4][0] == "Attempts per minute" and summary["facts"][4][1].replace(",", "").isdigit()


def test_seeds_times_repeats_on_a_sample(tmp_path):
    calls = {"n": 0}

    def flaky(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Seed 2 changes the answer to q7; q3 changes on every call, whatever the seed."""
        calls["n"] += 1
        seed, text = (info.model_settings or {}).get("seed"), messages[-1].parts[-1].content
        return ModelResponse(
            parts=[TextPart(f"{text}{calls['n'] if text == 'q3' else ''}{'!' if (seed, text) == (2, 'q7') else ''}")]
        )

    cases = [Case(turns=[f"q{i}"], expected=[f"q{i}"]) for i in range(40)]
    book = Casebook.write(tmp_path / "book.zip", cases)
    target = Target("t", FunctionModel(flaky))
    plan = RunPlan(book, Agent(), [target], seeds=[1, 2], repeats=2, sample=10, sample_seed=5)
    assert plan.attempts == ((1, 0), (1, 1), (2, 0), (2, 1)) and plan.total_attempts == 40
    drawn = plan.cases
    assert drawn is not None and len(drawn) == 10 and list(drawn) == sorted(drawn) and drawn != tuple(range(10))
    assert RunPlan(book, Agent(), [target], sample=10, sample_seed=5).cases == drawn  # the draw is repeatable
    plan = RunPlan(book, Agent(), [target], seeds=[1, 2], repeats=2, sample=40)  # all cases, in sample form
    with Journal.open(plan, parent=tmp_path / "staging") as journal:
        run(execute(plan, journal, progress=False))
        ledger = journal.seal(tmp_path / "run.zip")
    assert {(a["seed"], a["repeat"]) for a in ledger.attempts()} == {(1, 0), (1, 1), (2, 0), (2, 1)}

    from pydai_ledger.summary import build

    summary = build(ledger, book)
    blocks = summary["validation"]
    assert [b["text"] for b in blocks if b["kind"] == "h2"] == ["Setup", "Precision", "Trueness", "Accuracy"]
    tables = [b["table"] for b in blocks if b["kind"] == "table"]
    conditions = {row[0]: row[3:] for row in tables[0]["rows"]}  # ISO 5725: what was compared, what was identical
    assert conditions["Repeatability"] == ["80", "97.5 %"]  # 2 of 80 same-seed groups differ: q3 under each seed
    assert conditions["Seed robustness (intermediate precision)"] == ["40", "95.0 %"]  # q3 and q7 of 40
    assert conditions["Reproducibility"] == ["0", "not tested"]  # one target only
    assert tables[1]["rows"][0] == ["t", "97.5 %", "80", "95.0 %", "40"]
    assert tables[-1]["head"][1:] == ["stable and right", "stable but wrong", "unstable"]
    assert tables[-1]["rows"][0][1:] == ["38", "0", "2"]
    assert "2 of 40" in summary["verdict"]

    subset = RunPlan(book, Agent(), [target], sample=10, sample_seed=5)
    with Journal.open(subset, parent=tmp_path / "staging") as journal:
        run(execute(subset, journal, progress=False))
        sampled = journal.seal(tmp_path / "sample.zip")
    assert sorted(a["case_index"] for a in sampled.attempts()) == list(drawn)
    summary = build(sampled, book)
    assert summary["case_ids"] == list(drawn) and [c["i"] for c in summary["cases"]["cases"]][-1] in drawn
    assert "Random sample of 10" in summary["validation"][1]["text"] and "cannot" not in summary["verdict"]


def test_targets_side_by_side_or_one_after_another(tmp_path):
    active = {"now": 0, "peak": 0}

    async def serve(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.01)
        active["now"] -= 1
        return ModelResponse(parts=[TextPart("ok")])

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(6)])
    targets = [Target("a", FunctionModel(serve)), Target("b", FunctionModel(serve))]
    peaks = {}
    for sequential in (False, True):
        active.update(now=0, peak=0)
        plan = RunPlan(book, Agent(), targets, concurrency=2, sequential_targets=sequential)
        with Journal.open(plan, parent=tmp_path / "staging") as journal:
            run(execute(plan, journal, progress=False))
            ledger = journal.seal(tmp_path / f"run{sequential}.zip")
        assert ledger.manifest["plan"]["sequential_targets"] is sequential
        peaks[sequential] = active["peak"]
    assert peaks == {False: 4, True: 2}


def test_benchmark(tmp_path):
    from pydai_ledger import benchmark
    from pydai_ledger.benchmark import ladder

    assert [level for level, _ in ladder(5000)] == [1, 2, 4, 8, 16, 32, 64, 128]  # never beyond 128
    assert ladder(170) == [(1, 32), (2, 32), (4, 40), (8, 80), (16, 160)]  # 32 would need 320 cases

    limit = 4  # the pretend server handles four requests at a time
    slots = asyncio.Semaphore(limit)

    async def serve(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        async with slots:
            await asyncio.sleep(0.02)
        return ModelResponse(parts=[TextPart("ok")])

    book = Casebook.write(tmp_path / "c.zip", [Case(turns=[f"q{i}"]) for i in range(400)])
    target = Target("t", FunctionModel(serve))

    async def go():
        nonlocal slots
        slots = asyncio.Semaphore(limit)
        return await benchmark(
            book, Agent(), target, tmp_path / "tune.zip", parent=tmp_path, reset_prefix_cache=False,
            progress=False, min_gain=0.3,
        )  # fmt: skip

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run(go())
    measured = [level["concurrency"] for level in result.levels]
    assert result.best in (4, 8) and measured[-1] <= 16 and 32 not in measured
    assert result.ledger.manifest["complete"]  # skipped levels are not missing
    assert result.ledger.tunings()["t"]["skipped"]
    assert result.ledger.kind == "benchmark"
    plan = result.ledger.manifest["plan"]
    drawn = plan["sample"]["order"]
    assert drawn != sorted(drawn) and len(set(drawn)) == len(drawn)  # random cases, each once
    assert {a["case_index"] for a in result.ledger.attempts() if a["concurrency"] == 1} == set(drawn[:32])

    from pydai_ledger.summary import build

    summary = build(result.ledger)
    assert (
        summary["benchmark"] and summary["curve"]["series"][0]["points"] and summary["facts"][2][1] == str(result.best)
    )
    assert summary["facts"][3][0] == "Attempts per minute" and "\u00a0" in summary["facts"][3][2]
    assert summary["tuning"]["rows"][0][:2] == ["t", str(result.best)] and " of " in summary["tuning"]["rows"][0][5]

    # the recommendation fits any target, without a warning; the plan notes where it comes from
    tuning = result.tuning
    assert tuning.concurrency == result.best and tuning.target == "t" and tuning.rate and tuning.attempts_per_min
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        other = RunPlan(book, Agent(), [Target("elsewhere", FunctionModel(_always("ok")))], concurrency=tuning)
    assert other.concurrency == {"elsewhere": result.best}
    assert other.spec["tuning"]["run_id"] == result.ledger.manifest["run_id"]
    with result:  # closes the benchmark ledger
        pass


def test_targets_file(tmp_path, monkeypatch):
    from pydai_ledger import load_targets

    file = tmp_path / "targets.toml"
    file.write_text(
        '[targets.gpu-a]\nmodel = "qwen3"\nbase_url = "http://gpu-a:8000/v1"\napi_key_env = "MY_KEY"\n'
        'concurrency = 32\ninfo = { gpu = "2x RTX 4090" }\n\n[targets.other]\nmodel = "test"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MY_KEY", "secret-value")
    a, other = load_targets(file)
    assert (a.name, a.concurrency, a.info["gpu"], a.root_url) == ("gpu-a", 32, "2x RTX 4090", "http://gpu-a:8000")
    assert a.describe()["model_name"] == "qwen3" and "secret-value" not in str(a.describe())
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=["q"])])
    assert RunPlan(book, Agent(), [a, other]).concurrency == {"gpu-a": 32, "other": 8}  # the file's number, else 8
    assert RunPlan(book, Agent(), [a, other], concurrency=2).concurrency == {"gpu-a": 2, "other": 2}
    file.write_text('[targets.x]\nmodel = "m"\napi_key = "oops"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="api_key"):
        load_targets(file)


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


def test_stage_of_two_servers_runs_again_before_the_stop_is_lifted(tmp_path):
    from pydai_ledger._util import now_iso

    calls = 0

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("ok")])

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=["q1"]), Case(turns=["q2"])])
    plan = RunPlan(book, Agent(), [Target("t", FunctionModel(answer))], concurrency=2)
    with Journal.open(plan, parent=tmp_path) as journal:
        run(execute(plan, journal, progress=False))
        # What execute() records when the server changed during the stage: all attempts are there.
        stop = {"kind": "aborted", "target": "t", "reason": "the server changed", "at": now_iso(), "void_stage": 2}
        journal.append(stop)
        with pytest.raises(IncompleteRunError, match="Target t stopped"):
            journal.seal(tmp_path / "run.zip")
        with journal.seal(tmp_path / "stopped.zip", require_complete=False, cleanup=False) as stopped:
            assert not stopped.manifest["complete"]
        # A new start alone does not lift the stop: the stage has to be measured again.
        journal.append({"kind": "target", "target": "t", "at": now_iso(), "snapshot": {}})
        with pytest.raises(IncompleteRunError, match="Target t stopped"):
            journal.seal(tmp_path / "run.zip")
        assert calls == 2
        run(execute(plan, journal, progress=False))
        assert calls == 4  # both attempts ran again, on the server as it is now
        with journal.seal(tmp_path / "run.zip") as ledger:
            assert ledger.manifest["complete"] and ledger.manifest["attempt_count"] == 2
            assert ledger.events("aborted")  # the stop stays on record


def test_benchmark_takes_no_figures_from_a_stopped_stage(tmp_path):
    from pydai_ledger import RunAbortedError, benchmark
    from pydai_ledger.benchmark import BenchPlan
    from pydai_ledger.plan import Stage

    healthy = False

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not healthy:
            raise RuntimeError("down")
        return ModelResponse(parts=[TextPart("ok")])

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(3)])
    target = Target("t", FunctionModel(answer))
    plan = BenchPlan(book, Agent(), [target], ladder=(Stage(1, (0, 1, 2)),), gain=0.05)
    with Journal.open(plan, parent=tmp_path) as journal:
        with pytest.raises(RunAbortedError):
            run(execute(plan, journal, progress=False, warmup=0, max_consecutive_failures=1))
        kinds = [record["kind"] for record in journal.records()]
        assert "stage" not in kinds and "tuning" not in kinds  # nothing to recommend from
        healthy = True
        run(execute(plan, journal, progress=False, warmup=0, max_consecutive_failures=1))
        with journal.seal(tmp_path / "bench.zip", require_complete=False) as ledger:
            level = ledger.tunings()["t"]["levels"][0]
            assert (level["concurrency"], level["cases"]) == (1, 3) and level["rate"] > 0

    healthy = False  # every attempt fails, nothing stops the benchmark: still no recommendation
    with pytest.raises(RuntimeError, match="no level with a successful attempt"):
        run(
            benchmark(
                book, Agent(), target, tmp_path / "none.zip", parent=tmp_path / "none", levels=[1], warmup=0,
                reset_prefix_cache=False, progress=False, max_consecutive_failures=None,
            )
        )  # fmt: skip
    assert not (tmp_path / "none.zip").exists()
    healthy = True  # the journal was kept; the stage without a success runs again
    again = run(
        benchmark(
            book, Agent(), target, tmp_path / "none.zip", parent=tmp_path / "none", levels=[1], warmup=0,
            reset_prefix_cache=False, progress=False, max_consecutive_failures=None,
        )
    )  # fmt: skip
    assert again.best == 1 and again.levels[0]["rate"] > 0
    again.close()


def test_benchmark_ranks_all_levels_in_one_unit(tmp_path):
    from pydantic_ai.usage import RequestUsage

    from pydai_ledger.benchmark import BenchPlan
    from pydai_ledger.plan import Stage

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        text = str(messages[-1].parts[-1].content)
        tokens = 0 if text in ("q4", "q5", "q6", "q7") else 50  # the second level reports no tokens
        return ModelResponse(parts=[TextPart("ok")], usage=RequestUsage(input_tokens=1, output_tokens=tokens))

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(8)])
    ladder = (Stage(1, (0, 1, 2, 3)), Stage(2, (4, 5, 6, 7)))
    plan = BenchPlan(book, Agent(), [Target("t", FunctionModel(answer))], ladder=ladder, gain=0.0)
    with Journal.open(plan, parent=tmp_path) as journal:
        run(execute(plan, journal, progress=False, warmup=0))
        tuning = next(record for record in journal.records() if record["kind"] == "tuning")
        walls = {record["concurrency"]: record["wall_s"] for record in journal.records() if record["kind"] == "stage"}
    assert tuning["metric"] == "attempts_per_s"
    for level in tuning["levels"]:  # both in attempts per second, not one of them in tokens
        assert level["rate"] == pytest.approx(4 / walls[level["concurrency"]])


def test_benchmark_measures_a_broken_off_stage_again(tmp_path):
    import json

    from pydai_ledger.benchmark import BenchPlan
    from pydai_ledger.plan import Stage

    calls = 0

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("ok")])

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(8)])
    ladder = (Stage(1, (0, 1, 2, 3)), Stage(2, tuple(range(8))))
    plan = BenchPlan(book, Agent(), [Target("t", FunctionModel(answer))], ladder=ladder, gain=0.0)
    with Journal.open(plan, parent=tmp_path) as journal:
        run(execute(plan, journal, progress=False, warmup=0))
        folder = journal.directory
    assert calls == 12
    for path in folder.glob("records-*.jsonl"):  # as if the call had died right after the last attempt of level 2
        kept = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        kept = [
            line
            for line in kept
            if (record := json.loads(line))["kind"] != "tuning"
            and not (record["kind"] == "stage" and record["concurrency"] == 2)
        ]
        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    with Journal.open(plan, parent=tmp_path) as journal:
        run(execute(plan, journal, progress=False, warmup=0))
        with journal.seal(tmp_path / "bench.zip", require_complete=False) as ledger:
            assert calls == 20  # level 2 ran again as a whole, level 1 did not
            assert [level["concurrency"] for level in ledger.tunings()["t"]["levels"]] == [1, 2]
            assert ledger.manifest["attempt_count"] == 12


def test_ledgers_of_0_1_are_refused(tmp_path):
    import json
    import zipfile

    with zipfile.ZipFile(tmp_path / "old.zip", "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": "pydai-ledger/ledger", "version": 1}))
    with pytest.raises(ValueError, match=r"0\.1\.1"):
        Ledger.open(tmp_path / "old.zip")
