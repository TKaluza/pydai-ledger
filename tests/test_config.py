"""Configuration file, determinism points and note levels."""

import sys

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from support import _always, make_cases, run

from pydai_ledger import (
    Case,
    Casebook,
    Comparison,
    Config,
    DeterminismError,
    IncompleteRunError,
    Journal,
    RunPlan,
    Target,
    determinism,
    execute,
    write_report,
)
from pydai_ledger.target import REDACTED, probe, redact

execute_module = sys.modules["pydai_ledger.execute"]

SERVER = {
    "version": {"version": "0.33.0"},
    "server_info": {
        "vllm_config": "model='/m', seed=0, enable_prefix_caching=True, tensor_parallel_size=1, dtype=torch.bfloat16",
        "vllm_env": {"VLLM_BATCH_INVARIANT": False, "VLLM_ENABLE_V1_MULTIPROCESSING": True},
        "system_env": {"nvidia_gpu_models": "GPU X"},
    },
}


def test_load(tmp_path):
    assert Config.load().require == ()
    file = tmp_path / "pydai-ledger.toml"
    file.write_text(
        '[report.notes]\nno_seed = "error"\nprovider_only = "off"\n'
        '[report.remarks]\nprovider_only = "gateway"\n'
        '[determinism]\nrequire = ["request_seed"]\non_missing = "warning"\n',
        encoding="utf-8",
    )
    config = Config.load(file)
    assert config.level("no_seed") == "error" and config.level("provider_only") == "off"
    assert config.level("prefix_cache_not_reset") == "info" and config.level("determinism_request_seed") == "warning"
    assert config.require == ("request_seed",) and config.remarks == {"provider_only": "gateway"}
    file.write_text('[report.notes]\nno_seed = "red"\n[determinism]\nrequire = ["nope"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"no_seed.*nope"):
        Config.load(file)
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "missing.toml")


def test_determinism_points():
    points = determinism({"temperature": 0, "seed": 3}, {"a": SERVER, "b": {**SERVER, "version": {"version": "0.32"}}})
    a = points["a"]
    assert a["temperature_zero"][0] == a["request_seed"][0] == a["server_seed"][0] == a["single_gpu"][0] == "on"
    assert a["batch_invariant"][0] == a["prefix_cache_neutral"][0] == a["multiprocessing_off"][0] == "off"
    assert a["same_server_version"][0] == "off"
    reset = determinism({}, {"a": SERVER}, reset_prefix_cache=True)["a"]
    assert reset["prefix_cache_neutral"][0] == "on" and reset["temperature_zero"][0] == "off"
    assert reset["same_server_version"][0] == "n/a"
    assert determinism({}, {"a": {}})["a"]["batch_invariant"][0] == "unknown"


def test_required_points_stop_the_run_and_levels_shape_the_report(tmp_path):
    book = Casebook.write(tmp_path / "book.zip", make_cases())
    agent = Agent(model_settings={"temperature": 0, "seed": 1})
    plan = RunPlan(book, agent, [Target("fake", FunctionModel(_always("x")))])
    strict = tmp_path / "strict.toml"
    strict.write_text('[determinism]\nrequire = ["request_seed", "batch_invariant"]\n', encoding="utf-8")
    with Journal.open(plan, parent=tmp_path / "strict") as journal:
        with pytest.raises(DeterminismError, match="batch_invariant \\(fake: unknown\\)"):
            run(execute(plan, journal, progress=False, config=strict))
        assert not journal.done()

    relaxed = tmp_path / "relaxed.toml"
    relaxed.write_text(
        '[report.notes]\nprovider_only = "off"\n[report.remarks]\ndeterminism_batch_invariant = "known gap"\n'
        '[determinism]\nrequire = ["request_seed", "batch_invariant"]\non_missing = "warning"\n',
        encoding="utf-8",
    )
    with Journal.open(plan, parent=tmp_path / "relaxed") as journal:
        run(execute(plan, journal, progress=False, config=relaxed))
        ledger = journal.seal(tmp_path / "run.zip")
    assert ledger.events("config")[0]["require"] == ["request_seed", "batch_invariant"]

    html = write_report(ledger, tmp_path / "r.html", config=relaxed).read_text(encoding="utf-8")
    assert 'class="warning"><b>⚠ determinism_batch_invariant' in html and "known gap" in html
    assert "1 notes hidden by the configuration: provider_only" in html
    assert "<b>⚠ provider_only" not in html
    plain = write_report(ledger, tmp_path / "p.html", config=Config()).read_text(encoding="utf-8")
    assert 'class="warning"><b>⚠ provider_only' in plain and "determinism_batch_invariant" not in plain
    ledger.close()
    book.close()


def test_comparison_joins_two_runs_in_one_report(tmp_path):
    book = Casebook.write(tmp_path / "book.zip", make_cases())
    agent = Agent(model_settings={"temperature": 0, "seed": 1})
    ledgers = []
    for name, text in (("left", "x"), ("right", "y")):
        plan = RunPlan(book, agent, [Target(name, FunctionModel(_always(text)))])
        with Journal.open(plan, parent=tmp_path / name) as journal:
            run(execute(plan, journal, progress=False, config=Config()))
            ledgers.append(journal.seal(tmp_path / f"{name}.zip"))
    with pytest.raises(ValueError, match="repeated: left"):
        Comparison([ledgers[0], ledgers[0]])
    with Comparison(ledgers) as both:
        assert [t["name"] for t in both.manifest["plan"]["targets"]] == ["left", "right"]
        assert sum(1 for _ in both.attempts()) == sum(ledger.manifest["attempt_count"] for ledger in ledgers)
        assert both.verify() and both.agreement().rate == 0
        html = write_report(both, tmp_path / "both.html", casebook=book, config=Config()).read_text(encoding="utf-8")
        assert "comparison of 2 runs" in html and "Runs in this comparison" in html
        assert html.count(ledgers[1].stamp()) == 1
    book.close()


def test_secrets_in_server_info_are_redacted(monkeypatch):
    info = {
        "vllm_env": {"VLLM_API_KEY": "sk-abc", "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD": 64, "VLLM_USE_FASTOKENS": "x"},
        "system_env": {
            "env_vars": "HF_TOKEN=hf_abc\nAWS_SECRET_ACCESS_KEY='s3cr3t'\nMAX_TOKEN_COUNT=12\nCUDA_HOME=/usr"
        },
        "vllm_config": "model='/m', api_key=abc, seed=0, hf_token=None",
    }
    clean = redact(info)
    assert clean["vllm_env"] == {
        "VLLM_API_KEY": REDACTED,
        "VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD": 64,
        "VLLM_USE_FASTOKENS": "x",
    }
    env = clean["system_env"]["env_vars"]
    assert f"HF_TOKEN={REDACTED}\nAWS_SECRET_ACCESS_KEY={REDACTED}\nMAX_TOKEN_COUNT=12\nCUDA_HOME=/usr" == env
    assert clean["vllm_config"] == f"model='/m', api_key={REDACTED}, seed=0, hf_token=None"

    target = Target("fake", FunctionModel(_always("x")))

    async def answer(self, path):
        return info if path == "/server_info" else None

    monkeypatch.setattr(Target, "_get_json", answer)
    snapshot = run(probe(target))
    assert "sk-abc" not in str(snapshot) and "hf_abc" not in str(snapshot)


def test_changed_server_stops_the_target(tmp_path, monkeypatch):
    book = Casebook.write(tmp_path / "book.zip", make_cases())
    calls = 0

    def answer(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("x")])

    target = Target("fake", FunctionModel(answer))
    plan = RunPlan(book, Agent(), [target])
    models = {"data": [{"id": "m", "root": "/m", "max_model_len": 4096}]}
    versions = iter(["0.34.0", "0.33.0"])  # changed after the first call, as probed after the second

    async def snapshot(_target):
        return {"server_info": None, "version": {"version": "0.33.0"}, "models": models, "provenance": "provider_only"}

    async def now(_target):
        return {"version": next(versions), "models": sorted([["m", "/m", "4096"]])}

    monkeypatch.setattr(Target, "root_url", property(lambda self: "http://server"))
    monkeypatch.setattr(execute_module, "probe", snapshot)
    monkeypatch.setattr(execute_module, "current_identity", now)
    with Journal.open(plan, parent=tmp_path / "j") as journal:
        with pytest.raises(Exception, match="server changed during the run"):
            run(execute(plan, journal, progress=False, probe_capabilities=False))
        with pytest.raises(IncompleteRunError, match="Target fake stopped"):
            journal.seal(tmp_path / "run.zip")  # every attempt is there, the run is not complete all the same
        with journal.seal(tmp_path / "stopped.zip", require_complete=False, cleanup=False) as stopped:
            assert "0.34.0" in stopped.events("aborted")[0]["reason"] and not stopped.manifest["complete"]
            assert stopped.manifest["attempt_count"] == 4  # the attempts before the change are kept
            assert not stopped.events("stage")  # a stage of two servers has no measurement
        made = calls
        run(execute(plan, journal, progress=False, probe_capabilities=False))
        assert calls == 2 * made  # the whole stage ran again
        ledger = journal.seal(tmp_path / "run.zip")
    assert ledger.manifest["complete"] and ledger.manifest["attempt_count"] == 4 and len(ledger.events("stage")) == 1
    ledger.close()
    book.close()


def test_comparison_keeps_seeds_and_repeats_apart(tmp_path):
    from pydai_ledger.summary import build

    def answer(messages, info):
        return ModelResponse(parts=[TextPart(f"seed {(info.model_settings or {}).get('seed')}")])

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=["q"])])

    def ledger(name: str, **options):
        plan = RunPlan(book, Agent(), [Target(name, FunctionModel(answer, model_name="m"))], **options)
        with Journal.open(plan, parent=tmp_path / name) as journal:
            run(execute(plan, journal, progress=False, config=Config()))
            return journal.seal(tmp_path / f"{name}.zip")

    with Comparison([ledger("a", seeds=[1, 2]), ledger("b", seeds=[1], repeats=2)]) as both:
        blocks = build(both, config=Config())["validation"]
        tables = [b["table"] for b in blocks if b["kind"] == "table"]
    # every run says what it did itself
    assert blocks[1]["text"].startswith("a:") and "Seeds: 1, 2  |  Repeats per seed: 1" in blocks[1]["text"]
    assert blocks[2]["text"].startswith("b:") and "Seeds: 1  |  Repeats per seed: 2" in blocks[2]["text"]
    # a ran two seeds once each, b one seed twice: neither counts as the other
    assert tables[1]["rows"] == [["a", "–", "0", "0.0 %", "1"], ["b", "100.0 %", "1", "–", "0"]]


class FakeServer:
    """A target that claims to be a real server whose version the test can change."""

    def __init__(self, monkeypatch, version: str = "A") -> None:
        self.version, self.calls, self.fail_next = version, 0, 0
        models = {"data": [{"id": "m", "root": "/m", "max_model_len": 4096}]}

        async def snapshot(_target):
            return {"server_info": None, "version": {"version": self.version}, "models": models, "provenance": "x"}

        async def now(_target):
            return {"version": self.version, "models": sorted([["m", "/m", "4096"]])}

        monkeypatch.setattr(Target, "root_url", property(lambda _self: "http://server"))
        monkeypatch.setattr(execute_module, "probe", snapshot)
        monkeypatch.setattr(execute_module, "current_identity", now)

    def answer(self, messages, info):
        self.calls += 1
        if self.fail_next:
            self.fail_next -= 1
            raise RuntimeError("down")
        return ModelResponse(parts=[TextPart(f"from {self.version}")])

    def target(self, name: str = "t") -> Target:
        return Target(name, FunctionModel(self.answer))


def outputs(ledger) -> set[str]:
    return {turn["output"] for attempt in ledger.attempts() for turn in attempt["turns"]}


def test_attempts_of_a_changed_server_run_again(tmp_path, monkeypatch):
    server = FakeServer(monkeypatch)
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(4)])
    plan = RunPlan(book, Agent(), [server.target()], concurrency=1)
    with Journal.open(plan, parent=tmp_path) as journal:
        run(execute(plan, journal, progress=False, probe_capabilities=False))
        assert server.calls == 4
        # The server changes between two calls; the first new attempt fails and stops the target.
        server.version, server.fail_next = "B", 1
        with pytest.raises(Exception, match="stopped"):
            run(execute(plan, journal, progress=False, probe_capabilities=False, max_consecutive_failures=1))
        with pytest.raises(IncompleteRunError):
            journal.seal(tmp_path / "run.zip")
        before = server.calls
        run(execute(plan, journal, progress=False, probe_capabilities=False))
        assert server.calls == before + 4  # every case again, not only the one that failed
        with journal.seal(tmp_path / "run.zip") as ledger:
            assert ledger.manifest["complete"] and outputs(ledger) == {"from B"}


def test_a_later_repeat_by_another_worker_wins(tmp_path, monkeypatch):
    server = FakeServer(monkeypatch)
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(4)])
    plan = RunPlan(book, Agent(), [server.target()])
    with Journal.open(plan, parent=tmp_path, worker_id="worker-z") as journal:  # its file is read last
        run(execute(plan, journal, progress=False, probe_capabilities=False))
    server.version = "B"
    with Journal.open(plan, parent=tmp_path, worker_id="worker-a") as journal:
        run(execute(plan, journal, progress=False, probe_capabilities=False))
        assert server.calls == 8
        with journal.seal(tmp_path / "run.zip") as ledger:
            assert ledger.manifest["attempt_count"] == 4 and outputs(ledger) == {"from B"}


def test_shards_on_different_servers_do_not_make_one_run(tmp_path, monkeypatch):
    server = FakeServer(monkeypatch)
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(4)])
    plan = RunPlan(book, Agent(), [server.target()])
    with Journal.open(plan, parent=tmp_path, worker_id="w0") as journal:
        run(execute(plan, journal, progress=False, probe_capabilities=False, shard=(0, 2)))
    server.version = "B"
    with Journal.open(plan, parent=tmp_path, worker_id="w1") as journal:
        run(execute(plan, journal, progress=False, probe_capabilities=False, shard=(1, 2)))
        with pytest.raises(IncompleteRunError, match="different servers"):
            journal.seal(tmp_path / "run.zip")  # every attempt is there, half of them from A
    with Journal.open(plan, parent=tmp_path, worker_id="w0") as journal:
        run(execute(plan, journal, progress=False, probe_capabilities=False, shard=(0, 2)))
        assert server.calls == 6  # the first shard ran again
        with journal.seal(tmp_path / "run.zip") as ledger:
            assert ledger.manifest["complete"] and outputs(ledger) == {"from B"}


def test_comparison_lists_the_cases_of_every_run(tmp_path):
    from pydai_ledger.summary import build

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(8)])
    ledgers = []
    for name, options in (("a", {"limit": 1}), ("b", {"sample": 1, "sample_seed": 3, "repeats": 2})):
        plan = RunPlan(book, Agent(), [Target(name, FunctionModel(_always("x"), model_name="m"))], **options)
        with Journal.open(plan, parent=tmp_path / name) as journal:
            run(execute(plan, journal, progress=False, config=Config()))
            ledgers.append(journal.seal(tmp_path / f"{name}.zip"))
    drawn = ledgers[1].manifest["plan"]["cases"][0]
    assert drawn != 0
    with Comparison(ledgers) as both:
        summary = build(both, config=Config())
    assert summary["facts"][0][1] == "2" and summary["case_ids"] == [0, drawn]  # not only the first run's case
    assert [case["i"] for case in summary["cases"]["cases"]] == [0, drawn]


def test_failed_cache_reset_is_not_reported_as_done(tmp_path, monkeypatch):
    from pydai_ledger.summary import build

    async def snapshot(_target):
        return {**SERVER, "models": None, "provenance": "server_info"}

    async def no_reset(_target):
        return False

    monkeypatch.setattr(execute_module, "probe", snapshot)
    monkeypatch.setattr(execute_module, "reset_prefix_cache", no_reset)
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=["q"])])
    plan = RunPlan(book, Agent(), [Target("t", FunctionModel(_always("x")))])
    strict = Config(require=("prefix_cache_neutral",))
    with Journal.open(plan, parent=tmp_path / "strict") as journal:
        with pytest.raises(DeterminismError, match="could not be reset"):
            run(execute(plan, journal, progress=False, config=strict, reset_prefix_cache=True))
        assert not journal.done()  # nothing ran with the cache in an unknown state
    relaxed = Config(require=("prefix_cache_neutral",), on_missing="warning")
    with Journal.open(plan, parent=tmp_path / "relaxed") as journal:
        run(execute(plan, journal, progress=False, config=relaxed, reset_prefix_cache=True))
        with journal.seal(tmp_path / "run.zip") as ledger:
            summary = build(ledger, config=relaxed)
    cache = next(block for block in summary["validation"] if block["kind"] == "det")["items"]
    assert next(item for item in cache if "cache" in item["label"].lower())["state"] == "off"


def test_binaries_of_replaced_attempts_stay_out_of_the_ledger(tmp_path, monkeypatch):
    from pydantic_ai import BinaryImage
    from pydantic_ai.messages import FilePart
    from pydantic_ai.profiles import ModelProfile
    from support import PNG

    server = FakeServer(monkeypatch)

    def answer(messages, info):
        if server.version == "A":
            return ModelResponse(parts=[FilePart(content=BinaryImage(data=PNG, media_type="image/png"))])
        return ModelResponse(parts=[TextPart("from B")])

    book = Casebook.write(tmp_path / "book.zip", [Case(turns=["draw"])])
    model = FunctionModel(answer, profile=ModelProfile(supports_image_output=True))
    plan = RunPlan(book, Agent(output_type=BinaryImage | str), [Target("t", model)])
    with Journal.open(plan, parent=tmp_path) as journal:
        run(execute(plan, journal, progress=False, probe_capabilities=False))
        assert len(list((journal.directory / "blobs").iterdir())) == 1  # the image from A
        server.version = "B"
        run(execute(plan, journal, progress=False, probe_capabilities=False))
        with journal.seal(tmp_path / "run.zip", cleanup=False) as ledger:
            assert outputs(ledger) == {"from B"}
            assert ledger.manifest["blob_count"] == 0  # the image belongs to an attempt that is not in the ledger
            assert not [name for name in ledger._zf.namelist() if name.startswith("blobs/")]
            assert ledger.verify()


def test_report_names_the_server_the_results_came_from(tmp_path, monkeypatch):
    from pydai_ledger.summary import build

    server = FakeServer(monkeypatch)
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=["q"])])
    plan = RunPlan(book, Agent(), [server.target()])
    with Journal.open(plan, parent=tmp_path, worker_id="w0") as journal:
        run(execute(plan, journal, progress=False, probe_capabilities=False, shard=(0, 2)))
    server.version = "B"  # a second worker probes another server, but no case is its own
    with Journal.open(plan, parent=tmp_path, worker_id="w1") as journal:
        run(execute(plan, journal, progress=False, probe_capabilities=False, shard=(1, 2)))
        assert server.calls == 1
        with journal.seal(tmp_path / "run.zip") as ledger:
            assert outputs(ledger) == {"from A"}
            summary = build(ledger, config=Config())
    assert summary["snapshots"]["t"]["version"] == {"version": "A"}


def test_benchmark_does_not_return_the_recommendation_of_a_gone_server(tmp_path, monkeypatch):
    from pydai_ledger import benchmark

    server = FakeServer(monkeypatch)
    book = Casebook.write(tmp_path / "book.zip", [Case(turns=[f"q{i}"]) for i in range(3)])
    options = {"parent": tmp_path / "j", "levels": [1], "warmup": 0, "reset_prefix_cache": False, "progress": False,
                   "probe_capabilities": False, "max_consecutive_failures": None}  # fmt: skip
    monkeypatch.setattr(Journal, "seal", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):  # measured on A, but the ledger could not be written
        run(benchmark(book, Agent(), server.target(), tmp_path / "bench.zip", **options))
    monkeypatch.undo()
    server = FakeServer(monkeypatch, version="B")
    server.fail_next = 99
    with pytest.raises(RuntimeError, match="no level with a successful attempt"):
        run(benchmark(book, Agent(), server.target(), tmp_path / "bench.zip", **options))
    assert not (tmp_path / "bench.zip").exists()
