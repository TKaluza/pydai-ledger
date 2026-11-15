"""Configuration file, determinism points and note levels."""

import sys

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from support import _always, make_cases, run

from pydai_ledger import (
    Casebook,
    Comparison,
    Config,
    DeterminismError,
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
    target = Target("fake", FunctionModel(_always("x")))
    plan = RunPlan(book, Agent(), [target], concurrency=[1, 2])
    models = {"data": [{"id": "m", "root": "/m", "max_model_len": 4096}]}
    versions = iter(["0.33.0", "0.34.0"])

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
        ledger = journal.seal(tmp_path / "run.zip", require_complete=False)
    assert "0.34.0" in ledger.events("aborted")[0]["reason"]
    assert {a["concurrency"] for a in ledger.attempts()} == {1, 2}
    ledger.close()
    book.close()
