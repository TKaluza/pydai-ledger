"""Execute a RunPlan and write every attempt to a Journal."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, UserPromptPart
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_core import to_jsonable_python

from ._util import BLOB_PLACEHOLDER, canonical_json, environment, externalize, now_iso, sha256_hex
from .casebook import StoredCase
from .config import Config, DeterminismError, determinism, missing
from .journal import Journal
from .plan import RunPlan, Stage, attempt_id
from .target import (
    Target,
    current_identity,
    identity,
    probe,
    probe_capabilities,
    reset_prefix_cache,
    scrape_metrics,
    server_key,
)


class _Progress:
    """Prints overall throughput to stderr every few seconds."""

    def __init__(self, total: int, already_done: int, enabled: bool, interval: float = 2.0) -> None:
        self.total = total
        self.done = already_done
        self.new = 0
        self.errors = 0
        self.output_tokens = 0
        self.enabled = enabled
        self.interval = interval
        self.started = time.perf_counter()
        self._last = self.started

    def update(self, record: dict[str, Any]) -> None:
        """Count one finished attempt and print a line when the interval has passed."""
        self.new += 1
        if record["status"] == "ok":
            self.done += 1
        else:
            self.errors += 1
        self.output_tokens += sum(t["usage"].get("output_tokens") or 0 for t in record["turns"])
        if time.perf_counter() - self._last >= self.interval:
            self.report()

    def report(self) -> None:
        """Print the current totals and rates."""
        self._last = time.perf_counter()
        if not self.enabled:
            return
        elapsed = max(self._last - self.started, 1e-9)
        print(
            f"[pydai-ledger] {self.done}/{self.total} attempts ok, {self.errors} failed | "
            f"{self.new / elapsed:.2f} attempts/s | {self.output_tokens / elapsed:.1f} output tokens/s",
            file=sys.stderr,
        )


class RunAbortedError(RuntimeError):
    """Raised by :func:`execute` when a target was stopped early. Everything recorded so
    far stays in the journal; fix the cause and call execute() again to resume.
    """


def _slim(messages: list[ModelMessage], blob_shas: dict[int, str]) -> list[ModelMessage]:
    """Swap input binaries that came from the casebook for tiny placeholders before the
    messages are dumped, so large images are neither base64-encoded nor hashed again.
    """
    if not blob_shas:
        return messages
    out = []
    for message in messages:
        parts = None
        for position, part in enumerate(message.parts):
            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                continue
            if not any(id(item) in blob_shas for item in part.content):
                continue
            content = [
                BinaryContent(
                    BLOB_PLACEHOLDER + blob_shas[id(item)].encode("ascii"),
                    media_type=item.media_type,
                    identifier=item.identifier,
                    vendor_metadata=item.vendor_metadata,
                )
                if id(item) in blob_shas and isinstance(item, BinaryContent)
                else item
                for item in part.content
            ]
            parts = parts or list(message.parts)
            parts[position] = dataclasses.replace(part, content=content)
        out.append(dataclasses.replace(message, parts=parts) if parts else message)
    return out


class _Recorder(WrapperModel):
    """Wraps the target model and notes every request exactly as it is sent: the merged
    model settings, the tool and output definitions, and the time the request took.
    This uses only public Pydantic AI interfaces.
    """

    def __init__(self, wrapped: Any) -> None:
        super().__init__(wrapped)
        self.requests: list[dict[str, Any]] = []

    async def request(self, messages: Any, model_settings: Any, model_request_parameters: Any) -> Any:
        """Forward the request and note its settings, parameters and duration."""
        t0 = time.perf_counter()
        try:
            return await super().request(messages, model_settings, model_request_parameters)
        finally:
            self.requests.append(
                {
                    "duration_s": time.perf_counter() - t0,
                    "settings": model_settings,
                    "parameters": model_request_parameters,
                }
            )


@dataclass
class _State:
    """Shared bookkeeping of one execute() call."""

    plan: RunPlan
    journal: Journal
    done: set[str]
    progress: _Progress
    timeout: float | None = None
    max_consecutive_failures: int | None = 20
    reset_cache: bool = False
    shard: tuple[int, int] | None = None
    warmup: int = 0
    capabilities: bool = True
    seen_parameters: set[str] = field(default_factory=set)
    attempts_seen: dict[str, int] = field(default_factory=dict)
    consecutive_failures: dict[str, int] = field(default_factory=dict)
    aborted: dict[str, str] = field(default_factory=dict)
    warned: set[tuple[str, str]] = field(default_factory=set)
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    servers: dict[str, str | None] = field(default_factory=dict)  # target -> key of the server it runs on
    settings: Config = field(default_factory=Config)
    model_settings: dict[str, Any] = field(default_factory=dict)

    def warn(self, target: str, code: str, message: str) -> None:
        """Record a warning once per target and code, and emit it as a Python warning."""
        if (target, code) in self.warned:
            return
        self.warned.add((target, code))
        self.journal.append({"kind": "warning", "target": target, "code": code, "message": message, "at": now_iso()})
        if code != "provider_only":  # probe() already warned about that one
            warnings.warn(f"Target {target!r}: {message}", stacklevel=2)

    def note_result(self, target: str, record: dict[str, Any]) -> None:
        """Stop a target early when it is evidently broken instead of failing every case."""
        first = self.attempts_seen.get(target, 0) == 0
        self.attempts_seen[target] = self.attempts_seen.get(target, 0) + 1
        if record["status"] == "ok":
            self.consecutive_failures[target] = 0
            return
        failures = self.consecutive_failures.get(target, 0) + 1
        self.consecutive_failures[target] = failures
        if target in self.aborted:
            return
        if first and record["error_kind"] == "api":
            self.aborted[target] = f"the first request failed: {record['error']}"
        elif self.max_consecutive_failures and failures >= self.max_consecutive_failures:
            self.aborted[target] = f"{failures} attempts failed in a row, last error: {record['error']}"

    def note_requests(self, target: str, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Turn recorded requests into JSON, store new parameter sets once, check sampling."""
        out = []
        for request in requests:
            settings = to_jsonable_python(request["settings"], fallback=repr) or {}
            parameters = {
                k: v
                for k, v in to_jsonable_python(request["parameters"], fallback=repr).items()
                if not k.startswith("_")
            }
            sha = sha256_hex(canonical_json(parameters))[:16]
            if sha not in self.seen_parameters:
                self.seen_parameters.add(sha)
                self.journal.append({"kind": "request_parameters", "sha": sha, "parameters": parameters})
            if settings.get("seed") is None:
                self.warn(target, "no_seed", "no seed is set, so repeated runs are not comparable.")
            temperature = settings.get("temperature")
            if temperature is None:
                self.warn(target, "no_temperature", "no temperature is set; the server default applies.")
            elif temperature > 0:
                self.warn(target, "temperature_nonzero", f"temperature is {temperature}; outputs will vary.")
            out.append({"duration_s": request["duration_s"], "settings": settings, "parameters_sha": sha})
        return out


async def _run_attempt(state: _State, target: Target, concurrency: int, stored: StoredCase, rep: int) -> dict[str, Any]:
    """Run all turns of one case sequentially. Never raises for model/agent errors;
    a failure is recorded with the turns completed so far.
    """
    plan, journal = state.plan, state.journal
    recorder = _Recorder(target.model)
    seed, repeat = plan.attempts[rep]
    settings: Any = {"seed": seed} if seed is not None else None

    def sink(sha: str, data: bytes) -> None:
        if not plan.casebook.has_blob(sha):
            journal.store_blob(sha, data)

    record: dict[str, Any] = {
        "kind": "attempt",
        "attempt_id": attempt_id(target.name, concurrency, stored.index, rep),
        "target": target.name,
        "concurrency": concurrency,
        "case_index": stored.index,
        "case_id": stored.id,
        "rep": rep,
        "seed": seed,
        "repeat": repeat,
        "server": state.servers.get(target.name),
        "worker_id": journal.worker_id,
        "status": "ok",
        "error": None,
        "error_kind": None,
        "started_at": now_iso(),
        "turns": [],
    }
    began = time.perf_counter()
    history = None
    try:
        for index, turn in enumerate(stored.case.user_turns):
            turn_started = now_iso()
            t0 = time.perf_counter()
            async with asyncio.timeout(state.timeout):
                result = await plan.agent.run(
                    turn.content, message_history=history, model=recorder, model_settings=settings
                )
            duration = time.perf_counter() - t0
            history = result.all_messages()
            requests, recorder.requests = recorder.requests, []
            usage = result.usage
            output = result.output
            record["turns"].append(
                {
                    "index": index,
                    "started_at": turn_started,
                    "duration_s": duration,
                    "output": output.model_dump(mode="json")
                    if isinstance(output, BaseModel)
                    else to_jsonable_python(output, fallback=repr),
                    "usage": to_jsonable_python(dataclasses.asdict(usage), fallback=repr),
                    "model_name": result.response.model_name,
                    "finish_reason": result.response.finish_reason,
                    "requests": state.note_requests(target.name, requests),
                    "messages": externalize(
                        ModelMessagesTypeAdapter.dump_python(
                            _slim(result.new_messages(), stored.blob_shas), mode="json"
                        ),
                        sink,
                    ),
                }
            )
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        # "api": the server could not be reached or refused; "timeout"; "run": everything else
        record["error_kind"] = (
            "api"
            if isinstance(exc, (ModelAPIError, OSError)) and not isinstance(exc, TimeoutError)
            else "timeout"
            if isinstance(exc, TimeoutError)
            else "run"
        )
    record["duration_s"] = time.perf_counter() - began
    record["ended_at"] = now_iso()
    return record


async def _run_stage(state: _State, target: Target, stage: Stage) -> tuple[int, int, int]:
    """Run all pending attempts of one stage of a target. Cases are loaded only when a
    slot is free, so memory stays bounded for very large casebooks.
    """
    plan, journal, done, progress = state.plan, state.journal, state.done, state.progress
    concurrency = stage.concurrency
    slots = asyncio.Semaphore(concurrency)
    executed = ok = tokens = 0

    async def worker(stored: StoredCase, rep: int) -> None:
        nonlocal ok, tokens
        try:
            record = await _run_attempt(state, target, concurrency, stored, rep)
            journal.append(record)
            if record["status"] == "ok":
                ok += 1
                tokens += sum(t["usage"].get("output_tokens") or 0 for t in record["turns"])
            progress.update(record)
            state.note_result(target.name, record)
        finally:
            slots.release()

    async with asyncio.TaskGroup() as group:
        for raw in plan.records(stage):
            if target.name in state.aborted:
                break
            if state.shard is not None and raw["index"] % state.shard[1] != state.shard[0]:
                continue
            pending = [
                rep
                for rep in range(plan.attempts_per_case)
                if attempt_id(target.name, concurrency, raw["index"], rep) not in done
            ]
            if not pending:
                continue
            stored = plan.casebook.load(raw)
            for rep in pending:
                await slots.acquire()
                if target.name in state.aborted:
                    slots.release()
                    break
                task = group.create_task(worker(stored, rep))
                executed += 1
                if state.attempts_seen.get(target.name, 0) == 0:
                    await task  # the very first attempt runs alone as a connectivity check
    return executed, ok, tokens


def _forget(state: _State, target: str, concurrency: int) -> None:
    """Make every attempt of one stage pending again."""
    state.done.difference_update([done for done in state.done if done.startswith(f"{target}|c{concurrency}|")])


async def _warm_up(state: _State, target: Target) -> None:
    """Run the first cases once without recording them as attempts, so model loading
    and other cold-start effects stay out of the measured stages.
    """
    plan, journal = state.plan, state.journal
    started_at, t0, ok, count = now_iso(), time.perf_counter(), 0, 0
    for position, raw in enumerate(plan.records(plan.stages(target.name)[0])):
        if position >= state.warmup:
            break
        record = await _run_attempt(state, target, 0, plan.casebook.load(raw), 0)
        count += 1
        if record["status"] == "ok":
            ok += 1
        elif record["error_kind"] == "api":
            reason = f"the warm-up request failed: {record['error']}"
            journal.append({"kind": "aborted", "target": target.name, "reason": reason, "at": now_iso()})
            raise RunAbortedError(f"Target {target.name!r} stopped: {reason}")
    journal.append(
        {
            "kind": "warmup",
            "target": target.name,
            "worker_id": journal.worker_id,
            "started_at": started_at,
            "wall_s": time.perf_counter() - t0,
            "attempts": count,
            "attempts_ok": ok,
        }
    )


async def _run_target(state: _State, target: Target) -> None:
    """Probe one target, then run its warm-up and its stages (a run has one, a benchmark several)."""
    journal, plan = state.journal, state.plan
    snapshot = state.snapshots.get(target.name) or await probe(target)
    if state.capabilities and target.root_url:  # only real endpoints; three small requests
        snapshot["capabilities"] = await probe_capabilities(target)
    journal.append({"kind": "target", "target": target.name, "at": now_iso(), "snapshot": snapshot})
    if snapshot["provenance"] == "provider_only":
        state.warn(
            target.name, "provider_only", "/server_info not available; only provider-level information recorded."
        )
    stages = plan.stages(target.name)
    levels = [stage.concurrency for stage in stages]
    if not state.reset_cache and len(stages) > 1:
        state.warn(
            target.name,
            "prefix_cache_not_reset",
            "the server's prefix cache is not reset between stages, so later stages can be "
            "faster. Pass reset_prefix_cache=True (needs VLLM_SERVER_DEV_MODE=1).",
        )
    if state.warmup:
        await _warm_up(state, target)

    # Benchmark: figures of stages measured so far, including those of an earlier, resumed call.
    measured: dict[int, list[float]] = {}  # concurrency -> [attempts ok, output tokens, wall seconds]
    skipped: set[int] = set()
    server = state.servers.get(target.name)
    for record in journal.records():
        if record.get("target") != target.name:
            continue
        if record.get("server") != server:
            continue  # measured on a server that is gone
        if record.get("kind") == "stage" and record.get("attempts_ok"):  # one without success runs again
            entry = measured.setdefault(record["concurrency"], [0, 0, 0.0])
            entry[0] += record["attempts_ok"]
            entry[1] += record.get("output_tokens", 0)
            entry[2] += record["wall_s"]
        elif record.get("kind") == "tuning":
            skipped = set(record["skipped"])

    # A stage that a server change stopped holds outputs of two servers. It runs again as a
    # whole; only its new measurement lifts the stop (see Journal.seal).
    void = {level for name, level, shard in journal.void_stages() if name == target.name and shard == state.shard}

    def in_tokens() -> bool:
        """One unit for the whole benchmark: tokens only when every level reported them."""
        return all(entry[1] for entry in measured.values() if entry[0])

    def rate(level: int) -> float:
        ok, tokens, wall = measured[level]
        return (tokens if in_tokens() else ok) / wall if wall else 0.0

    def gains(position: int) -> bool:
        """Whether the level at ``position`` is faster than every lower one by ``min_gain``."""
        earlier = [rate(level) for level in levels[:position] if level in measured]
        return not earlier or rate(levels[position]) >= max(earlier) * (1 + (plan.min_gain or 0.0))

    for position, stage in enumerate(stages):
        concurrency = stage.concurrency
        if concurrency in skipped:
            continue
        if plan.min_gain is not None:
            if concurrency in measured:  # measured by an earlier call; a stage is never topped up
                if not gains(position):
                    skipped = set(levels[position + 1 :])
                    break
                continue
            # Attempts without the stage's measurement come from a call that broke off; their
            # timing is worth nothing, so the whole stage runs again.
            _forget(state, target.name, concurrency)
        elif concurrency in void:
            _forget(state, target.name, concurrency)
        cache_reset = None
        if state.reset_cache:
            cache_reset = await reset_prefix_cache(target)
            if not cache_reset:
                state.warn(
                    target.name,
                    "prefix_cache_reset_failed",
                    "reset_prefix_cache=True was requested but /reset_prefix_cache is not available.",
                )
                points = determinism(state.model_settings, {target.name: snapshot})  # as it is without a reset
                if state.settings.on_missing == "error" and "prefix_cache_neutral" in missing(state.settings, points):
                    raise DeterminismError(
                        f"Target {target.name!r}: the prefix cache is required to be neutral, but it is on "
                        "and could not be reset before the stage."
                    )
        metrics_before = await scrape_metrics(target)
        started_at = now_iso()
        t0 = time.perf_counter()
        executed, ok, tokens = await _run_stage(state, target, stage)
        wall = time.perf_counter() - t0
        changed = None
        if executed and target.root_url and target.name not in state.aborted:
            before, after = identity(snapshot), await current_identity(target)
            if before is not None and after is not None and before != after:  # restarted or swapped mid-run
                changed = state.aborted[target.name] = f"the server changed during the run: {before} became {after}"
        # No measurement for a stage of two servers, and none for a benchmark stage that stopped:
        # its figures would stand for a level that was never measured in full.
        if executed and not changed and not (plan.min_gain is not None and target.name in state.aborted):
            entry = measured.setdefault(concurrency, [0, 0, 0.0])
            entry[0] += ok
            entry[1] += tokens
            entry[2] += wall
            journal.append(
                {
                    "kind": "stage",
                    "target": target.name,
                    "concurrency": concurrency,
                    "server": server,
                    "shard": list(state.shard) if state.shard else None,
                    "stopped": target.name in state.aborted,
                    "worker_id": journal.worker_id,
                    "started_at": started_at,
                    "ended_at": now_iso(),
                    "wall_s": wall,
                    "attempts_executed": executed,
                    "attempts_ok": ok,
                    "output_tokens": tokens,
                    "prefix_cache_reset": cache_reset,
                    "metrics_before": metrics_before,
                    "metrics_after": await scrape_metrics(target),
                }
            )
        if target.name in state.aborted:
            reason = state.aborted[target.name]
            stop = {"kind": "aborted", "target": target.name, "reason": reason, "at": now_iso()}
            shard = list(state.shard) if state.shard else None
            journal.append({**stop, "void_stage": concurrency, "shard": shard} if changed else stop)
            raise RunAbortedError(f"Target {target.name!r} stopped: {reason}")
        if plan.min_gain is not None and concurrency in measured and not gains(position):
            skipped = set(levels[position + 1 :])
            break
    levels = [level for level in levels if level in measured and measured[level][0]]  # with a successful attempt
    if plan.min_gain is not None and levels:
        journal.append(
            {
                "kind": "tuning",
                "target": target.name,
                "server": server,
                "at": now_iso(),
                "metric": "output_tokens_per_s" if in_tokens() else "attempts_per_s",
                "min_gain": plan.min_gain,
                "levels": [
                    {"concurrency": level, "rate": rate(level), "cases": int(measured[level][0])} for level in levels
                ],
                "best": next(level for level in levels if rate(level) >= max(map(rate, levels)) * (1 - plan.min_gain)),
                "skipped": sorted(skipped),
            }
        )


async def execute(
    plan: RunPlan,
    journal: Journal,
    *,
    progress: bool = True,
    timeout: float | None = None,
    max_consecutive_failures: int | None = 20,
    reset_prefix_cache: bool = False,
    shard: tuple[int, int] | None = None,
    warmup: int = 0,
    probe_capabilities: bool = True,
    config: str | os.PathLike[str] | Config | None = None,
) -> None:
    """Run every planned attempt that is not yet successful in ``journal``.

    Targets run side by side, each with its own concurrency, because they are
    independent servers; ``RunPlan(sequential_targets=True)`` runs them one after
    another. Every attempt records the seed it was sent with. Calling this again on the same journal resumes: successful attempts are skipped,
    failed and missing ones are executed.

    ``timeout``
        Seconds allowed per turn; a turn that takes longer is recorded as failed.
    ``max_consecutive_failures``
        A target is stopped with :class:`RunAbortedError` after this many failures in a
        row, or at once when its very first request cannot reach the server. ``None``
        disables the threshold.
    ``reset_prefix_cache``
        Call vLLM's ``/reset_prefix_cache`` before the target starts (needs dev mode).
        Off by default.
    ``warmup``
        Number of cases to run once per target before its attempts. Their results are
        discarded (only a ``warmup`` event is recorded). Off (0) by default.
    ``probe_capabilities``
        Send three small requests to every HTTP target before the run to record whether
        it supports tool output, native JSON-schema output and image input. On by default.
    ``config``
        A :class:`Config` or the path of a ``pydai-ledger.toml``; by default the file in
        the working directory is used when it exists. Determinism points it requires are
        checked against all servers first, and :class:`DeterminismError` is raised
        before any attempt runs when one does not hold (``on_missing = "error"``).
    ``shard``
        ``(i, n)`` runs only cases with ``index % n == i``. Start ``n`` processes, each
        with its own shard and its own ``Journal.open(..., worker_id=...)`` on the same
        ``parent`` directory, then seal once.
    """
    if shard is not None and not 0 <= shard[0] < shard[1]:
        raise ValueError(f"shard must be (i, n) with 0 <= i < n, got {shard!r}.")
    if plan.min_gain is not None and shard is not None:
        raise ValueError("a benchmark measures whole stages and cannot be combined with shard.")
    done = journal.done()
    planned = set(plan.attempt_ids(shard))
    tracker = _Progress(len(planned), len(done & planned), progress)
    journal.append({"kind": "environment", "at": now_iso(), "worker_id": journal.worker_id, **environment()})
    journal.append(
        {
            "kind": "execution",
            "at": now_iso(),
            "worker_id": journal.worker_id,
            "options": {
                "timeout": timeout,
                "max_consecutive_failures": max_consecutive_failures,
                "reset_prefix_cache": reset_prefix_cache,
                "shard": list(shard) if shard else None,
                "warmup": warmup,
                "probe_capabilities": probe_capabilities,
            },
        }
    )
    state = _State(
        plan,
        journal,
        done,
        tracker,
        timeout=timeout,
        max_consecutive_failures=max_consecutive_failures,
        reset_cache=reset_prefix_cache,
        shard=shard,
        warmup=warmup,
        capabilities=probe_capabilities,
    )
    settings = Config.load(config)
    journal.append({"kind": "config", "at": now_iso(), "worker_id": journal.worker_id, **settings.describe()})
    probed = await asyncio.gather(*(probe(target) for target in plan.targets))
    state.snapshots = {target.name: snapshot for target, snapshot in zip(plan.targets, probed, strict=True)}
    state.servers = {name: server_key(snapshot) for name, snapshot in state.snapshots.items()}
    # Attempts of a server that has changed since do not count: they run again on the one that is there now.
    stale = done - journal.done(state.servers)
    done -= stale
    tracker.done -= len(stale & planned)
    recorded = to_jsonable_python(plan.agent.model_settings, fallback=repr)
    model_settings = dict(recorded) if isinstance(recorded, dict) else {}
    if plan.seeds:  # the plan's seeds replace the agent's
        model_settings["seed"] = ", ".join(map(str, plan.seeds))
    state.settings, state.model_settings = settings, model_settings
    if settings.require and settings.on_missing == "error":
        points = determinism(model_settings, state.snapshots, reset_prefix_cache=reset_prefix_cache)
        gaps = missing(settings, points)
        if gaps:
            detail = "; ".join(
                f"{point} ({', '.join(f'{name}: {points[name][point][0]}' for name in names)})"
                for point, names in gaps.items()
            )
            raise DeterminismError(
                f"Required determinism points do not hold: {detail}. "
                f"Fix the setup or change [determinism] in {settings.source}."
            )
    results: list[BaseException | None]
    if plan.sequential_targets:
        results = []
        for target in plan.targets:
            try:
                results.append(await _run_target(state, target))
            except Exception as exc:  # the other targets still get their turn
                results.append(exc)
    else:
        results = await asyncio.gather(*(_run_target(state, target) for target in plan.targets), return_exceptions=True)
    tracker.report()
    errors = [r for r in results if isinstance(r, BaseException)]
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("several targets failed", errors)
