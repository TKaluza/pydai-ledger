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
from .plan import RunPlan, attempt_id
from .target import (
    Target,
    current_identity,
    identity,
    probe,
    probe_capabilities,
    reset_prefix_cache,
    scrape_metrics,
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
    autotune: bool = False
    min_gain: float = 0.05
    seen_parameters: set[str] = field(default_factory=set)
    attempts_seen: dict[str, int] = field(default_factory=dict)
    consecutive_failures: dict[str, int] = field(default_factory=dict)
    aborted: dict[str, str] = field(default_factory=dict)
    warned: set[tuple[str, str]] = field(default_factory=set)
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)

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
                result = await plan.agent.run(turn.content, message_history=history, model=recorder)
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


async def _run_stage(state: _State, target: Target, concurrency: int) -> tuple[int, int, int]:
    """Run all pending attempts of one (target, concurrency) stage. Cases are loaded
    only when a slot is free, so memory stays bounded for very large casebooks.
    """
    plan, journal, done, progress = state.plan, state.journal, state.done, state.progress
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
        for position, raw in enumerate(plan.casebook.records()):
            if position >= plan.case_count or target.name in state.aborted:
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


async def _warm_up(state: _State, target: Target) -> None:
    """Run the first cases once without recording them as attempts, so model loading
    and other cold-start effects stay out of the measured stages.
    """
    plan, journal = state.plan, state.journal
    started_at, t0, ok, count = now_iso(), time.perf_counter(), 0, 0
    for position, raw in enumerate(plan.casebook.records()):
        if position >= min(state.warmup, plan.case_count):
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
    """Probe one target, then run its warm-up and all its concurrency stages."""
    journal, plan = state.journal, state.plan
    snapshot = state.snapshots.get(target.name) or await probe(target)
    if state.capabilities and target.root_url:  # only real endpoints; three small requests
        snapshot["capabilities"] = await probe_capabilities(target)
    journal.append({"kind": "target", "target": target.name, "at": now_iso(), "snapshot": snapshot})
    if snapshot["provenance"] == "provider_only":
        state.warn(
            target.name, "provider_only", "/server_info not available; only provider-level information recorded."
        )
    if not state.reset_cache and (len(plan.levels) > 1 or plan.attempts_per_case > 1):
        state.warn(
            target.name,
            "prefix_cache_not_reset",
            "the server's prefix cache is not reset between stages, so later stages and "
            "repetitions can be faster. Pass reset_prefix_cache=True (needs "
            "VLLM_SERVER_DEV_MODE=1) if that matters for your comparison.",
        )
    if state.warmup:
        await _warm_up(state, target)

    # Autotune: figures of stages measured so far, including those of an earlier, resumed call.
    measured: dict[int, list[float]] = {}  # concurrency -> [attempts ok, output tokens, wall seconds]
    skipped: set[int] = set()
    for record in journal.records():
        if record.get("target") != target.name:
            continue
        if record.get("kind") == "stage":
            entry = measured.setdefault(record["concurrency"], [0, 0, 0.0])
            entry[0] += record.get("attempts_ok", 0)
            entry[1] += record.get("output_tokens", 0)
            entry[2] += record["wall_s"]
        elif record.get("kind") == "autotune":
            skipped = set(record["skipped"])

    def rate(level: int) -> float:
        ok, tokens, wall = measured[level]
        return (tokens or ok) / wall if wall else 0.0

    for position, concurrency in enumerate(plan.levels):
        if concurrency in skipped:
            continue
        cache_reset = None
        if state.reset_cache:
            cache_reset = await reset_prefix_cache(target)
            if not cache_reset:
                state.warn(
                    target.name,
                    "prefix_cache_reset_failed",
                    "reset_prefix_cache=True was requested but /reset_prefix_cache is not available.",
                )
        metrics_before = await scrape_metrics(target)
        started_at = now_iso()
        t0 = time.perf_counter()
        executed, ok, tokens = await _run_stage(state, target, concurrency)
        wall = time.perf_counter() - t0
        if executed:
            entry = measured.setdefault(concurrency, [0, 0, 0.0])
            entry[0] += ok
            entry[1] += tokens
            entry[2] += wall
            journal.append(
                {
                    "kind": "stage",
                    "target": target.name,
                    "concurrency": concurrency,
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
        if executed and target.root_url and target.name not in state.aborted:
            before, after = identity(snapshot), await current_identity(target)
            if before is not None and after is not None and before != after:  # restarted or swapped mid-run
                state.aborted[target.name] = f"the server changed during the run: {before} became {after}"
        if target.name in state.aborted:
            reason = state.aborted[target.name]
            journal.append({"kind": "aborted", "target": target.name, "reason": reason, "at": now_iso()})
            raise RunAbortedError(f"Target {target.name!r} stopped: {reason}")
        if state.autotune and concurrency in measured:
            earlier = [rate(level) for level in plan.levels[:position] if level in measured]
            if earlier and rate(concurrency) < max(earlier) * (1 + state.min_gain):
                skipped = set(plan.levels[position + 1 :])
                break
    if state.autotune and measured:
        levels = [level for level in plan.levels if level in measured]
        journal.append(
            {
                "kind": "autotune",
                "target": target.name,
                "at": now_iso(),
                "metric": "output_tokens_per_s" if any(measured[level][1] for level in levels) else "attempts_per_s",
                "min_gain": state.min_gain,
                "levels": [{"concurrency": level, "rate": rate(level)} for level in levels],
                "best": next(level for level in levels if rate(level) >= max(map(rate, levels)) * (1 - state.min_gain)),
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
    autotune: bool = False,
    min_gain: float = 0.05,
    config: str | os.PathLike[str] | Config | None = None,
) -> None:
    """Run every planned attempt that is not yet successful in ``journal``.

    Targets run in parallel (they are independent servers). Per target the concurrency
    levels run one after another so their throughput figures do not disturb each other.
    Calling this again on the same journal resumes: successful attempts are skipped,
    failed and missing ones are executed.

    ``timeout``
        Seconds allowed per turn; a turn that takes longer is recorded as failed.
    ``max_consecutive_failures``
        A target is stopped with :class:`RunAbortedError` after this many failures in a
        row, or at once when its very first request cannot reach the server. ``None``
        disables the threshold.
    ``reset_prefix_cache``
        Call vLLM's ``/reset_prefix_cache`` before every stage (needs dev mode). Off by
        default; a warning is recorded when stages or repetitions could be affected.
    ``warmup``
        Number of cases to run once per target before the first stage. Their results
        are discarded (only a ``warmup`` event is recorded), which keeps model loading
        out of the measurements. Off (0) by default.
    ``probe_capabilities``
        Send three small requests to every HTTP target before the run to record whether
        it supports tool output, native JSON-schema output and image input. On by default.
    ``autotune``
        Treat ``plan.concurrency`` as a ladder: run the levels in order and stop a target
        as soon as a level is not at least ``min_gain`` (5 %) faster than the best level
        before it. The remaining levels are skipped and do not count as missing. See
        :func:`pydai_ledger.autotune` for the ready-made search.
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
    if autotune and shard is not None:
        raise ValueError("autotune measures whole stages and cannot be combined with shard.")
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
                "autotune": autotune,
                "min_gain": min_gain,
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
        autotune=autotune,
        min_gain=min_gain,
    )
    settings = Config.load(config)
    journal.append({"kind": "config", "at": now_iso(), "worker_id": journal.worker_id, **settings.describe()})
    probed = await asyncio.gather(*(probe(target) for target in plan.targets))
    state.snapshots = {target.name: snapshot for target, snapshot in zip(plan.targets, probed, strict=True)}
    if settings.require and settings.on_missing == "error":
        recorded = to_jsonable_python(plan.agent.model_settings, fallback=repr)
        model_settings = recorded if isinstance(recorded, dict) else {}
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
    results = await asyncio.gather(*(_run_target(state, target) for target in plan.targets), return_exceptions=True)
    tracker.report()
    errors = [r for r in results if isinstance(r, BaseException)]
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("several targets failed", errors)
