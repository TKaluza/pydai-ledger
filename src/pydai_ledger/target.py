"""Targets (where a run executes) and what can be learned about them over HTTP."""

from __future__ import annotations

import asyncio
import os
import re
import struct
import time
import tomllib
import warnings
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
from pydantic import BaseModel
from pydantic_ai import Agent, BinaryContent, NativeOutput, ToolOutput
from pydantic_ai.models import Model, infer_model
from pydantic_core import to_jsonable_python

from ._util import canonical_json, sha256_hex


@dataclass(frozen=True)
class Target:
    """A named model endpoint. ``info`` is free-form operator knowledge (GPU, host, notes)
    that is stored verbatim with every run. ``concurrency`` is how many attempts a
    :class:`~pydai_ledger.RunPlan` keeps in flight on this target unless the plan says
    otherwise, for example the number a benchmark recommended.
    """

    name: str
    model: Model | str
    info: Mapping[str, Any] = field(default_factory=dict)
    concurrency: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.model, str):
            object.__setattr__(self, "model", infer_model(self.model))

    @property
    def root_url(self) -> str | None:
        """Server root without the ``/v1`` suffix, or None if the model has no base URL."""
        base = getattr(self.model, "base_url", None)
        if not base:
            return None
        base = str(base).rstrip("/")
        return base[: -len("/v1")] if base.endswith("/v1") else base

    def describe(self) -> dict[str, Any]:
        """The target as plain JSON; this is what the ledger stores."""
        model = self.model
        return {
            "name": self.name,
            "model_name": getattr(model, "model_name", None),
            "system": getattr(model, "system", None),
            "base_url": getattr(model, "base_url", None),
            "settings": to_jsonable_python(getattr(model, "settings", None), fallback=repr),
            "info": to_jsonable_python(dict(self.info), fallback=repr),
        }

    async def _get(self, path: str, method: str = "GET") -> httpx2.Response | None:
        """Request against the server root; None on any failure."""
        if not self.root_url:
            return None
        headers = {}
        api_key = getattr(getattr(self.model, "client", None), "api_key", None)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx2.AsyncClient(headers=headers, timeout=10.0) as client:
                response = await client.request(method, self.root_url + path)
                response.raise_for_status()
                return response
        except Exception:
            return None

    async def _get_json(self, path: str) -> Any:
        """GET a path and parse JSON; None on any failure."""
        response = await self._get(path)
        try:
            return response.json() if response is not None else None
        except ValueError:
            return None


def load_targets(path: str | os.PathLike[str] = "targets.toml") -> list[Target]:
    """Read targets from a TOML file, so servers are named in one place and not in code::

        [targets.gpu-a]
        model = "Qwen/Qwen3-32B"                  # the served model name
        base_url = "http://gpu-a:8000/v1"         # an OpenAI-compatible endpoint
        api_key_env = "VLLM_API_KEY"              # optional: name of the variable holding the key
        concurrency = 32                          # optional, e.g. what a benchmark recommended
        info = { gpu = "2x RTX 4090" }            # optional, stored verbatim with every run

    The file holds no secret, only the name of the environment variable. Without
    ``base_url``, ``model`` is a Pydantic AI model string such as ``"openai:gpt-5"``.
    Models that cannot be described this way are built in code as before.
    """
    file = Path(path)
    data = tomllib.loads(file.read_text(encoding="utf-8")).get("targets") or {}
    if not data:
        raise ValueError(f"{file} defines no [targets.<name>] table.")
    targets = []
    for name, entry in data.items():
        unknown = set(entry) - {"model", "base_url", "api_key_env", "concurrency", "info"}
        if unknown or "model" not in entry:
            raise ValueError(f"{file}: target {name!r} needs 'model'; unknown keys: {sorted(unknown) or 'none'}.")
        model: Model | str = entry["model"]
        if "base_url" in entry:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider

            key = os.environ.get(entry.get("api_key_env", ""), "EMPTY")
            model = OpenAIChatModel(entry["model"], provider=OpenAIProvider(base_url=entry["base_url"], api_key=key))
        targets.append(Target(name, model, entry.get("info") or {}, entry.get("concurrency")))
    return targets


REDACTED = "[redacted]"
_SECRET_NAME = re.compile(r"(?:^|_)(?:TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIALS?)(?:$|_)", re.IGNORECASE)
_PLAIN_VALUE = re.compile(r"(?:|none|null|true|false|[-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"\b(\w+)(\s*=\s*)([^\s,;)]+)")


def _is_secret(name: str, value: object) -> bool:
    """A name such as ``HF_TOKEN`` or ``VLLM_API_KEY`` with a value that is text, not a number or flag."""
    return bool(_SECRET_NAME.search(name)) and isinstance(value, str) and not _PLAIN_VALUE.fullmatch(value.strip("'\""))


def redact(value: Any) -> Any:
    """Replace values that look like secrets, so they reach neither the ledger nor a report.

    Applies to mapping entries and to ``NAME=value`` text whose name has a part ``TOKEN``,
    ``KEY``, ``SECRET``, ``PASSWORD``, ``PASSWD`` or ``CREDENTIAL(S)``. Numbers and flags
    stay, so settings such as ``..._TOKEN_THRESHOLD=64`` remain readable.
    """
    if isinstance(value, dict):
        return {k: REDACTED if _is_secret(str(k), v) else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _ASSIGNMENT.sub(lambda m: m[1] + m[2] + REDACTED if _is_secret(m[1], m[3]) else m[0], value)
    return value


async def probe(target: Target) -> dict[str, Any]:
    """Collect a server snapshot.

    Values of ``/server_info`` that look like secrets are replaced, see :func:`redact`.

    With ``VLLM_SERVER_DEV_MODE=1`` vLLM exposes ``/server_info`` (full server
    configuration). Without it only provider-level facts are available; this is
    recorded as ``provenance: "provider_only"`` and a warning is emitted.
    """
    snapshot: dict[str, Any] = {"server_info": None, "version": None, "models": None}
    for key, path in (("server_info", "/server_info"), ("version", "/version"), ("models", "/v1/models")):
        snapshot[key] = await target._get_json(path)
    snapshot["server_info"] = redact(snapshot["server_info"])
    if snapshot["server_info"] is None:
        snapshot["provenance"] = "provider_only"
        warnings.warn(
            f"Target {target.name!r}: /server_info is not available, so only provider-level "
            "information is recorded. Start vLLM with VLLM_SERVER_DEV_MODE=1 to capture the "
            "full server configuration.",
            stacklevel=2,
        )
    else:
        snapshot["provenance"] = "server_info"
    return snapshot


def identity(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """What must stay the same during a run: server version and the served models."""
    models = (snapshot.get("models") or {}).get("data")
    if models is None:
        return None
    return {
        "version": (snapshot.get("version") or {}).get("version"),
        "models": sorted([str(m.get("id")), str(m.get("root")), str(m.get("max_model_len"))] for m in models),
    }


def server_key(snapshot: dict[str, Any]) -> str | None:
    """A short key for :func:`identity`, recorded with every attempt and stage; None when
    the server does not say what it is.
    """
    found = identity(snapshot)
    return None if found is None else sha256_hex(canonical_json(found).encode())[:16]


async def current_identity(target: Target) -> dict[str, Any] | None:
    """Ask the server again for its version and models; None when it does not answer."""
    return identity({"version": await target._get_json("/version"), "models": await target._get_json("/v1/models")})


async def scrape_metrics(target: Target) -> dict[str, float] | None:
    """Read the server's Prometheus ``/metrics`` as ``{sample: value}`` (histogram buckets
    dropped). Works for vLLM (``vllm:...``) and other servers that expose the endpoint.
    """
    response = await target._get("/metrics")
    if response is None:
        return None
    samples: dict[str, float] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#") or "_bucket{" in line:
            continue
        name, _, value = line.rpartition(" ")
        try:
            samples[name] = float(value)
        except ValueError:
            continue
    return samples


async def reset_prefix_cache(target: Target) -> bool:
    """Ask vLLM to drop its prefix cache (``POST /reset_prefix_cache``, dev mode only)."""
    return await target._get("/reset_prefix_cache", method="POST") is not None


class _Probe(BaseModel):
    """Tiny structured output used to test a target's capabilities."""

    ok: bool


def _tiny_png() -> bytes:
    """A valid 16 x 16 red PNG for the image capability test."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + b"\xff\x00\x00" * 16 for _ in range(16))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


async def probe_capabilities(target: Target, *, timeout: float = 60.0) -> dict[str, dict[str, Any]]:
    """Find out what a target can do by sending three small requests: structured output
    through a tool call, structured output through a native JSON schema, and an image.

    Returns ``{"tool_output" | "native_output" | "image_input": {"ok", "error", "duration_s"}}``.
    """
    settings: Any = {"temperature": 0, "max_tokens": 256}
    image = BinaryContent(_tiny_png(), media_type="image/png")
    checks: dict[str, tuple[Agent[Any, Any], Any]] = {
        "tool_output": (Agent(output_type=ToolOutput(_Probe), model_settings=settings), "Reply with ok = true."),
        "native_output": (Agent(output_type=NativeOutput(_Probe), model_settings=settings), "Reply with ok = true."),
        "image_input": (Agent(model_settings=settings), ["Name the colour of this image in one word.", image]),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (agent, prompt) in checks.items():
        t0 = time.perf_counter()
        error = None
        try:
            async with asyncio.timeout(timeout):
                await agent.run(prompt, model=target.model)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:400]
        result[name] = {"ok": error is None, "error": error, "duration_s": time.perf_counter() - t0}
    return result
