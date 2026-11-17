"""Optional settings from ``pydai-ledger.toml``: note levels and determinism requirements.

The file never changes what a ledger records. It decides how notes are shown in
reports and which determinism points must hold before a run may start::

    [report.notes]            # error | warning | info | off
    prefix_cache_not_reset = "info"
    no_seed = "error"

    [report.remarks]          # your own background text, shown next to the note
    provider_only = "The gateway hides /server_info on purpose."

    [determinism]
    require = ["temperature_zero", "request_seed", "batch_invariant"]
    on_missing = "error"      # "error" stops the run before it starts, "warning" only reports
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FILENAME = "pydai-ledger.toml"
LEVELS = ("error", "warning", "info", "off")

# Level of every note a report can show, unless the configuration says otherwise.
DEFAULT_LEVELS = {
    "stopped": "error",
    "incomplete": "error",
    "failed_attempts": "error",
    "invalid_signature": "error",
    "no_seed": "warning",
    "no_temperature": "warning",
    "temperature_nonzero": "warning",
    "provider_only": "warning",
    "prefix_cache_reset_failed": "warning",
    "prefix_cache_not_reset": "info",
}

# Points that matter for bit-identical outputs. A point is "on", "off", "unknown" or "n/a".
POINTS = (
    "temperature_zero",
    "request_seed",
    "server_seed",
    "batch_invariant",
    "prefix_cache_neutral",
    "single_gpu",
    "multiprocessing_off",
    "same_server_version",
)


class DeterminismError(RuntimeError):
    """Raised before a run starts when a required determinism point does not hold."""


@dataclass(frozen=True)
class Config:
    """Settings read from ``pydai-ledger.toml``; the defaults apply when there is no file."""

    notes: dict[str, str] = field(default_factory=dict)
    remarks: dict[str, str] = field(default_factory=dict)
    require: tuple[str, ...] = ()
    on_missing: str = "error"
    source: str | None = None

    @classmethod
    def load(cls, path: str | os.PathLike[str] | Config | None = None) -> Config:
        """Read ``path``, or ``pydai-ledger.toml`` in the working directory when it exists."""
        if isinstance(path, Config):
            return path
        file = Path(path) if path is not None else Path(FILENAME)
        if not file.exists():
            if path is not None:
                raise FileNotFoundError(file)
            return cls()
        data = tomllib.loads(file.read_text(encoding="utf-8"))
        report, determinism = data.get("report", {}), data.get("determinism", {})
        notes = {str(k): str(v) for k, v in report.get("notes", {}).items()}
        require = tuple(determinism.get("require", ()))
        on_missing = str(determinism.get("on_missing", "error"))
        bad = [f"report.notes.{k} = {v!r}" for k, v in notes.items() if v not in LEVELS]
        bad += [f"determinism.require: {p!r}" for p in require if p not in POINTS]
        if on_missing not in ("error", "warning"):
            bad.append(f"determinism.on_missing = {on_missing!r}")
        if bad:
            raise ValueError(f"{file}: invalid entries: {'; '.join(bad)}")
        remarks = {str(k): str(v) for k, v in report.get("remarks", {}).items()}
        return cls(notes, remarks, require, on_missing, str(file))

    def level(self, code: str) -> str:
        """The level a note is shown with: configured, else its default, else ``warning``."""
        default = self.on_missing if code.startswith("determinism_") else DEFAULT_LEVELS.get(code, "warning")
        return self.notes.get(code, default)

    def describe(self) -> dict[str, Any]:
        """The settings as plain data, for the journal and the reports."""
        return {
            "source": self.source,
            "notes": dict(self.notes),
            "remarks": dict(self.remarks),
            "require": list(self.require),
            "on_missing": self.on_missing,
        }


def _setting(config: str, name: str) -> str | None:
    """One ``name=value`` entry of the ``vllm_config`` string."""
    found = re.search(rf"\b{name}=([^,)]+)", config)
    return found.group(1).strip("'\" ") if found else None


def _flag(value: bool | None) -> str:
    """Map a known or unknown truth value to a point state."""
    return "unknown" if value is None else "on" if value else "off"


def determinism(
    model_settings: dict[str, Any] | None,
    snapshots: dict[str, dict[str, Any]],
    *,
    reset_prefix_cache: bool | Mapping[str, bool] = False,
) -> dict[str, dict[str, tuple[str, str]]]:
    """Per target, the state and observed value of every determinism point.

    Server points need ``/server_info`` (vLLM dev mode); without it they are "unknown".
    The point that compares targets is "n/a" when there is only one target.
    ``reset_prefix_cache`` says whether the cache was emptied before every stage, for all
    targets or per target; pass what was recorded, not what was asked for.
    """
    settings = model_settings or {}
    temperature, seed = settings.get("temperature"), settings.get("seed")
    out: dict[str, dict[str, tuple[str, str]]] = {}
    versions: dict[str, str | None] = {}
    for name, snapshot in snapshots.items():
        info = snapshot.get("server_info") or {}
        known = bool(info)
        config, env = str(info.get("vllm_config") or ""), info.get("vllm_env") or {}
        server_seed = _setting(config, "seed")
        caching = _setting(config, "enable_prefix_caching")
        tensor = _setting(config, "tensor_parallel_size")
        invariant, multi = env.get("VLLM_BATCH_INVARIANT"), env.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        reset = reset_prefix_cache.get(name, False) if isinstance(reset_prefix_cache, Mapping) else reset_prefix_cache
        neutral = None if not known or caching is None else caching == "False" or reset
        out[name] = {
            "temperature_zero": (_flag(temperature == 0), "–" if temperature is None else str(temperature)),
            "request_seed": (_flag(seed is not None), "–" if seed is None else str(seed)),
            "server_seed": (
                _flag(None if not known else server_seed not in (None, "None")),
                server_seed or "–",
            ),
            "batch_invariant": (_flag(None if invariant is None else bool(invariant)), str(invariant or "–")),
            "prefix_cache_neutral": (
                _flag(neutral),
                "–" if caching is None else f"caching {caching}, reset {reset}",
            ),
            "single_gpu": (_flag(None if tensor is None else tensor == "1"), tensor or "–"),
            "multiprocessing_off": (_flag(None if multi is None else not multi), str(multi if known else "–")),
        }
        versions[name] = (snapshot.get("version") or {}).get("version")
    values = list(versions.values())
    if len(values) < 2:
        state = "n/a"
    elif None in values:
        state = "unknown"
    else:
        state = "on" if len(set(values)) == 1 else "off"
    for name in snapshots:
        out[name]["same_server_version"] = (state, versions[name] or "–")
    return out


def missing(config: Config, points: dict[str, dict[str, tuple[str, str]]]) -> dict[str, list[str]]:
    """Required points that are not "on" (or "n/a"), each with the targets concerned."""
    out: dict[str, list[str]] = {}
    for point in config.require:
        names = [name for name, states in points.items() if states[point][0] not in ("on", "n/a")]
        if names:
            out[point] = names
    return out
