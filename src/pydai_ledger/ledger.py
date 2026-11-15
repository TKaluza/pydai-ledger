"""Ledger: the sealed, read-only record of one run.

A ledger is a ZIP64 file::

    manifest.json              plan (agent, targets), server snapshots, stage metrics, hashes
    attempts/00000.jsonl.xz    one record per attempt with all turns, messages and timings
    blobs/<sha256>             binaries produced during the run (inputs stay in the casebook)

It references its casebook by id instead of copying the inputs.
"""

from __future__ import annotations

import json
import os
import statistics
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from . import signing
from ._util import canonical_json, load_jsonl_xz, sha256_hex

FORMAT = "pydai-ledger/ledger"
VERSION = 1


def casefold(output: Any) -> Any:
    """Normaliser for comparisons: ignore case and surrounding whitespace of all strings."""
    if isinstance(output, str):
        return output.strip().casefold()
    if isinstance(output, list):
        return [casefold(item) for item in output]
    if isinstance(output, dict):
        return {key: casefold(value) for key, value in output.items()}
    return output


@dataclass(frozen=True)
class Agreement:
    """Result of comparing outputs of the same case turn across targets and repetitions."""

    compared: int
    agreeing: int
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rate(self) -> float:
        """Share of compared turns that agree; 1.0 when nothing was compared."""
        return self.agreeing / self.compared if self.compared else 1.0


class Ledger:
    """Read access to a sealed run; see the module docstring."""

    def __init__(self, path: Path, zf: zipfile.ZipFile, manifest: dict[str, Any], prefix: str = "") -> None:
        self.path = path
        self.manifest = manifest
        self._zf = zf
        self._prefix = prefix

    @classmethod
    def open(cls, path: str | os.PathLike[str], *, prefix: str = "") -> Ledger:
        """Open a ledger file. ``prefix`` selects a ledger embedded in a bundle."""
        path = Path(path)
        zf = zipfile.ZipFile(path, "r")
        manifest = json.loads(zf.read(prefix + "manifest.json"))
        if manifest.get("format") != FORMAT:
            zf.close()
            raise ValueError(f"{path} is not a ledger (format={manifest.get('format')!r}).")
        return cls(path, zf, manifest, prefix)

    # -- raw access ---------------------------------------------------------------

    def attempts(self) -> Iterator[dict[str, Any]]:
        """All attempt records, shard by shard."""
        for name in self.manifest["members"]:
            yield from load_jsonl_xz(self._zf.read(self._prefix + name))

    def blob(self, sha: str) -> bytes:
        """A binary produced during the run, by SHA-256."""
        return self._zf.read(f"{self._prefix}blobs/{sha}")

    def events(self, kind: str) -> list[dict[str, Any]]:
        """Run events of one kind: ``"target"`` (server snapshots), ``"stage"`` (timings,
        metrics), ``"environment"``, ``"request_parameters"`` or ``"warning"``.
        """
        return [e for e in self.manifest["events"] if e.get("kind") == kind]

    def warnings(self) -> list[dict[str, Any]]:
        """Distinct warnings of the run (missing seed, temperature, no /server_info).
        A report should show these prominently: they limit reproducibility.
        """
        seen: dict[tuple[str, str], dict[str, Any]] = {}
        for event in self.events("warning"):
            seen.setdefault((event["target"], event["code"]), event)
        return list(seen.values())

    @property
    def signer(self) -> str | None:
        """Public key (hex) of a valid signature, None when unsigned."""
        return signing.signer(self._zf, self._prefix)

    def stamp(self) -> str:
        """Short identifier of this ledger: run id, manifest hash and, if signed, public
        key and signature. Reports print it as a QR code; see ``signing.verify_stamp``.
        """
        return signing.stamp(self._zf, self._prefix, self.manifest["run_id"])

    def autotune(self) -> dict[str, dict[str, Any]]:
        """Result of an autotune run per target: ``best`` concurrency, the measured
        ``levels`` with their rate, the ``metric`` and the ``skipped`` levels.
        """
        return {event["target"]: event for event in self.events("autotune")}

    def verify(self, public_key: str | None = None) -> bool:
        """Check shards and blobs against their SHA-256 and the signature, if any.
        With ``public_key`` the file must be signed by exactly that key.
        """
        for name, expected in self.manifest["members"].items():
            if sha256_hex(self._zf.read(self._prefix + name)) != expected:
                return False
        blobs = self._prefix + "blobs/"
        for name in self._zf.namelist():
            if name.startswith(blobs) and sha256_hex(self._zf.read(name)) != name[len(blobs) :]:
                return False
        return signing.signature_ok(self._zf, self._prefix, public_key)

    # -- analysis -----------------------------------------------------------------

    def rows(self) -> Iterator[dict[str, Any]]:
        """One flat row per executed turn, suitable for a dataframe."""
        for attempt in self.attempts():
            for turn in attempt["turns"]:
                usage = turn["usage"]
                yield {
                    "run_id": self.manifest["run_id"],
                    "attempt_id": attempt["attempt_id"],
                    "target": attempt["target"],
                    "concurrency": attempt["concurrency"],
                    "case_index": attempt["case_index"],
                    "case_id": attempt["case_id"],
                    "rep": attempt["rep"],
                    "status": attempt["status"],
                    "turn": turn["index"],
                    "started_at": turn["started_at"],
                    "duration_s": turn["duration_s"],
                    "model_request_s": sum(r["duration_s"] for r in turn.get("requests", [])),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "requests": usage.get("requests"),
                    "tool_calls": usage.get("tool_calls"),
                    "model_name": turn["model_name"],
                    "finish_reason": turn["finish_reason"],
                    "output": canonical_json(turn["output"]),
                }

    def agreement(self, normalize: Callable[[Any], Any] | None = None) -> Agreement:
        """Compare outputs per (case, turn) over all successful attempts.

        Outputs agree when their canonical JSON is identical. ``normalize`` is applied
        to every output first, for example :func:`casefold` to ignore case, or your own
        function that rounds numbers or drops fields. Identical bytes are not guaranteed
        by an LLM server even with a fixed seed, so read this as a rate.
        """
        variants: dict[tuple[int, int], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for row in self.rows():
            if row["status"] == "ok":
                output = row["output"]
                if normalize is not None:
                    output = canonical_json(normalize(json.loads(output)))
                variants[(row["case_index"], row["turn"])][output].append(row["attempt_id"])
        compared = agreeing = 0
        disagreements = []
        for (case_index, turn), outputs in sorted(variants.items()):
            if sum(len(ids) for ids in outputs.values()) < 2:
                continue  # nothing to compare against
            compared += 1
            if len(outputs) == 1:
                agreeing += 1
            else:
                disagreements.append({"case_index": case_index, "turn": turn, "variants": dict(outputs)})
        return Agreement(compared=compared, agreeing=agreeing, disagreements=disagreements)

    def throughput(self) -> list[dict[str, Any]]:
        """Throughput and latency per (target, concurrency) stage.

        Wall time is the union of the recorded stage intervals, so both a resumed run
        (several intervals) and parallel workers (overlapping intervals) count correctly.
        """
        intervals: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
        for stage in self.events("stage"):
            intervals[(stage["target"], stage["concurrency"])].append(
                (
                    datetime.fromisoformat(stage["started_at"]).timestamp(),
                    datetime.fromisoformat(stage["ended_at"]).timestamp(),
                )
            )
        wall: dict[tuple[str, int], float] = {}
        for key, spans in intervals.items():
            total, end = 0.0, float("-inf")
            for start, stop in sorted(spans):
                if stop > end:
                    total += stop - max(start, end)
                    end = stop
            wall[key] = total
        stats: dict[tuple[str, int], dict[str, Any]] = defaultdict(
            lambda: {"ok": 0, "errors": 0, "output_tokens": 0, "durations": []}
        )
        for attempt in self.attempts():
            entry = stats[(attempt["target"], attempt["concurrency"])]
            if attempt["status"] != "ok":
                entry["errors"] += 1
                continue
            entry["ok"] += 1
            entry["durations"].append(attempt["duration_s"])
            entry["output_tokens"] += sum(t["usage"].get("output_tokens") or 0 for t in attempt["turns"])
        result = []
        for (target, concurrency), entry in sorted(stats.items()):
            seconds = wall.get((target, concurrency), 0.0)
            durations = sorted(entry["durations"])
            result.append(
                {
                    "target": target,
                    "concurrency": concurrency,
                    "attempts_ok": entry["ok"],
                    "attempts_failed": entry["errors"],
                    "wall_s": seconds,
                    "attempts_per_s": entry["ok"] / seconds if seconds else None,
                    "output_tokens_per_s": entry["output_tokens"] / seconds if seconds else None,
                    "latency_mean_s": statistics.fmean(durations) if durations else None,
                    "latency_p50_s": durations[len(durations) // 2] if durations else None,
                    "latency_p95_s": durations[min(len(durations) - 1, int(len(durations) * 0.95))]
                    if durations
                    else None,
                }
            )
        return result

    def to_parquet(self, path: str | os.PathLike[str]) -> Path:
        """Export :meth:`rows` as a Parquet file. Needs ``pydai-ledger[analysis]``."""
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "to_parquet() needs polars. Install it with: pip install 'pydai-ledger[analysis]'"
            ) from exc
        path = Path(path)
        pl.DataFrame(list(self.rows())).write_parquet(path)
        return path

    def close(self) -> None:
        """Close the underlying file."""
        self._zf.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Ledger({self.path.name!r}, attempts={self.manifest['attempt_count']}, "
            f"complete={self.manifest['complete']}, run={self.manifest['run_id'][:12]})"
        )
