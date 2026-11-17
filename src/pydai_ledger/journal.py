"""Journal: the append-only staging directory of a run in progress.

Layout::

    <parent>/run-<plan id>/
        run.json                 plan spec
        records-<worker>.jsonl   one JSON line per event (target, stage, attempt)
        blobs/<sha256>           binaries produced during the run

One file per worker means no file locking is needed, on Linux or Windows. A record
written later for the same attempt replaces the earlier one when the journal is sealed.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from . import signing
from ._util import dump_jsonl_xz, environment, now_iso, sha256_hex
from .ledger import FORMAT, VERSION, Ledger
from .plan import RunPlan


class IncompleteRunError(RuntimeError):
    """Raised by :meth:`Journal.seal` when planned attempts are missing or failed."""


def _blob_refs(obj: Any) -> Iterator[str]:
    """SHA-256 of every binary a record refers to (see ``_util.externalize``)."""
    if isinstance(obj, dict):
        if obj.get("kind") == "binary" and isinstance(obj.get("blob"), str):
            yield obj["blob"]
        for value in obj.values():
            yield from _blob_refs(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _blob_refs(value)


class Journal:
    """Append-only staging directory of one run; see the module docstring."""

    def __init__(self, plan: RunPlan, directory: Path, worker_id: str) -> None:
        self.plan = plan
        self.directory = directory
        self.worker_id = worker_id
        self._file = (directory / f"records-{worker_id}.jsonl").open("a", encoding="utf-8")

    @classmethod
    def open(cls, plan: RunPlan, *, parent: str | os.PathLike[str] = "tmp", worker_id: str = "worker-1") -> Journal:
        """Open the journal for ``plan``, creating it or resuming an existing one."""
        directory = Path(parent) / f"run-{plan.id[:12]}"
        (directory / "blobs").mkdir(parents=True, exist_ok=True)
        run_file = directory / "run.json"
        if not run_file.exists():
            run_file.write_text(
                json.dumps({"run_id": plan.id, "plan": plan.spec, "created_at": now_iso()}, indent=2),
                encoding="utf-8",
            )
        return cls(plan, directory, worker_id)

    # -- writing ------------------------------------------------------------------

    def append(self, record: dict[str, Any]) -> None:
        """Write one record and flush it to disk."""
        self._file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._file.flush()

    def store_blob(self, sha: str, data: bytes) -> None:
        """Store a binary produced during the run under its SHA-256."""
        path = self.directory / "blobs" / sha
        if not path.exists():
            tmp = path.with_name(f"{sha}.{self.worker_id}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)

    # -- reading ------------------------------------------------------------------

    def records(self) -> Iterator[dict[str, Any]]:
        """All records of all workers in this journal directory."""
        self._file.flush()
        for path in sorted(self.directory.glob("records-*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue  # truncated last line after a crash

    def latest(self) -> dict[str, dict[str, Any]]:
        """Per attempt id, its latest record by the time it ended: ``position`` among all
        records, ``status``, ``target`` and the ``server`` it ran on. The order of the
        workers' files plays no part, so a later repeat by another worker wins.
        """
        found: dict[str, dict[str, Any]] = {}
        for position, record in enumerate(self.records()):
            if record.get("kind") != "attempt":
                continue
            order = (record.get("ended_at") or "", position)
            if record["attempt_id"] not in found or order > found[record["attempt_id"]]["order"]:
                found[record["attempt_id"]] = {
                    "order": order,
                    "position": position,
                    "status": record["status"],
                    "target": record["target"],
                    "server": record.get("server"),
                }
        return found

    def done(self, servers: Mapping[str, str | None] | None = None) -> set[str]:
        """Attempt ids whose latest record is successful. With ``servers`` (target name to
        server key) only attempts that ran on that server count: outputs of a server that
        has changed since are not carried over.
        """
        return {
            attempt
            for attempt, entry in self.latest().items()
            if entry["status"] == "ok"
            and (servers is None or entry["target"] not in servers or entry["server"] == servers[entry["target"]])
        }

    def void_stages(self) -> set[tuple[str, int, tuple[int, ...] | None]]:
        """Stages (target, concurrency, shard) that a server change stopped and that have not
        run through again since. Such a stage holds outputs of two servers.
        """
        stopped: dict[tuple[str, int, tuple[int, ...] | None], str] = {}
        measured: dict[tuple[str, int, tuple[int, ...] | None], str] = {}
        for record in self.records():
            shard = tuple(record["shard"]) if record.get("shard") else None
            if record.get("kind") == "aborted" and "void_stage" in record:
                key = (record["target"], record["void_stage"], shard)
                stopped[key] = max(record["at"], stopped.get(key, ""))
            elif record.get("kind") == "stage" and not record.get("stopped"):
                key = (record["target"], record["concurrency"], shard)
                measured[key] = max(record["ended_at"], measured.get(key, ""))
        return {key for key, at in stopped.items() if measured.get(key, "") <= at}

    # -- finalising ---------------------------------------------------------------

    def seal(
        self,
        destination: str | os.PathLike[str],
        *,
        require_complete: bool = True,
        cleanup: bool = True,
        shard_size: int = 10_000,
        signing_key: Any = None,
    ) -> Ledger:
        """Write the run to a ledger file. The journal is removed only after success.

        ``signing_key`` (path to an Ed25519 PEM file, see :mod:`pydai_ledger.signing`)
        adds a signature.
        """
        tunings: dict[str, dict[str, Any]] = {}
        for record in self.records():
            if record.get("kind") == "tuning" and record["at"] >= tunings.get(record["target"], {"at": ""})["at"]:
                tunings[record["target"]] = record
        skipped = {  # levels the latest benchmark of a target decided not to measure
            f"{target}|c{level}|" for target, record in tunings.items() for level in record["skipped"]
        }
        latest = self.latest()
        missing = {
            attempt
            for attempt in set(self.plan.attempt_ids()) - self.done()
            if not any(attempt.startswith(prefix) for prefix in skipped)
        }
        # A target that stopped is not complete, even when every attempt is there. Starting it
        # again lifts the stop, except after a server change: then the stage has to run through
        # again, because it holds outputs of two servers.
        void = self.void_stages()
        stops: dict[str, dict[str, Any]] = {}
        started: dict[str, str] = {}
        for record in self.records():
            kind, target = record.get("kind"), record.get("target", "")
            if kind == "aborted" and record["at"] >= stops.get(target, {"at": ""})["at"]:
                stops[target] = record
            elif kind == "target":
                started[target] = max(record["at"], started.get(target, ""))
        stopped = sorted(
            {target for target, stop in stops.items() if started.get(target, "") < stop["at"]}
            | {target for target, _, _ in void}
        )
        # One target, one server: successful attempts recorded on different servers do not
        # make one result.
        servers: dict[str, set[str | None]] = {}
        for entry in latest.values():
            if entry["status"] == "ok":
                servers.setdefault(entry["target"], set()).add(entry["server"])
        mixed = sorted(target for target, keys in servers.items() if len(keys) > 1)
        if mixed and require_complete:
            raise IncompleteRunError(
                f"The attempts of target {', '.join(mixed)} come from different servers. Run execute() "
                f"again (with every shard), so that all of them run on the server as it is now; the "
                f"journal is kept at {self.directory}."
            )
        if stopped and require_complete:
            raise IncompleteRunError(
                f"Target {', '.join(stopped)} stopped before the run was over. Fix the cause and run "
                f"execute() again; the journal is kept at {self.directory}."
            )
        if missing and require_complete:
            raise IncompleteRunError(
                f"{len(missing)} of {self.plan.total_attempts} planned attempts are missing or "
                f"failed. Run execute() again to resume; the journal is kept at {self.directory}."
            )

        # Pass 1: the latest record of every attempt stays, and the small events.
        events = [record for record in self.records() if record.get("kind") != "attempt"]
        keep = {entry["position"] for entry in latest.values()}

        destination = Path(destination)
        tmp = destination.with_name(destination.name + ".tmp")
        members: dict[str, str] = {}
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            buffer: list[dict[str, Any]] = []

            def flush() -> None:
                if buffer:
                    name = f"attempts/{len(members):05d}.jsonl.xz"
                    data = dump_jsonl_xz(buffer)
                    zf.writestr(name, data)
                    members[name] = sha256_hex(data)
                    buffer.clear()

            # Pass 2: write the surviving attempt records in shards.
            used: set[str] = set()  # binaries the surviving records refer to
            for position, record in enumerate(self.records()):
                if position in keep:
                    used.update(_blob_refs(record))
                    buffer.append(record)
                    if len(buffer) >= shard_size:
                        flush()
            flush()
            blob_count = 0
            # Only binaries of attempts that are in the ledger: those of a replaced attempt stay behind.
            for blob in sorted((self.directory / "blobs").iterdir()):
                if blob.name in used:
                    zf.write(blob, f"blobs/{blob.name}")
                    blob_count += 1
            manifest = {
                "format": FORMAT,
                "version": VERSION,
                "run_id": self.plan.id,
                "plan": self.plan.spec,
                "events": events,
                "sealed_with": environment(),
                "attempt_count": len(latest),
                "complete": not missing and not stopped and not mixed,
                "blob_count": blob_count,
                "members": members,
                "sealed_at": now_iso(),
            }
            manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode()
            zf.writestr("manifest.json", manifest_bytes)
            if signing_key is not None:
                zf.writestr(signing.SIGNATURE, signing.signature_bytes(manifest_bytes, signing_key))
        os.replace(tmp, destination)

        if cleanup:
            self.close()
            shutil.rmtree(self.directory)
        return Ledger.open(destination)

    def close(self) -> None:
        """Close this worker's record file."""
        self._file.close()

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
