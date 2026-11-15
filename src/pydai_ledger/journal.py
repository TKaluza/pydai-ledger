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
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import signing
from ._util import dump_jsonl_xz, environment, now_iso, sha256_hex
from .ledger import FORMAT, VERSION, Ledger
from .plan import RunPlan


class IncompleteRunError(RuntimeError):
    """Raised by :meth:`Journal.seal` when planned attempts are missing or failed."""


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

    def done(self) -> set[str]:
        """Attempt ids whose latest record is successful."""
        status: dict[str, str] = {}
        for record in self.records():
            if record.get("kind") == "attempt":
                status[record["attempt_id"]] = record["status"]
        return {attempt for attempt, state in status.items() if state == "ok"}

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
        skipped = {  # levels an autotune run decided not to measure
            f"{record['target']}|c{level}|"
            for record in self.records()
            if record.get("kind") == "autotune"
            for level in record["skipped"]
        }
        missing = {
            attempt
            for attempt in set(self.plan.attempt_ids()) - self.done()
            if not any(attempt.startswith(prefix) for prefix in skipped)
        }
        if missing and require_complete:
            raise IncompleteRunError(
                f"{len(missing)} of {self.plan.total_attempts} planned attempts are missing or "
                f"failed. Run execute() again to resume; the journal is kept at {self.directory}."
            )

        # Pass 1: remember where the latest record of every attempt is, keep small events.
        latest: dict[str, int] = {}
        events: list[dict[str, Any]] = []
        for position, record in enumerate(self.records()):
            if record.get("kind") == "attempt":
                latest[record["attempt_id"]] = position
            else:
                events.append(record)
        keep = set(latest.values())

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
            for position, record in enumerate(self.records()):
                if position in keep:
                    buffer.append(record)
                    if len(buffer) >= shard_size:
                        flush()
            flush()
            blob_count = 0
            for blob in sorted((self.directory / "blobs").iterdir()):
                if not blob.name.endswith(".tmp"):
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
                "complete": not missing,
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
