"""Cases (what to ask) and the immutable Casebook archive that stores them.

A casebook is a ZIP64 file::

    manifest.json           format, id, counts, shard list
    cases/00000.jsonl.xz    case records, binaries replaced by sha256 references
    blobs/<sha256>          raw bytes (images, PDFs, long payloads), stored once
"""

from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import BinaryContent
from pydantic_core import to_jsonable_python

from . import signing
from ._util import canonical_json, dump_jsonl_xz, load_jsonl_xz, now_iso, sha256_hex

FORMAT = "pydai-ledger/casebook"
VERSION = 1

TurnContent = str | Sequence[str | BinaryContent]


@dataclass(frozen=True)
class Turn:
    """One user prompt. Content is text, or a list of text and ``BinaryContent`` items."""

    content: TurnContent

    def __post_init__(self) -> None:
        if isinstance(self.content, str):
            return
        items = tuple(self.content)
        for item in items:
            if not isinstance(item, (str, BinaryContent)):
                raise TypeError(
                    f"Turn content items must be str or BinaryContent, got {type(item).__name__}. "
                    "URLs are not accepted because their content is not reproducible."
                )
        object.__setattr__(self, "content", items)


@dataclass(frozen=True)
class Case:
    """One task: one or more turns that run sequentially with shared message history.

    ``expected`` optionally holds the desired output per turn (same length as ``turns``,
    ``None`` where unknown). It is stored alongside the case and never evaluated here.
    """

    turns: Sequence[Turn | str]
    expected: Sequence[Any] | None = None
    key: str | None = None
    metadata: Mapping[str, Any] | None = None

    @property
    def user_turns(self) -> tuple[Turn, ...]:
        """The turns as :class:`Turn` objects, also where plain strings were given."""
        return tuple(t if isinstance(t, Turn) else Turn(t) for t in self.turns)

    def __post_init__(self) -> None:
        raw = [self.turns] if isinstance(self.turns, (str, Turn)) else list(self.turns)
        turns = tuple(t if isinstance(t, Turn) else Turn(t) for t in raw)
        if not turns:
            raise ValueError("A Case needs at least one turn.")
        object.__setattr__(self, "turns", turns)
        if self.expected is not None:
            expected = tuple(
                e.model_dump(mode="json") if isinstance(e, BaseModel) else to_jsonable_python(e) for e in self.expected
            )
            if len(expected) != len(turns):
                raise ValueError(f"expected has {len(expected)} entries but the case has {len(turns)} turns.")
            object.__setattr__(self, "expected", expected)


@dataclass(frozen=True)
class StoredCase:
    """A case as read from a casebook, with its position and content hash."""

    index: int
    id: str
    case: Case
    # id(BinaryContent object) -> sha256 of its blob, so a run can reference inputs
    # without hashing them again.
    blob_shas: dict[int, str] = field(default_factory=dict, compare=False, repr=False)


def _turn_to_record(turn: Turn, blobs: dict[str, bytes]) -> Any:
    """Turn one user turn into its stored form and collect its binaries in ``blobs``."""
    if isinstance(turn.content, str):
        return turn.content
    items: list[Any] = []
    for item in turn.content:
        if isinstance(item, str):
            items.append({"text": item})
        else:
            sha = sha256_hex(item.data)
            blobs[sha] = item.data
            items.append({"blob": sha, "media_type": item.media_type})
    return items


class Casebook:
    """Read access to a casebook file. Create one with :meth:`write`."""

    def __init__(self, path: Path, zf: zipfile.ZipFile, manifest: dict[str, Any], prefix: str = "") -> None:
        self.path = path
        self.manifest = manifest
        self._zf = zf
        self._prefix = prefix
        self._names = set(zf.namelist())

    # -- creation -----------------------------------------------------------------

    @classmethod
    def write(
        cls,
        path: str | os.PathLike[str],
        cases: Iterable[Case],
        *,
        shard_size: int = 10_000,
        signing_key: Any = None,
    ) -> Casebook:
        """Write ``cases`` to a new casebook. Cases are consumed lazily, so a generator
        can feed very large sets without holding all images in memory.

        ``signing_key`` (path to an Ed25519 PEM file, see :mod:`pydai_ledger.signing`)
        adds a signature.
        """
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        seen_blobs: set[str] = set()
        case_ids: list[str] = []
        shards: dict[str, str] = {}
        buffer: list[dict[str, Any]] = []

        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:

            def flush() -> None:
                if buffer:
                    name = f"cases/{len(shards):05d}.jsonl.xz"
                    data = dump_jsonl_xz(buffer)
                    zf.writestr(name, data)
                    shards[name] = sha256_hex(data)
                    buffer.clear()

            for case in cases:
                blobs: dict[str, bytes] = {}
                turns = [_turn_to_record(t, blobs) for t in case.user_turns]
                for sha, data in blobs.items():
                    if sha not in seen_blobs:
                        zf.writestr(f"blobs/{sha}", data)
                        seen_blobs.add(sha)
                case_id = sha256_hex(canonical_json({"turns": turns}))
                buffer.append(
                    {
                        "index": len(case_ids),
                        "id": case_id,
                        "turns": turns,
                        "expected": list(case.expected) if case.expected is not None else None,
                        "key": case.key,
                        "metadata": to_jsonable_python(case.metadata),
                    }
                )
                case_ids.append(case_id)
                if len(buffer) >= shard_size:
                    flush()
            flush()
            manifest = {
                "format": FORMAT,
                "version": VERSION,
                "id": sha256_hex("\n".join(case_ids)),
                "case_count": len(case_ids),
                "blob_count": len(seen_blobs),
                "shards": list(shards),
                "members": shards,
                "created_at": now_iso(),
            }
            manifest_bytes = json.dumps(manifest, indent=2).encode()
            zf.writestr("manifest.json", manifest_bytes)
            if signing_key is not None:
                zf.writestr(signing.SIGNATURE, signing.signature_bytes(manifest_bytes, signing_key))
        os.replace(tmp, path)
        return cls.open(path)

    @classmethod
    def open(cls, path: str | os.PathLike[str], *, prefix: str = "") -> Casebook:
        """Open a casebook file. ``prefix`` selects a casebook embedded in a bundle."""
        path = Path(path)
        zf = zipfile.ZipFile(path, "r")
        manifest = json.loads(zf.read(prefix + "manifest.json"))
        if manifest.get("format") != FORMAT:
            zf.close()
            raise ValueError(f"{path} is not a casebook (format={manifest.get('format')!r}).")
        return cls(path, zf, manifest, prefix)

    # -- reading ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """Hash of all case ids; a ledger refers to its casebook by this."""
        return self.manifest["id"]

    def __len__(self) -> int:
        return self.manifest["case_count"]

    def records(self) -> Iterator[dict[str, Any]]:
        """Raw case records without loading any blob."""
        for name in self.manifest["shards"]:
            yield from load_jsonl_xz(self._zf.read(self._prefix + name))

    def load(self, record: Mapping[str, Any]) -> StoredCase:
        """Turn a raw record into a :class:`StoredCase`, reading its blobs."""
        turns = []
        blob_shas: dict[int, str] = {}
        for content in record["turns"]:
            if isinstance(content, str):
                turns.append(Turn(content))
                continue
            items: list[str | BinaryContent] = []
            for item in content:
                if "text" in item:
                    items.append(item["text"])
                else:
                    binary = BinaryContent(self.blob(item["blob"]), media_type=item["media_type"])
                    blob_shas[id(binary)] = item["blob"]
                    items.append(binary)
            turns.append(Turn(items))
        case = Case(
            turns=turns,
            expected=record.get("expected"),
            key=record.get("key"),
            metadata=record.get("metadata"),
        )
        return StoredCase(index=record["index"], id=record["id"], case=case, blob_shas=blob_shas)

    def __iter__(self) -> Iterator[StoredCase]:
        for record in self.records():
            yield self.load(record)

    @property
    def signer(self) -> str | None:
        """Public key (hex) of a valid signature, None when unsigned."""
        return signing.signer(self._zf, self._prefix)

    def verify(self, public_key: str | None = None) -> bool:
        """Check shards and blobs against their SHA-256 and the signature, if any.
        With ``public_key`` the file must be signed by exactly that key.
        """
        for name, expected in self.manifest.get("members", {}).items():
            if sha256_hex(self._zf.read(self._prefix + name)) != expected:
                return False
        blobs = self._prefix + "blobs/"
        for name in self._names:
            if name.startswith(blobs) and sha256_hex(self._zf.read(name)) != name[len(blobs) :]:
                return False
        return signing.signature_ok(self._zf, self._prefix, public_key)

    def has_blob(self, sha: str) -> bool:
        """Whether the casebook stores a binary with this SHA-256."""
        return f"{self._prefix}blobs/{sha}" in self._names

    def blob(self, sha: str) -> bytes:
        """The stored binary with this SHA-256."""
        return self._zf.read(f"{self._prefix}blobs/{sha}")

    def close(self) -> None:
        """Close the underlying file."""
        self._zf.close()

    def __enter__(self) -> Casebook:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Casebook({self.path.name!r}, cases={len(self)}, blobs={self.manifest['blob_count']}, id={self.id[:12]})"
        )
