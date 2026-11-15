"""Small shared helpers: canonical JSON, hashing, compressed JSONL, blob references."""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import platform
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

# Stand-in payload for a binary whose blob is already known: prefix + sha256.
BLOB_PLACEHOLDER = b"pydai-ledger-blob:"


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes | str) -> str:
    """SHA-256 of bytes or UTF-8 text as lower-case hex."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def now_iso() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def environment() -> dict[str, Any]:
    """Software versions of the executing machine."""
    packages = {}
    for name in ("pydai-ledger", "pydantic-ai-slim", "pydantic", "openai", "httpx2"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"platform": platform.platform(), "python": platform.python_version(), "packages": packages}


def dump_jsonl_xz(records: Iterable[Any]) -> bytes:
    """Serialise records as xz-compressed JSON lines."""
    text = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records)
    return lzma.compress(text.encode("utf-8"))


def load_jsonl_xz(data: bytes) -> Iterator[Any]:
    """Yield the records of an xz-compressed JSON lines blob."""
    for line in lzma.decompress(data).decode("utf-8").splitlines():
        if line:
            yield json.loads(line)


def externalize(obj: Any, sink: Callable[[str, bytes], None]) -> Any:
    """Replace inline base64 binaries in dumped Pydantic AI messages by blob references.

    A dumped ``BinaryContent`` looks like ``{"kind": "binary", "data": "<base64>", ...}``.
    The bytes are handed to ``sink(sha256, raw)`` and the dict keeps ``{"blob": sha256}``.
    """
    if isinstance(obj, dict):
        if obj.get("kind") == "binary" and isinstance(obj.get("data"), str):
            # Pydantic dumps bytes as URL-safe base64; this also accepts the standard alphabet.
            data = obj["data"]
            raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
            if raw.startswith(BLOB_PLACEHOLDER):
                sha = raw[len(BLOB_PLACEHOLDER) :].decode("ascii")  # already stored, see execute._slim
            else:
                sha = sha256_hex(raw)
                sink(sha, raw)
            out = {k: v for k, v in obj.items() if k != "data"}
            out["blob"] = sha
            return out
        return {k: externalize(v, sink) for k, v in obj.items()}
    if isinstance(obj, list):
        return [externalize(v, sink) for v in obj]
    return obj


def internalize(obj: Any, load: Callable[[str], bytes]) -> Any:
    """Inverse of :func:`externalize`."""
    if isinstance(obj, dict):
        if obj.get("kind") == "binary" and "blob" in obj:
            out = {k: v for k, v in obj.items() if k != "blob"}
            out["data"] = base64.urlsafe_b64encode(load(obj["blob"])).decode("ascii")
            return out
        return {k: internalize(v, load) for k, v in obj.items()}
    if isinstance(obj, list):
        return [internalize(v, load) for v in obj]
    return obj
