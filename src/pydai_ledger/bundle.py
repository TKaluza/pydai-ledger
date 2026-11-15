"""Bundle: a casebook and its ledgers packed into one self-contained file.

Layout::

    manifest.json        format, casebook id, run ids
    casebook/...         the complete casebook
    runs/<run id>/...    one complete ledger per run
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections.abc import Iterable
from pathlib import Path

from ._util import now_iso
from .casebook import Casebook
from .ledger import Ledger

FORMAT = "pydai-ledger/bundle"
VERSION = 1


def _copy(source: zipfile.ZipFile, source_prefix: str, target: zipfile.ZipFile, target_prefix: str) -> None:
    """Stream all members below ``source_prefix`` into ``target`` without loading them."""
    for info in source.infolist():
        if info.is_dir() or not info.filename.startswith(source_prefix):
            continue
        name = target_prefix + info.filename[len(source_prefix) :]
        with source.open(info) as reader, target.open(name, "w", force_zip64=True) as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)


def pack(casebook: Casebook, ledgers: Iterable[Ledger], destination: str | os.PathLike[str]) -> Path:
    """Write ``casebook`` and all ``ledgers`` into one file: inputs, runs and outputs together."""
    ledgers = list(ledgers)
    run_ids = [ledger.manifest["run_id"] for ledger in ledgers]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("The same run was passed more than once.")
    for ledger in ledgers:
        if ledger.manifest["plan"]["casebook_id"] != casebook.id:
            raise ValueError(f"{ledger.path.name} belongs to a different casebook than {casebook.path.name}.")
    destination = Path(destination)
    tmp = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        _copy(casebook._zf, casebook._prefix, zf, "casebook/")
        for ledger in ledgers:
            _copy(ledger._zf, ledger._prefix, zf, f"runs/{ledger.manifest['run_id']}/")
        manifest = {
            "format": FORMAT,
            "version": VERSION,
            "casebook_id": casebook.id,
            "runs": run_ids,
            "packed_at": now_iso(),
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    os.replace(tmp, destination)
    return destination


def open_bundle(path: str | os.PathLike[str]) -> tuple[Casebook, list[Ledger]]:
    """Open the casebook and all ledgers inside a bundle."""
    with zipfile.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path} is not a bundle (format={manifest.get('format')!r}).")
    casebook = Casebook.open(path, prefix="casebook/")
    return casebook, [Ledger.open(path, prefix=f"runs/{run_id}/") for run_id in manifest["runs"]]
