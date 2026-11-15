"""Optional signatures for casebooks and ledgers (``pip install pydai-ledger[sign]``).

The signature covers the SHA-256 of the exact bytes of ``manifest.json``, which lists the
SHA-256 of every shard; blobs are named by their SHA-256. It is stored next to the
manifest as ``signature.json``. Because only the hash is signed, the short :func:`stamp`
(shown as a QR code in reports) can be checked on its own with :func:`verify_stamp`.

Three algorithms are understood, so a key file as well as a smart card or security key
can sign:

``ed25519-sha256``     key file made by :func:`generate_key`, or an Ed25519 card key
``ecdsa-sha256``       NIST P-256/P-384 keys, the usual choice on PKI cards
``rsa-pkcs1-sha256``   RSA keys on older cards

A card never hands out its private key. Wrap whatever signs for you in an
:class:`ExternalSigner`; :func:`pkcs11_signer` builds one for PKCS#11 tokens.

A valid signature shows who sealed the file and that it is unchanged since. It does not
show that a server really produced the recorded outputs.
"""

from __future__ import annotations

import contextlib
import getpass
import hashlib
import json
import os
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._util import now_iso

SIGNATURE = "signature.json"
STAMP = "PYDAI-LEDGER"
ED25519, ECDSA, RSA = "ed25519-sha256", "ecdsa-sha256", "rsa-pkcs1-sha256"
_HINT = "Signing needs the 'cryptography' package: pip install pydai-ledger[sign]"
_SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")


class InvalidSignatureError(ValueError):
    """The file carries a signature that does not match its manifest."""


@dataclass(frozen=True)
class ExternalSigner:
    """A key that lives outside this process, for example on a smart card.

    ``sign`` receives the 32-byte SHA-256 of the manifest and returns the signature:
    64 raw bytes for Ed25519, DER for ECDSA, the raw block for RSA PKCS#1 v1.5.
    ``public_key`` is hex: the raw 32 bytes for Ed25519, DER SubjectPublicKeyInfo
    otherwise. ``certificate`` (PEM) is stored with the signature but not validated.
    """

    algorithm: str
    public_key: str
    sign: Callable[[bytes], bytes]
    certificate: str | None = None


def _crypto() -> Any:
    """Import cryptography lazily so the package works without the extra."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_HINT) from exc
    return serialization, ed25519


def _load(key: Any, password: str | None) -> Any:
    """Return an Ed25519 private key from a key object or a PEM file."""
    serialization, ed25519 = _crypto()
    if isinstance(key, ed25519.Ed25519PrivateKey):
        return key
    loaded = serialization.load_pem_private_key(Path(key).read_bytes(), password.encode() if password else None)
    if not isinstance(loaded, ed25519.Ed25519PrivateKey):
        raise TypeError(f"{key} is not an Ed25519 private key.")
    return loaded


def _public_hex(private_key: Any) -> str:
    """Raw public key of an Ed25519 private key as hex."""
    serialization, _ = _crypto()
    raw: bytes = private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return raw.hex()


def generate_key(path: str | os.PathLike[str], *, password: str | None = None) -> str:
    """Create an Ed25519 private key file (PEM) and return the public key as hex.

    Keep the file secret; hand the returned public key to whoever verifies your files.
    """
    serialization, ed25519 = _crypto()
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite a key.")
    key = ed25519.Ed25519PrivateKey.generate()
    encryption = serialization.BestAvailableEncryption(password.encode()) if password else serialization.NoEncryption()
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, encryption))
    with contextlib.suppress(OSError):  # e.g. unusual Windows file systems
        path.chmod(0o600)
    return _public_hex(key)


def public_key(private_key: Any, *, password: str | None = None) -> str:
    """Public key (hex) that belongs to a private key file or an external signer."""
    if isinstance(private_key, ExternalSigner):
        return private_key.public_key.lower()
    return _public_hex(_load(private_key, password))


def signature_bytes(manifest: bytes, private_key: Any, password: str | None = None) -> bytes:
    """Content of ``signature.json`` for the given manifest bytes."""
    digest = hashlib.sha256(manifest).digest()
    record: dict[str, Any]
    if isinstance(private_key, ExternalSigner):
        record = {
            "algorithm": private_key.algorithm,
            "public_key": private_key.public_key.lower(),
            "signature": private_key.sign(digest).hex(),
        }
        if private_key.certificate:
            record["certificate"] = private_key.certificate
        _check(record["algorithm"], record["public_key"], record["signature"], digest)  # catch a wrong card key now
    else:
        key = _load(private_key, password)
        record = {"algorithm": ED25519, "public_key": _public_hex(key), "signature": key.sign(digest).hex()}
    record["signed_at"] = now_iso()
    return json.dumps(record, indent=2).encode()


def sign(path: str | os.PathLike[str], private_key: Any, *, password: str | None = None) -> str:
    """Sign an existing casebook or ledger file in place; returns the public key (hex).

    Close every open ``Casebook``/``Ledger`` of that file first. Bundles are not signed
    as a whole: sign the parts before packing, their signatures travel with them.
    """
    with zipfile.ZipFile(path, "a", zipfile.ZIP_STORED) as zf:
        if SIGNATURE in zf.namelist():
            raise ValueError(f"{path} is already signed.")
        manifest = zf.read("manifest.json")
        if json.loads(manifest).get("format", "").endswith("/bundle"):
            raise ValueError("Sign the casebook and the ledgers before packing them into a bundle.")
        data = signature_bytes(manifest, private_key, password)
        zf.writestr(SIGNATURE, data)
    return str(json.loads(data)["public_key"])


def _check(algorithm: str, key_hex: str, signature_hex: str, digest: bytes) -> None:
    """Raise InvalidSignatureError unless the signature over ``digest`` is valid."""
    serialization, ed25519 = _crypto()
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa, utils

    try:
        key_bytes, signature = bytes.fromhex(key_hex), bytes.fromhex(signature_hex)
        if algorithm in (ED25519, "ed25519"):
            ed25519.Ed25519PublicKey.from_public_bytes(key_bytes).verify(signature, digest)
            return
        key = serialization.load_der_public_key(key_bytes)
        prehashed = utils.Prehashed(hashes.SHA256())
        if algorithm == ECDSA and isinstance(key, ec.EllipticCurvePublicKey):
            key.verify(signature, digest, ec.ECDSA(prehashed))
        elif algorithm == RSA and isinstance(key, rsa.RSAPublicKey):
            key.verify(signature, digest, padding.PKCS1v15(), prehashed)
        else:
            raise ValueError(f"unsupported signature algorithm {algorithm!r}")
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise InvalidSignatureError("The signature does not match the manifest.") from exc


def signer(zf: zipfile.ZipFile, prefix: str = "") -> str | None:
    """Public key (hex) of a valid signature, None for an unsigned file.

    Raises :class:`InvalidSignatureError` when a signature is present but wrong.
    """
    try:
        record = json.loads(zf.read(prefix + SIGNATURE))
    except KeyError:
        return None
    digest = hashlib.sha256(zf.read(prefix + "manifest.json")).digest()
    try:
        _check(record["algorithm"], record["public_key"], record["signature"], digest)
    except KeyError as exc:
        raise InvalidSignatureError("The signature record is incomplete.") from exc
    return str(record["public_key"])


def signature_ok(zf: zipfile.ZipFile, prefix: str, public_key: str | None) -> bool:
    """True when the signature (if any) is valid and, with ``public_key`` given, made by it."""
    try:
        found = signer(zf, prefix)
    except InvalidSignatureError:
        return False
    return found == public_key.lower() if public_key else True


def stamp(zf: zipfile.ZipFile, prefix: str, run_id: str) -> str:
    """One line that identifies a ledger.

    ``PYDAI-LEDGER:1:<run id>:<manifest sha256>`` and, for a signed ledger,
    ``:<algorithm>:<public key>:<signature>``. Upper case so that a QR code can use its
    compact alphanumeric mode.
    """
    parts = [STAMP, "1", run_id[:16], hashlib.sha256(zf.read(prefix + "manifest.json")).hexdigest()]
    try:
        record = json.loads(zf.read(prefix + SIGNATURE))
        parts += [record["algorithm"], record["public_key"], record["signature"]]
    except KeyError:
        pass
    return ":".join(parts).upper()


def verify_stamp(text: str, public_key: str | None = None) -> dict[str, Any]:
    """Read a stamp (for example scanned from a report) and check its signature.

    Returns ``{"run", "manifest_sha256", "algorithm", "signer"}``; ``signer`` is None for
    an unsigned stamp. Raises :class:`InvalidSignatureError` when the signature is wrong,
    or when ``public_key`` is given and the stamp is not signed by it. The public key
    inside a stamp is only a claim: compare it with a key you trust, and compare
    ``manifest_sha256`` with ``Ledger.stamp()`` of the file you hold.
    """
    parts = "".join(text.split()).lower().split(":")
    if len(parts) not in (4, 7) or parts[0] != STAMP.lower() or parts[1] != "1":
        raise ValueError("Not a pydai-ledger stamp.")
    result: dict[str, Any] = {"run": parts[2], "manifest_sha256": parts[3], "algorithm": None, "signer": None}
    if len(parts) == 7:
        try:
            digest = bytes.fromhex(parts[3])
        except ValueError as exc:
            raise InvalidSignatureError("The stamp is damaged.") from exc
        _check(parts[4], parts[5], parts[6], digest)
        result.update(algorithm=parts[4], signer=parts[5])
    if public_key and result["signer"] != public_key.lower():
        raise InvalidSignatureError("The stamp is not signed by the given key.")
    return result


def pkcs11_signer(
    module: str | os.PathLike[str],
    *,
    token_label: str | None = None,
    key_label: str | None = None,
    key_id: bytes | None = None,
    pin: str | None = None,
    certificate: str | None = None,
) -> ExternalSigner:
    """Signer for a key on a PKCS#11 token: PKI smart card, Nitrokey, YubiKey, HSM.

    ``module`` is the PKCS#11 library of the card driver, for example OpenSC's
    ``opensc-pkcs11.so`` (Linux) or ``opensc-pkcs11.dll`` (Windows). The token is chosen
    by ``token_label`` (the first one if omitted), the key by ``key_label`` or
    ``key_id``. Without ``pin`` it is asked for on the terminal. Needs the
    ``python-pkcs11`` package (``pydai-ledger[card]``). The session stays open for the
    life of the returned signer.

    Written against the PKCS#11 interface but not yet tested with hardware.
    """
    try:
        import pkcs11
        from pkcs11 import Attribute, KeyType, Mechanism, ObjectClass
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Card signing needs python-pkcs11: pip install 'pydai-ledger[card]'") from exc
    serialization, _ = _crypto()
    from cryptography.hazmat.primitives.asymmetric import rsa, utils

    library = pkcs11.lib(str(module))
    token = library.get_token(token_label=token_label) if token_label else next(iter(library.get_tokens()))
    session = token.open(user_pin=pin if pin is not None else getpass.getpass(f"PIN for {token.label}: "))
    query: dict[str, Any] = {}
    if key_label is not None:
        query["label"] = key_label
    if key_id is not None:
        query["id"] = key_id
    private: Any = session.get_key(object_class=ObjectClass.PRIVATE_KEY, **query)
    public: Any = session.get_key(object_class=ObjectClass.PUBLIC_KEY, **query)

    if private.key_type == KeyType.EC:
        from pkcs11.util.ec import encode_ec_public_key

        def sign_ec(digest: bytes) -> bytes:
            raw = bytes(private.sign(digest, mechanism=Mechanism.ECDSA))
            half = len(raw) // 2
            return utils.encode_dss_signature(int.from_bytes(raw[:half], "big"), int.from_bytes(raw[half:], "big"))

        return ExternalSigner(ECDSA, bytes(encode_ec_public_key(public)).hex(), sign_ec, certificate)
    if private.key_type == KeyType.RSA:
        numbers = rsa.RSAPublicNumbers(
            int.from_bytes(public[Attribute.PUBLIC_EXPONENT], "big"), int.from_bytes(public[Attribute.MODULUS], "big")
        )
        spki = numbers.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )

        def sign_rsa(digest: bytes) -> bytes:
            return bytes(private.sign(_SHA256_DIGEST_INFO + digest, mechanism=Mechanism.RSA_PKCS))

        return ExternalSigner(RSA, spki.hex(), sign_rsa, certificate)
    if private.key_type == KeyType.EC_EDWARDS:
        point = bytes(public[Attribute.EC_POINT])  # the point is wrapped in a DER octet string

        def sign_ed(digest: bytes) -> bytes:
            return bytes(private.sign(digest, mechanism=Mechanism.EDDSA))

        return ExternalSigner(ED25519, point[-32:].hex(), sign_ed, certificate)
    raise TypeError(f"Unsupported key type on the token: {private.key_type!r}")
