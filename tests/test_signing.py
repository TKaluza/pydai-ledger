import warnings

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from support import Answer, make_cases, run

from pydai_ledger import (
    Casebook,
    Journal,
    Ledger,
    RunPlan,
    Target,
    execute,
    open_bundle,
    pack,
)


def test_signing(tmp_path):
    import zipfile

    from pydai_ledger import generate_key, sign

    key = tmp_path / "key.pem"
    public = generate_key(key)
    other = generate_key(tmp_path / "other.pem")

    book = Casebook.write(tmp_path / "c.zip", make_cases(), signing_key=key)
    assert book.signer == public and book.verify(public) and not book.verify(other)

    plan = RunPlan(book, Agent(output_type=Answer), [Target("t", TestModel())])
    with Journal.open(plan, parent=tmp_path) as journal:
        run(execute(plan, journal, progress=False))
        ledger = journal.seal(tmp_path / "l.zip", signing_key=key)
    assert ledger.signer == public and ledger.verify(public)

    from pydai_ledger import InvalidSignatureError, verify_stamp

    stamp = ledger.stamp()
    assert stamp == stamp.upper() and verify_stamp(stamp, public)["signer"] == public
    with pytest.raises(InvalidSignatureError):
        verify_stamp(stamp, other)
    with pytest.raises(InvalidSignatureError):
        verify_stamp(stamp.replace(stamp.split(":")[3], "0" * 64))
    assert stamp.split(":")[4] == "ED25519-SHA256"

    bundle = pack(book, [ledger], tmp_path / "b.zip")
    packed_book, (packed_ledger,) = open_bundle(bundle)
    assert packed_book.verify(public) and packed_ledger.verify(public)

    # Signing afterwards, and an unsigned file never satisfies a required key.
    plain = Casebook.write(tmp_path / "plain.zip", make_cases())
    assert plain.signer is None and plain.verify() and not plain.verify(public)
    plain.close()
    assert sign(tmp_path / "plain.zip", key) == public
    assert Casebook.open(tmp_path / "plain.zip").verify(public)

    # A changed manifest invalidates the signature.
    forged = tmp_path / "forged.zip"
    with zipfile.ZipFile(tmp_path / "l.zip") as src, zipfile.ZipFile(forged, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            dst.writestr(
                name, data.replace(b'"complete": true', b'"complete":  true') if name == "manifest.json" else data
            )
    assert not Ledger.open(forged).verify()


def test_external_signers_like_a_card(tmp_path):
    """ECDSA and RSA keys held outside the library, the way a smart card would sign."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa, utils

    from pydai_ledger import ExternalSigner, InvalidSignatureError, verify_stamp

    def spki(key) -> str:
        return (
            key.public_key()
            .public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
            .hex()
        )

    prehashed = utils.Prehashed(hashes.SHA256())
    ec_key = ec.generate_private_key(ec.SECP256R1())
    rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signers = [
        ExternalSigner("ecdsa-sha256", spki(ec_key), lambda digest: ec_key.sign(digest, ec.ECDSA(prehashed))),
        ExternalSigner(
            "rsa-pkcs1-sha256", spki(rsa_key), lambda digest: rsa_key.sign(digest, padding.PKCS1v15(), prehashed)
        ),
    ]
    book = Casebook.write(tmp_path / "c.zip", make_cases())
    plan = RunPlan(book, Agent(), [Target("t", TestModel())])
    for number, card in enumerate(signers):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Journal.open(plan, parent=tmp_path / f"j{number}") as journal:
                run(execute(plan, journal, progress=False))
                ledger = journal.seal(tmp_path / f"l{number}.zip", signing_key=card)
        assert ledger.signer == card.public_key and ledger.verify(card.public_key)
        assert verify_stamp(ledger.stamp(), card.public_key)["algorithm"] == card.algorithm

    wrong = ExternalSigner("ecdsa-sha256", spki(ec.generate_private_key(ec.SECP256R1())), signers[0].sign)
    with pytest.raises(InvalidSignatureError):  # the key on the card is not the one announced
        Casebook.write(tmp_path / "bad.zip", make_cases(), signing_key=wrong)
