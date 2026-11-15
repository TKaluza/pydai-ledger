"""pydai-ledger: reproducible, fully recorded batch runs of Pydantic AI agents."""

from .autotune import Autotune, autotune
from .bundle import open_bundle, pack
from .casebook import Case, Casebook, StoredCase, Turn
from .compare import Comparison
from .config import Config, DeterminismError, determinism
from .execute import RunAbortedError, execute
from .journal import IncompleteRunError, Journal
from .ledger import Agreement, Ledger, casefold
from .pdf import write_pdf
from .plan import RunPlan, agent_fingerprint
from .report import write_report
from .signing import ExternalSigner, InvalidSignatureError, generate_key, pkcs11_signer, sign, verify_stamp
from .target import Target, probe, probe_capabilities, scrape_metrics

__all__ = [
    "Agreement",
    "Autotune",
    "Case",
    "Casebook",
    "Comparison",
    "Config",
    "DeterminismError",
    "ExternalSigner",
    "IncompleteRunError",
    "InvalidSignatureError",
    "Journal",
    "Ledger",
    "RunAbortedError",
    "RunPlan",
    "StoredCase",
    "Target",
    "Turn",
    "agent_fingerprint",
    "autotune",
    "casefold",
    "determinism",
    "execute",
    "generate_key",
    "open_bundle",
    "pack",
    "pkcs11_signer",
    "probe",
    "probe_capabilities",
    "scrape_metrics",
    "sign",
    "verify_stamp",
    "write_pdf",
    "write_report",
]
