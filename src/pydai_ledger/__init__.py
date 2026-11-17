"""pydai-ledger: reproducible, fully recorded batch runs of Pydantic AI agents."""

from .benchmark import Benchmark, benchmark
from .bundle import Bundle, open_bundle, pack
from .casebook import Case, Casebook, StoredCase, Turn
from .compare import Comparison
from .config import Config, DeterminismError, determinism
from .execute import RunAbortedError, execute
from .header import ReportHeader
from .journal import IncompleteRunError, Journal
from .ledger import Agreement, Ledger, casefold
from .pdf import write_pdf
from .plan import RunPlan, agent_fingerprint
from .report import write_report
from .signing import ExternalSigner, InvalidSignatureError, generate_key, pkcs11_signer, sign, verify_stamp
from .target import Target, load_targets, probe, probe_capabilities, scrape_metrics
from .tuning import Tuning

__all__ = [
    "Agreement",
    "Benchmark",
    "Bundle",
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
    "ReportHeader",
    "RunAbortedError",
    "RunPlan",
    "StoredCase",
    "Target",
    "Tuning",
    "Turn",
    "agent_fingerprint",
    "benchmark",
    "casefold",
    "determinism",
    "execute",
    "generate_key",
    "load_targets",
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
