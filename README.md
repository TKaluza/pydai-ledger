# pydai-ledger

Reproducible, fully recorded batch runs of [Pydantic AI](https://ai.pydantic.dev) agents,
built for vLLM servers and other OpenAI-compatible endpoints.

**Status:** version 0.1, an early release. The API and the file formats may still change.
Developed and tested on Linux; Windows is covered by the test suite in CI only.

- [Installation](#installation)
- [Usage](#usage)
- [Concept](#concept): the idea, the steps of a run, what is recorded and what is not
- [Security](#security)
- [How this project was made](#how-this-project-was-made)

## Installation

Python 3.12 or newer. Until there is a release on PyPI, install from the repository:

```bash
pip install "pydai-ledger[report,pdf,sign] @ git+https://github.com/TKaluza/pydai-ledger"
```

The core records runs and needs nothing else. Extras add the optional parts:

| Extra | Adds | Needed for |
|---|---|---|
| `report` | pillow, segno | `write_report()` (HTML) |
| `pdf` | typst, segno | `write_pdf()` |
| `sign` | cryptography | signing and verifying with a key file |
| `card` | cryptography, python-pkcs11 | signing with a smart card or token |
| `analysis` | polars | `ledger.to_parquet()` |

## Usage

```python
book = Casebook.write("demo.casebook.zip", [Case(turns=["Say 'A'", "Say 'B'", "What is that together?"])])
plan = RunPlan(
    book,
    agent,
    targets=[Target("gpu-a", model_a), Target("gpu-b", model_b)],
    attempts_per_case=3,
    concurrency=[1, 4, 16],
)
with Journal.open(plan, parent="tmp") as journal:
    await execute(plan, journal)
    ledger = journal.seal("demo.ledger.zip")

ledger.agreement()  # same output across targets and repetitions?
ledger.throughput()  # per target and concurrency level
ledger.warnings()  # missing seed, temperature > 0, no /server_info
ledger.to_parquet("demo.parquet")  # needs pydai-ledger[analysis]

pack(book, [ledger], "demo.bundle.zip")  # inputs, runs and outputs in one file
book, ledgers = open_bundle("demo.bundle.zip")

head = dict(title="Colour recognition", author="Tim", number="R-2026-007", date="2026-11-13")
write_report(ledger, "demo.html", casebook=book, lang="de", **head)  # needs pydai-ledger[report]
write_pdf(ledger, "demo.pdf", casebook=book, lang="de", **head)  # needs pydai-ledger[pdf]
```

A complete script is in [examples/quickstart.py](examples/quickstart.py).
[examples/synthetic_report.py](examples/synthetic_report.py) writes sample reports (HTML and
PDF, English and German) into `reports/` from a pretend run and needs no server.

### Options of `execute()`

- `timeout=300` – seconds per turn; a slower turn is recorded as failed.
- `max_consecutive_failures=20` – stop a target with `RunAbortedError` after that many
  failures in a row, or at once when its first request cannot reach the server.
- `reset_prefix_cache=True` – call vLLM's `/reset_prefix_cache` before every stage
  (dev mode). Off by default; a warning is recorded when it could matter.
- `warmup=2` – run the first cases once per target before measuring, results discarded.
  Off by default.
- `probe_capabilities=True` – three small requests per HTTP target before the run to
  record whether tool output, native JSON-schema output and image input work. On by default.
- `shard=(i, n)` – run only cases with `index % n == i`. Start `n` processes with their
  own `worker_id` on the same journal directory and seal once.
- `config="pydai-ledger.toml"` – note levels and determinism requirements, see below.
  Without it the file in the working directory is used when it exists.

### Configuration: `pydai-ledger.toml`

The file never changes what is recorded. It sets how notes appear in reports and which
determinism points must hold before a run starts. A commented sample is in
[examples/pydai-ledger.toml](examples/pydai-ledger.toml).

```toml
[report.notes]              # error (red) | warning (amber) | info (grey) | off
prefix_cache_not_reset = "info"
no_seed = "error"

[report.remarks]            # your own background text for a note
provider_only = "The gateway hides /server_info on purpose."

[determinism]
require = ["temperature_zero", "request_seed", "batch_invariant"]
on_missing = "error"        # or "warning"
```

- **Notes.** Every note has a code and a default level: `stopped`, `incomplete`,
  `failed_attempts`, `invalid_signature` are errors; `no_seed`, `no_temperature`,
  `temperature_nonzero`, `provider_only`, `prefix_cache_reset_failed` are warnings;
  `prefix_cache_not_reset` is information and stands with the throughput figures, because it
  does not touch the answers. The HTML report explains each note under "Background".
  A report names the notes that the configuration hides.
- **Determinism points.** `temperature_zero`, `request_seed`, `server_seed`,
  `batch_invariant` (`VLLM_BATCH_INVARIANT=1`), `prefix_cache_neutral`, `single_gpu`,
  `multiprocessing_off`, `same_server_version`. Every report lists them in a short
  section as yes, no or unknown, naming targets only where they differ. Server points need
  vLLM dev mode; without it they are unknown and count as missing.
- **Start check.** With `on_missing = "error"` `execute()` reads all servers first and
  raises `DeterminismError` before the first attempt when a required point is missing.
  With `"warning"` the run starts and the report shows the gap.
- The points are preconditions. Whether outputs really were identical is measured: see
  agreement across repetitions and concurrency levels in the report.

- **Server must not change.** After every stage `execute()` asks the server again for its
  version and models. If either differs from the start (restart, update, another model),
  the target is stopped with `RunAbortedError` and the report shows an error.

Pass the same file to `write_report(..., config=...)` and `write_pdf(..., config=...)`.

### Comparing runs

Runs of the same casebook can be reported together, even when they were made on
different days or machines. Target names must differ.

```python
from pydai_ledger import Comparison, Ledger, write_pdf

with Comparison([Ledger.open("vllm.ledger.zip"), Ledger.open("other.ledger.zip")]) as both:
    write_pdf(both, "compare.pdf", casebook=book)
```

The comparison is a view; nothing is written or re-signed. The report lists the stamp of
every run in the run details, and its own QR code is a hash over those stamps.

### Autotune

```python
tune = await autotune(book, agent, target, "tune.ledger.zip", sample=64)
tune.best  # concurrency with the highest throughput
tune.levels  # [{"concurrency": 1, "rate": 29.1}, ...] in output tokens per second
```

Runs the first `sample` cases at 1, 2, 4, 8, ... parallel requests and stops as soon as
a level is not at least `min_gain=0.05` faster than the best before it. The prefix cache
is reset before every level, because all levels run the same cases. The measurements
are an ordinary ledger and can be reported like any run.

### Reports

`write_report()` writes one self-contained HTML file, `write_pdf()` a printable PDF
(typeset with Typst). Both show the same figures in English or German (`lang="en" | "de"`):
key figures, warnings, a models table (server, GPU, context, weights, capabilities),
throughput per concurrency level, vLLM's own means per stage, agreement between targets,
and the cases.

- `title`, `author`, `number` and `date` head the report; the date defaults to the sealing time.
- `normalize=casefold` (or your own function) is applied to outputs before they are compared,
  so "blue" and "Blue" count as equal. `ledger.agreement(casefold)` does the same in code.
- PDF `style`: `"plain"` (space only, default), `"ruled"` (one line under table headers) or
  `"banded"` (shaded rows).
- Figures follow scientific practice: one figure with panels (a) to (c), a shared legend, axes
  with arrows, quantity and unit, and a numbered caption below. They are left out when a run
  has a single concurrency level.
- A result grid shows every case per target as a green, yellow or red cell. In the HTML report a
  cell shows its case on hover and opens it in the case table on click.
- The PDF footer names report number, run, manifest hash and date on every page, so a single
  loose page can be assigned.
- In the PDF, prompts and outputs longer than `excerpt_chars=240` are shortened in the notable
  cases; their full text is set in the last appendix.
- HTML: compact case table with filter, search, sorting and paging. Cases are embedded
  compressed, so thousands stay small; details open on click.
- PDF: up to `max_notable=50` notable cases in detail and all cases as a dense appendix.
  A further appendix prints what every server reported: `/server_info` with vLLM
  configuration, environment variables, GPU and library versions.
- Image previews for the first `max_images=30` images (needs Pillow).
- Fonts are bundled: Jost for text, Ubuntu Mono for code and numbers
  (see `src/pydai_ledger/fonts/` for their licences).
- A QR code carries the ledger stamp (see below).

### Signing (optional)

Needs `pydai-ledger[sign]`. An Ed25519 signature over the SHA-256 of the manifest, which
lists the SHA-256 of every shard, is stored inside the file and travels into bundles.

```python
public = generate_key("ledger-key.pem")  # keep the file secret, share `public`
book = Casebook.write("demo.casebook.zip", cases, signing_key="ledger-key.pem")
ledger = journal.seal("demo.ledger.zip", signing_key="ledger-key.pem")
sign("older.ledger.zip", "ledger-key.pem")  # sign an existing file
ledger.verify(public)  # hashes intact and signed by exactly this key
ledger.signer  # public key of a valid signature, or None

stamp = ledger.stamp()  # PYDAI-LEDGER:1:<run>:<manifest sha256>[:<algorithm>:<public key>:<signature>]
verify_stamp(stamp, public)  # checks a stamp scanned from a report, without the file
```

A key on a smart card or security key (PKI card, Nitrokey, YubiKey, HSM) can sign as well.
ECDSA P-256/P-384, RSA and Ed25519 keys are understood:

```python
card = pkcs11_signer("/usr/lib/opensc-pkcs11.so", key_label="Signature key")  # asks for the PIN
ledger = journal.seal("demo.ledger.zip", signing_key=card)  # needs pydai-ledger[card]
```

`pkcs11_signer` is written against PKCS#11 but not yet tested with hardware. Any other way
to sign works through `ExternalSigner(algorithm, public_key, sign)`, where `sign` receives
the SHA-256 of the manifest and returns the signature.

The stamp is what the QR code in a report contains. It ties a printed report to exactly
one ledger file (`ledger.stamp()` of the file must be equal) and, for a signed ledger,
to the key that sealed it. The public key inside a stamp is only a claim; compare it
with a key you trust. A signature shows who sealed a file and that it is unchanged. It
does not prove that a server really produced the recorded outputs.

### Notes

- Start vLLM with `VLLM_SERVER_DEV_MODE=1` so `/server_info` is recorded. Without it a
  warning is emitted and the run is marked `provenance: "provider_only"`. Dev mode exposes
  unprotected control endpoints; use it on trusted networks only.
- Structured output through tool calls needs vLLM's `--enable-auto-tool-choice` and
  `--tool-call-parser`. Without them use `Agent(output_type=NativeOutput(MyModel))`.
- Images are passed as Pydantic AI `BinaryContent`. URLs are rejected because their
  content is not reproducible.

### Known limits

- Agents with `deps` are not supported yet.
- The capability check sends one turn. A model can pass it for tool output and still
  answer in plain text in a later turn of a conversation; `NativeOutput` avoids that.
- A report keeps all outputs in memory and the PDF appendix prints every case. For runs
  with very many cases use `max_cases` and `all_cases=False`. Memory use of runs with
  more than a few thousand cases has not been measured.
- Signing with PKCS#11 hardware is implemented but has not been tested on a device.
- Windows has not been used outside the test suite.

### Development

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ruff format . && uv run ruff check . && uv run ty check
```

A release is a tag: set the version in `pyproject.toml`, then push `v<version>`. The
release workflow tests, builds, publishes to PyPI (trusted publishing, environment `pypi`)
and creates the GitHub release.

## Concept

### Idea

A run should be repeatable and provable later: what was asked, of which model on which
server, with which settings, and what came back, how fast. So the question and the
answer are kept in two kinds of files that never change after they are written.

| Part | What it is | Changes? |
|---|---|---|
| **Casebook** | The task: cases with one or more user turns, images, optional expected outputs, keys and metadata. Knows nothing about agent, model or server. | never |
| **RunPlan** | Casebook × agent × targets × concurrency levels × repetitions. Its hash names the run. | in memory only |
| **Journal** | Working directory of a run in progress; one append-only file per worker. | grows until sealed |
| **Ledger** | The sealed result of one plan. Refers to its casebook by hash. | never |
| **Bundle** | Casebook and any number of its ledgers in one file. | never |

### Terms

| Term | Meaning |
|---|---|
| **Case** | One conversation to run: one or more user turns, optionally with images and expected outputs. |
| **Turn** | One user message and the model's answer to it. Later turns see the whole history. |
| **Target** | One model on one server, with a name of your choice. Reports compare targets. |
| **Attempt** | One execution of one case on one target at one concurrency level. Repetitions are further attempts. |
| **Concurrency level** | How many attempts are in flight at once. Each level is its own **stage**, measured separately. |
| **Stage** | All attempts of one target at one concurrency level. |
| **Output** | The final result of a turn, as validated by Pydantic AI (text or structured data). |
| **Expected output** | What a turn should return, stored in the casebook. "As expected" counts matches. |
| **Agreement / identical outputs** | Share of turns for which all successful attempts, across targets, repetitions and levels, returned the same output. |
| **Throughput** | Successful attempts per second and output tokens per second of a stage, by wall time. |
| **Latency** | Duration of one attempt (all its turns); mean, median (p50) and 95th percentile (p95). |
| **Output mode** | How structured output is requested: `tool` (tool call), `native` (JSON schema enforced by the server), `text`. |
| **Capabilities** | Whether tool output, native output and image input worked in three small test requests before the run. |
| **Server snapshot** | What the server reported at the start: `/server_info` (vLLM dev mode), `/version`, `/v1/models`. |
| **provider_only** | No `/server_info` was available, so only model name and base URL identify the server. |
| **Note** | A finding about the run with a level: error (result affected), warning (comparability limited), information (context). |
| **Determinism point** | One precondition for bit-identical outputs, such as temperature 0 or batch invariance. |
| **Prefix cache** | The server keeps computed prompt beginnings; repeated prompts get faster. Affects speed, not answers. |
| **Batch invariance** | vLLM option that makes an output independent of what else is in the same batch; costs throughput. |
| **Run id** | Hash of the plan. The same plan always has the same id, which makes a run resumable. |
| **Stamp** | Short identifier of a ledger: run id, manifest hash and, if signed, key and signature. Printed as QR code. |
| **Comparison** | Several ledgers of one casebook read as one run for a joint report. |

Both file types are ZIP64 archives with a `manifest.json`, xz-compressed JSONL shards of
10,000 records and binaries stored once under their SHA-256. This holds a few cases as
well as 150,000 images.

### Steps of a run

1. **Define** – write the cases to a casebook. Each case gets the hash of its turns as
   id; the casebook id is the hash of all case ids.
2. **Plan** – combine casebook, agent and targets. The plan id is the hash of the
   casebook id, the agent fingerprint, the targets, the levels and the repetitions.
   The same plan always leads to the same journal, which makes a run resumable.
3. **Probe** – per target: `/server_info`, `/version`, `/v1/models`, and three small
   requests for capabilities.
4. **Warm up** (optional) – a few unrecorded cases so model loading is not measured.
5. **Execute** – per target one stage per concurrency level, one after another; targets
   run side by side. Server metrics are read before and after each stage. Every attempt
   is written to the journal the moment it ends.
6. **Resume** – calling `execute()` again skips successful attempts and repeats failed
   and missing ones.
7. **Seal** – the latest record of every attempt goes into the ledger with hashes of
   all parts; optionally signed. The journal is deleted only after that.
8. **Analyse and report** – agreement, throughput, Parquet export, HTML and PDF.
9. **Pack** (optional) – casebook and ledgers into one bundle for hand-over.

### What is recorded

| Area | Content |
|---|---|
| Inputs | Every turn with text and images (by hash), expected outputs, key, metadata. |
| Agent | Instructions, system prompts, tool definitions, output schema, model settings, Pydantic AI version. |
| Target | Name, model name, base URL, model settings, free-form `info` from the operator. |
| Server | Full `/server_info` (vLLM configuration, environment variables, GPU, driver, library versions), `/version`, `/v1/models`, tested capabilities. |
| Client | Platform, Python version, versions of pydai-ledger, pydantic-ai, pydantic, openai, httpx2; the options `execute()` was called with. |
| Each request | Merged settings as sent (temperature, seed, ...), tool and output definitions (once per distinct set), duration. |
| Each turn | All new messages including tool calls and results, output, token usage, model name returned by the server, finish reason, start time, duration. |
| Each attempt | Target, concurrency level, case, repetition, worker, status, error text and kind (`api`, `timeout`, `run`), start and end. |
| Each stage | Start, end, wall time, attempts, output tokens, whether the prefix cache was reset, all `/metrics` samples before and after. |
| Configuration | The settings of `pydai-ledger.toml` in force when `execute()` ran. |
| Run | Warnings, warm-up, autotune decision, early stops, hashes of all parts, signature. |

### What is not recorded, and what to do about it

| Gap | Why | Remedy |
|---|---|---|
| Identity of the model weights | vLLM reports only the path or name it was started with, no hash. | Put revision or checksum into `Target(info={...})`. |
| Chat template and server-side sampling defaults | Not part of `/server_info` in full. | Set temperature and seed explicitly (a warning reminds you); note the template in `info`. |
| Raw HTTP exchange | Recorded at the level of Pydantic AI messages, not bytes on the wire. Hidden retries of the HTTP client make a request look slower, not different. | Compare with the server-side means from `/metrics` in the report. |
| Dependencies of an agent (`deps`) | Agents with `deps` are not supported yet. | – |
| Tool behaviour | Tool calls and results are recorded, the tool code is not. A replay needs the same tools. | Version the tool code with the casebook. |
| Client host name and network | Left out on purpose; latency includes the network. | Add the host to `info` if it matters. |
| Proof of origin | A signature proves who sealed the file, not that a server produced the outputs. | – |

Identical outputs are not guaranteed by a server even with a fixed seed (batching and
hardware differences), so agreement is reported as a rate.

## Security

- `/server_info`, `/reset_prefix_cache` and the other dev-mode endpoints of vLLM
  (`VLLM_SERVER_DEV_MODE=1`) need no authentication. Run such a server on a trusted
  network only.
- A ledger and its reports contain the server snapshot: configuration, environment
  variables, installed packages, CPU and GPU. Values whose name has a part `TOKEN`, `KEY`,
  `SECRET`, `PASSWORD`, `PASSWD` or `CREDENTIAL(S)` are replaced by `[redacted]` before
  they are stored. Other values are kept as the server reports them, so read a report
  before you pass it on.
- Ledgers and reports contain the inputs and outputs of every case in full.
- A signing key is a secret. It belongs neither in the repository nor in a report; the
  QR code of a report carries the ledger stamp only.

## How this project was made

| Part | Done by |
|---|---|
| Architecture and design | Human |
| Programming | AI (Claude, Anthropic), directed and reviewed by the author |
| Testing | AI (automated tests) and human (live runs against vLLM and review of the reports) |
