"""PDF report of one ledger, typeset with Typst (``pip install pydai-ledger[pdf]``).

The figures come from :mod:`pydai_ledger.summary`, shared with the HTML report. They are
written as ``data.json`` next to chart SVGs, preview images and the QR code into a
temporary folder; a fixed Typst template reads them. No text from a run is pasted into
Typst markup, so prompts and outputs cannot change the document.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .casebook import Casebook
from .config import Config
from .header import ReportHeader
from .ledger import Ledger
from .report import FONTS
from .summary import GRID_COLOURS, MAX_SERIES, PRINT_PALETTE, _curve, _result_grid, build

_TEMPLATE = r"""
#let d = json("data.json")
#let L = d.L
#let ink = rgb("#111111")
#let soft = ink  // all text is black, so a print is clean; emphasis comes from size, weight and italics
#let red = rgb("#c0392b")
#let amber = rgb("#a86b00")
#let green = rgb("#1a7f37")
#let band = luma(245)
#let mono = ("Ubuntu Mono", "DejaVu Sans Mono")

// Page: equal side margins, bottom larger than top; header and footer carry little, no rules.
#set document(title: d.title, author: if d.author != none { d.author } else { () })
#set page(
  paper: d.paper,
  margin: (x: 21mm, top: 25mm, bottom: 32mm),
  header: context {
    if counter(page).get().first() > 1 {
      set text(8pt, fill: soft)
      grid(columns: (1fr, auto), d.title, d.run)
    }
  },
  // The footer identifies a single loose page: report number, run, manifest hash, date.
  footer: context {
    set text(8pt, fill: soft)
    grid(columns: (1fr, auto), d.foot, [#L.page #counter(page).display() / #counter(page).final().first()])
  },
)
// Type: one family, two weights, four sizes (title, heading, body, small).
#set text(font: ("Lato", "New Computer Modern"), size: 9.5pt, fill: ink, lang: d.lang)
#set par(leading: 0.55em)
#show raw: set text(font: mono, size: 8.5pt)
// Emphasis is either size or weight, never both: title and headings are larger and
// regular, sub-headings and case ids are body size and bold.
#show heading.where(level: 1): it => block(above: 22pt, below: 9pt, sticky: true, text(13pt, weight: "regular", it.body))
#show heading.where(level: 2): it => block(above: 12pt, below: 5pt, sticky: true, text(9.5pt, weight: "bold", it.body))

#let small(body) = text(8pt, fill: soft, body)

// Status letters with their colour; the letter keeps it readable in greyscale.
#let marks(flags) = {
  if calc.rem(flags, 2) == 1 { text(fill: red)[F ] }
  if calc.rem(calc.quo(flags, 2), 2) == 1 { text(fill: amber)[D ] }
  if calc.quo(flags, 4) == 1 { text(fill: red)[X] }
}
#let th(h) = text(weight: "bold", h)
// A caption above a table, with room above it; a remark is a sentence about how something ran.
#let cap(body) = block(above: 15pt, below: 5pt, sticky: true, small(body))
#let remark(body) = block(above: 6pt, text(8.5pt, style: "italic", body))
// A figure: content, then a numbered caption below it.
#let fig(number, caption, body) = block(width: 100%, above: 6pt, below: 10pt, breakable: false)[
  #body
  #v(-2pt)
  #text(8.5pt)[#L.figure #number. #text(fill: soft, caption)]
]
#let key(colour, name) = { box(baseline: -0.5pt, rect(width: 7pt, height: 5pt, fill: rgb(colour))); h(3pt); small(name); h(10pt) }

// Tables: no lines at all; every other row is shaded.
#let row-fill(y) = if y > 0 and calc.odd(y) { band } else { none }
#let tbl(t, size: 8.5pt) = {
  set text(size)
  table(
    columns: t.cols.map(c => if c == 0 { auto } else { c * 1fr }),
    align: (x, y) => if t.align.at(x) == "r" { right } else if t.align.at(x) == "c" { center } else { left },
    stroke: none,
    fill: (x, y) => row-fill(y),
    inset: (x: 4pt, y: 3pt),
    table.header(..t.head.map(th)),
    ..t.rows.map(r => r.enumerate().map(((x, c)) => if t.align.at(x) == "r" { text(font: mono, c) } else { [#c] })).flatten(),
  )
}

// ---- summary -------------------------------------------------------------
#grid(
  columns: (1fr, auto),
  column-gutter: 8mm,
  [
    #text(17pt, d.title)
    #v(-3pt)
    #for line in d.head { small(line); linebreak() }
  ],
  if d.qr != none { image(d.qr, width: 19mm) },
)
#v(8pt)
#grid(
  columns: d.facts.map(f => 1fr),
  column-gutter: 8pt,
  ..d.facts.map(f => stack(spacing: 5pt, small(f.at(0)), text(15pt, font: mono, f.at(1)), small(f.at(2)))),
)
#v(4pt)
// Notes: the label stands in its own column, so wrapped text keeps one left edge.
// The bar shows the level: red for errors, amber for warnings, grey for information.
#let notes(list) = if list.len() > 0 {
  set text(8.5pt)
  grid(
    columns: (auto, 1fr),
    column-gutter: 10pt,
    row-gutter: 6pt,
    ..list.map(n => (
      grid.cell(
        stroke: (left: 2pt + if n.level == "error" { red } else if n.level == "warning" { amber } else { soft }),
        inset: (left: 6pt, y: 1pt),
        text(weight: "bold", n.code),
      ),
      grid.cell(inset: (y: 1pt), n.text),
    )).flatten(),
  )
}
#notes(d.notes.filter(n => n.section == ""))
#if d.config_line != "" { small(d.config_line) }

#if d.show.targets [
  = #L.models
  #block(below: 5pt, small(L.hardware_title))
  #tbl(d.models)
  #cap(L.setup_title)
  #tbl(d.capabilities)
]

#if d.show.benchmark [
  = #L.throughput
  #block(sticky: true, notes(d.notes.filter(n => n.section == "throughput")))
  #if d.curve != none {
    fig(d.chart_number, d.chart_caption)[
      #for s in d.series { key(s.at(1), s.at(0)) }
      #align(center, image(d.curve, width: 82%))
    ]
  }
  #v(6pt)
  #tbl(d.throughput)
  #if d.tuning != none [
    #cap(L.result_title)
    #tbl(d.tuning)
  ]
  #if d.server != none [
    == #L.server_means
    #tbl(d.server)
  ]
]

// Reproducibility between targets with one model, as wide as its content. A cell turns
// amber as the share of identical outputs falls below 100 %.
#let shade(v) = amber.transparentize(calc.max(0%, 100% - (1 - v) * 400%))
#let matrix(b) = {
  set text(8.5pt)
  table(
    columns: (auto,) * (b.targets.len() + 1),
    align: (x, y) => if x == 0 { left } else { right },
    stroke: none,
    inset: (x: 6pt, y: 3.5pt),
    fill: (x, y) => if x > 0 and y > 0 {
      let v = b.values.at(y - 1).at(x - 1)
      if v != none and v < 1 { shade(v) } else { none }
    } else { none },
    table.header([], ..b.targets.map(th)),
    ..b.targets.enumerate().map(((y, name)) => (th(name), ..b.rows.at(y).map(c => text(font: mono, c)))).flatten(),
  )
}
// The preconditions for identical outputs: one line per point, set in two columns.
#let det(items) = {
  set text(8.5pt)
  block(above: 7pt, grid(
    columns: (auto, 1fr, auto, 1fr),
    column-gutter: (7pt, 16pt, 7pt),
    row-gutter: 6pt,
    ..items.map(i => (
      text(font: mono, fill: if i.state == "on" { green } else if i.state == "off" { red } else if i.state == "mixed" { amber } else { soft }, i.word),
      [#i.label#if i.required [ #text(fill: soft)[(#L.d_required)]]#if i.value != "" [ #h(3pt)#text(7.5pt, font: mono, fill: soft, i.value)]],
    )).flatten(),
  ))
}
// Validation after ISO 5725: setup, precision, trueness, accuracy, block by block.
#if d.show.validation [
= #L.validation
#for b in d.validation {
  if b.kind == "h2" { heading(level: 2, b.text) }
  else if b.kind == "text" { block(above: 5pt, below: 5pt, small(b.text)) }
  else if b.kind == "det" { det(b.items) }
  else {
    block(breakable: false)[
      #if b.cap != none { cap(b.cap) } else { v(2pt) }
      #if b.kind == "matrix" { matrix(b) } else { tbl(b.table) }
    ]
  }
}
#if d.model_pairs != none [
  #block(breakable: false)[
    = #L.model_comparison
    #block(below: 5pt, small(L.model_comparison_note))
    #tbl(d.model_pairs)
  ]
]
#block(breakable: false)[
  = #L.execution_title
  #tbl(d.per_target)
  #if d.execution != "" { remark(d.execution) }
]
]

// ---- notable cases ---------------------------------------------------------
// One case is one block: its id stands at the far left, every turn is a row with the
// prompt on the left and the outputs on the right.
#let case-block(c, shaded) = {
  let id = [#text(weight: "bold", c.id) #h(3pt) #marks(c.f)]
  let body = {
    for (n, t) in c.t.enumerate() {
      grid(
        columns: (4mm, 62mm, 1fr),
        column-gutter: 5pt,
        small[#(n + 1)],
        grid(
          columns: (1fr, auto),
          column-gutter: 4pt,
          text(8.5pt, t.p.join(" ")),
          for m in t.m { if "file" in m { image(m.file, height: 7mm) } else { small[#m.y #m.b] } },
        ),
        grid(
          columns: (auto, auto, 1fr, auto),
          column-gutter: 5pt,
          row-gutter: 3pt,
          ..t.rows.map(r => (small(r.at(0)), small(r.at(1)), text(8.5pt, font: mono, r.at(2)), text(8pt, fill: if r.at(4) { red } else { soft }, r.at(3)))).flatten(),
        ),
      )
      v(4pt, weak: true)
    }
    for x in c.x { text(8pt, fill: red)[#x.at(0): #x.at(1)]; linebreak() }
    if c.note != none { small(c.note) }
  }
  block(
    breakable: false, width: 100%, above: 10pt, below: 0pt,
    fill: if shaded { band } else { none },
    inset: 4pt,
    grid(columns: (6mm, 1fr), column-gutter: 4pt, rotate(-90deg, reflow: true, text(8.5pt, id)), body),
  )
}
#if d.show.validation [
= #L.results
#fig(d.grid_number, d.grid_caption)[
  #for k in d.grid_keys { key(k.at(0), k.at(1)) }
  #image("grid.svg", width: 100%)
]
]

#if d.notable.len() > 0 [
  = #L.notable
  #block(sticky: true, small(d.notable_note))
  #for (n, c) in d.notable.enumerate() { case-block(c, calc.odd(n)) }
]

// ---- appendix A ------------------------------------------------------------
#if d.all != none [
  #pagebreak(weak: true)
  = #L.appendix #d.letters.all · #L.all_cases
  #small(d.all_note)
  #{
    set text(7.5pt)
    show table.cell: it => if it.x == 0 or it.x >= 3 { text(font: mono, it) } else { it }
    table(
      columns: (auto, auto, 2fr) + d.targets.map(t => 3fr) + (auto, auto),
      align: (x, y) => if x == 0 or x >= 3 + d.targets.len() { right } else { left },
      stroke: none,
      fill: (x, y) => row-fill(y),
      inset: (x: 3pt, y: 1.8pt),
      table.header(..(("#", "", L.js_prompt.split(" ").first()) + d.targets + ("s", "tok")).map(th)),
      ..d.all.map(r => ([#r.at(0)], marks(r.at(1)), [#r.at(2)]) + r.at(3).map(o => [#o]) + ([#r.at(4)], [#r.at(5)])).flatten(),
    )
  }
]

// ---- appendix B ------------------------------------------------------------
#pagebreak(weak: true)
= #L.appendix #d.letters.run · #L.run_details
#tbl(d.run_facts)
#if d.runs.len() > 0 [
  == #L.runs
  #for r in d.runs [
    #block(above: 6pt, below: 2pt, text(8.5pt)[#r.targets.join(", ") #h(4pt) #text(fill: soft, r.sealed_at)])
    #raw(r.stamp.clusters().chunks(96).map(c => c.join()).join("\n"), block: true)
  ]
]
#if d.stamp != none [
  == #L.stamp
  #raw(d.stamp.clusters().chunks(96).map(c => c.join()).join("\n"), block: true)
]
== #L.instructions
#for text in d.instructions { raw(text, block: true) }
== #L.settings
#raw(d.agent_json, block: true)

#let lines(s) = s.split("\n").join(linebreak())
// ---- appendix: what every server reported -------------------------------------
#if d.servers.len() > 0 [
  #pagebreak(weak: true)
  = #L.appendix #d.letters.server · #L.server_info
  #small(L.server_info_note)
  #for s in d.servers [
    == #s.name
    #if s.note != none { small(s.note) }
    #for sec in s.sections {
      block(above: 10pt, below: 4pt, sticky: true, text(8.5pt, weight: "bold", sec.title))
      if sec.kind == "rows" {
        let body = {
          set text(7pt, font: mono)
          grid(
            columns: (auto, 1fr),
            column-gutter: 8pt,
            row-gutter: 3pt,
            ..sec.rows.map(r => (text(fill: soft, r.at(0)), r.at(1))).flatten(),
          )
        }
        if sec.cols == 2 { columns(2, gutter: 8mm, body) } else { body }
      } else {
        text(7pt, font: mono, lines(sec.text))
      }
    }
  ]
]

// ---- last appendix: full texts ------------------------------------------------
#if d.full.len() > 0 [
  #pagebreak(weak: true)
  = #L.appendix #d.letters.full · #L.full_texts
  #small(L.full_note)
  // two columns keep the lines near 60 characters, which long text needs to stay readable
  #columns(2, gutter: 8mm, for c in d.full {
    block(above: 14pt, below: 4pt, sticky: true)[#text(weight: "bold", c.id) #h(3pt) #marks(c.f)]
    for (n, t) in c.t.enumerate() {
      block(above: 8pt, below: 3pt, sticky: true, small[#L.turn #(n + 1) · #L.prompt])
      for p in t.p { par(text(8pt, lines(p))) }
      for o in t.o {
        block(above: 6pt, below: 2pt, sticky: true, small[#L.output #o.at(0) #o.at(1)])
        text(8pt, font: mono, lines(o.at(2)))
      }
      if t.e != none {
        block(above: 6pt, below: 2pt, sticky: true, small(L.expected))
        text(8pt, font: mono, lines(t.e))
      }
    }
  })
]
"""


def _short(text: Any, limit: int) -> str:
    """Collapse whitespace and cut the text to ``limit`` characters."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _best(case: dict[str, Any], target: int) -> str:
    """Most frequent output of the last turn that has one for this target."""
    for turn in reversed(case["t"]):
        variants = sorted((v for v in turn["o"] if v[0] == target), key=lambda v: -v[1])
        if variants:
            return variants[0][2] + (f" +{len(variants) - 1}" if len(variants) > 1 else "")
    return "–"


def _split(config: str) -> list[list[str]]:
    """Split a ``name=value, name=value`` string at its top-level commas into rows."""
    parts, depth, start = [], 0, 0
    for position, char in enumerate(config):
        depth += char in "([{"
        depth -= char in ")]}"
        if char == "," and depth == 0:
            parts.append(config[start:position])
            start = position + 1
    parts.append(config[start:])
    return [[*part.strip().split("=", 1), ""][:2] for part in parts if part.strip()]


def _servers(S: dict[str, Any]) -> list[dict[str, Any]]:
    """Per target, the recorded server snapshot arranged into short tables and text blocks."""
    L, out = S["L"], []
    for target in S["plan_targets"]:
        snapshot = S["snapshots"].get(target["name"], {})
        info = snapshot.get("server_info") or {}
        system = info.get("system_env") or {}
        models = (snapshot.get("models") or {}).get("data") or []
        facts = [[k, str(v)] for k, v in target.items() if k != "name" and v not in (None, "", {})]
        facts += [["version", str((snapshot.get("version") or {}).get("version") or "–")]]
        facts += [
            ["model", ", ".join(f"{k} {m[k]}" for k in ("id", "root", "max_model_len", "owned_by") if m.get(k))]
            for m in models
        ]
        sections: list[dict[str, Any]] = [{"title": "target", "kind": "rows", "cols": 1, "rows": facts}]
        short = [[k, str(v)] for k, v in system.items() if v not in (None, "") and "\n" not in str(v)]
        if short:
            sections.append({"title": "system_env", "kind": "rows", "cols": 1, "rows": short})
        if info.get("vllm_config"):
            rows = _split(str(info["vllm_config"]))
            sections.append({"title": "vllm_config", "kind": "rows", "cols": 1, "rows": rows})
        if info.get("vllm_env"):
            rows = [[k, str(v)] for k, v in info["vllm_env"].items()]
            sections.append({"title": "vllm_env", "kind": "rows", "cols": 1, "rows": rows})
        sections += [
            {"title": f"system_env.{k}", "kind": "text", "text": str(v).strip()[:6000]}
            for k, v in system.items()
            if "\n" in str(v)
        ]
        out.append({"name": target["name"], "note": None if info else L["si_none"], "sections": sections})
    return out


def write_pdf(
    ledger: Ledger,
    path: str | os.PathLike[str],
    *,
    casebook: Casebook | None = None,
    lang: str = "en",
    max_notable: int = 50,
    all_cases: bool = True,
    max_images: int = 30,
    thumbnail_px: int = 256,
    header: ReportHeader | None = None,
    sections: Sequence[str] | None = None,
    paper: str = "a4",
    excerpt_chars: int = 240,
    normalize: Callable[[Any], Any] | None = None,
    config: str | os.PathLike[str] | Config | None = None,
) -> Path:
    """Write the PDF report of ``ledger`` to ``path``.

    ``header`` is a :class:`ReportHeader` with title, authors, number and date. As
    with :func:`pydai_ledger.write_report`, a run ledger gives the run report and a
    benchmark ledger the benchmark report; ``sections`` limits either.
    ``lang`` is ``"en"`` or ``"de"``. The PDF holds a one-page summary with the ledger
    stamp as QR code, the validation (precision, trueness, accuracy), up to ``max_notable`` notable cases in
    detail (failed, differing, not as expected) and, with ``all_cases``, a dense table
    of every case. For the ledger of a :func:`pydai_ledger.benchmark` it is the
    benchmark report instead: latency over throughput per concurrency level and the
    recommendation, without cases. Prompts and outputs longer than ``excerpt_chars`` are shortened in the notable
    cases; their full text follows in the last appendix. An appendix lists what every
    server reported (``/server_info``, when vLLM runs in dev mode). ``config`` sets the
    level of each note, see :func:`pydai_ledger.write_report`.
    Needs ``pydai-ledger[pdf]``; image previews need Pillow.
    """
    try:
        import typst
    except ImportError as exc:
        raise ImportError("write_pdf() needs Typst. Install it with: pip install 'pydai-ledger[pdf]'") from exc

    S = build(
        ledger,
        casebook,
        lang=lang,
        header=header,
        sections=sections,
        max_images=max_images,
        thumbnail_px=thumbnail_px,
        normalize=normalize,
        max_text=20_000,
        config=config,
    )
    L, targets, payload = S["L"], S["targets"], S["cases"]
    num = (
        (lambda v, d: "–" if v is None else f"{v:,.{d}f}".translate(str.maketrans(",.", ".,")))
        if lang == "de"
        else (lambda v, d: "–" if v is None else f"{v:,.{d}f}")
    )
    flagged = [c for c in payload["cases"] if c["f"]] if S["show"]["cases"] else []

    with tempfile.TemporaryDirectory(prefix="pydai-ledger-pdf-") as folder:
        root = Path(folder)
        for index, uri in enumerate(payload["thumbs"]):
            (root / f"thumb{index}.jpg").write_bytes(base64.b64decode(uri.split(",", 1)[1]))
        curve = None
        if S["curve"]:
            (root / "curve.svg").write_text(_curve(**S["curve"], bare=True), encoding="utf-8")
            curve = "curve.svg"
        per_cell = 1
        all_cases = all_cases and S["show"]["cases"]
        if S["show"]["validation"]:
            grid_svg, per_cell = _result_grid(S["states"], L["axis_case"], bare=True)
            (root / "grid.svg").write_text(grid_svg, encoding="utf-8")
        letters = dict(zip(("all", "run", "server", "full"), "ABCD" if all_cases else " ABC", strict=True))
        qr = None
        try:
            import segno

            segno.make(S["stamp"], error="m").save(str(root / "qr.svg"), border=0, dark="#111111")
            qr = "qr.svg"
        except ImportError:
            pass

        notable, full = [], []
        for case in flagged[:max_notable]:
            turns, long = [], False
            for turn in case["t"]:
                texts = [*turn["p"], *(v[2] for v in turn["o"]), *([turn["e"]] if turn["e"] is not None else [])]
                long = long or any(len(text) > excerpt_chars for text in texts)
                rows = [
                    [
                        targets[v[0]] if len(targets) > 1 else "",
                        f"{v[1]}×",
                        _short(v[2], excerpt_chars),
                        "" if v[3] is None else L["is_expected"] if v[3] else L["not_expected"],
                        v[3] is False,
                    ]
                    for v in turn["o"]
                ]
                if turn["e"] is not None:
                    rows.append([L["expected"], "", _short(turn["e"], excerpt_chars), "", False])
                media = [{"file": f"thumb{m['u']}.jpg"} if "u" in m else m for m in turn["m"]]
                turns.append({"p": [_short(p, excerpt_chars) for p in turn["p"]], "m": media, "rows": rows})
            errors = [list(e) for e in dict.fromkeys((targets[x[0]], _short(x[1], 300)) for x in case["x"])]
            label = _short(str(case["k"]) if case["k"] is not None else f"#{case['i']}", 28)
            note = None
            if long:
                size = sum(len(p) for t in case["t"] for p in t["p"]) + sum(
                    len(v[2]) for t in case["t"] for v in t["o"]
                )
                note = L["shortened"].format(n=num(size, 0), a=letters["full"])
                full.append(
                    {
                        "id": label,
                        "f": case["f"],
                        "t": [
                            {"p": t["p"], "o": [[targets[v[0]], f"{v[1]}×", v[2]] for v in t["o"]], "e": t["e"]}
                            for t in case["t"]
                        ],
                    }
                )
            notable.append({"id": label, "f": case["f"], "t": turns, "x": errors, "note": note})

        every = None
        if all_cases:
            width = max(18, 78 // max(len(targets), 1))
            every = [
                [
                    str(c["i"]),
                    c["f"],
                    _short(
                        c["k"] if c["k"] is not None else (c["t"][0]["p"][0] if c["t"] and c["t"][0]["p"] else ""), 34
                    ),
                    [_short(_best(c, t), width) for t in range(len(targets))],
                    num(c["l"], 2),
                    num(c["o"], 0),
                ]
                for c in sorted(payload["cases"], key=lambda c: c["i"])
            ]

        run_facts = [
            [L["run"], S["run_id"]],
            [L["casebook"], S["casebook_id"]],
            [L["sealed"], S["sealed_at"]],
            [L["signature"], S["signer"] or S["signed"]],
        ]
        run_facts += [
            [k, _short(v if not isinstance(v, dict) else ", ".join(f"{a} {b}" for a, b in v.items()), 160)]
            for k, v in S["environment"].items()
        ]
        document = {
            "L": L,
            "lang": S["lang"],
            "benchmark": S["benchmark"],
            "execution": S["execution"],
            "author": S["author"],
            "head": S["head"],
            "show": S["show"],
            "validation": S["validation"],
            "verdict": S["verdict"],
            "model_pairs": S["model_pairs"],
            "title": S["title"],
            "paper": paper,
            "run": S["number"] or f"{L['run']} {S['run_id'][:12]}",
            "foot": " · ".join(
                part
                for part in (
                    S["number"],
                    f"{L['run']} {S['run_id'][:12]}",
                    f"{L['manifest']} {S['stamp'].split(':')[3][:16].lower()}",
                    S["date"],
                )
                if part
            ),
            "qr": qr,
            "stamp": S["stamp"],
            "facts": S["facts"],
            "notes": [
                {**n, "text": _short(n["text"] + (f" {L['remark']}: {n['remark']}" if n["remark"] else ""), 400)}
                for n in S["notes"]
            ],
            "config_line": S["config_line"],
            "servers": _servers(S) if S["show"]["targets"] else [],
            "runs": S["runs"],
            "models": S["models"],
            "capabilities": S["capabilities"],
            "series": [[name, PRINT_PALETTE[i]] for i, name in enumerate(S["series_names"][:MAX_SERIES])],
            "curve": curve,
            "chart_number": 1,
            "chart_caption": S["chart_caption"],
            "grid_number": 2 if curve else 1,
            "grid_caption": L["fig_grid"].format(agg=L["grid_agg"].format(k=per_cell) if per_cell > 1 else "")
            + (L["grid_rows"].format(names=", ".join(targets)) if len(targets) > 1 else ""),
            "grid_keys": [
                [GRID_COLOURS[state], L[name]]
                for state, name in ((1, "grid_ok"), (2, "grid_dev"), (3, "grid_fail"), (0, "grid_none"))
            ],
            "letters": letters,
            "full": full,
            "throughput": S["throughput"],
            "tuning": S["tuning"],
            "server": S["server"],
            "per_target": S["per_target"],
            "targets": targets,
            "notable": notable,
            "notable_note": L["notable_note"].format(n=len(notable), total=len(flagged)) + " " + L["legend_text"],
            "all": every,
            "all_note": L["all_note"].format(n=num(len(every or []), 0)) + " " + L["legend_text"],
            "run_facts": {"head": ["", ""], "cols": [0, 1], "align": ["l", "l"], "rows": run_facts},
            "instructions": [text[:3000] for text in S["instructions"]],
            "agent_json": json.dumps(
                {k: S["agent"].get(k) for k in ("model_settings", "tools", "output_schema")},
                indent=2,
                ensure_ascii=False,
                default=str,
            )[:6000],
        }
        (root / "data.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        (root / "report.typ").write_text(_TEMPLATE, encoding="utf-8")
        path = Path(path)
        typst.compile(str(root / "report.typ"), output=str(path), root=str(root), font_paths=[str(FONTS)])
    return path
