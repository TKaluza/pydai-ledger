"""Everything a report shows, computed once from a ledger.

:func:`build` returns a plain dictionary with display-ready tables in the chosen
language. The HTML and the PDF report only lay it out, so both always show the same
figures.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from html import escape
from typing import Any

from . import signing
from ._util import canonical_json
from .casebook import Casebook
from .config import LEVELS, POINTS, Config, determinism, missing
from .i18n import Lang
from .ledger import Ledger

MAX_SERIES = 8  # the categorical palette has eight validated slots
MAX_TEXT = 4000  # characters of one prompt or output kept in a report
PRINT_PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def _nice_max(value: float) -> float:
    """Round an axis maximum up to a value that divides into four readable ticks."""
    if value <= 0:
        return 1.0
    exponent = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 4, 5, 8, 10):  # each divides into four readable ticks
        if value <= step * exponent:
            return step * exponent
    return 10 * exponent


GRID_COLOURS = {0: "#d9d9d9", 1: "#3fa66b", 2: "#f2c744", 3: "#b3261e"}  # not run, ok, deviating, failed


def _print(svg: str, width: int, height: int) -> str:
    """Give an SVG fixed size, colours and fonts so it can be used outside the HTML page."""
    svg = svg.replace("<svg ", f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" ', 1)
    for name, colour in (("grid", "#d0d0d0"), ("surface", "#ffffff"), ("text", "#111111"), ("text2", "#333333")):
        svg = svg.replace(f"var(--{name})", colour)
    for index, colour in enumerate(PRINT_PALETTE):
        svg = svg.replace(f"var(--s{index + 1})", colour)
    return svg.replace("<text ", '<text font-family="Jost" ')


def _panel(
    label: str,
    y_label: str,
    x_label: str,
    levels: list[int],
    series: dict[str, list[float | None]],
    *,
    bare: bool = False,
) -> str:
    """One panel of a figure: a quantity over the concurrency levels, one line per target.

    Both axes carry an arrow, ticks and a label with quantity and unit; the panel is
    identified by ``label`` such as "(a)" and explained in the figure caption. The x
    axis is logarithmic when the levels span a factor of four or more.
    """
    names = list(series)[:MAX_SERIES]
    width, height, left, right, top, bottom = 320, 250, 58, 18, 22, 50
    plot_w, plot_h = width - left - right, height - top - bottom
    peak = max((v for name in names for v in series[name] if v is not None), default=0.0)
    y_max = _nice_max(peak)
    low, high = min(levels), max(levels)
    log_x = high / low >= 4

    def x(level: int) -> float:
        if high == low:
            return left + plot_w / 2
        share = (math.log2(level / low) / math.log2(high / low)) if log_x else (level - low) / (high - low)
        return left + 10 + (plot_w - 20) * share

    def y(value: float) -> float:
        return top + plot_h * (1 - value / y_max)

    base = top + plot_h
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(y_label)}">']
    for tick in range(5):
        value = y_max * tick / 4
        if tick:
            parts.append(
                f'<line x1="{left}" x2="{left + plot_w}" y1="{y(value):.1f}" y2="{y(value):.1f}" stroke="var(--grid)"/>'
            )
        parts.append(
            f'<line x1="{left - 4}" x2="{left}" y1="{y(value):.1f}" y2="{y(value):.1f}" stroke="var(--text)"/>'
            f'<text font-size="12" fill="var(--text2)" x="{left - 7}" y="{y(value) + 4:.1f}" text-anchor="end">{value:g}</text>'
        )
    shown = levels if len(levels) <= 7 else levels[:: math.ceil(len(levels) / 7)]
    for level in shown:
        parts.append(
            f'<line x1="{x(level):.1f}" x2="{x(level):.1f}" y1="{base}" y2="{base + 4}" stroke="var(--text)"/>'
            f'<text font-size="12" fill="var(--text2)" x="{x(level):.1f}" y="{base + 17}" text-anchor="middle">{level}</text>'
        )
    # axes with arrow heads
    tip_x, tip_y = left + plot_w + 10, top - 12
    parts.append(
        f'<line x1="{left}" x2="{tip_x}" y1="{base}" y2="{base}" stroke="var(--text)"/>'
        f'<path d="M{tip_x + 6},{base} l-8,-3.5 v7 z" fill="var(--text)"/>'
        f'<line x1="{left}" x2="{left}" y1="{base}" y2="{tip_y}" stroke="var(--text)"/>'
        f'<path d="M{left},{tip_y - 6} l-3.5,8 h7 z" fill="var(--text)"/>'
        f'<text font-size="12.5" fill="var(--text)" x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle">{escape(x_label)}</text>'
        f'<text font-size="12.5" fill="var(--text)" transform="translate(14,{top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>'
        f'<text font-size="13" fill="var(--text)" x="{left + 8}" y="{top + 2}">{escape(label)}</text>'
    )
    for index, name in enumerate(names):
        colour = f"var(--s{index + 1})"
        points = [(x(levels[i]), y(v), v, levels[i]) for i, v in enumerate(series[name]) if v is not None]
        if len(points) > 1:
            path = " ".join(f"{px:.1f},{py:.1f}" for px, py, _, _ in points)
            parts.append(f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for px, py, value, level in points:
            tip = escape(f"{name}\n{level}: {value:,.2f}", quote=True)
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{colour}" stroke="var(--surface)" stroke-width="1.5"/>'
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="12" fill="transparent" data-tip="{tip}"/>'
            )
    parts.append("</svg>")
    svg = "".join(parts)
    return _print(svg, width, height) if bare else svg


def _states(
    case_count: int,
    targets: list[str],
    data: dict[str, Any],
    expected: dict[tuple[int, int], str],
    norm: Callable[[str], str],
) -> dict[str, list[int]]:
    """Result of every case per target: 0 not run, 1 fine, 2 deviating, 3 failed.

    A target deviates in a case when one of its outputs differs from the expected
    output or, where none is given, from the most frequent output of all targets.
    """
    states = {target: [0] * case_count for target in targets}
    for (case, turn), by_target in data["outputs"].items():
        if case >= case_count:
            continue
        want = expected.get((case, turn))
        if want is None:
            totals: Counter[str] = Counter()
            for variants in by_target.values():
                for output, count in variants.items():
                    totals[norm(output)] += count
            reference = totals.most_common(1)[0][0]
        else:
            reference = norm(want)
        for target, variants in by_target.items():
            deviates = any(norm(output) != reference for output in variants)
            states[target][case] = max(states[target][case], 2 if deviates else 1)
    for case, failures in data["errors"].items():
        for failure in failures:
            if case < case_count:
                states[failure["target"]][case] = 3
    return states


def _result_grid(states: dict[str, list[int]], x_label: str, *, bare: bool = False) -> tuple[str, int]:
    """Heat map of case results: one cell per case, one row per target, 100 cells per band.

    Outside print (``bare``) every cell carries its case, target row and state as data
    attributes, which the HTML report uses for tooltips and to jump to the case.
    The number at the left of a band is its first case, the scale on top the offset
    within the band, so a cell is found by adding the two. Very large runs put several
    cases into one cell and show the worst result. Returns the SVG and the number of
    cases per cell.
    """
    targets = list(states)
    count = len(next(iter(states.values()), []))
    per_cell = max(1, math.ceil(count / 6000))
    cells = math.ceil(count / per_cell)
    per_band, pitch, size = 100, 7, 6
    bands = max(1, math.ceil(cells / per_band))
    left, right, top = 52, 6, 30
    band_h = len(targets) * pitch + (9 if len(targets) > 1 else 3)
    width, height = left + per_band * pitch + right, top + bands * band_h + 4
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(x_label)}">']
    parts.append(f'<text font-size="10.5" fill="var(--text)" x="{left}" y="10">{escape(x_label)}</text>')
    for offset in range(0, per_band, 10):
        parts.append(
            f'<text font-size="9.5" fill="var(--text2)" x="{left + offset * pitch}" y="{top - 5}">+{offset * per_cell}</text>'
        )
    for band in range(bands):
        y0 = top + band * band_h
        parts.append(
            f'<text font-size="9.5" fill="var(--text2)" x="{left - 6}" y="{y0 + size}" text-anchor="end">{band * per_band * per_cell}</text>'
        )
        for row, target in enumerate(targets):
            values = states[target]
            for column in range(per_band):
                first = (band * per_band + column) * per_cell
                if first >= count:
                    break
                state = max(values[first : first + per_cell])
                parts.append(
                    f'<rect x="{left + column * pitch}" y="{y0 + row * pitch}" width="{size}" height="{size}" fill="{GRID_COLOURS[state]}"'
                    + ("" if bare else f' data-c="{first}" data-t="{row}" data-s="{state}"')
                    + "/>"
                )
    parts.append("</svg>")
    svg = "".join(parts)
    return (_print(svg, width, height) if bare else svg), per_cell


def _thumbnail(data: bytes, media_type: str, size: int) -> str | None:
    """Data URI of a preview image, or None if it cannot be embedded sensibly."""
    if not media_type.startswith("image/"):
        return None
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(data))
        image.thumbnail((size, size))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, "JPEG", quality=80)
        data, media_type = buffer.getvalue(), "image/jpeg"
    except ImportError:
        if len(data) > 256 * 1024:
            return None
    except Exception:
        return None
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def _collect(ledger: Ledger) -> dict[str, Any]:
    """One pass over all attempts."""
    outputs: dict[tuple[int, int], dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    errors: dict[int, list[dict[str, Any]]] = defaultdict(list)
    per_target: dict[str, Counter[str]] = defaultdict(Counter)
    per_case: dict[int, list[float]] = defaultdict(lambda: [0, 0.0, 0])  # ok attempts, seconds, output tokens
    for attempt in ledger.attempts():
        counts = per_target[attempt["target"]]
        counts["ok" if attempt["status"] == "ok" else "failed"] += 1
        if attempt["status"] != "ok":
            errors[attempt["case_index"]].append({"target": attempt["target"], "error": attempt["error"]})
            continue
        stats = per_case[attempt["case_index"]]
        stats[0] += 1
        stats[1] += attempt["duration_s"]
        for turn in attempt["turns"]:
            stats[2] += turn["usage"].get("output_tokens") or 0
            counts["input_tokens"] += turn["usage"].get("input_tokens") or 0
            counts["output_tokens"] += turn["usage"].get("output_tokens") or 0
            counts["requests"] += len(turn.get("requests", []))
            outputs[(attempt["case_index"], turn["index"])][attempt["target"]][canonical_json(turn["output"])] += 1
    return {"outputs": outputs, "errors": errors, "per_target": per_target, "per_case": per_case}


def _normalizer(normalize: Callable[[Any], Any] | None) -> Callable[[str], str]:
    """Map a canonical output string to the string that is compared."""
    if normalize is None:
        return lambda text: text
    cache: dict[str, str] = {}

    def norm(text: str) -> str:
        if text not in cache:
            cache[text] = canonical_json(normalize(json.loads(text)))
        return cache[text]

    return norm


def _pair_matrix(
    targets: list[str], outputs: dict[tuple[int, int], dict[str, Counter[str]]], norm: Callable[[str], str]
) -> dict[tuple[str, str], tuple[int, int]]:
    """(agreeing, compared) turns for every pair of targets; the diagonal is a target's
    consistency with itself over repetitions and concurrency levels.
    """
    matrix: dict[tuple[str, str], tuple[int, int]] = {}
    for a_index, a in enumerate(targets):
        for b in targets[a_index:]:
            agreeing = compared = 0
            for by_target in outputs.values():
                if a not in by_target or b not in by_target:
                    continue
                if a == b and sum(by_target[a].values()) < 2:
                    continue
                compared += 1
                agreeing += len({norm(o) for o in set(by_target[a]) | set(by_target[b])}) == 1
            matrix[(a, b)] = matrix[(b, a)] = (agreeing, compared)
    return matrix


def _expected(
    casebook: Casebook | None,
    targets: list[str],
    outputs: dict[tuple[int, int], dict[str, Counter[str]]],
    norm: Callable[[str], str],
) -> tuple[dict[tuple[int, int], str], dict[str, list[int]]]:
    """Expected outputs per (case, turn) and, per target, [matching, checked] outputs."""
    expected: dict[tuple[int, int], str] = {}
    hits: dict[str, list[int]] = {t: [0, 0] for t in targets}
    if casebook is not None:
        for record in casebook.records():
            for turn, value in enumerate(record.get("expected") or []):
                if value is not None:
                    expected[(record["index"], turn)] = canonical_json(value)
        for key, want in expected.items():
            for target, variants in outputs.get(key, {}).items():
                hits[target][0] += sum(n for out, n in variants.items() if norm(out) == norm(want))
                hits[target][1] += sum(variants.values())
    return expected, hits


def _clip(text: str, limit: int = MAX_TEXT) -> str:
    """Cut very long prompts and outputs so a report stays small."""
    return text if len(text) <= limit else text[:limit] + " … [shortened]"


def _server_metrics(stage: dict[str, Any]) -> dict[str, float | None]:
    """Means over one stage from the server's own counters (vLLM ``/metrics``)."""
    before, after = stage.get("metrics_before") or {}, stage.get("metrics_after") or {}

    def delta(name: str) -> float:
        return sum(v - before.get(k, 0.0) for k, v in after.items() if k.split("{")[0] == name)

    def mean(name: str) -> float | None:
        count = delta(f"vllm:{name}_seconds_count")
        return delta(f"vllm:{name}_seconds_sum") / count if count else None

    queries = delta("vllm:prefix_cache_queries_total")
    return {
        "ttft_s": mean("time_to_first_token"),
        "queue_s": mean("request_queue_time"),
        "prefill_s": mean("request_prefill_time"),
        "decode_s": mean("request_decode_time"),
        "cache_hit": delta("vllm:prefix_cache_hits_total") / queries if queries else None,
    }


def _model_facts(target: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """The few facts about a target that fit into one table row."""
    info = snapshot.get("server_info") or {}
    config = str(info.get("vllm_config") or "")
    system = info.get("system_env") or {}

    def setting(name: str) -> str | None:
        found = re.search(rf"\b{name}=([^,]+)", config)
        return found.group(1).strip("'") if found else None

    listed = next(
        (m for m in (snapshot.get("models") or {}).get("data", []) if m.get("id") == target.get("model_name")), {}
    )
    version = (snapshot.get("version") or {}).get("version")
    weights = [
        w for w in (setting("quantization"), (setting("dtype") or "").removeprefix("torch.")) if w and w != "None"
    ]
    max_seq_len = setting("max_seq_len")
    return {
        "model": str(target.get("model_name") or "–"),
        "server": f"vLLM {version}" if version and info else str(version or target.get("system") or "–"),
        "gpu": str(system.get("nvidia_gpu_models") or (target.get("info") or {}).get("gpu") or "–").strip(),
        "context": listed.get("max_model_len") or (int(max_seq_len) if max_seq_len else None),
        "weights": ", ".join(weights) or "–",
        "capabilities": snapshot.get("capabilities") or {},
    }


def _cases_payload(
    case_count: int,
    targets: list[str],
    data: dict[str, Any],
    expected: dict[tuple[int, int], str],
    differing: set[int],
    casebook: Casebook | None,
    max_cases: int | None,
    max_images: int,
    thumbnail_px: int,
    norm: Callable[[str], str],
    max_text: int = MAX_TEXT,
) -> dict[str, Any]:
    """Everything the case table shows, in short keys because it is repeated per case."""
    outputs, errors, per_case = data["outputs"], data["errors"], data["per_case"]
    target_index = {name: i for i, name in enumerate(targets)}
    turns_of: dict[int, int] = defaultdict(int)
    wrong: set[int] = set()
    for (case, turn), by_target in outputs.items():
        turns_of[case] = max(turns_of[case], turn + 1)
        want = expected.get((case, turn))
        if want is not None and any(norm(out) != norm(want) for variants in by_target.values() for out in variants):
            wrong.add(case)

    def priority(index: int) -> tuple[bool, bool, bool, int]:
        return (index not in errors, index not in differing, index not in wrong, index)

    chosen = sorted(range(case_count), key=priority)
    note = None
    if max_cases is not None and len(chosen) > max_cases:
        chosen = chosen[:max_cases]
        note = [max_cases, case_count]
    records: dict[int, dict[str, Any]] = {}
    if casebook is not None:
        wanted = set(chosen)
        records = {r["index"]: r for r in casebook.records() if r["index"] in wanted}

    thumbs: list[str] = []
    cases = []
    for index in chosen:
        record = records.get(index)
        turns, images = [], 0
        for turn in range(max(turns_of[index], len(record["turns"]) if record else 0)):
            texts: list[str] = []
            media: list[dict[str, Any]] = []
            if record and turn < len(record["turns"]):
                content = record["turns"][turn]
                for item in [{"text": content}] if isinstance(content, str) else content:
                    if "text" in item:
                        texts.append(_clip(item["text"], max_text))
                        continue
                    images += 1
                    uri = None
                    if casebook is not None and len(thumbs) < max_images:  # previews for the most interesting cases
                        uri = _thumbnail(casebook.blob(item["blob"]), item["media_type"], thumbnail_px)
                    if uri:
                        thumbs.append(uri)
                        media.append({"u": len(thumbs) - 1})
                    else:
                        media.append({"y": item["media_type"], "b": item["blob"][:12]})
            want = expected.get((index, turn))
            variants = [
                [
                    target_index[target],
                    count,
                    _clip(output, max_text),
                    None if want is None else norm(output) == norm(want),
                ]
                for target in targets
                for output, count in outputs.get((index, turn), {}).get(target, {}).items()
            ]
            turns.append(
                {"p": texts, "m": media, "e": _clip(want, max_text) if want is not None else None, "o": variants}
            )
        ok, seconds, tokens = per_case.get(index, (0, 0.0, 0))
        cases.append(
            {
                "i": index,
                "k": record.get("key") if record else None,
                "f": (index in errors) | (index in differing) << 1 | (index in wrong) << 2,
                "n": images,
                "l": round(seconds / ok, 3) if ok else None,
                "o": round(tokens / ok, 1) if ok else None,
                "t": turns,
                "x": [
                    [target_index[e["target"]], _clip(e["error"] or "", max_text)] for e in errors.get(index, [])[:5]
                ],
            }
        )
    return {"cases": cases, "thumbs": thumbs, "note": note}


def _table(head: list[str], align: str, rows: list[list[str]], cols: list[int] | None = None) -> dict[str, Any]:
    """``align`` has one letter per column (l, r, c); ``cols`` gives each column's share
    of the free width, 0 for as narrow as its content.
    """
    return {"head": head, "align": list(align), "cols": cols or [1] + [0] * (len(head) - 1), "rows": rows}


def build(
    ledger: Ledger,
    casebook: Casebook | None = None,
    *,
    lang: str = "en",
    title: str | None = None,
    author: str | None = None,
    number: str | None = None,
    date: str | None = None,
    max_cases: int | None = None,
    max_images: int = 30,
    thumbnail_px: int = 256,
    normalize: Callable[[Any], Any] | None = None,
    max_text: int = MAX_TEXT,
    config: str | os.PathLike[str] | Config | None = None,
) -> dict[str, Any]:
    """Compute everything a report shows; see the module docstring."""
    L = Lang(lang)
    manifest, plan = ledger.manifest, ledger.manifest["plan"]
    if casebook is not None and casebook.id != plan["casebook_id"]:
        raise ValueError("The casebook does not belong to this ledger.")
    targets = [t["name"] for t in plan["targets"]]
    data = _collect(ledger)
    outputs, errors, per_target = data["outputs"], data["errors"], data["per_target"]
    norm = _normalizer(normalize)
    agreement, throughput = ledger.agreement(normalize), ledger.throughput()
    expected, expected_hits = _expected(casebook, targets, outputs, norm)
    ok = sum(c["ok"] for c in per_target.values())
    failed = sum(c["failed"] for c in per_target.values())

    invalid = False
    try:
        signer = ledger.signer
        signed = f"{L('signed_by')} {signer[:16]}…" if signer else L("unsigned")
    except signing.InvalidSignatureError:
        signer, signed, invalid = None, L("invalid_signature").upper(), True
    except ImportError:
        signer, signed = None, L("unchecked")

    if manifest.get("runs"):
        signed = f"{L('comparison', n=len(manifest['runs']))} · {signed}"

    # -- key figures ---------------------------------------------------------------
    hit, hit_total = sum(v[0] for v in expected_hits.values()), sum(v[1] for v in expected_hits.values())
    facts = [
        [L("cases"), L.num(plan["case_count"]), f"{len(targets)} {L('target' if len(targets) == 1 else 'targets')}"],
        [L("attempts"), L.num(ok), f"{L.num(failed)} {L('failed')}"],
        [
            L("identical"),
            L.pct(agreement.agreeing, agreement.compared),
            f"{L.num(agreement.compared)} {L('turns_compared')}",
        ],
        [
            L("as_expected"),
            L.pct(hit, hit_total),
            f"{L.num(hit_total)} {L('outputs_checked')}" if hit_total else L("no_expected"),
        ],
    ]
    fastest = max(throughput, key=lambda r: r["output_tokens_per_s"] or r["attempts_per_s"] or 0, default=None)
    if fastest:
        tokens = fastest["output_tokens_per_s"]
        facts.append(
            [
                L("best_throughput"),
                f"{L.num(tokens, 0)} tok/s" if tokens else f"{L.num(fastest['attempts_per_s'])} /s",
                f"{L.num(fastest['attempts_per_s'])} {L('at_concurrency')} {fastest['concurrency']}",
            ]
        )

    settings = Config.load(config)
    snapshots = {e["target"]: e["snapshot"] for e in ledger.events("target")}
    options = (ledger.events("execution")[-1:] or [{}])[0].get("options") or {}
    points = determinism(
        plan["agent"].get("model_settings"),
        {t["name"]: snapshots.get(t["name"], {}) for t in plan["targets"]},
        reset_prefix_cache=bool(options.get("reset_prefix_cache")),
    )
    found: list[tuple[str, str, str]] = [  # code, text, section it belongs to ("" = summary)
        ("stopped", f"{L('w_stopped')} ({e['target']}: {e['reason']})", "") for e in ledger.events("aborted")
    ]
    if not manifest["complete"]:
        found.append(("incomplete", L("w_incomplete"), ""))
    if failed:
        found.append(("failed_attempts", L("w_failed_attempts", n=L.num(failed)), ""))
    if invalid:
        found.append(("invalid_signature", L("w_invalid_signature"), ""))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for warning in ledger.warnings():
        grouped.setdefault(warning["code"], []).append(warning)
    for code, group in grouped.items():  # one note per kind of warning, listing its targets
        names = ", ".join(w["target"] for w in group)
        section = "throughput" if code.startswith("prefix_cache") else ""
        found.append((code, f"{L.warning(code, group[0]['message'])} ({names})", section))
    for point, names in missing(settings, points).items():
        found.append((f"determinism_{point}", f"{L('w_determinism', point=L('p_' + point))} ({', '.join(names)})", ""))
    notes, hidden = [], []
    for code, text, section in found:
        level = settings.level(code)
        if level == "off":
            hidden.append(code)
            continue
        background = L.texts.get(f"b_{code}") or (L("b_determinism") if code.startswith("determinism_") else None)
        notes.append(
            {
                "level": level,
                "code": code,
                "text": text,
                "section": section,
                "background": background,
                "remark": settings.remarks.get(code),
            }
        )
    notes.sort(key=lambda n: LEVELS.index(n["level"]))
    config_line = " ".join(
        ([L("config_line", source=settings.source)] if settings.source else [])
        + ([L("notes_hidden", n=len(hidden), codes=", ".join(hidden))] if hidden else [])
    )
    states = {"on": L("yes"), "off": L("no"), "unknown": L("unknown"), "n/a": "–"}
    items = []
    for point in POINTS:  # one compact entry per point; targets are named only where they differ
        found_states = [points[t][point][0] for t in targets]
        values = [v for v in dict.fromkeys(points[t][point][1] for t in targets) if v not in ("–", "True", "False")]
        if set(found_states) == {"n/a"}:
            continue
        if len(set(found_states)) == 1:
            state, word, value = found_states[0], states[found_states[0]], ", ".join(values)
        else:
            state, word = "mixed", L("partly")
            value = ", ".join(f"{t}: {states[points[t][point][0]]}" for t in targets)
        items.append(
            {
                "state": state,
                "word": word,
                "label": L("p_" + point),
                "value": value,
                "required": point in settings.require,
            }
        )
    gaps = [L("p_" + point) for point in missing(settings, points)]
    determinism_view = {
        "summary": L("d_summary", on=sum(i["state"] == "on" for i in items), n=len(items))
        + (" " + L("d_missing", points="; ".join(gaps)) if gaps else ""),
        "items": items,
    }

    # -- models ----------------------------------------------------------------------
    def capability(result: dict[str, Any] | None) -> str:
        return "–" if result is None else L("yes") if result["ok"] else L("no")

    model_rows = []
    for target in plan["targets"]:
        m = _model_facts(target, snapshots.get(target["name"], {}))
        model_rows.append(
            [target["name"], m["model"], m["server"], m["gpu"], L.num(m["context"]), m["weights"]]
            + [capability(m["capabilities"].get(k)) for k in ("tool_output", "native_output", "image_input")]
        )
    modes = sorted({str(e["parameters"].get("output_mode")) for e in ledger.events("request_parameters")})
    used = " · ".join(f"{k} {v}" for k, v in (plan["agent"].get("model_settings") or {}).items()) or "–"

    # -- throughput --------------------------------------------------------------------
    levels = sorted({row["concurrency"] for row in throughput})
    by_key = {(row["target"], row["concurrency"]): row for row in throughput}
    charts = [
        {
            "label": f"({letter})",
            "y_label": L(key),
            "series": {t: [(by_key.get((t, level)) or {}).get(field) for level in levels] for t in targets},
        }
        for letter, key, field in (
            ("a", "axis_attempts", "attempts_per_s"),
            ("b", "axis_tokens", "output_tokens_per_s"),
            ("c", "axis_latency", "latency_p50_s"),
        )
    ]
    if len(levels) < 2:  # a single point per line says nothing the table does not say
        charts = []
    log_x = len(levels) > 1 and max(levels) / min(levels) >= 4
    autotune = []
    for name, tune in ledger.autotune().items():
        unit = "tok/s" if tune["metric"] == "output_tokens_per_s" else L("h_attempts_s")
        best = next(level for level in tune["levels"] if level["concurrency"] == tune["best"])
        rest = f"; {L('not_measured')}: {', '.join(map(str, tune['skipped']))}" if tune["skipped"] else ""
        autotune.append(
            L("autotune", name=name, best=tune["best"], rate=L.num(best["rate"], 1), unit=unit) + rest + "."
        )
    warmups = ledger.events("warmup")
    warmup = None
    if warmups:
        warmup = f"{L('warmup')}: " + ", ".join(
            L("warmup_item", target=w["target"], n=w["attempts"], s=L.num(w["wall_s"], 1)) for w in warmups
        )
    server_rows = []
    for stage in ledger.events("stage"):
        m = _server_metrics(stage)
        if any(v is not None for v in m.values()):
            server_rows.append(
                [stage["target"], str(stage["concurrency"])]
                + [L.num(m[k], 3) for k in ("queue_s", "ttft_s", "prefill_s", "decode_s")]
                + [
                    L.pct(m["cache_hit"], 1) if m["cache_hit"] is not None else "–",
                    L("yes") if stage.get("prefix_cache_reset") else L("no"),
                ]
            )

    # -- agreement ---------------------------------------------------------------------
    matrix = _pair_matrix(targets, outputs, norm)

    # -- cases -------------------------------------------------------------------------
    differing = {d["case_index"] for d in agreement.disagreements}
    payload = _cases_payload(
        plan["case_count"],
        targets,
        data,
        expected,
        differing,
        casebook,
        max_cases,
        max_images,
        thumbnail_px,
        norm,
        max_text,
    )
    if payload["note"]:
        payload["note"] = L("limited", n=L.num(payload["note"][0]), total=L.num(payload["note"][1]))
    wrong = sum(1 for c in payload["cases"] if c["f"] & 4)

    environment = (ledger.events("environment")[-1:] or [manifest.get("sealed_with") or {}])[0]
    agent = plan["agent"]
    return {
        "lang": L.code,
        "L": L.texts,
        "title": title or L("report"),
        "run_id": manifest["run_id"],
        "casebook_id": plan["casebook_id"],
        "sealed_at": manifest["sealed_at"],
        "complete": manifest["complete"],
        "signed": signed,
        "signer": signer,
        "stamp": ledger.stamp(),
        "author": author,
        "number": number,
        "date": date or f"{manifest['sealed_at'][:16].replace('T', ' ')} UTC",
        "status": f"{L('complete' if manifest['complete'] else 'incomplete')} · {signed}",
        "meta": " · ".join(
            part
            for part in (
                number,
                date or f"{manifest['sealed_at'][:16].replace('T', ' ')} UTC",
                author,
                L("complete" if manifest["complete"] else "incomplete"),
                signed,
            )
            if part
        ),
        "facts": facts,
        "notes": notes,
        "config_line": config_line,
        "config": settings.describe(),
        "determinism": determinism_view,
        "runs": manifest.get("runs") or [],
        "targets": targets,
        "models": _table(
            [L("target"), L("model"), L("server"), "GPU", L("context"), L("weights")],
            "llllrl",
            [row[:6] for row in model_rows],
            [0, 0, 0, 3, 0, 2],
        ),
        "capabilities": _table(
            [L("target"), L("tool_output"), L("native_output"), L("image_input"), L("mode_col"), L("settings_col")],
            "lcccll",
            [[row[0], *row[6:], ", ".join(modes) or "–", used] for row in model_rows],
            [2, 0, 0, 0, 0, 3],
        ),
        "levels": levels,
        "charts": charts,
        "chart_caption": L("fig_throughput") + (L("log_scale") if log_x else ""),
        "states": _states(plan["case_count"], targets, data, expected, norm),
        "charts_note": L("charts_limited", n=MAX_SERIES) if len(targets) > MAX_SERIES else None,
        "throughput": _table(
            [
                L("target"),
                L("concurrency"),
                "ok",
                L("h_failed"),
                L("h_wall"),
                L("h_attempts_s"),
                L("h_tokens_s"),
                L("h_mean"),
                "p50 s",
                "p95 s",
            ],
            "lrrrrrrrrr",
            [
                [
                    r["target"],
                    str(r["concurrency"]),
                    L.num(r["attempts_ok"]),
                    L.num(r["attempts_failed"]),
                    L.num(r["wall_s"], 1),
                    L.num(r["attempts_per_s"]),
                    L.num(r["output_tokens_per_s"], 1),
                    L.num(r["latency_mean_s"]),
                    L.num(r["latency_p50_s"]),
                    L.num(r["latency_p95_s"]),
                ]
                for r in throughput
            ],
        ),
        "warmup": warmup,
        "autotune": autotune,
        "server": _table(
            [
                L("target"),
                L("concurrency"),
                L("h_queue"),
                L("h_ttft"),
                "prefill s",
                "decode s",
                L("h_cache"),
                L("h_reset"),
            ],
            "lrrrrrrr",
            server_rows,
        )
        if server_rows
        else None,
        "matrix_values": [
            [(matrix[(a, b)][0] / matrix[(a, b)][1]) if matrix[(a, b)][1] else None for b in targets] for a in targets
        ],
        "matrix": _table(
            ["", *targets],
            "l" + "r" * len(targets),
            [[a] + [L.pct(*matrix[(a, b)]) if matrix[(a, b)][1] else "–" for b in targets] for a in targets],
        ),
        "per_target": _table(
            [L("target"), "ok", L("h_failed"), L("h_requests"), L("h_input"), L("h_output"), L("h_expected")],
            "lrrrrrr",
            [
                [
                    t,
                    L.num(per_target[t]["ok"]),
                    L.num(per_target[t]["failed"]),
                    L.num(per_target[t]["requests"]),
                    L.num(per_target[t]["input_tokens"]),
                    L.num(per_target[t]["output_tokens"]),
                    L.pct(*expected_hits[t]) if expected_hits[t][1] else "–",
                ]
                for t in targets
            ],
        ),
        "cases": payload,
        "cases_note": L(
            "cases_note",
            n=L.num(len(payload["cases"])),
            total=L.num(plan["case_count"]),
            f=L.num(len(errors)),
            d=L.num(len(differing)),
            x=L.num(wrong),
        ),
        "has_casebook": casebook is not None,
        "instructions": [str(text) for text in agent.get("instructions") or []],
        "agent": {k: agent.get(k) for k in ("model_settings", "tools", "output_schema", "system_prompts")},
        "request_parameters": ledger.events("request_parameters"),
        "plan_targets": plan["targets"],
        "snapshots": snapshots,
        "environment": {
            **{k: v for k, v in environment.items() if k not in ("kind", "at", "worker_id")},
            **({"execute": ledger.events("execution")[-1]["options"]} if ledger.events("execution") else {}),
        },
    }
