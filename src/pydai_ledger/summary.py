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
from collections.abc import Callable, Sequence
from datetime import datetime
from html import escape
from typing import Any

from . import signing
from ._util import canonical_json
from .casebook import Casebook
from .config import LEVELS, POINTS, Config, determinism, missing
from .header import ReportHeader
from .i18n import Lang
from .ledger import Ledger
from .target import server_key

SECTIONS = ("targets", "benchmark", "validation", "cases")  # what a report can be limited to
NBSP = "\u00a0"
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
    return svg.replace("<text ", '<text font-family="Lato" ')


def _curve(
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
    *,
    bare: bool = False,
) -> str:
    """Latency over throughput, one line per target.

    Every point is one concurrency level and carries its number; the points are joined
    in the order of the levels. The solid line is the median latency, the dashed one the
    95th percentile. The recommended level is filled. Where the line stops going right
    and turns up, more concurrency only costs latency.
    """
    series = series[:MAX_SERIES]
    width, height, left, right, top, bottom = 620, 300, 74, 30, 24, 50
    plot_w, plot_h = width - left - right, height - top - bottom
    points = [point for one in series for point in one["points"]]
    x_max = _nice_max(max((point["x"] for point in points), default=0.0))
    y_max = _nice_max(max((point["p95"] or point["p50"] or 0.0 for point in points), default=0.0))

    def x(value: float) -> float:
        return left + plot_w * value / x_max

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
        value = x_max * tick / 4
        parts.append(
            f'<line x1="{x(value):.1f}" x2="{x(value):.1f}" y1="{base}" y2="{base + 4}" stroke="var(--text)"/>'
            f'<text font-size="12" fill="var(--text2)" x="{x(value):.1f}" y="{base + 17}" text-anchor="middle">{value:g}</text>'
        )
    tip_x, tip_y = left + plot_w + 12, top - 12
    parts.append(
        f'<line x1="{left}" x2="{tip_x}" y1="{base}" y2="{base}" stroke="var(--text)"/>'
        f'<path d="M{tip_x + 6},{base} l-8,-3.5 v7 z" fill="var(--text)"/>'
        f'<line x1="{left}" x2="{left}" y1="{base}" y2="{tip_y}" stroke="var(--text)"/>'
        f'<path d="M{left},{tip_y - 6} l-3.5,8 h7 z" fill="var(--text)"/>'
        f'<text font-size="12.5" fill="var(--text)" x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle">{escape(x_label)}</text>'
        f'<text font-size="12.5" fill="var(--text)" transform="translate(14,{top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>'
    )
    for index, one in enumerate(series):
        colour = f"var(--s{index + 1})"
        for key, style in (("p95", 'stroke-width="1.2" stroke-dasharray="4 3"'), ("p50", 'stroke-width="2"')):
            line = [(x(point["x"]), y(point[key])) for point in one["points"] if point[key] is not None]
            if len(line) > 1:
                path = " ".join(f"{px:.1f},{py:.1f}" for px, py in line)
                parts.append(f'<polyline points="{path}" fill="none" stroke="{colour}" {style}/>')
        for point in one["points"]:
            px = x(point["x"])
            if point["p95"] is not None:
                parts.append(f'<circle cx="{px:.1f}" cy="{y(point["p95"]):.1f}" r="2.2" fill="{colour}"/>')
            if point["p50"] is None:
                continue
            py = y(point["p50"])
            tip = escape(
                f"{one['name']}\n{point['level']}: {point['x']:,.1f} · p50 {point['p50']:,.3f} s"
                + (f" · p95 {point['p95']:,.3f} s" if point["p95"] is not None else ""),
                quote=True,
            )
            weight = ' font-weight="700"' if point["best"] else ""
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{colour if point["best"] else "var(--surface)"}" '
                f'stroke="{colour}" stroke-width="2"/>'
                f'<text font-size="12"{weight} fill="var(--text)" x="{px + 8:.1f}" y="{py + 14:.1f}">{point["level"]}</text>'
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="12" fill="transparent" data-tip="{tip}"/>'
            )
    parts.append("</svg>")
    svg = "".join(parts)
    return _print(svg, width, height) if bare else svg


def _states(
    case_ids: list[int],
    targets: list[str],
    data: dict[str, Any],
    expected: dict[tuple[int, int], str],
    norm: Callable[[str], str],
) -> dict[str, list[int]]:
    """Result of every case per target: 0 not run, 1 fine, 2 deviating, 3 failed.

    A target deviates in a case when one of its outputs differs from the expected
    output or, where none is given, from the most frequent output of all targets.
    """
    position = {case: at for at, case in enumerate(case_ids)}
    states = {target: [0] * len(case_ids) for target in targets}
    for (case, turn), by_target in data["outputs"].items():
        if case not in position:
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
            states[target][position[case]] = max(states[target][position[case]], 2 if deviates else 1)
    for case, failures in data["errors"].items():
        for failure in failures:
            if case in position:
                states[failure["target"]][position[case]] = 3
    return states


def _result_grid(
    states: dict[str, list[int]], x_label: str, *, ids: list[int] | None = None, bare: bool = False
) -> tuple[str, int]:
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
                    + ("" if bare else f' data-c="{ids[first] if ids else first}" data-t="{row}" data-s="{state}"')
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
    # An attempt of a case is told by the seed and the repeat it recorded, not by its number:
    # in a comparison the same number can mean another seed in another run.
    per_rep: dict[tuple[Any, int], dict[str, Any]] = defaultdict(
        lambda: {"ok": 0, "failed": 0, "seeds": set(), "order": math.inf}
    )
    runs: dict[tuple[str, int, int], dict[tuple[Any, int], str]] = defaultdict(dict)  # (target, case, turn) -> ...
    servers: dict[str, Counter[str | None]] = defaultdict(Counter)  # target -> server of its successful attempts
    for attempt in ledger.attempts():
        counts = per_target[attempt["target"]]
        counts["ok" if attempt["status"] == "ok" else "failed"] += 1
        key = (attempt["seed"], attempt["repeat"])
        of_rep = per_rep[key]
        of_rep["order"] = min(of_rep["order"], attempt["rep"])
        of_rep["ok" if attempt["status"] == "ok" else "failed"] += 1
        of_rep["seeds"].update(
            request["settings"].get("seed") for turn in attempt["turns"] for request in turn.get("requests", [])
        )
        if attempt["status"] != "ok":
            errors[attempt["case_index"]].append({"target": attempt["target"], "error": attempt["error"]})
            continue
        servers[attempt["target"]][attempt.get("server")] += 1
        stats = per_case[attempt["case_index"]]
        stats[0] += 1
        stats[1] += attempt["duration_s"]
        for turn in attempt["turns"]:
            stats[2] += turn["usage"].get("output_tokens") or 0
            counts["input_tokens"] += turn["usage"].get("input_tokens") or 0
            counts["output_tokens"] += turn["usage"].get("output_tokens") or 0
            counts["requests"] += len(turn.get("requests", []))
            output = canonical_json(turn["output"])
            outputs[(attempt["case_index"], turn["index"])][attempt["target"]][output] += 1
            runs[(attempt["target"], attempt["case_index"], turn["index"])][key] = output
    return {
        "outputs": outputs,
        "errors": errors,
        "per_target": per_target,
        "per_case": per_case,
        "per_rep": per_rep,
        "runs": runs,
        "servers": servers,
    }


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
    consistency with itself over all its attempts.
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


def _attempt_rows(data: dict[str, Any], norm: Callable[[str], str], L: Lang) -> list[list[str]]:
    """One row per attempt of a case: its seed and repeat, how many ran, and the share of
    outputs that are identical to those of the first attempt (same target, case and turn).
    """
    per_rep, runs = data["per_rep"], data["runs"]
    rows = []
    keys = sorted(per_rep, key=lambda key: (per_rep[key]["order"], str(key)))
    first = keys[0] if keys else None
    for position, number in enumerate(keys):
        of_rep = per_rep[number]
        seeds = sorted(str(seed) for seed in of_rep["seeds"] if seed is not None)
        shared = [by_rep for by_rep in runs.values() if number in by_rep and first in by_rep]
        same = sum(norm(by_rep[number]) == norm(by_rep[first]) for by_rep in shared)
        rows.append(
            [
                str(position + 1),
                ", ".join(seeds) or "–",
                str(number[1] + 1),
                L.num(of_rep["ok"]),
                L.num(of_rep["failed"]),
                "–" if number == first or not shared else L.pct(same, len(shared)),
            ]
        )
    return rows


def _validation(
    data: dict[str, Any], expected: dict[tuple[int, int], str], norm: Callable[[str], str]
) -> dict[str, dict[str, int]]:
    """Per target, what its attempts say about precision and accuracy.

    ``repeat``: groups of attempts with the same seed, and how many gave one output
    (repeatability). ``seed``: case turns run with several seeds, and how many gave the
    same output for every seed (seed robustness). The last
    three class every case turn: stable and right, stable and wrong (needs an expected
    output), unstable.
    """
    keys = ("repeat", "repeat_same", "seed", "seed_same", "right", "wrong", "unstable")
    out: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(keys, 0))
    for (target, case, turn), by_rep in data["runs"].items():
        stats = out[target]
        by_seed: dict[Any, list[str]] = defaultdict(list)
        for seed, repeat in sorted(by_rep, key=lambda key: key[1]):
            by_seed[seed].append(norm(by_rep[(seed, repeat)]))
        for group in by_seed.values():
            if len(group) > 1:
                stats["repeat"] += 1
                stats["repeat_same"] += len(set(group)) == 1
        if len(by_seed) > 1:
            stats["seed"] += 1
            stats["seed_same"] += len({group[0] for group in by_seed.values()}) == 1
        distinct = {output for group in by_seed.values() for output in group}
        want = expected.get((case, turn))
        if len(distinct) > 1:
            stats["unstable"] += 1
        elif want is not None:
            stats["right" if distinct == {norm(want)} else "wrong"] += 1
    return out


def _wall_seconds(stages: list[dict[str, Any]]) -> float:
    """Wall time covered by the stages; overlapping stages of parallel targets count once."""
    total, end = 0.0, float("-inf")
    for start, stop in sorted(
        (datetime.fromisoformat(s["started_at"]).timestamp(), datetime.fromisoformat(s["ended_at"]).timestamp())
        for s in stages
    ):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def _throughput_part(ledgers: list[Ledger], L: Lang, normalize: Callable[[Any], Any] | None) -> dict[str, Any]:
    """Everything the throughput section shows, from one or more benchmark ledgers."""
    rows = [row for ledger in ledgers for row in ledger.throughput()]
    names = list(dict.fromkeys(row["target"] for row in rows))
    levels = sorted({row["concurrency"] for row in rows})
    by_key = {(row["target"], row["concurrency"]): row for row in rows}
    tokens = any(row["output_tokens_per_s"] for row in rows)
    rate = "output_tokens_per_s" if tokens else "attempts_per_s"
    best = {name: tune["best"] for ledger in ledgers for name, tune in ledger.tunings().items()}
    curve = None
    if len(levels) > 1:  # a single point says nothing the table does not say
        curve = {
            "x_label": L("axis_tokens" if tokens else "axis_attempts"),
            "y_label": L("axis_latency"),
            "series": [
                {
                    "name": name,
                    "points": [
                        {
                            "level": level,
                            "x": by_key[(name, level)][rate],
                            "p50": by_key[(name, level)]["latency_p50_s"],
                            "p95": by_key[(name, level)]["latency_p95_s"],
                            "best": best.get(name) == level,
                        }
                        for level in levels
                        if (name, level) in by_key and by_key[(name, level)][rate]
                    ],
                }
                for name in names
            ],
        }
    tuning_rows, server_rows = [], []
    for ledger in ledgers:
        warmups = {w["target"]: w for w in ledger.events("warmup")}
        across = ledger.agreement(normalize)
        for name, tune in ledger.tunings().items():
            unit = "tok/s" if tune["metric"] == "output_tokens_per_s" else L("h_attempts_s")
            best = next(level for level in tune["levels"] if level["concurrency"] == tune["best"])
            warm = warmups.get(name)
            tuning_rows.append(
                [
                    name,
                    str(tune["best"]),
                    f"{L.num(best['rate'], 1)} {unit}",
                    ", ".join(map(str, tune["skipped"])) or "–",
                    L("warmup_item", n=warm["attempts"], s=L.num(warm["wall_s"], 1)) if warm else "–",
                    f"{L.num(across.agreeing)} {L('js_of')} {L.num(across.compared)}" if across.compared else "–",
                ]
            )
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
    head = [L("target"), L("concurrency"), "ok", L("h_failed"), L("h_wall"), L("h_attempts_min")]
    head += [L("h_tokens_s"), L("h_mean"), "p50 s", "p95 s"]
    return {
        "series_names": names,
        "curve": curve,
        "chart_caption": L("fig_curve"),
        "charts_note": L("charts_limited", n=MAX_SERIES) if len(names) > MAX_SERIES else None,
        "throughput": _table(
            head,
            "lrrrrrrrrr",
            [
                [
                    r["target"],
                    str(r["concurrency"]),
                    L.num(r["attempts_ok"]),
                    L.num(r["attempts_failed"]),
                    L.num(r["wall_s"], 1),
                    L.num(r["attempts_per_s"] * 60 if r["attempts_per_s"] else None, 0),
                    L.num(r["output_tokens_per_s"], 1),
                    L.num(r["latency_mean_s"]),
                    L.num(r["latency_p50_s"]),
                    L.num(r["latency_p95_s"]),
                ]
                for r in rows
            ],
        ),
        "tuning": _table(
            [L("target"), L("recommended_short"), L("h_rate"), L("h_skipped"), L("h_warmup"), L("h_levels_same")],
            "lrrrrr",
            tuning_rows,
        )
        if tuning_rows
        else None,
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
    }


def _levels(plan: dict[str, Any], target: str) -> str:
    """The concurrency a target ran at, as text."""
    given = plan["concurrency"]
    value = given.get(target) if isinstance(given, dict) else given
    return ", ".join(map(str, value)) if isinstance(value, list) else str(value if value is not None else "–")


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
    case_ids: list[int],
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

    chosen = sorted(case_ids, key=priority)
    note = None
    if max_cases is not None and len(chosen) > max_cases:
        chosen = chosen[:max_cases]
        note = [max_cases, len(case_ids)]
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


def _key_figures(
    ledger: Ledger, L: Lang, agreement: Any, expected_hits: dict[str, list[int]], ok: int, failed: int
) -> list[list[str]]:
    """The figures that head a report, each as label, value and a short note."""
    plan = ledger.manifest["plan"]
    targets, benchmark = plan["targets"], plan.get("kind") == "benchmark"
    hit, hit_total = sum(v[0] for v in expected_hits.values()), sum(v[1] for v in expected_hits.values())
    facts = [
        [L("cases"), L.num(plan["case_count"]), f"{len(targets)} {L('target' if len(targets) == 1 else 'targets')}"],
        [
            L("attempts"),
            L.num(ok),
            f"{L.num(failed)} {L('h_failed')}"
            + ("" if benchmark else f" · {plan['attempts_per_case']} {L('per_case')}"),
        ],
    ]
    if not benchmark:  # precision and trueness are the two verdicts of a validation
        facts += [
            [
                L("precision"),
                L.pct(agreement.agreeing, agreement.compared),
                f"{L.num(agreement.compared)} {L('turns_compared')}",
            ],
            [
                L("trueness"),
                L.pct(hit, hit_total),
                f"{L.num(hit_total)} {L('outputs_checked')}" if hit_total else L("no_expected"),
            ],
        ]
        seconds = _wall_seconds(ledger.events("stage"))
        if seconds:
            took = f"{L.num(seconds, 0)} s" if seconds < 120 else f"{L.num(seconds / 60, 1)} min"
            facts.append([L("attempts_min"), L.num(ok * 60 / seconds, 0), f"{L('duration')} {took}".replace(" ", NBSP)])
    for name in ledger.tunings() if benchmark else ():
        tuning = ledger.tuning(name)
        at = f"{L('at_concurrency')} {tuning.concurrency}".replace(" ", NBSP)
        facts.append([L("recommended"), str(tuning.concurrency), name])
        facts.append([L("attempts_min"), L.num(tuning.attempts_per_min, 0), at])
        if tuning.metric == "output_tokens_per_s":
            facts.append([L("tokens_s"), L.num(tuning.rate, 0), at])
    return facts


def _notes(
    ledger: Ledger,
    L: Lang,
    settings: Config,
    snapshots: dict[str, Any],
    sent: dict[str, Any],
    failed: int,
    invalid: bool,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """The notes of a report with their levels, the line about the configuration, and the
    preconditions for identical outputs as one compact entry per point.
    """
    manifest, plan = ledger.manifest, ledger.manifest["plan"]
    targets = [t["name"] for t in plan["targets"]]
    benchmark = plan.get("kind") == "benchmark"
    options = (ledger.events("execution")[-1:] or [{}])[0].get("options") or {}
    points = determinism(
        sent,
        {t["name"]: snapshots.get(t["name"], {}) for t in plan["targets"]},
        reset_prefix_cache={  # what happened, not what was asked for
            name: bool(options.get("reset_prefix_cache"))
            and all(s["prefix_cache_reset"] for s in ledger.events("stage") if s["target"] == name)
            for name in targets
        },
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
        section = "throughput" if benchmark and code.startswith("prefix_cache") else ""
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
    return notes, config_line, determinism_view


def _validation_part(
    L: Lang,
    plan: dict[str, Any],
    targets: list[str],
    models: dict[str, str],
    data: dict[str, Any],
    norm: Callable[[str], str],
    agreement: Any,
    expected: dict[tuple[int, int], str],
    expected_hits: dict[str, list[int]],
    case_ids: list[int],
    determinism_view: dict[str, Any],
    casebook: Casebook | None,
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, list[list[str]]]:
    """The validation section as a list of blocks (headings, texts, tables), laid out along
    ISO 5725: setup, precision under its three conditions, trueness, accuracy. Also the
    sentence about precision overall and the rows of the model comparison.
    """
    outputs = data["outputs"]
    checks = _validation(data, expected, norm)

    def share(stats: dict[str, int], same: str, total: str) -> str:
        return L.pct(stats[same], stats[total]) if stats[total] else "–"

    if agreement.compared and agreement.agreeing == agreement.compared:
        verdict = L("precision_all", n=L.num(agreement.compared))
    elif agreement.compared:
        verdict = L("precision_found", k=L.num(agreement.compared - agreement.agreeing), n=L.num(agreement.compared))
    else:
        verdict = L("precision_none")
    matrix = _pair_matrix(targets, outputs, norm)
    pairs = [(a, b) for i, a in enumerate(targets) for b in targets[i + 1 :]]
    same_model = [(a, b) for a, b in pairs if models[a] == models[b]]

    def total(same: str, of: str) -> tuple[int, int]:
        return sum(checks[t][same] for t in targets), sum(checks[t][of] for t in targets)

    conditions = [  # the three conditions of precision in ISO 5725, from narrow to wide
        ("repeatability_long", "c_target_seed", "c_nothing", total("repeat_same", "repeat")),
        ("seed_robustness_long", "c_target", "c_seed", total("seed_same", "seed")),
        (
            "reproducibility_long",
            "c_model",
            "c_target_changed",
            (sum(matrix[pair][0] for pair in same_model), sum(matrix[pair][1] for pair in same_model)),
        ),
    ]

    def setup(of: dict[str, Any], names: list[str]) -> str:
        """What one plan did, in one line."""
        seeds, drawn = of.get("seeds") or [], of.get("cases")
        line = L(
            "setup_text",
            t=len(names),
            m=len({models[name] for name in names}),
            c=L.num(len(drawn) if drawn is not None else of["case_count"]),
            seeds=L("seeds_are", s=", ".join(map(str, seeds))) if seeds else L("no_seeds"),
            r=of.get("repeats", 1),
            a=of["attempts_per_case"],
        )
        if drawn is not None:
            line += "  |  " + L("sampled", n=L.num(len(drawn)), seed=of.get("sample_seed"))
        return line

    # A comparison joins runs with plans of their own, so each run gets its own line.
    setups = [f"{', '.join(run['targets'])}:  {setup(run['plan'], run['targets'])}" for run in runs] or [
        setup(plan, targets)
    ]
    # A matrix pays off from three targets with one model on; below that the pairs say it all.
    crowd = [t for t in targets if sum(models[o] == models[t] for o in targets) >= 3]
    checked = any(expected_hits[t][1] for t in targets)
    validation: list[dict[str, Any]] = [
        {"kind": "h2", "text": L("setup")},
        *({"kind": "text", "text": line} for line in setups),
        {"kind": "text", "text": determinism_view["summary"]},
        {"kind": "det", "items": determinism_view["items"]},
        {"kind": "h2", "text": L("precision")},
        {"kind": "text", "text": L("precision_text")},
        {
            "kind": "table",
            "cap": None,
            "table": _table(
                [L("h_condition"), L("h_kept"), L("h_changed"), L("h_compared"), L("h_identical")],
                "lllrr",
                [
                    [L(name), L(kept), L(changed), L.num(of), L.pct(same, of) if of else L("not_tested")]
                    for name, kept, changed, (same, of) in conditions
                ],
                [2, 1, 2, 0, 0],
            ),
        },
        {"kind": "text", "text": verdict},
        {
            "kind": "table",
            "cap": L("per_target_title"),
            "table": _table(
                [L("target"), L("repeatability"), L("h_compared"), L("seed_robustness"), L("h_compared")],
                "lrrrr",
                [
                    [
                        t,
                        share(checks[t], "repeat_same", "repeat"),
                        L.num(checks[t]["repeat"]),
                        share(checks[t], "seed_same", "seed"),
                        L.num(checks[t]["seed"]),
                    ]
                    for t in targets
                ],
            ),
        },
    ]
    if same_model:
        validation.append(
            {
                "kind": "table",
                "cap": L("repro_title"),
                "table": _table(
                    [L("target"), L("target"), L("model"), L("h_compared"), L("h_identical")],
                    "lllrr",
                    [
                        [
                            a,
                            b,
                            models[a],
                            L.num(matrix[(a, b)][1]),
                            L.pct(*matrix[(a, b)]) if matrix[(a, b)][1] else "–",
                        ]
                        for a, b in same_model
                    ],
                    [1, 1, 1, 0, 0],
                ),
            }
        )
    if crowd:
        cells = [
            [matrix[(a, b)] if a != b and models[a] == models[b] and matrix[(a, b)][1] else None for b in crowd]
            for a in crowd
        ]
        validation.append(
            {
                "kind": "matrix",
                "cap": L("mx_title"),
                "targets": crowd,
                "rows": [[L.pct(*cell) if cell else "" for cell in row] for row in cells],
                "values": [[cell[0] / cell[1] if cell else None for cell in row] for row in cells],
            }
        )
    validation += [
        {
            "kind": "table",
            "cap": L("attempts_title"),
            "table": _table(
                [L("h_attempt"), L("h_seed"), L("h_repeat"), "ok", L("h_failed"), L("h_same")],
                "lrrrrr",
                _attempt_rows(data, norm, L),
            ),
        },
        {"kind": "h2", "text": L("trueness")},
        {
            "kind": "text",
            "text": L("trueness_text" if checked else "no_casebook" if casebook is None else "trueness_none"),
        },
    ]
    if checked:
        validation.append(
            {
                "kind": "table",
                "cap": None,
                "table": _table(
                    [L("target"), L("outputs_checked"), L("h_expected"), L("trueness")],
                    "lrrr",
                    [
                        [t, L.num(expected_hits[t][1]), L.num(expected_hits[t][0]), L.pct(*expected_hits[t])]
                        if expected_hits[t][1]
                        else [t, "–", "–", "–"]
                        for t in targets
                    ],
                ),
            }
        )
    validation += [
        {"kind": "h2", "text": L("accuracy")},
        {"kind": "text", "text": L("accuracy_text")},
        {
            "kind": "table",
            "cap": None,
            "table": _table(
                [L("target"), L("h_right"), L("h_wrong"), L("h_unstable")],
                "lrrr",
                [
                    [
                        t,
                        L.num(checks[t]["right"]) if expected_hits[t][1] else "–",
                        L.num(checks[t]["wrong"]) if expected_hits[t][1] else "–",
                        L.num(checks[t]["unstable"]),
                    ]
                    for t in targets
                ],
            ),
        },
    ]
    other_model = [
        [a, models[a], b, models[b], L.num(matrix[(a, b)][1]), L.pct(*matrix[(a, b)]) if matrix[(a, b)][1] else "–"]
        for a, b in pairs
        if models[a] != models[b]
    ]
    return validation, verdict, other_model


def build(
    ledger: Ledger,
    casebook: Casebook | None = None,
    *,
    lang: str = "en",
    header: ReportHeader | None = None,
    sections: Sequence[str] | None = None,
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
    benchmark = plan.get("kind") == "benchmark"
    unknown = set(sections or ()) - set(SECTIONS)
    if unknown:
        raise ValueError(f"Unknown sections {sorted(unknown)}; choose from {', '.join(SECTIONS)}.")
    header = header or ReportHeader()
    case_ids = [] if benchmark else list(plan.get("cases") or range(plan["case_count"]))
    models = {t["name"]: str(t.get("model_name") or t["name"]) for t in plan["targets"]}
    data = _collect(ledger)
    outputs, errors, per_target = data["outputs"], data["errors"], data["per_target"]
    norm = _normalizer(normalize)
    # Precision compares only what should be equal: the outputs of one model, on whatever target.
    agreement = ledger.agreement(normalize, groups=models)
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

    facts = _key_figures(ledger, L, agreement, expected_hits, ok, failed)

    settings = Config.load(config)
    # The server the results came from: the latest probe that matches the successful attempts.
    # A worker that probed another server later, but ran nothing, does not count. Without a
    # match (no success, or a ledger without server keys) the latest probe stands.
    snapshots: dict[str, Any] = {}
    matched: set[str] = set()
    for e in sorted(ledger.events("target"), key=lambda e: e.get("at") or ""):
        used = data["servers"].get(e["target"])
        fits = bool(used) and server_key(e["snapshot"]) == used.most_common(1)[0][0]
        if fits or e["target"] not in matched:
            snapshots[e["target"]] = e["snapshot"]
        if fits:
            matched.add(e["target"])
    sent = dict(plan["agent"].get("model_settings") or {})
    if plan.get("seeds"):  # the plan's seeds replace the agent's
        sent["seed"] = ", ".join(map(str, plan["seeds"]))
    notes, config_line, determinism_view = _notes(ledger, L, settings, snapshots, sent, failed, invalid)

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

    # -- throughput: only a benchmark ledger has it ---------------------------------------
    part = _throughput_part([ledger], L, normalize) if benchmark else None

    validation, verdict, other_model = _validation_part(
        L,
        plan,
        targets,
        models,
        data,
        norm,
        agreement,
        expected,
        expected_hits,
        case_ids,
        determinism_view,
        casebook,
        manifest.get("runs") or [],
    )
    execution = [L("targets_sequential" if plan.get("sequential_targets") else "targets_parallel")] * (len(targets) > 1)
    if not benchmark and ledger.events("warmup"):
        execution.append(
            f"{L('warmup')}: "
            + ", ".join(
                f"{w['target']} {L('warmup_item', n=w['attempts'], s=L.num(w['wall_s'], 1))}"
                for w in ledger.events("warmup")
            )
            + "."
        )
    show = {"targets": True, "benchmark": part is not None, "validation": not benchmark, "cases": not benchmark}
    if sections is not None:
        show = {name: on and name in sections for name, on in show.items()}
    dated = header.date or f"{manifest['sealed_at'][:16].replace('T', ' ')} UTC"
    authors = header.names
    authors_line = f"{L('authors' if len(authors) > 1 else 'author')}: {', '.join(authors)}" if authors else ""
    # -- cases -------------------------------------------------------------------------
    differing = {d["case_index"] for d in agreement.disagreements}
    payload = _cases_payload(
        case_ids,  # empty for a benchmark report, which lists no cases
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
        "title": header.title or L("benchmark_report" if benchmark else "report"),
        "benchmark": benchmark,
        "show": show,
        "validation": validation,
        "verdict": verdict,
        "model_pairs": _table(
            [L("target"), L("model"), L("target"), L("model"), L("h_compared"), L("h_identical")],
            "llllrr",
            other_model,
            [1, 1, 1, 1, 0, 0],
        )
        if other_model
        else None,
        "execution": " ".join(execution),
        "run_id": manifest["run_id"],
        "casebook_id": plan["casebook_id"],
        "sealed_at": manifest["sealed_at"],
        "complete": manifest["complete"],
        "signed": signed,
        "signer": signer,
        "stamp": ledger.stamp(),
        "author": ", ".join(authors) or None,
        "number": header.number,
        "date": dated,
        "status": f"{L('complete' if manifest['complete'] else 'incomplete')} · {signed}",
        "head": [
            line
            for line in (
                " · ".join(part for part in (header.number, dated, header.organisation) if part),
                authors_line,
                f"{L('complete' if manifest['complete'] else 'incomplete')} · {signed}",
            )
            if line
        ],
        "facts": facts,
        "notes": notes,
        "config_line": config_line,
        "config": settings.describe(),
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
        "case_ids": case_ids,
        "states": {} if benchmark else _states(case_ids, targets, data, expected, norm),
        **(
            part
            or {
                "series_names": [],
                "curve": None,
                "chart_caption": "",
                "charts_note": None,
                "throughput": _table([""], "l", []),
                "tuning": None,
                "server": None,
            }
        ),
        "per_target": _table(
            [
                L("target"),
                L("concurrency"),
                "ok",
                L("h_failed"),
                L("h_requests"),
                L("h_input"),
                L("h_output"),
            ],
            "lrrrrrr",
            [
                [
                    t,
                    _levels(plan, t),
                    L.num(per_target[t]["ok"]),
                    L.num(per_target[t]["failed"]),
                    L.num(per_target[t]["requests"]),
                    L.num(per_target[t]["input_tokens"]),
                    L.num(per_target[t]["output_tokens"]),
                ]
                for t in targets
            ],
        ),
        "cases": payload,
        "cases_note": L(
            "cases_note",
            n=L.num(len(payload["cases"])),
            total=L.num(len(case_ids)),
            f=L.num(len(errors)),
            d=L.num(len(differing)),
            x=L.num(wrong),
        ),
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
