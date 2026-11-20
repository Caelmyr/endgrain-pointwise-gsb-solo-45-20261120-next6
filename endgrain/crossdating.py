"""Dual-transect cross-dating into one shared annual sequence.

Two transects are sampled from the inner end outward with the same signal
machinery as the single :mod:`rings` tool.  Their contrast crossings are then
aligned by ring-width similarity with explicit gap penalties so that rings
which only one transect can resolve are kept instead of being silently merged.

The alignment never treats the distance across a stretch without image
evidence (a check, an unsupported band or a transect end) as a single annual
increment: such intervals are flagged ``evidence_gap`` and leave the affected
ring width undecided until the operator resolves the surrounding ambiguity.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .errors import WorkbenchError
from .geometry import Plane, Raster, pixel_point
from .imaging import safe_ratio, symmetric_overlay
from .rings import (
    Signal, add_positions, auto_crossings, manual_crossings,
    runs_of, transect_signal,
)
from .validation import choice, document, fields, integer, list_of, number

TRANSECTS = ("a", "b")
#: alignment step codes
MATCH = "match"          # one crossing from each transect, same annual ring
GAP_A = "only_a"         # ring resolved on transect A only
GAP_B = "only_b"         # ring resolved on transect B only
AMBIGUOUS = "ambiguous"   # tentative pairing that needs operator confirmation

# Gap penalty in the same 0..1 similarity units as the match score.
GAP_PENALTY = 0.30
# A pairing whose margin over the next-best alignment is below this is ambiguous.
AMBIGUITY_MARGIN = 0.16
# Width ratio above which two widths are considered incompatible.
WIDTH_RATIO_LIMIT = 2.2
MAX_CONFIRMATIONS = 120


@dataclass(frozen=True)
class TransectReading:
    code: str
    config: dict
    signal: Signal
    crossings: list[dict]


def _transect_config(raw: Any, plane: Plane, code: str, defaults: list[list[float]],
                     shared: dict) -> dict:
    body = document(raw, f"transect {code.upper()} settings")
    # Detection controls are shared at the top level; per-transect detection
    # keys are only tolerated when re-validating a stored run produced before
    # the shared controls were introduced.
    allowed = {"endpoints", "manual_crossings", "band", "prominence", "minimum_gap", "polarity"}
    fields(body, allowed, f"transect {code.upper()}")
    w, h = plane.width, plane.height
    ends = [
        pixel_point(point, f"transect {code.upper()} endpoint {idx}", w, h)
        for idx, point in enumerate(list_of(body.get("endpoints", defaults), f"transect {code.upper()} endpoints", 2, 2), 1)
    ]
    if float(np.linalg.norm(np.asarray(ends[1]) - ends[0])) < 20:
        raise WorkbenchError(
            f"Place transect {code.upper()} endpoints at least twenty measuring pixels apart.",
            f"transect_{code}_endpoints",
        )
    crossings = list_of(body.get("manual_crossings", []), f"transect {code.upper()} manual crossings", 0, 120)
    return {
        "endpoints": ends,
        "band": integer(body.get("band", shared["band"]), "transect band in pixels", 1, 41),
        "prominence": number(body.get("prominence", shared["prominence"]), "boundary prominence", 0.003, 0.45),
        "minimum_gap": integer(body.get("minimum_gap", shared["minimum_gap"]),
                               "candidate minimum separation", 2, 200),
        "polarity": choice(body.get("polarity", shared["polarity"]), "boundary contrast",
                           ("dark", "light")),
        "manual_crossings": sorted({number(value, "crossing fraction", 0, 1) for value in crossings}),
    }


def normalize_params(values: Any, plane: Plane) -> dict:
    body = document(values, "cross-dating settings")
    fields(body, {"transects", "width_tolerance", "band", "prominence", "minimum_gap",
                  "polarity", "decisions"}, "cross-dating")
    w, h = plane.width, plane.height
    shared = {
        "band": integer(body.get("band", 7), "transect band in pixels", 1, 41),
        "prominence": number(body.get("prominence", 0.024), "boundary prominence", 0.003, 0.45),
        "minimum_gap": integer(body.get("minimum_gap", 6), "candidate minimum separation", 2, 200),
        "polarity": choice(body.get("polarity", "dark"), "boundary contrast", ("dark", "light")),
    }
    default_a = [[w * 0.30, h * 0.50], [w * 0.62, h * 0.46]]
    default_b = [[w * 0.40, h * 0.66], [w * 0.78, h * 0.62]]
    raw_transects = document(body.get("transects", {}), "transect settings")
    fields(raw_transects, set(TRANSECTS), "transect settings")
    configs = {
        "a": _transect_config(raw_transects.get("a", {}), plane, "a", default_a, shared),
        "b": _transect_config(raw_transects.get("b", {}), plane, "b", default_b, shared),
    }
    tolerance = number(body.get("width_tolerance", 0.35), "ring width tolerance", 0.05, 0.90)
    decisions = _normalize_decisions(body.get("decisions", []))
    return {
        "width_tolerance": tolerance,
        **shared,
        "transects": configs,
        "decisions": decisions,
    }


def _normalize_decisions(raw: Any) -> list[dict]:
    decisions = []
    seen = set()
    for entry in list_of(raw, "cross-dating decisions", 0, MAX_CONFIRMATIONS):
        item = document(entry, "cross-dating decision")
        fields(item, {"ambiguity_id", "choice"}, "cross-dating decision")
        ambiguity_id = integer(item.get("ambiguity_id"), "ambiguity identifier", 0, 10_000)
        choice_code = choice(item.get("choice"), "ambiguity choice", (MATCH, GAP_A, GAP_B))
        if ambiguity_id in seen:
            raise WorkbenchError("Each ambiguous zone can only carry one decision.", "decisions")
        seen.add(ambiguity_id)
        decisions.append({"ambiguity_id": ambiguity_id, "choice": choice_code})
    return decisions


def _read_transect(raster: Raster, config: dict, code: str) -> TransectReading:
    signal = transect_signal(raster, config)
    if config["manual_crossings"]:
        crossings = manual_crossings(signal, config)
    else:
        crossings = auto_crossings(signal, config)
    if len(crossings) < 2:
        raise WorkbenchError(
            f"Transect {code.upper()} needs at least two ring crossings to build a sequence. "
            "Add manual crossings or adjust its endpoints.",
            f"transect_{code}_endpoints",
        )
    return TransectReading(code, config, signal, add_positions(signal, crossings))


def _width_series(reading: TransectReading) -> np.ndarray:
    gaps = np.asarray([item["gap_from_previous"] for item in reading.crossings], dtype=np.float64)
    # First crossing has no previous crossing inside the sequence.
    gaps[0] = np.nan
    return gaps


def _support_break(reading: TransectReading, left_index: int, right_index: int) -> bool:
    """True when the interval between two crossings loses image evidence."""
    if right_index <= left_index or left_index < 0:
        return False
    signal = reading.signal
    if not bool(signal.valid[left_index:right_index + 1].all()):
        return True
    return any(
        bool(reading.crossings[ix].get("gap_has_missing_support"))
        for ix in range(left_index + 1, right_index + 1)
    )


def _pair_score(width_a: float, width_b: float, tolerance: float) -> float:
    """Similarity in [0, 1]; higher means the two widths describe one year."""
    if not np.isfinite(width_a) or not np.isfinite(width_b) or width_a <= 0 or width_b <= 0:
        return 0.0
    ratio = max(width_a, width_b) / min(width_a, width_b)
    if ratio > WIDTH_RATIO_LIMIT:
        return 0.0
    relative = abs(width_a - width_b) / max(width_a, width_b)
    return float(np.clip(1.0 - relative / max(tolerance * 2.0, 1e-6), 0.0, 1.0))


@dataclass
class AlignmentCell:
    score: float
    origin: str  # "match", "gap_a", "gap_b"
    i: int
    j: int


def _dp_cell(
    i: int, j: int, grid: list[list["AlignmentCell"]], widths_a: np.ndarray, widths_b: np.ndarray,
    reading_a: TransectReading, reading_b: TransectReading, tolerance: float,
    force: str | None = None,
) -> "AlignmentCell":
    """Fill one alignment cell, optionally forcing a specific step kind."""
    wa, wb = widths_a[i - 1], widths_b[j - 1]
    # An interval that spans an evidence break cannot be scored as a clean
    # annual width; only pair such crossings weakly.
    broken_a = _support_break(reading_a, i - 2, i - 1) if i >= 2 else False
    broken_b = _support_break(reading_b, j - 2, j - 1) if j >= 2 else False
    similarity = _pair_score(wa, wb, tolerance)
    if broken_a or broken_b:
        similarity *= 0.35
    options = {
        MATCH: (grid[i - 1][j - 1].score + similarity, i - 1, j - 1),
        GAP_A: (grid[i - 1][j].score - GAP_PENALTY, i - 1, j - 1),
        GAP_B: (grid[i][j - 1].score - GAP_PENALTY, i - 1, j - 1),
    }
    if force is not None:
        # A hard operator constraint dominates the data-driven score.
        score, fi, fj = options[force]
        return AlignmentCell(score + 1_000.0, force, fi, fj)
    best_score, origin = max(((value[0], kind) for kind, value in options.items()),
                             key=lambda item: item[0])
    return AlignmentCell(best_score, origin, options[origin][1], options[origin][2])


def _traceback(grid: list[list["AlignmentCell"]], n: int, m: int) -> list[dict]:
    steps = []
    i, j = n, m
    while i > 0 or j > 0:
        cell = grid[i][j]
        if cell.origin == MATCH:
            steps.append({"kind": MATCH, "a": i, "b": j})
            i, j = i - 1, j - 1
        elif cell.origin == GAP_A:
            steps.append({"kind": GAP_A, "a": i, "b": None})
            i -= 1
        else:
            steps.append({"kind": GAP_B, "a": None, "b": j})
            j -= 1
    steps.reverse()
    return steps


def _align(
    widths_a: np.ndarray,
    widths_b: np.ndarray,
    reading_a: TransectReading,
    reading_b: TransectReading,
    tolerance: float,
    constraints: dict[tuple[int, int], str] | None = None,
) -> list[dict]:
    """Sequence alignment of crossings (index 0 is the innermost crossing).

    Returns steps in inner-to-outer order with 1-based crossing ordinals.
    ``constraints`` maps a (1-based A ordinal, 1-based B ordinal) match cell to
    a forced step kind, carrying confirmed operator decisions through the whole
    alignment instead of editing a single row in place.
    """
    constraints = constraints or {}
    n, m = len(widths_a), len(widths_b)
    grid = [[AlignmentCell(-np.inf, "", 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    grid[0][0] = AlignmentCell(0.0, "start", 0, 0)
    for i in range(1, n + 1):
        grid[i][0] = AlignmentCell(grid[i - 1][0].score - GAP_PENALTY, GAP_A, i - 1, 0)
    for j in range(1, m + 1):
        grid[0][j] = AlignmentCell(grid[0][j - 1].score - GAP_PENALTY, GAP_B, 0, j - 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            grid[i][j] = _dp_cell(
                i, j, grid, widths_a, widths_b, reading_a, reading_b, tolerance,
                force=constraints.get((i, j)),
            )
    return _traceback(grid, n, m)


def _decision_map(decisions: list[dict]) -> dict[int, str]:
    return {item["ambiguity_id"]: item["choice"] for item in decisions}


def _ambiguous_zones(steps: list[dict], readings: dict[str, TransectReading],
                     widths: dict[str, np.ndarray], tolerance: float) -> list[dict]:
    """Find zones where two distinct alignments are both plausible.

    A zone is anchored on a match whose width agreement is weak and offers the
    alternative of shifting one transect by one ring (an ``only_a`` / ``only_b``
    interpretation).  This captures both repeated narrow rings (false ring /
    double dark line) and a ring lost beside a check.
    """
    reading_a, reading_b = readings["a"], readings["b"]
    zones = []
    for position, step in enumerate(steps):
        if step["kind"] != MATCH:
            continue
        ia, ib = step["a"] - 1, step["b"] - 1
        direct = _pair_score(widths["a"][ia], widths["b"][ib], tolerance)
        alternatives = []
        next_match = next(
            (other for other in steps[position + 1:] if other["kind"] == MATCH), None,
        )
        # Alternative 1: A's crossing is a ring B does not have (or missed).
        if next_match is not None:
            shifted_b = _pair_score(widths["a"][ia], widths["b"][next_match["b"] - 1], tolerance)
            if shifted_b >= direct - AMBIGUITY_MARGIN:
                alternatives.append({"choice": GAP_A, "score": round(shifted_b, 4)})
        # Alternative 2: B's crossing is a ring A does not have (or missed).
        if next_match is not None:
            shifted_a = _pair_score(widths["b"][ib], widths["a"][next_match["a"] - 1], tolerance)
            if shifted_a >= direct - AMBIGUITY_MARGIN:
                alternatives.append({"choice": GAP_B, "score": round(shifted_a, 4)})
        weak = direct < 0.55
        broken = (
            (ia >= 1 and _support_break(reading_a, ia - 1, ia))
            or (ib >= 1 and _support_break(reading_b, ib - 1, ib))
        )
        if alternatives and (weak or broken):
            zones.append({
                "ambiguity_id": len(zones),
                "a_crossing": step["a"],
                "b_crossing": step["b"],
                "match_score": round(direct, 4),
                "alternatives": alternatives,
                "near_evidence_gap": bool(broken),
            })
    return zones


def _build_constraints(zones: list[dict], decisions: dict[int, str]) -> dict[tuple[int, int], str]:
    """Turn confirmed decisions into hard DP alignment constraints.

    A ``match`` decision forces the zone's A/B cell to pair.  An ``only_a``
    decision forces A's crossing into a gap step (the matched B crossing stays
    for the following row), and symmetrically for ``only_b``.
    """
    constraints: dict[tuple[int, int], str] = {}
    for zone in zones:
        choice_code = decisions.get(zone["ambiguity_id"])
        if choice_code is None:
            continue
        cell = (zone["a_crossing"], zone["b_crossing"])
        if choice_code == MATCH:
            constraints[cell] = MATCH
        elif choice_code == GAP_A:
            # A's crossing is its own ring: emit it before consuming B's.
            constraints[cell] = GAP_A
        else:
            constraints[cell] = GAP_B
    return constraints


def _apply_decisions(steps: list[dict], zones: list[dict], decisions: dict[int, str]) -> list[dict]:
    """Mark the steps that belong to confirmed or pending ambiguous zones.

    After a constrained re-alignment a confirmed gap decision shifts steps, so
    the zone's A crossing may appear on an ``only_a`` step and its B crossing on
    the next ``match`` step.  We therefore tag every step touching either of a
    zone's crossings and resolve the confirmed kind from the decision.
    """
    by_a = {zone["a_crossing"]: zone for zone in zones}
    by_b = {zone["b_crossing"]: zone for zone in zones}
    annotated = []
    for step in steps:
        entry = dict(step)
        zone = by_a.get(step.get("a")) or by_b.get(step.get("b"))
        if zone is not None and (
            step.get("a") == zone["a_crossing"] or step.get("b") == zone["b_crossing"]
        ):
            entry["ambiguity_id"] = zone["ambiguity_id"]
            choice_code = decisions.get(zone["ambiguity_id"])
            entry["status"] = "confirmed_" + choice_code if choice_code is not None else AMBIGUOUS
        else:
            entry["status"] = step["kind"]
        annotated.append(entry)
    return annotated


def _build_rows(
    steps: list[dict],
    readings: dict[str, TransectReading],
    widths: dict[str, np.ndarray],
    decisions: dict[int, str],
) -> list[dict]:
    """Assemble the shared annual sequence, one row per annual ring."""
    reading_a, reading_b = readings["a"], readings["b"]
    rows = []
    for step in steps:
        kind = step["kind"]
        ia = step["a"] - 1 if step["a"] else None
        ib = step["b"] - 1 if step["b"] else None
        ambiguity_id = step.get("ambiguity_id")
        confirmed = step["status"].startswith("confirmed_")

        # After a constrained re-alignment the step kind already reflects the
        # confirmed decision, so each crossing is consumed exactly once.
        a_used = kind == MATCH or kind == GAP_A
        b_used = kind == MATCH or kind == GAP_B

        # Evidence gaps: a width that spans unsupported samples is never a
        # single-year width on that transect.
        gap_a = a_used and ia is not None and ia >= 1 and _support_break(reading_a, ia - 1, ia)
        gap_b = b_used and ib is not None and ib >= 1 and _support_break(reading_b, ib - 1, ib)

        width_a = float(widths["a"][ia]) if a_used and ia is not None and np.isfinite(widths["a"][ia]) and not gap_a else None
        width_b = float(widths["b"][ib]) if b_used and ib is not None and np.isfinite(widths["b"][ib]) and not gap_b else None
        shared_width = None
        width_status = "measured"
        candidates = [value for value in (width_a, width_b) if value is not None]
        if candidates:
            shared_width = float(np.mean(candidates))
        # The innermost crossing has no preceding crossing to measure, so it
        # anchors the sequence but carries no annual width of its own.
        inner_anchor = width_a is None and width_b is None and not gap_a and not gap_b
        if inner_anchor:
            width_status = "inner_anchor"
        elif gap_a or gap_b:
            width_status = "evidence_gap" if not candidates else "partial_evidence"

        cross_a = reading_a.crossings[ia] if a_used and ia is not None else None
        cross_b = reading_b.crossings[ib] if b_used and ib is not None else None

        if confirmed:
            row_status = "confirmed"
        elif step["status"] == AMBIGUOUS:
            row_status = AMBIGUOUS
        elif width_status == "evidence_gap":
            row_status = "evidence_gap"
        elif inner_anchor:
            # The first crossing anchors the sequence but has no increment.
            row_status = "inner_anchor"
        elif kind == GAP_A:
            row_status = "only_a"
        elif kind == GAP_B:
            row_status = "only_b"
        else:
            row_status = "matched"

        rows.append({
            "year_index": None,
            "status": row_status,
            "ambiguity_id": ambiguity_id,
            "a_crossing": (ia + 1) if a_used and ia is not None else None,
            "b_crossing": (ib + 1) if b_used and ib is not None else None,
            "a_distance": cross_a["distance"] if cross_a else None,
            "b_distance": cross_b["distance"] if cross_b else None,
            "a_x": cross_a["x"] if cross_a else None,
            "a_y": cross_a["y"] if cross_a else None,
            "b_x": cross_b["x"] if cross_b else None,
            "b_y": cross_b["y"] if cross_b else None,
            "a_origin": cross_a["origin"] if cross_a else None,
            "b_origin": cross_b["origin"] if cross_b else None,
            "width_a": width_a,
            "width_b": width_b,
            "shared_width": shared_width,
            "width_status": width_status,
            "evidence_gap_a": bool(gap_a),
            "evidence_gap_b": bool(gap_b),
        })
    # Number years consecutively over the rows that actually carry a ring.
    ring_number = 0
    for row in rows:
        ring_number += 1
        row["year_index"] = ring_number
    return rows


def _metrics(rows: list[dict], readings: dict[str, TransectReading], zones: list[dict],
             decisions: dict[int, str]) -> dict:
    measured = [row["shared_width"] for row in rows if row["shared_width"] is not None]
    pending = [zone for zone in zones if zone["ambiguity_id"] not in decisions]
    only_a = sum(1 for row in rows if row["status"] == "only_a")
    only_b = sum(1 for row in rows if row["status"] == "only_b")
    evidence_gaps = sum(1 for row in rows if row["status"] == "evidence_gap")
    return {
        "shared_rings": sum(1 for row in rows if row["year_index"] is not None),
        "transect_a_crossings": len(readings["a"].crossings),
        "transect_b_crossings": len(readings["b"].crossings),
        "paired_rings": sum(1 for row in rows if row["a_crossing"] and row["b_crossing"]),
        "single_line_rings": only_a + only_b,
        "ambiguities": len(pending),
        "resolved_ambiguities": len(zones) - len(pending),
        "evidence_gap_rings": evidence_gaps,
        "mean_width": float(np.mean(measured)) if measured else None,
        "median_width": float(np.median(measured)) if measured else None,
        "measured_widths": len(measured),
        "supported_transect_a": safe_ratio(int(readings["a"].signal.valid.sum()), len(readings["a"].signal.valid)),
        "supported_transect_b": safe_ratio(int(readings["b"].signal.valid.sum()), len(readings["b"].signal.valid)),
    }


def _draw_transect(pen: ImageDraw.ImageDraw, reading: TransectReading, color: tuple,
                   flank_color: tuple, size: int) -> None:
    signal = reading.signal
    start, end = signal.points[[0, -1]]
    heading = end - start
    norm = float(np.linalg.norm(heading))
    tangent = np.array([-heading[1], heading[0]]) / norm
    flank = max(4, signal.half_band)
    for side in (-1, 1):
        pen.line([tuple(start + side * tangent * flank), tuple(end + side * tangent * flank)],
                 fill=flank_color, width=1)
    pen.line([tuple(start), tuple(end)], fill=color, width=size)
    for a, b in runs_of(~signal.valid):
        pen.line([tuple(signal.points[a]), tuple(signal.points[b - 1])], fill=(205, 102, 75), width=size + 2)


def crossdating_preview(raster: Raster, readings: dict[str, TransectReading],
                        rows: list[dict], zones: list[dict],
                        decisions: dict[int, str]) -> np.ndarray:
    output = Image.fromarray(symmetric_overlay(raster.rgb, [], support=raster.support))
    pen = ImageDraw.Draw(output, mode="RGB")
    size = max(2, int(round(max(raster.plane.width, raster.plane.height) / 430)))
    _draw_transect(pen, readings["a"], (231, 232, 220), (232, 200, 129), size)
    _draw_transect(pen, readings["b"], (214, 226, 238), (140, 184, 214), size)

    pending = {zone["ambiguity_id"] for zone in zones if zone["ambiguity_id"] not in decisions}
    # Offset labels perpendicular to each line so adjacent years never collide
    # and the two transects' labels for one shared year sit on opposite sides.
    tangents = {}
    for code in TRANSECTS:
        start, end = readings[code].signal.points[[0, -1]]
        heading = end - start
        norm = float(np.linalg.norm(heading))
        tangents[code] = np.array([-heading[1], heading[0]]) / norm
    label_side = {"a": 1, "b": -1}
    for row in rows:
        if row["year_index"] is None:
            continue
        text = str(row["year_index"])
        is_pending = row["status"] == AMBIGUOUS and row["ambiguity_id"] in pending
        is_gap = row["status"] == "evidence_gap"
        fill = (205, 102, 75) if is_gap else ((224, 148, 66) if is_pending else (45, 58, 42))
        for code in TRANSECTS:
            x, y = row.get(code + "_x"), row.get(code + "_y")
            if x is None:
                continue
            tone = (252, 234, 183) if code == "a" else (190, 222, 244)
            pen.ellipse((x - size - 1, y - size - 1, x + size + 1, y + size + 1),
                        fill=fill, outline=tone, width=1)
            offset = tangents[code] * label_side[code] * (size * 9 + 4)
            pen.text((x + offset[0] - 3, y + offset[1] - 5), text, fill=tone)
    # Inner-end markers show the reading direction (inner -> outer).
    for reading, tone in ((readings["a"], (252, 234, 183)), (readings["b"], (190, 222, 244))):
        x, y = reading.signal.points[0]
        r = size + 3
        pen.ellipse((x - r, y - r, x + r, y + r), fill=tone, outline=(55, 40, 28), width=2)
        pen.text((x + r + 2, y - 6), reading.code.upper() + " inner", fill=tone)
    return np.asarray(output)


CROSSDATING_COLUMNS = [
    "shared_year", "row_status", "ambiguity_id",
    "a_crossing", "a_distance", "a_width", "a_evidence_gap", "a_origin",
    "b_crossing", "b_distance", "b_width", "b_evidence_gap", "b_origin",
    "shared_width", "width_status",
]


def _table_rows(rows: list[dict]) -> list[list]:
    return [[
        row["year_index"], row["status"], row["ambiguity_id"],
        row["a_crossing"], row["a_distance"], row["width_a"], row["evidence_gap_a"], row["a_origin"],
        row["b_crossing"], row["b_distance"], row["width_b"], row["evidence_gap_b"], row["b_origin"],
        row["shared_width"], row["width_status"],
    ] for row in rows]


def _profile_chart(reading: TransectReading, rows: list[dict], which: str) -> dict:
    signal = reading.signal
    selected = np.linspace(0, len(signal.points) - 1, min(len(signal.points), 720), dtype=int)
    crossing_key = which + "_crossing"
    markers = []
    for row in rows:
        ordinal = row[crossing_key]
        if ordinal:
            markers.append(reading.crossings[ordinal - 1]["distance"])
    return {
        "kind": "profile",
        "x": signal.distance[selected].tolist(),
        "y": [float(signal.raw[index]) if signal.valid[index] else None for index in selected],
        "markers": markers,
        "x_label": "Distance transect " + which.upper(),
        "y_label": "Luminance",
    }


def measure(raster: Raster, params: dict) -> tuple[dict, np.ndarray, list[list]]:
    readings = {
        "a": _read_transect(raster, params["transects"]["a"], "a"),
        "b": _read_transect(raster, params["transects"]["b"], "b"),
    }
    widths = {code: _width_series(reading) for code, reading in readings.items()}
    # First pass, unconstrained: discover the candidate ambiguous zones.
    baseline_steps = _align(widths["a"], widths["b"], readings["a"], readings["b"],
                            params["width_tolerance"])
    zones = _ambiguous_zones(baseline_steps, readings, widths, params["width_tolerance"])
    decisions = _decision_map(params["decisions"])
    unknown = sorted(set(decisions) - {zone["ambiguity_id"] for zone in zones})
    if unknown:
        raise WorkbenchError(
            "A decision refers to an ambiguous zone that no longer exists. Re-open the latest reading.",
            "decisions",
        )
    # Second pass: confirmed decisions become hard constraints on a full
    # re-alignment, so no crossing is ever consumed by two shared rings.
    constraints = _build_constraints(zones, decisions)
    steps = _align(widths["a"], widths["b"], readings["a"], readings["b"],
                   params["width_tolerance"], constraints)
    annotated = _apply_decisions(steps, zones, decisions)
    rows = _build_rows(annotated, readings, widths, decisions)
    pending = [zone for zone in zones if zone["ambiguity_id"] not in decisions]

    notices = [
        "The shared sequence is a proposal. Confirm every highlighted ambiguous zone before treating ring counts as annual growth.",
        "Widths spanning unsupported samples or checks are withheld; the distance across an evidence gap is never reported as one ring width.",
        "Rings resolved on only one transect are retained and marked only_a / only_b.",
    ]
    if pending:
        notices.append(f"{len(pending)} ambiguous zone(s) still need your confirmation.")
    if not pending and zones:
        notices.append("All ambiguous zones are resolved; the shared sequence is operator-confirmed.")
    output = {
        "metrics": _metrics(rows, readings, zones, decisions),
        "notices": notices,
        "unit": raster.plane.unit,
        "rows": rows,
        "ambiguities": zones,
        "pending_ambiguities": [zone["ambiguity_id"] for zone in pending],
        "charts": {
            "a": _profile_chart(readings["a"], rows, "a"),
            "b": _profile_chart(readings["b"], rows, "b"),
        },
        "transects": {
            code: {
                "endpoints": reading.config["endpoints"],
                "crossings": [
                    {key: mark[key] for key in ("sample", "x", "y", "distance", "origin", "gap_from_previous", "gap_has_missing_support")}
                    for mark in reading.crossings
                ],
            }
            for code, reading in readings.items()
        },
    }
    visual = crossdating_preview(raster, readings, rows, zones, decisions)
    return output, visual, _table_rows(rows)
