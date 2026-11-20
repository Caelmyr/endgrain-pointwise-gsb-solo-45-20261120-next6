from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from math import isfinite
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .database import Database, as_dict
from .errors import WorkbenchError
from .imaging import encode_png
from .measurements import TOOLS
from .reports import render_csv, render_metric
from .validation import choice, document, fields, number, string

REPORTABLE_FORMATS = {"%", "area", "linear", "px", "ratio", "count", "density", "°"}
MAX_LOT_ROWS = 64


@dataclass(frozen=True)
class Survey:
    records: list[dict]
    context: dict
    warnings: list[str]


def lot_options(db: Database) -> list[dict]:
    with db.connection() as conn:
        rows = conn.execute(
            """SELECT lot, COUNT(*) AS inspections, MAX(updated_at) AS latest
               FROM inspections WHERE lot <> '' GROUP BY lot ORDER BY MAX(updated_at) DESC, lot LIMIT 80"""
        ).fetchall()
    return [dict(row) for row in rows]


def available_metrics() -> list[dict]:
    return [
        {
            "kind": code,
            "title": entry.title,
            "readings": [
                {"field": field, "title": label, "mode": mode}
                for field, (label, mode) in entry.metrics.items() if mode in REPORTABLE_FORMATS
            ],
        }
        for code, entry in TOOLS.items()
    ]


def chosen_metric(raw: Any) -> dict:
    body = document(raw, "lot survey")
    fields(body, {"lot", "kind", "metric", "below", "above", "units", "revisions"}, "lot survey")
    lot = string(body.get("lot"), "lot", 72, required=True)
    kind = choice(body.get("kind"), "measurement", tuple(TOOLS))
    spec = {name: info for name, info in TOOLS[kind].metrics.items() if info[1] in REPORTABLE_FORMATS}
    metric = choice(body.get("metric"), "metric", tuple(spec))
    title, mode = spec[metric]
    if mode == "°" and (body.get("below") is not None or body.get("above") is not None):
        raise WorkbenchError("Use the tool's angular tolerance for directional screening.", "metric")
    ceiling = 1 if mode == "%" else 1_000_000_000
    lower = None if body.get("below") is None else number(body["below"], "low screening limit", 0, ceiling)
    upper = None if body.get("above") is None else number(body["above"], "high screening limit", 0, ceiling)
    if lower is not None and upper is not None and lower >= upper:
        raise WorkbenchError("Set the lower screening limit below the upper limit.", "below")
    return {
        "lot": lot, "kind": kind, "metric": metric, "metric_title": title, "mode": mode,
        "below": lower, "above": upper,
        "units": choice(body.get("units", "metric"), "comparison units", ("metric", "pixels")),
        "revisions": choice(body.get("revisions", "current"), "wanted revisions", ("current", "latest")),
    }


def first_runs(db: Database, params: dict) -> list[dict]:
    sql = """SELECT i.id AS inspection_id, i.title, i.material, i.surface,
                    i.capture_id, i.revision AS active_revision, c.metadata,
                    r.id AS run_id, r.revision, r.params, r.result, r.snapshot, r.created_at
             FROM inspections i JOIN captures c ON c.id = i.capture_id
             LEFT JOIN runs r ON r.id = (
                 SELECT id FROM runs m WHERE m.inspection_id = i.id AND m.kind = ?
                 {restriction} ORDER BY created_at DESC, id LIMIT 1
             )
             WHERE i.lot = ? ORDER BY i.title COLLATE NOCASE, i.id LIMIT ?"""
    restriction = "AND m.revision = i.revision" if params["revisions"] == "current" else ""
    with db.transaction(write=False) as conn:
        results = conn.execute(sql.format(restriction=restriction), (params["kind"], params["lot"], MAX_LOT_ROWS + 1)).fetchall()
    return [as_dict(item, "metadata") for item in results]


def clean_record(row: dict) -> dict:
    capture = row["metadata"]
    return {
        "inspection_id": row["inspection_id"],
        "capture_id": row["capture_id"],
        "title": row["title"],
        "material": row["material"],
        "surface": row["surface"],
        "active_revision": row["active_revision"],
        "synthetic": capture["synthetic"],
        "run_id": row["run_id"],
        "revision": row["revision"],
        "recorded_at": row["created_at"],
    }


def safety_band(record: dict, val: float, limits: dict) -> str:
    lo, hi = limits["below"], limits["above"]
    if val is None:
        return "undefined"
    if lo is not None and val < lo:
        return "below"
    if hi is not None and val > hi:
        return "above"
    if record["revision"] != record["active_revision"]:
        return "historic"
    return "measured"


def screening_entry(row: dict, context: dict) -> dict:
    result = clean_record(row)
    if not row["run_id"]:
        return {**result, "value": None, "status": "missing", "unit": "", "display": "—"}
    import json
    reading = json.loads(row["result"])
    measure = reading["metrics"]
    snapshot = json.loads(row["snapshot"])
    mode = context["mode"]
    val = measure.get(context["metric"])
    unit = reading["unit"]
    required_unit = "mm" if context["units"] == "metric" else "px"
    comparable = (mode not in {"area", "linear", "density"}) or unit == required_unit
    defined = isinstance(val, (int, float)) and not isinstance(val, bool) and isfinite(val)
    can_measure = comparable and defined
    checked = "incompatible_scale" if not comparable else (safety_band(row, val, context) if defined else "undefined")
    basis = reading.get("basis") or {}
    measured_area = basis.get("survey_area", measure.get("survey_area"))
    return {
        **result,
        "status": checked,
        "value": val if can_measure else None,
        "recorded_value": val if defined else None,
        "display": render_metric(val if defined else None, mode, unit, measure),
        "unit": unit,
        "measured_area": measured_area,
        "measured_area_display": (
            None if measured_area is None
            else f"{measured_area:,.2f} mm²" if unit == "mm" else f"{measured_area:,.0f} px²"
        ),
        "density_basis": basis.get("density_basis", measure.get("density_basis")),
        "calibration": snapshot["plane"]["calibration"]["mode"],
        "roi": bool(snapshot["state"]["region"]),
        "lighting": bool(snapshot["state"]["lighting"]),
        "settings": json.loads(row["params"]),
    }


def summarize_numeric(records: list[dict], params: dict) -> dict:
    finite = [item["value"] for item in records if item.get("value") is not None]
    base = {"usable": len(finite), "median": None, "p10": None, "p90": None, "mean_axis": None, "concentration": None}
    if not finite:
        return base
    if params["mode"] == "°":
        doubled = np.radians(finite) * 2
        direction = np.mean(np.exp(1j * doubled))
        return {
            **base, "mean_axis": float(np.degrees(np.angle(direction) / 2) % 180) if abs(direction) >= 0.005 else None,
            "concentration": float(abs(direction)),
        }
    p10, center, p90 = np.percentile(finite, [10, 50, 90])
    return {**base, "p10": float(p10), "median": float(center), "p90": float(p90)}


def audit_groups(records: list[dict]) -> list[str]:
    real = [entry for entry in records if entry["run_id"]]
    if not real:
        return []
    notices = []
    if len({entry["synthetic"] for entry in real}) > 1:
        notices.append("The lot contains both synthetic and operator-supplied images.")
    if len({(entry["roi"], entry["lighting"], entry["calibration"]) for entry in real}) > 1:
        notices.append("Region, calibration or illumination settings differ across faces. Review the overlay context before drawing conclusions.")
    by_params = set()
    for entry in real:
        local = entry["settings"].copy()
        for field in ("endpoints", "manual_crossings", "swatches", "corners", "seed"):
            local.pop(field, None)
        by_params.add(repr(sorted(local.items())))
    if len(by_params) > 1:
        notices.append("Tool thresholds or window sizes changed between these measurements.")
    return notices


def measured_area_notice(records: list[dict], params: dict) -> list[str]:
    """State which measured area each density or area reading was divided by."""
    if params["kind"] != "pores" or params["mode"] not in {"density", "area"}:
        return []
    real = [entry for entry in records if entry["run_id"] and entry["measured_area"] is not None]
    if not real:
        return []
    shown = {(entry["measured_area"], entry["unit"]) for entry in real}
    if len(shown) == 1:
        area, unit = shown.pop()
        text = f"{area:,.2f} mm²" if unit == "mm" else f"{area:,.0f} px²"
        return [f"Every reading uses the same measurable area: {text} of supported pixels inside each face's drawn region."]
    return [
        "Faces use different measurable areas. Compare each density against its own face area rather than summing across the lot.",
    ]


def build_survey(db: Database, raw: Any) -> Survey:
    params = chosen_metric(raw)
    source = first_runs(db, params)
    warnings = []
    if len(source) > MAX_LOT_ROWS:
        warnings.append(f"Only the first {MAX_LOT_ROWS} inspected faces in this lot are displayed.")
    entries = [screening_entry(row, params) for row in source[:MAX_LOT_ROWS]]
    context = {
        **params,
        "statuses": dict(Counter(row["status"] for row in entries)),
        "summary": summarize_numeric(entries, params),
        "faces": len(entries),
    }
    return Survey(entries, context, warnings + audit_groups(entries) + measured_area_notice(entries, params))


def payload(value: Survey) -> dict:
    return {"context": value.context, "items": value.records, "notices": value.warnings}


def to_csv(value: Survey) -> bytes:
    return render_csv(
        [
            "lot", "inspection", "wood_species", "surface", "capture", "measurement",
            "snapshot_revision", "current_revision", "value", "unit", "status", "synthetic",
            "measurable_area", "measurable_area_unit", "density_basis",
        ],
        [
            [
                value.context["lot"], item["title"], item["material"], item["surface"],
                item["capture_id"], item["run_id"], item["revision"], item["active_revision"],
                item.get("recorded_value"), item["unit"], item["status"], item["synthetic"],
                item.get("measured_area"), item["unit"] if item.get("measured_area") is not None else "",
                item.get("density_basis"),
            ]
            for item in value.records
        ],
    )


def clipped_text(pen: ImageDraw.ImageDraw, line: str, length: int) -> str:
    if pen.textlength(line) <= length:
        return line
    while len(line) > 2 and pen.textlength(line + "…") > length:
        line = line[:-1]
    return line + "…"


def thumbnail_for(db: Database, record: dict, size: tuple[int, int]) -> Image.Image:
    if record["run_id"]:
        content = db.run_attachment(record["inspection_id"], record["run_id"], "preview")
    else:
        content, _ = db.capture_blob(record["capture_id"], "thumb")
    with Image.open(BytesIO(content)) as image:
        return ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)


def contact_sheet(db: Database, survey: Survey) -> bytes:
    columns = 4
    card = (276, 205)
    border = 14
    chosen = survey.records[:24]
    rows = max(1, (len(chosen) + columns - 1) // columns)
    canvas = Image.new(
        "RGB",
        (columns * (card[0] + border) + border, 60 + rows * (card[1] + border)),
        (237, 235, 223),
    )
    pen = ImageDraw.Draw(canvas)
    label = "Endgrain Lab / Lot " + survey.context["lot"] + " / " + survey.context["metric_title"]
    pen.text((border, 13), clipped_text(pen, label, canvas.width - 30), fill=(55, 66, 48))
    legend = "Overlay where measured, source image where missing."
    if len(chosen) < len(survey.records):
        legend += " First 24 faces."
    pen.text((border, 31), legend, fill=(90, 96, 75))
    for index, record in enumerate(chosen):
        x = border + (index % columns) * (card[0] + border)
        y = 51 + (index // columns) * (card[1] + border)
        artwork = thumbnail_for(db, record, (card[0] - 12, card[1] - 51))
        if record["status"] in {"above", "below", "incompatible_scale"}:
            ink = (157, 74, 51)
        elif record["status"] in {"missing", "undefined"}:
            ink = (111, 116, 105)
        else:
            ink = (68, 91, 58)
        pen.rectangle((x, y, x + card[0], y + card[1]), fill=(252, 250, 239), outline=ink, width=2)
        canvas.paste(artwork, (x + (card[0] - artwork.width) // 2, y + 6 + (card[1] - 51 - artwork.height) // 2))
        title = ("[Synthetic] " if record["synthetic"] else "") + record["title"]
        pen.text((x + 10, y + card[1] - 39), clipped_text(pen, title, card[0] - 20), fill=(46, 58, 41))
        state = record["status"].replace("_", " ").capitalize()
        note = record["display"] + " / " + state
        pen.text((x + 10, y + card[1] - 22), clipped_text(pen, note, card[0] - 20), fill=ink)
    return encode_png(np.asarray(canvas))
