from collections.abc import Iterable
from html import escape
from io import StringIO
from math import isfinite
from typing import Any
import csv

from .comparison import Pair
from .database import Database
from .errors import WorkbenchError
from .measurements import TOOLS
from .validation import choice, document, expect_revision, fields, list_of, string

DISPOSITIONS = {"recorded": "Recorded", "hold": "Hold for review", "recapture": "Recapture required"}


def form_values(raw: Any) -> tuple[int, dict, list[str]]:
    data = document(raw, "report")
    fields(data, {"revision", "title", "reviewer", "disposition", "summary", "run_ids"}, "report")
    revision = expect_revision(data)
    details = {
        "title": string(data.get("title"), "report title", 120, required=True),
        "reviewer": string(data.get("reviewer"), "reviewer", 80, required=True),
        "disposition": choice(data.get("disposition"), "disposition", tuple(DISPOSITIONS)),
        "summary": string(data.get("summary"), "review summary", 3000, multiline=True),
    }
    ids = [
        string(item, "measurement identifier", 60, required=True)
        for item in list_of(data.get("run_ids"), "measurements to issue", 1, 12)
    ]
    if len(set(ids)) != len(ids):
        raise WorkbenchError("A report cannot include the same measurement twice.", "run_ids")
    return revision, details, ids


def issue(db: Database, inspection_id: str, raw: Any) -> dict:
    revision, report, run_ids = form_values(raw)
    return describe(db.create_report(inspection_id, revision, report, run_ids))


def describe(report: dict) -> dict:
    item = dict(report)
    manifest = report.get("manifest")
    if manifest:
        inspection = manifest["inspection"]
        item.update(manifest["review"])
        item["inspection_id"] = inspection["id"]
        item["measurement_count"] = len(manifest["measurements"])
        item["marker_count"] = len(manifest["markers"])
    if "inspection_id" in item:
        base = "/api/inspections/" + item["inspection_id"] + "/reports/" + item["id"]
        item["links"] = {
            "html": base + "/print",
            "zip": base + "/bundle.zip",
            "manifest": base,
            "plate": base + "/scope.png",
            "review": base + "/review.zip",
        }
    return item


def format_number(item: float | int | None, max_decimals: int = 3) -> str:
    if item is None or not isfinite(item):
        return "—"
    if isinstance(item, int) or abs(item) >= 10000:
        return f"{item:,.0f}"
    return f"{item:,.{max_decimals}f}".rstrip("0").rstrip(".")


def render_metric(value, formatting: str, unit: str, source: dict) -> str:
    if value is None:
        return "—"
    if formatting == "bool":
        return "Yes" if value else "No"
    if not isinstance(value, (int, float)):
        return str(value)
    if formatting == "%":
        return format_number(value * 100, 1) + "%"
    if formatting == "area":
        return format_number(value, 2) + " " + unit + "²"
    if formatting == "linear":
        return format_number(value, 2) + " " + unit
    if formatting == "density":
        return format_number(value, 2) + " / " + source.get("density_basis", unit + "²")
    if formatting == "°":
        return format_number(value, 1) + "°"
    if formatting == "px":
        return format_number(value) + " px"
    return format_number(value, 3)


def csv_item(val: Any) -> Any:
    if val is None:
        return ""
    if isinstance(val, str) and val.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + val
    if isinstance(val, bool):
        return "true" if val else "false"
    return val


def render_csv(columns: Iterable[str], contents: Iterable[Iterable[Any]]) -> bytes:
    stream = StringIO()
    writer = csv.writer(stream, lineterminator="\r\n")
    writer.writerow(csv_item(val) for val in columns)
    writer.writerows([csv_item(val) for val in row] for row in contents)
    return stream.getvalue().encode("utf-8-sig")


def export_run_csv(db: Database, inspection_id: str, run_id: str) -> bytes:
    record = db.run_attachment(inspection_id, run_id, "table")
    return render_csv(record["columns"], record["rows"])


def compare_csv(pair: Pair) -> bytes:
    columns = ["metric", "description", "native_format", "run_a", "run_b", "difference"]
    rows = (
        [row["key"], row["title"], row["mode"], row["left"], row["right"], row["difference"]]
        for row in pair.metric_rows
    )
    return render_csv(columns, rows)


def metadata_rows(inspection: dict, review: dict | None = None) -> str:
    capture = inspection["capture_metadata"]
    details = {
        "Inspection": inspection["title"],
        "Lot": inspection["lot"],
        "Material": inspection["material"],
        "Surface": inspection["surface"],
        "Source": capture["filename"],
        "Oriented size": f"{capture['width']} × {capture['height']} px",
        "Plane": inspection["state"]["calibration"]["mode"],
        "Selected region": "Polygon" if inspection["state"]["region"] else "Whole frame",
        "Illumination adjustment": "Applied" if inspection["state"]["lighting"] else "Original",
        "Revision": str(inspection["revision"]),
        "Capture type": "Synthetic demonstration image" if capture["synthetic"] else "Operator-supplied",
    }
    if review:
        details["Delivered extent"] = "Selected region only"
        details["Source capture"] = "Withheld outside the selected region"
        details["Illumination reference"] = "Withheld from this copy"
    return "".join("<dt>" + escape(name) + "</dt><dd>" + escape(value or "—") + "</dd>" for name, value in details.items())


def measurement_section(record: dict, image_url: str, csv_url: str, allowance: dict | None = None) -> str:
    kind = TOOLS[record["kind"]]
    outcome = record["result"]
    metric_values = outcome["metrics"]
    lines = []
    for field, (label, mode) in kind.metrics.items():
        val = render_metric(metric_values.get(field), mode, outcome["unit"], metric_values)
        lines.append("<tr><th>" + escape(label) + "</th><td>" + escape(val) + "</td></tr>")
    params = [
        "<li><code>" + escape(key.replace("_", " ")) + "</code> " + escape(str(value)) + "</li>"
        for key, value in record["params"].items()
    ]
    notices = "".join("<li>" + escape(notice) + "</li>" for notice in outcome["notices"])
    return (
        '<section class="measurement">'
        '<h2>' + escape(kind.title) + '</h2>'
        '<p>' + escape(kind.description) + '</p>'
        '<figure><img src="' + escape(image_url, quote=True) + '" alt="' + escape(kind.title, quote=True) + ' overlay"></figure>'
        + allowance_html(allowance)
        + '<table>' + "".join(lines) + '</table>'
        '<ul>' + notices + '</ul>'
        '<p><a href="' + escape(csv_url, quote=True) + '">Download measurement table</a></p>'
        '<details><summary>Analysis settings</summary><ul>' + "".join(params) + '</ul></details>'
        '</section>'
    )


def allowance_html(allowance: dict | None) -> str:
    if allowance is None:
        return ""
    label = "Re-measurable from this package" if allowance["can_remeasure"] else "Compare against issued evidence only"
    lines = "".join("<li>" + escape(reason) + "</li>" for reason in allowance["reasons"])
    body = "<ul>" + lines + "</ul>" if lines else ""
    return (
        '<p class="allowance ' + ("ok" if allowance["can_remeasure"] else "limited") + '">'
        + escape(label) + '</p>' + body
    )


def marker_section(entries: list[dict], review: dict | None = None) -> str:
    if not entries:
        return ""
    allowed = {item["label"]: item for item in (review["markers"] if review else [])}
    rows = []
    for marker in entries:
        x, y = [format_number(value, 1) for value in marker["point"]]
        if review:
            entry = allowed.get(marker["label"], {})
            shown = "Included" if entry.get("in_package") else "Coordinates only — outside the delivered region"
        else:
            shown = "Included"
        rows.append(
            "<tr><th>" + escape(marker["label"]) + "</th><td>" + escape(x + ", " + y) + "</td>"
            "<td>" + escape(shown) + "</td>"
            "<td>" + escape(marker["comment"]) + "</td></tr>"
        )
    heading = "<th>Picture</th>" if review else ""
    note = (
        '<p class="review-note">Markers listed as coordinates only fall outside the delivered region and are not pictured.</p>'
        if review and any(not item["in_package"] for item in review["markers"]) else ""
    )
    return (
        '<section><h2>Inspection markers</h2>' + note + '<table><thead>'
        '<tr><th>Marker</th><th>Measuring coordinates (px)</th>' + heading + '<th>Observation</th></tr>'
        "</thead><tbody>" + "".join(rows) + "</tbody></table></section>"
    )


PRINT_STYLES = """
:root { color: #252820; background: #ebe9e0; font: 15px/1.6 system-ui, sans-serif; }
body { max-width: 830px; margin: 34px auto; padding: 48px; background: #fdfcf6; }
header { padding-bottom: 26px; border-bottom: 2px solid #b7bda7; }
h1,h2 { font-family: Georgia, serif; font-weight: 500; line-height: 1.12; }
h1 { margin: .3em 0 .5em; font-size: 2.8rem; }
h2 { font-size: 1.8rem; margin: 1.1em 0 .55em; }
p { margin: .45em 0 1em; }
.brand { text-transform: uppercase; letter-spacing: .14em; font-weight: 700; color: #59604b; }
section { padding: 16px 0; border-bottom: 1px solid #b7bda7; }
dl { display: grid; grid-template-columns: 145px minmax(0,1fr); }
dt { font-weight: 650; }
dd { margin: 0 0 .3em; }
table { width: 100%; border-collapse: collapse; }
th,td { text-align: left; padding: .45em .6em; border-bottom: 1px solid #cccabb; }
figure { margin: 0; }
figure img { display: block; max-width: 100%; max-height: 460px; margin: 0 auto 20px; }
code { overflow-wrap: anywhere; }
.summary { white-space: pre-wrap; }
.allowance { font-weight: 650; margin: .7em 0 .3em; }
.allowance.ok { color: #3f6448; }
.allowance.limited { color: #8a5a2c; }
.review-note { border-left: 3px solid #b7bda7; padding-left: 12px; color: #4c5142; }
a { color: #4b6049; }
@media print {
  :root { background: white; }
  body { padding: 0; margin: 0; max-width: none; }
  header, figure, tr { break-inside: avoid; }
  .measurement { break-before: page; }
}
"""


def render_print(report: dict, *, offline: bool = False, review: dict | None = None) -> str:
    evidence = report["manifest"]
    inspection = evidence["inspection"]
    review_details = evidence["review"]
    root = "/api/inspections/" + inspection["id"]
    allowances = {item["run_id"]: item for item in review["measurements"]} if review else {}
    sections = []
    for seq, run in enumerate(evidence["measurements"], 1):
        name = f"measurements/{seq:02d}-{run['kind']}"
        endpoint = root + "/runs/" + run["id"]
        image_url = name + ".png" if offline else endpoint + "/preview.png"
        csv_url = name + ".csv" if offline else endpoint + "/table.csv"
        sections.append(measurement_section(run, image_url, csv_url, allowances.get(run["id"])))
    scope_url = "scope.png" if offline else root + "/reports/" + report["id"] + "/scope.png"
    scope = '<figure><img src="' + escape(scope_url, quote=True) + '" alt="Source and measuring support with calibration and markers"></figure>'
    disposition = DISPOSITIONS[review_details["disposition"]]
    notices = (
        '<p class="summary">' + escape(review_details["summary"]) + '</p>'
        if review_details["summary"] else ""
    )
    stamp = evidence["created_at"].replace("T", " ")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>' + escape(review_details["title"]) + '</title><style>' + PRINT_STYLES + '</style></head><body>'
        '<header><div class="brand">Endgrain Lab · ' + ("External review copy" if review else "Inspection record") + '</div>'
        '<h1>' + escape(review_details["title"]) + '</h1>'
        '<p>' + escape(disposition) + ' &nbsp;·&nbsp; ' + escape(review_details["reviewer"]) + ' &nbsp;·&nbsp; ' + escape(stamp) + '</p>'
        '</header><section><h2>Scope</h2><dl>' + metadata_rows(inspection, review) + '</dl>' + notices + scope + '</section>'
        + review_section(review)
        + "".join(sections) + marker_section(evidence["markers"], review)
        + '<footer><p>' + ("Issued source SHA-256" if review else "Original SHA-256")
        + ' <code>' + escape(inspection["source_digest"]) + '</code></p>'
        + '<p>Manifest SHA-256 <code>' + escape(report["digest"]) + '</code></p></footer></body></html>'
    )


def review_section(review: dict | None) -> str:
    if not review:
        return ""
    support = review["support"]
    limits = review["limitations"]
    bounds = support["measured_bounds"]
    extent = (
        f"{bounds[0]}–{bounds[2]} × {bounds[1]}–{bounds[3]} measuring px"
        if bounds else "no supported pixels"
    )
    rows = [
        ("Report ID", review["record"]["report_id"]),
        ("Delivered extent", extent + " (selected region only)"),
        ("Withheld", "Everything outside the selected region; illumination reference and gain map"),
        ("Scale", f"{support['mm_per_measuring_x']:.4g} × {support['mm_per_measuring_y']:.4g} mm per measuring pixel"
         if support.get("mm_per_measuring_x") is not None else "Uncalibrated measuring pixels"),
        ("Illumination", limits["illumination"]),
        ("Calibration anchors", limits["calibration_anchors"]),
        ("Outside the region", limits["outside_region"]),
        ("Re-measurable tools", ", ".join(limits["remeasurable"]) or "None"),
        ("Compare-only tools", ", ".join(limits["compare_only"]) or "None"),
    ]
    body = "".join("<dt>" + escape(name) + "</dt><dd>" + escape(value) + "</dd>" for name, value in rows)
    return (
        '<section class="review-note"><h2>External review scope</h2>'
        '<p>' + escape(review["disclosure"][0]) + '</p>'
        + "".join('<p class="review-note">' + escape(line) + '</p>' for line in review["disclosure"][1:])
        + '<dl>' + body + '</dl>'
        "<p>Every picture excludes the source capture and the illumination reference. "
        "Transparent areas are empty: they hold no original colour values.</p></section>"
    )
