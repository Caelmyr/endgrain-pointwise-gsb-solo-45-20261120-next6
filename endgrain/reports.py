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
        item["links"] = {"html": base + "/print", "zip": base + "/bundle.zip", "manifest": base, "plate": base + "/scope.png"}
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


def provenance_rows(inspection: dict) -> str:
    join = inspection["state"].get("join")
    if not join:
        return ""
    sources = join.get("sources", [])
    labels = ["First frame", "Second frame"]
    pieces = []
    for index, source in enumerate(sources[:2]):
        pieces.append(
            "<dt>" + labels[index] + "</dt><dd>" + escape(source.get("title") or source["capture_id"])
            + " <code>" + escape(source["capture_id"][:12]) + "</code></dd>"
        )
    offset = join.get("offset") or [0, 0]
    pieces.append("<dt>Second frame offset</dt><dd>" + escape("x " + str(offset[0]) + ", y " + str(offset[1])) + "</dd>")
    pieces.append("<dt>Alignment basis</dt><dd>" + escape(
        "Reviewed and confirmed after competing matches" if join.get("ambiguous_at_save")
        else "Single strong automatic match") + "</dd>")
    pieces.append("<dt>Seam source</dt><dd>Per-pixel single-frame choice; overlap pixels are not averaged</dd>")
    fraction = join.get("uncertain_fraction")
    if fraction is not None:
        pieces.append("<dt>Unclear seam</dt><dd>" + escape(f"{fraction * 100:.1f}% of the overlap") + "</dd>")
    return (
        '<section><h2>Joined capture</h2><dl>' + "".join(pieces) + '</dl>'
        '<p>The joined working image was formed from two overlapping photographs at one magnification. '
        'Each overlap pixel comes from exactly one frame, so a sharp vessel edge in one frame is not '
        'replaced by an averaged ghost. The seam source map in this evidence bundle shows which frame '
        'supplied each area.</p></section>'
    )


def metadata_rows(inspection: dict) -> str:
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
        "Capture type": "Joined working image" if inspection["state"].get("join")
        else "Synthetic demonstration image" if capture["synthetic"] else "Operator-supplied",
    }
    return "".join("<dt>" + escape(name) + "</dt><dd>" + escape(value or "—") + "</dd>" for name, value in details.items())


def measurement_section(record: dict, image_url: str, csv_url: str) -> str:
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
        '<table>' + "".join(lines) + '</table>'
        '<ul>' + notices + '</ul>'
        '<p><a href="' + escape(csv_url, quote=True) + '">Download measurement table</a></p>'
        '<details><summary>Analysis settings</summary><ul>' + "".join(params) + '</ul></details>'
        '</section>'
    )


def marker_section(entries: list[dict]) -> str:
    if not entries:
        return ""
    rows = []
    for marker in entries:
        x, y = [format_number(value, 1) for value in marker["point"]]
        rows.append(
            "<tr><th>" + escape(marker["label"]) + "</th><td>" + escape(x + ", " + y) + "</td>"
            "<td>" + escape(marker["comment"]) + "</td></tr>"
        )
    return (
        '<section><h2>Inspection markers</h2><table><thead>'
        '<tr><th>Marker</th><th>Measuring coordinates (px)</th><th>Observation</th></tr>'
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
a { color: #4b6049; }
@media print {
  :root { background: white; }
  body { padding: 0; margin: 0; max-width: none; }
  header, figure, tr { break-inside: avoid; }
  .measurement { break-before: page; }
}
"""


def render_print(report: dict, *, offline: bool = False) -> str:
    evidence = report["manifest"]
    inspection = evidence["inspection"]
    review = evidence["review"]
    root = "/api/inspections/" + inspection["id"]
    sections = []
    for seq, run in enumerate(evidence["measurements"], 1):
        name = f"measurements/{seq:02d}-{run['kind']}"
        endpoint = root + "/runs/" + run["id"]
        image_url = name + ".png" if offline else endpoint + "/preview.png"
        csv_url = name + ".csv" if offline else endpoint + "/table.csv"
        sections.append(measurement_section(run, image_url, csv_url))
    scope_url = "scope.png" if offline else root + "/reports/" + report["id"] + "/scope.png"
    scope = '<figure><img src="' + escape(scope_url, quote=True) + '" alt="Source and measuring support with calibration and markers"></figure>'
    disposition = DISPOSITIONS[review["disposition"]]
    notices = (
        '<p class="summary">' + escape(review["summary"]) + '</p>'
        if review["summary"] else ""
    )
    stamp = evidence["created_at"].replace("T", " ")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>' + escape(review["title"]) + '</title><style>' + PRINT_STYLES + '</style></head><body>'
        '<header><div class="brand">Endgrain Lab · Inspection record</div>'
        '<h1>' + escape(review["title"]) + '</h1>'
        '<p>' + escape(disposition) + ' &nbsp;·&nbsp; ' + escape(review["reviewer"]) + ' &nbsp;·&nbsp; ' + escape(stamp) + '</p>'
        '</header><section><h2>Scope</h2><dl>' + metadata_rows(inspection) + '</dl>' + notices + scope + '</section>'
        + provenance_rows(inspection)
        + "".join(sections) + marker_section(evidence["markers"])
        + '<footer><p>Original SHA-256 <code>' + escape(inspection["source_digest"]) + '</code></p>'
        + '<p>Manifest SHA-256 <code>' + escape(report["digest"]) + '</code></p></footer></body></html>'
    )
