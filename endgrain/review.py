from hashlib import sha256
from io import BytesIO
import json
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError

from .database import Database
from .errors import Limits, WorkbenchError
from .exchange import zput
from .geometry import mask_bounds
from .imaging import encode_png
from .plates import external_plate, external_working_panel, working_anchor_points
from .reports import export_run_csv, render_print
from .validation import document, fields, integer, list_of, string
from .workspaces import frame_for

REVIEW_VERSION = "endgrain-review/1"
REVIEW_MANIFEST = "review.json"
REVIEW_CHECKSUM = "review.sha256"
REVIEW_MAX_BYTES = 138 * 1024 * 1024
REVIEW_MAX_JSON = 512 * 1024
REVIEW_MAX_MEMBERS = 40
MARKER_LIMIT = 400
REGION_ONLY = (
    "Every picture is limited to the selected measuring region and to source pixels fully opaque there. "
    "Areas outside that region, and every transparent pixel, hold no original colour values."
)


def digest_of(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def packaged_cutout(raster) -> bytes:
    keep = raster.support
    colour = raster.rgb.copy()
    alpha = np.where(keep, 255, 0).astype(np.uint8)
    premultiplied = np.rint(
        colour.astype(np.float32) * (alpha.astype(np.float32) / 255.0)[..., None]
    ).astype(np.uint8)
    premultiplied[~keep] = 0
    return encode_png(np.dstack((premultiplied, alpha)))


def masked_overlay(preview: bytes, support: np.ndarray) -> bytes:
    try:
        with Image.open(BytesIO(preview)) as source:
            rgb = np.asarray(source.convert("RGB"))
    except (UnidentifiedImageError, OSError):
        raise WorkbenchError("A saved measurement overlay could not be read.", "report") from None
    if rgb.shape[:2] != support.shape:
        raise WorkbenchError("A saved measurement overlay does not match the measuring plane.", "report")
    keep = support
    alpha = np.where(keep, 255, 0).astype(np.uint8)
    premultiplied = np.rint(
        rgb.astype(np.float32) * (alpha.astype(np.float32) / 255.0)[..., None]
    ).astype(np.uint8)
    premultiplied[~keep] = 0
    return encode_png(np.dstack((premultiplied, alpha)))


def support_statement(frame) -> dict:
    raster = frame.raster
    plane = raster.plane
    bounds = mask_bounds(raster.support)
    measured = int(np.count_nonzero(raster.support))
    region_pixels = int(np.count_nonzero(raster.inspected))
    region_supported = int(np.count_nonzero(raster.supported & raster.inspected))
    return {
        "unit": plane.unit,
        "measuring_width": plane.width,
        "measuring_height": plane.height,
        "mm_per_measuring_x": plane.mm_x,
        "mm_per_measuring_y": plane.mm_y,
        "region_polygon": frame.inspection["state"]["region"],
        "measured_bounds": list(bounds) if bounds else None,
        "measured_pixels": measured,
        "region_pixels": region_pixels,
        "supported_region_pixels": region_supported,
        "measured_area": measured * plane.pixel_area,
        "region_area": region_pixels * plane.pixel_area,
        "region_fully_supported": measured == region_supported == region_pixels,
        "calibration": plane.calibration,
    }


def anchor_positions(frame) -> list[dict]:
    points, _ = working_anchor_points(frame)
    names = {
        2: ("Ruler start", "Ruler end"),
        4: ("Rectangle corner 1", "Rectangle corner 2", "Rectangle corner 3", "Rectangle corner 4"),
    }.get(len(points), tuple(f"Calibration anchor {index}" for index in range(1, len(points) + 1)))
    return [{"name": name, "point": point} for name, point in zip(names, points, strict=True)]


def reference_points(run: dict) -> list[dict]:
    params = run["params"]
    if run["kind"] == "rings":
        return [
            {"name": f"Transect endpoint {index}", "point": point}
            for index, point in enumerate(params.get("endpoints", []), 1)
        ]
    if run["kind"] == "colors":
        return [{"name": "Swatch " + item["label"], "point": item["point"]} for item in params.get("swatches", [])]
    if run["kind"] == "edge_loss":
        return [
            {"name": f"Expected stock corner {index}", "point": point}
            for index, point in enumerate(params.get("corners", []), 1)
        ]
    if run["kind"] == "footprint" and params.get("seed"):
        return [{"name": "Footprint selection point", "point": params["seed"]}]
    return []


def pictured(frame, point) -> bool:
    support = frame.raster.support
    height, width = support.shape
    x, y = point
    if not (-0.5 <= x <= width - 0.5 and -0.5 <= y <= height - 0.5):
        return False
    column = min(width - 1, max(0, int(round(x))))
    row = min(height - 1, max(0, int(round(y))))
    return bool(support[row, column])


def table_of(db: Database, run: dict) -> dict:
    return db.run_attachment(run["inspection_id"], run["id"], "table")


def table_coordinates(table: dict) -> list[list[float]]:
    columns = table.get("columns") or []
    pairs = (
        ("x_px", "y_px"), ("x", "y"), ("x0", "y0"),
        ("expected_x_px", "expected_y_px"), ("observed_x_px", "observed_y_px"),
    )
    result = []
    for xs, ys in pairs:
        if xs not in columns or ys not in columns:
            continue
        ix, iy = columns.index(xs), columns.index(ys)
        for row in table.get("rows") or []:
            try:
                point = [float(row[ix]), float(row[iy])]
            except (IndexError, TypeError, ValueError):
                continue
            if point not in result:
                result.append(point)
    return result


def entitlement(run: dict, table: dict, frame) -> dict:
    reasons = []
    hidden = [item["name"] for item in reference_points(run) if not pictured(frame, item["point"])]
    if hidden:
        reasons.append(
            "Positions outside the delivered region are withheld, so these references cannot be re-placed: "
            + ", ".join(hidden) + "."
        )
    outside_rows = sum(1 for point in table_coordinates(table) if not pictured(frame, point))
    if outside_rows:
        reasons.append(
            f"{outside_rows} reading row(s) locate material outside the delivered region and are not pictured."
        )
    if run["snapshot"].get("plane") != frame.raster.plane.json():
        reasons.append("The saved measurement plane differs from this report's measuring plane.")
    if run["snapshot"].get("state") != frame.inspection["state"]:
        reasons.append("The saved measuring state differs from this report's state.")
    return {
        "reasons": reasons,
        "reference_points_outside": hidden,
        "rows_outside_package": outside_rows,
        "can_remeasure": not reasons,
    }


def marker_entries(manifest: dict, frame) -> tuple[list[dict], int]:
    entries = []
    outside = 0
    for marker in manifest["markers"][:MARKER_LIMIT]:
        visible = pictured(frame, marker["point"])
        outside += 0 if visible else 1
        entries.append({
            "label": marker["label"],
            "comment": marker["comment"],
            "point": marker["point"],
            "in_package": visible,
        })
    return entries, outside


def limitations(frame, runs: list[dict]) -> dict:
    lighting = frame.inspection["state"]["lighting"]
    hidden = [item["name"] for item in anchor_positions(frame) if not pictured(frame, item["point"])]
    bounds = mask_bounds(frame.raster.support)
    if bounds is None:
        anchor_note = "The delivered region has no supported pixels."
    elif hidden:
        anchor_note = (
            "Calibration anchors outside the delivered region are withheld: " + ", ".join(hidden)
            + ". Their coordinates stay in the review record but cannot be checked against the pictures."
        )
    else:
        anchor_note = "Every calibration anchor lies inside the delivered region."
    return {
        "illumination": (
            "The illumination reference picture is withheld. Delivered pictures show the corrected measuring "
            "plane, so values can be re-read and checked, but the brightness correction cannot be replayed."
            if lighting else
            "No illumination correction was applied to this report."
        ),
        "outside_region": (
            "Material or equipment outside the selected region is withheld, so its appearance cannot be checked here."
        ),
        "calibration_anchors": anchor_note,
        "remeasurable": sorted({run["kind"] for run in runs if run["can_remeasure"]}),
        "compare_only": sorted({run["kind"] for run in runs if not run["can_remeasure"]}),
    }


def build(db: Database, inspection_id: str, report_id: str) -> dict:
    report = db.report(inspection_id, report_id)
    manifest = report["manifest"]
    inspection = manifest["inspection"]
    if int(report["revision"]) != int(inspection["revision"]):
        raise WorkbenchError("The issued report does not match its own captured revision.", "report")
    frame = frame_for(db, inspection)
    if not frame.raster.support.any():
        raise WorkbenchError("The reported region has no supported source pixels to package.", "report")
    plane_cut = packaged_cutout(frame.raster)
    previews = {}
    runs = []
    for index, run in enumerate(manifest["measurements"], 1):
        table = table_of(db, run)
        item = entitlement(run, table, frame)
        stem = f"measurements/{index:02d}-{run['kind']}"
        previews[stem + ".png"] = masked_overlay(
            db.run_attachment(inspection["id"], run["id"], "preview"), frame.raster.support,
        )
        item.update({
            "sequence": index,
            "run_id": run["id"],
            "kind": run["kind"],
            "revision": run["revision"],
            "stem": stem,
            "unit": run["result"].get("unit", frame.raster.plane.unit),
            "readings": len(table.get("rows") or []),
            "verification": "remeasure" if item["can_remeasure"] else "compare",
        })
        runs.append(item)
    markers, outside_markers = marker_entries(manifest, frame)
    scope = external_working_panel(frame, manifest["markers"], (637, 471))
    document = {
        "format": REVIEW_VERSION,
        "package_kind": "external-review",
        "record": {
            "report_id": report["id"],
            "inspection_id": inspection["id"],
            "revision": report["revision"],
            "issued_at": manifest["created_at"],
            "manifest_sha256": report["digest"],
            "source_sha256": inspection["source_digest"],
            "measurement_runs": [run["id"] for run in manifest["measurements"]],
        },
        "report": {
            "title": manifest["review"]["title"],
            "reviewer": manifest["review"]["reviewer"],
            "disposition": manifest["review"]["disposition"],
            "summary": manifest["review"]["summary"],
        },
        "inspection": {
            key: inspection[key] for key in ("title", "material", "lot", "surface")
        } | {
            "source_digest": inspection["source_digest"],
            "source_filename": inspection["capture_metadata"]["filename"],
            "synthetic": inspection["capture_metadata"]["synthetic"],
        },
        "plane": frame.raster.plane.json(),
        "support": support_statement(frame),
        "measurements": runs,
        "markers": markers,
        "markers_outside_package": outside_markers,
        "limitations": limitations(frame, runs),
        "disclosure": [REGION_ONLY] + (
            []
            if frame.inspection["state"]["region"]
            else ["This report had no region polygon, so the whole measuring frame is the delivered region."]
        ),
        "files": {},
    }
    return {
        "report": report,
        "plane_cut": plane_cut,
        "plate": external_plate(
            plane_cut, scope, "REGION CUTOUT / TRANSPARENCY HOLDS NO COLOUR",
        ),
        "previews": previews,
        "document": document,
    }


def members_for(db: Database, payload: dict) -> dict[str, bytes]:
    report = payload["report"]
    inspection = report["manifest"]["inspection"]
    output = {
        "report.html": render_print(report, offline=True, review=payload["document"]).encode("utf-8"),
        "scope.png": payload["plate"],
        "measuring-plane.png": payload["plane_cut"],
    }
    output.update(payload["previews"])
    for item in payload["document"]["measurements"]:
        output[item["stem"] + ".csv"] = export_run_csv(db, inspection["id"], item["run_id"])
    return output


def seal(output: dict[str, bytes], document: dict) -> dict[str, bytes]:
    declared = {
        name: {"size": len(payload), "sha256": digest_of(payload)}
        for name, payload in sorted(output.items())
    }
    total = sum(item["size"] for item in declared.values())
    if total > REVIEW_MAX_BYTES:
        raise Limits("This review package is too large. Issue a report with fewer measurements.", "file")
    document["files"] = declared
    body = json.dumps(
        document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    if len(body) > REVIEW_MAX_JSON:
        raise Limits("The review record is too large for an external review package.", "file")
    sealed = dict(output)
    sealed[REVIEW_MANIFEST] = body
    sealed[REVIEW_CHECKSUM] = (digest_of(body) + "  " + REVIEW_MANIFEST + "\n").encode("ascii")
    return sealed


def package(db: Database, inspection_id: str, report_id: str) -> dict:
    payload = build(db, inspection_id, report_id)
    sealed = seal(members_for(db, payload), payload["document"])
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zipped:
        for name in sorted(sealed):
            zput(zipped, name, sealed[name])
    return {"name": f"endgrain-review-{report_id[:10]}.zip", "bytes": buffer.getvalue()}


def read_entry(zipped: ZipFile, inventory: dict, name: str, budget: int) -> bytes:
    if name not in inventory:
        raise WorkbenchError(f"The review package is missing {name}.", "file")
    item = inventory[name]
    if item.file_size < 1 or item.file_size > budget:
        raise Limits(f"{name} is empty or larger than the review package limit.", "file")
    with zipped.open(item) as raw:
        payload = raw.read(budget + 1)
    if len(payload) != item.file_size or len(payload) > budget:
        raise Limits(f"{name} exceeds its recorded size.", "file")
    return payload


def entry_digest(zipped: ZipFile, item) -> str:
    running = sha256()
    count = 0
    with zipped.open(item) as raw:
        while chunk := raw.read(128 * 1024):
            count += len(chunk)
            if count > item.file_size:
                raise WorkbenchError("A review file exceeds its declared size.", "file")
            running.update(chunk)
    if count != item.file_size:
        raise WorkbenchError("A review file did not expand to the expected size.", "file")
    return running.hexdigest()


def image_dimensions(payload: bytes) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(payload)) as source:
            source.verify()
        with Image.open(BytesIO(payload)) as source:
            return source.size
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise WorkbenchError("A review picture could not be read as an image.", "file") from None


def plane_image_issues(
    payload: bytes, bounds: list[int], plane_size: tuple[int, int], region_only: bool,
) -> list[str]:
    try:
        with Image.open(BytesIO(payload)) as source:
            source.verify()
        with Image.open(BytesIO(payload)) as source:
            rgba = np.asarray(source.convert("RGBA"))
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        return ["could not be read as an image."]
    alpha = rgba[..., 3]
    issues = []
    if region_only:
        clear = alpha == 0
        if not clear.any():
            issues.append(
                "shows the whole capture instead of just the selected region."
                if np.all(alpha == 255) else
                "has no empty area to show that material outside the region was withheld."
            )
            return issues
        if np.any(rgba[..., :3][clear] != 0):
            issues.append("keeps colour values behind transparent pixels.")
        if not np.all((alpha == 0) | (alpha == 255)):
            issues.append("uses partial transparency, which can blend withheld colours.")
        rows = np.flatnonzero(alpha.any(axis=1))
        cols = np.flatnonzero(alpha.any(axis=0))
        within = bool(len(rows) and len(cols)) and (
            bounds[0] <= cols[0] and cols[-1] + 1 <= bounds[2]
            and bounds[1] <= rows[0] and rows[-1] + 1 <= bounds[3]
        )
        if not within:
            issues.append("shows pixels outside the recorded measured extent.")
    elif (rgba.shape[1], rgba.shape[0]) != plane_size:
        issues.append("does not match the recorded measuring plane size.")
    return issues


def read_record(payload: bytes) -> dict:
    try:
        body = document(json.loads(payload.decode("utf-8")), "review record")
    except (ValueError, UnicodeError, RecursionError):
        raise WorkbenchError("The review record is unreadable.", "file") from None
    fields(body, {
        "format", "package_kind", "record", "report", "inspection", "plane", "support",
        "measurements", "markers", "markers_outside_package", "limitations", "disclosure", "files",
    }, "review record")
    if body.get("format") != REVIEW_VERSION or body.get("package_kind") != "external-review":
        raise WorkbenchError("Choose an Endgrain Lab external review package.", "file")
    record = document(body.get("record"), "review record")
    fields(record, {
        "report_id", "inspection_id", "revision", "issued_at", "manifest_sha256",
        "source_sha256", "measurement_runs",
    }, "review record")
    string(record.get("manifest_sha256"), "issued manifest checksum", 64, required=True)
    string(record.get("source_sha256"), "issued source checksum", 64, required=True)
    integer(record.get("revision"), "review revision", 1, 2**31 - 1)
    report = document(body.get("report"), "review details")
    fields(report, {"title", "reviewer", "disposition", "summary"}, "review details")
    title = string(report.get("title"), "report title", 120, required=True)
    runs = list_of(body.get("measurements"), "reviewed measurements", 1, 12)
    for index, item in enumerate(runs, 1):
        entry = document(item, "reviewed measurement")
        expected = f"measurements/{index:02d}-"
        stem = string(entry.get("stem"), "measurement stem", 60, required=True)
        fields(entry, {
            "sequence", "run_id", "kind", "revision", "stem", "unit", "readings",
            "verification", "can_remeasure", "reasons", "reference_points_outside",
            "rows_outside_package",
        }, "reviewed measurement")
        if stem != expected + entry.get("kind", "") or entry.get("sequence") != index:
            raise WorkbenchError("The review record lists measurements out of order.", "file")
    markers = list_of(body.get("markers"), "reviewed markers", 0, MARKER_LIMIT)
    declared = document(body.get("files"), "review file checksums")
    return {
        "record": record,
        "title": title,
        "report": report,
        "measurements": runs,
        "markers": markers,
        "limitations": document(body.get("limitations"), "review limitations"),
        "support": document(body.get("support"), "review support"),
        "files": declared,
    }


def inspect(zipped: ZipFile, inventory: dict) -> dict:
    names = set(inventory)
    if REVIEW_MANIFEST not in names:
        raise WorkbenchError("This is not an external review package.", "file")
    forbidden = ("original.", "reference.", "capture.", "manifest.")
    if any(name.startswith(forbidden) for name in names):
        raise WorkbenchError("A review package must not carry the original capture or manifest.", "file")
    body = read_entry(zipped, inventory, REVIEW_MANIFEST, REVIEW_MAX_JSON)
    checksum = read_entry(zipped, inventory, REVIEW_CHECKSUM, 256)
    if checksum != (digest_of(body) + "  " + REVIEW_MANIFEST + "\n").encode("ascii"):
        raise WorkbenchError("The review record does not match its declared checksum.", "file")
    parsed = read_record(body)
    declared = parsed["files"]
    if set(declared) | {REVIEW_MANIFEST, REVIEW_CHECKSUM} != names:
        raise WorkbenchError("The review file set differs from the record.", "file")
    required = {"report.html", "scope.png", "measuring-plane.png"}
    if not required <= set(declared) or REVIEW_MANIFEST in declared:
        raise WorkbenchError("The review package is missing a required report file.", "file")
    plane_size = (
        int(parsed["support"].get("measuring_width", 0)),
        int(parsed["support"].get("measuring_height", 0)),
    )
    bounds = parsed["support"].get("measured_bounds")
    if not bounds or plane_size[0] < 2 or plane_size[1] < 2:
        raise WorkbenchError("The review record has no measured extent.", "file")
    wanted = set(required)
    for item in parsed["measurements"]:
        if item.get("verification") not in {"remeasure", "compare"}:
            raise WorkbenchError("The review record has an unknown verification state.", "file")
        wanted.update({item["stem"] + ".png", item["stem"] + ".csv"})
    if wanted != set(declared):
        raise WorkbenchError("The review record does not describe the expected file set.", "file")
    banner = []
    checks = []
    region_only = bool(parsed["support"].get("region_polygon"))
    for name in sorted(declared):
        declared_entry = document(declared[name], "review file checksum")
        size = declared_entry.get("size")
        expected = declared_entry.get("sha256")
        if not isinstance(size, int) or not isinstance(expected, str) or len(expected) != 64:
            raise WorkbenchError("A review file has an invalid checksum entry.", "file")
        item = inventory[name]
        actual = entry_digest(zipped, item)
        same = actual == expected and item.file_size == size
        checks.append({"name": name, "size": item.file_size, "verified": same})
        if not name.endswith(".png"):
            continue
        payload = read_entry(zipped, inventory, name, size)
        if name == "scope.png":
            try:
                image_dimensions(payload)
            except WorkbenchError:
                banner.append("scope.png could not be read as an image.")
            continue
        issues = plane_image_issues(
            payload, bounds, plane_size, region_only and name != "scope.png",
        )
        banner.extend(name + " " + issue for issue in issues)
    if not any(item["verification"] == "remeasure" for item in parsed["measurements"]):
        banner.append("No measurement can be re-measured from this package; use the issued evidence ZIP.")
    return {
        "package_kind": "external-review",
        "report_id": parsed["record"]["report_id"],
        "report_title": parsed["title"],
        "revision": parsed["record"]["revision"],
        "manifest_sha256": parsed["record"]["manifest_sha256"],
        "source_sha256": parsed["record"].get("source_sha256"),
        "measurement_count": len(parsed["measurements"]),
        "markers_outside_package": parsed.get("markers_outside_package", 0),
        "remeasurable": parsed["limitations"].get("remeasurable", []),
        "compare_only": parsed["limitations"].get("compare_only", []),
        "verified": all(item["verified"] for item in checks) and not banner,
        "files": checks,
        "warnings": banner,
        "notice": (
            "Checksums confirm agreement within this package. The record names the issued manifest and source "
            "checksums; compare them with the internal evidence ZIP to confirm both describe the same issue."
        ),
    }


def looks_like_review(data: bytes) -> bool:
    try:
        with ZipFile(BytesIO(data)) as zipped:
            return REVIEW_MANIFEST in set(zipped.namelist())
    except (BadZipFile, RuntimeError, NotImplementedError):
        return False


def inspect_zip(data: bytes) -> dict:
    if len(data) > REVIEW_MAX_BYTES:
        raise Limits("Choose a review package no larger than 138 MiB.", "file")
    try:
        with ZipFile(BytesIO(data)) as zipped:
            inventory = {}
            accumulated = 0
            for item in zipped.infolist():
                if (
                    item.is_dir() or item.filename.startswith("/") or ".." in item.filename
                    or "\\" in item.filename or item.flag_bits & 1 or item.filename in inventory
                ):
                    raise WorkbenchError("The review package has an invalid entry.", "file")
                accumulated += item.file_size
                if accumulated > REVIEW_MAX_BYTES or len(inventory) >= REVIEW_MAX_MEMBERS:
                    raise Limits("The review package expands beyond the allowed size.", "file")
                inventory[item.filename] = item
            return inspect(zipped, inventory)
    except (BadZipFile, RuntimeError, NotImplementedError):
        raise WorkbenchError("The review package is damaged or uses unsupported compression.", "file") from None
