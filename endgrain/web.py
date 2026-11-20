from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge, UnsupportedMediaType

from . import calibration_preview, comparison, exchange, lots, measurements, plates, probe, recipes, reports, review, workspaces
from .captures import (
    SURFACES, image_etag, ingest_inspection, ingest_reference, select_references,
)
from .database import Database
from .demo import demo_payloads
from .errors import Limits, WorkbenchError
from .imaging import MAX_UPLOAD_BYTES, encode_png
from .quality import prepare_download_image
from .validation import choice, document, expect_revision, fields, integer, string

STATIC = Path(__file__).resolve().parents[1] / "static"


def read_document() -> dict:
    try:
        body = request.get_json()
    except (BadRequest, UnsupportedMediaType):
        raise WorkbenchError("Send a JSON request body.", "request") from None
    return document(body)


def file_attachment(max_size: int, label: str):
    entry = request.files.get("file")
    if entry is None or not entry.filename:
        raise WorkbenchError(f"Choose {label} to upload.", "file")
    contents = entry.stream.read(max_size + 1)
    if len(contents) > max_size:
        raise Limits(f"The selected upload is larger than {max_size // (1024 * 1024)} MiB.", "file")
    return contents, entry.filename


def binary(
    raw: bytes, mime: str, *, name: str | None = None, etag: str | None = None, immutable: bool = False,
) -> Response:
    response = send_file(
        BytesIO(raw),
        mimetype=mime,
        download_name=name,
        as_attachment=bool(name),
        max_age=86400 if immutable else 0,
        etag=etag,
        conditional=etag is not None,
    )
    return response


def build_folder(path: str | None) -> Path:
    return Path(path or os.environ.get("ENDGRAIN_DATA") or ".endgrain").expanduser()


def response_report(report: dict, inspection_id: str) -> dict:
    return reports.describe({"inspection_id": inspection_id, **report})


def create_app(data_folder: str | None = None) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/assets")
    app.config.update(MAX_CONTENT_LENGTH=140 * 1024 * 1024, MAX_FORM_MEMORY_SIZE=380_000, MAX_FORM_PARTS=24)
    app.json.sort_keys = False
    app.json.ensure_ascii = False
    db = Database(build_folder(data_folder))
    app.extensions["endgrain_db"] = db

    @app.after_request
    def add_headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "same-origin"
        if request.path.startswith("/api/"):
            resp.headers.setdefault("Cache-Control", "no-cache")
        return resp

    @app.errorhandler(WorkbenchError)
    def domain_failure(error: WorkbenchError):
        return jsonify(error.payload()), error.status

    @app.errorhandler(RequestEntityTooLarge)
    def upload_limit(_error):
        failure = Limits("This request exceeds the allowed upload size.", "file")
        return jsonify(failure.payload()), failure.status

    @app.errorhandler(404)
    def missing_route(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "This workspace resource is unavailable."}), 404
        return app.send_static_file("index.html")

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/health")
    def health():
        db.check()
        return jsonify({"status": "ok", "app": "Endgrain Lab", "archive": exchange.INSPECTION_VERSION})

    @app.get("/api/tools")
    def tools():
        return jsonify({"tools": measurements.catalog(), "surfaces": SURFACES, "dispositions": reports.DISPOSITIONS})

    @app.get("/api/inspections")
    def inspections():
        search = string(request.args.get("q"), "search", 100)
        surface = request.args.get("surface", "")
        if surface:
            choice(surface, "surface", SURFACES)
        limit = request.args.get("limit", 75, type=int)
        integer(limit, "limit", 1, 100)
        results = db.inspections(search, surface, limit)
        return jsonify({"items": [workspaces.project_metadata(item) for item in results]})

    @app.post("/api/inspections")
    def upload():
        content, filename = file_attachment(MAX_UPLOAD_BYTES, "an end-face photograph")
        body = ingest_inspection(db, content, filename, request.form.to_dict())
        return jsonify(workspaces.project_metadata(body)), 201

    @app.post("/api/demo")
    def demo():
        samples, controls = demo_payloads()
        imports = db.seed_inspections(samples)
        with db.transaction() as conn:
            for reference in controls:
                db.put_capture(conn, reference)
        return jsonify({"items": [workspaces.project_metadata(item) for item in imports]}), 201

    @app.get("/api/inspections/<inspection_id>")
    def inspect(inspection_id):
        return jsonify(workspaces.project_metadata(db.inspection(inspection_id)))

    @app.patch("/api/inspections/<inspection_id>")
    def update(inspection_id):
        return jsonify(workspaces.update(db, inspection_id, read_document()))

    @app.get("/api/inspections/<inspection_id>/stats")
    def frame_stats(inspection_id):
        record = db.inspection(inspection_id)
        frame = workspaces.frame_for(db, record)
        return jsonify({"revision": record["revision"], **workspaces.frame_stats(frame)})

    @app.get("/api/inspections/<inspection_id>/working.png")
    def working(inspection_id):
        record = db.inspection(inspection_id)
        show = choice(request.args.get("view", "display"), "view", ("display", "clean", "region"))
        frame = workspaces.frame_for(db, record)
        image = prepare_download_image(frame.raster, show)
        output = BytesIO()
        image.save(output, format="PNG", compress_level=4)
        attach = f"endgrain-{inspection_id[:8]}-r{record['revision']}-{show}.png" if request.args.get("download") == "1" else None
        return binary(output.getvalue(), "image/png", name=attach)

    @app.get("/api/inspections/<inspection_id>/lighting.png")
    def lighting(inspection_id):
        record = db.inspection(inspection_id)
        show = choice(request.args.get("view", "gain"), "illumination view", ("original", "corrected", "gain"))
        frame = workspaces.frame_for(db, record)
        return binary(encode_png(workspaces.lighting_view(frame, show)), "image/png")

    @app.get("/api/captures/<capture_id>/<variant>")
    def capture_file(capture_id, variant):
        options = {"original": "original", "canonical.png": "canonical", "thumb.png": "thumb"}
        source = choice(variant, "capture version", tuple(options))
        raw, metadata = db.capture_blob(capture_id, options[source])
        media = metadata["metadata"]
        name = media["filename"] if source == "original" else None
        return binary(
            raw,
            media["mime"] if source == "original" else "image/png",
            name=name,
            etag=image_etag(metadata, raw, source),
            immutable=True,
        )

    @app.get("/api/references")
    def references():
        results = select_references(db)
        items = [{**item, "thumbnail": "/api/captures/" + item["id"] + "/thumb.png"} for item in results]
        return jsonify({"items": items})

    @app.post("/api/references")
    def upload_reference():
        raw, filename = file_attachment(MAX_UPLOAD_BYTES, "a flat-field reference")
        result = ingest_reference(db, raw, filename, request.form.get("title"))
        return jsonify(result), 201

    @app.get("/api/inspections/<inspection_id>/runs")
    def runs(inspection_id):
        records = db.runs(inspection_id)
        return jsonify({"items": [measurements.describe_run(item) for item in records]})

    @app.post("/api/inspections/<inspection_id>/runs")
    def measure(inspection_id):
        return jsonify(measurements.execute(db, inspection_id, read_document())), 201

    @app.get("/api/inspections/<inspection_id>/runs/<run_id>")
    def run(inspection_id, run_id):
        return jsonify(measurements.describe_run(db.run(inspection_id, run_id)))

    @app.get("/api/inspections/<inspection_id>/runs/<run_id>/preview.png")
    def preview(inspection_id, run_id):
        result = db.run_attachment(inspection_id, run_id, "preview")
        name = f"endgrain-overlay-{run_id[:10]}.png" if request.args.get("download") == "1" else None
        return binary(result, "image/png", name=name, etag=run_id, immutable=True)

    @app.get("/api/inspections/<inspection_id>/runs/<run_id>/table.csv")
    def measurement_table(inspection_id, run_id):
        table = reports.export_run_csv(db, inspection_id, run_id)
        return binary(table, "text/csv", name=f"endgrain-reading-{run_id[:10]}.csv")

    def compare_query(inspection_id):
        return comparison.pair(db, inspection_id, request.args.get("left"), request.args.get("right"))

    @app.get("/api/inspections/<inspection_id>/compare")
    def compare(inspection_id):
        result = compare_query(inspection_id)
        payload = comparison.pair_document(result)
        pair = f"left={result.left['id']}&right={result.right['id']}"
        root = "/api/inspections/" + inspection_id + "/compare"
        payload["links"] = {"image": root + ".png?" + pair, "table": root + ".csv?" + pair}
        if comparison.same_pixel_frame(result.left, result.right):
            payload["links"]["delta"] = root + "-delta.png?" + pair
        return jsonify(payload)

    @app.get("/api/inspections/<inspection_id>/compare.png")
    def compare_overlay(inspection_id):
        result = compare_query(inspection_id)
        return binary(comparison.compare_sheet(db, result), "image/png")

    @app.get("/api/inspections/<inspection_id>/compare-delta.png")
    def compare_delta(inspection_id):
        result = compare_query(inspection_id)
        return binary(comparison.delta_overlay(db, result), "image/png")

    @app.get("/api/inspections/<inspection_id>/compare.csv")
    def compare_table(inspection_id):
        result = compare_query(inspection_id)
        return binary(reports.compare_csv(result), "text/csv", name=f"endgrain-comparison-{inspection_id[:10]}.csv")

    @app.get("/api/inspections/<inspection_id>/notes")
    def notes(inspection_id):
        revision = request.args.get("revision", type=int)
        if revision is not None:
            integer(revision, "revision", 1, 2**31 - 1)
        return jsonify({"items": db.notes(inspection_id, revision)})

    @app.post("/api/inspections/<inspection_id>/notes")
    def add_note(inspection_id):
        return jsonify(workspaces.marker(db, inspection_id, read_document())), 201

    @app.delete("/api/inspections/<inspection_id>/notes/<note_id>")
    def remove_note(inspection_id, note_id):
        body = read_document()
        fields(body, {"revision"})
        db.drop_note(inspection_id, note_id, expect_revision(body))
        return Response(status=204)

    @app.get("/api/inspections/<inspection_id>/events")
    def events(inspection_id):
        return jsonify({"items": db.events(inspection_id)})

    @app.get("/api/inspections/<inspection_id>/reports")
    def inspections_reports(inspection_id):
        return jsonify({"items": [response_report(item, inspection_id) for item in db.reports(inspection_id)]})

    @app.post("/api/inspections/<inspection_id>/reports")
    def issue_report(inspection_id):
        return jsonify(reports.issue(db, inspection_id, read_document())), 201

    @app.get("/api/inspections/<inspection_id>/reports/<report_id>")
    def one_report(inspection_id, report_id):
        return jsonify(reports.describe(db.report(inspection_id, report_id)))

    @app.get("/api/inspections/<inspection_id>/reports/<report_id>/print")
    def print_report(inspection_id, report_id):
        report = db.report(inspection_id, report_id)
        return Response(reports.render_print(report), mimetype="text/html")

    @app.get("/api/inspections/<inspection_id>/reports/<report_id>/bundle.zip")
    def packaged_report(inspection_id, report_id):
        archive = exchange.report_bundle(db, inspection_id, report_id)
        return binary(archive.bytes, "application/zip", name=archive.name)

    @app.get("/api/inspections/<inspection_id>/reports/<report_id>/review.zip")
    def review_package(inspection_id, report_id):
        archive = review.package(db, inspection_id, report_id)
        return binary(archive["bytes"], "application/zip", name=archive["name"])

    @app.get("/api/inspections/<inspection_id>/archive.zip")
    def export_archive(inspection_id):
        bundle = exchange.prepare(db, inspection_id)
        return binary(bundle.bytes, "application/zip", name=bundle.name)

    @app.post("/api/inspections/restore")
    def restore_archive():
        archive, _ = file_attachment(exchange.ARCHIVE_MAX, "an inspection archive")
        item = exchange.restore(db, archive)
        return jsonify(workspaces.project_metadata(item)), 201

    @app.get("/api/inspections/<inspection_id>/reports/<report_id>/scope.png")
    def report_scope(inspection_id, report_id):
        result = db.report(inspection_id, report_id)
        payload = plates.report_plate(db, result)
        return binary(payload, "image/png", etag=result["digest"], immutable=True)

    @app.post("/api/evidence/check")
    def check_evidence():
        archive, _ = file_attachment(review.REVIEW_MAX_BYTES, "a report evidence ZIP")
        if review.looks_like_review(archive):
            return jsonify(review.inspect_zip(archive))
        return jsonify(exchange.inspect_evidence(archive))

    @app.post("/api/review/check")
    def check_review():
        archive, _ = file_attachment(review.REVIEW_MAX_BYTES, "an external review ZIP")
        return jsonify(review.inspect_zip(archive))

    @app.get("/api/inspections/<inspection_id>/calibration")
    def calibration_reading(inspection_id):
        _, data, rev = calibration_preview.prepare(db, inspection_id)
        return jsonify({**data.payload(), "revision": rev})

    @app.get("/api/inspections/<inspection_id>/calibration.png")
    def calibration_overlay(inspection_id):
        raster, data, rev = calibration_preview.prepare(db, inspection_id)
        name = f"endgrain-calibration-r{rev}.png" if request.args.get("download") == "1" else None
        return binary(calibration_preview.stage_overlay(raster, data), "image/png", name=name)

    def probe_inputs() -> dict:
        args = request.args
        missing = {"x", "y", "revision"} - set(args)
        if missing:
            raise WorkbenchError("Provide a measuring position and revision for the pixel probe.", "point")
        return {
            "point": [args.get("x", type=float), args.get("y", type=float)],
            "revision": args.get("revision", type=int),
            "radius": args.get("radius", 12, type=int),
            "scale": args.get("scale", 8, type=int),
        }

    @app.get("/api/inspections/<inspection_id>/probe")
    def point_probe(inspection_id):
        selection = probe.load(db, inspection_id, probe_inputs())
        return jsonify(probe.describe(selection))

    @app.get("/api/inspections/<inspection_id>/probe.png")
    def point_preview(inspection_id):
        selection = probe.load(db, inspection_id, probe_inputs())
        return binary(probe.preview(selection), "image/png")

    @app.get("/api/inspections/<inspection_id>/probe.csv")
    def point_csv(inspection_id):
        selection = probe.load(db, inspection_id, probe_inputs())
        return binary(probe.csv_projection(selection), "text/csv", name="endgrain-pixel-probe.csv")

    @app.get("/api/recipes")
    def list_recipes():
        return jsonify({"items": recipes.catalog(db)})

    @app.post("/api/recipes")
    def save_recipe():
        return jsonify(recipes.create(db, read_document())), 201

    @app.post("/api/recipes/<recipe_id>/prepare")
    def preview_recipe(recipe_id):
        result = recipes.prepare(db, recipe_id, read_document())
        return jsonify(recipes.overview(result))

    @app.post("/api/recipes/<recipe_id>/run")
    def apply_recipe(recipe_id):
        result = recipes.prepare(db, recipe_id, read_document())
        return jsonify({**recipes.overview(result), "runs": recipes.replay(db, result)}), 201

    @app.get("/api/lots")
    def list_lots():
        return jsonify({"items": lots.lot_options(db), "tools": lots.available_metrics()})

    def survey_query():
        params = {}
        for key in ("lot", "kind", "metric", "units", "revisions"):
            if key in request.args:
                params[key] = request.args[key]
        for key in ("below", "above"):
            if key in request.args:
                val = request.args.get(key, type=float)
                if val is None:
                    raise WorkbenchError("Enter a numeric screening limit.", key)
                params[key] = val
        return lots.build_survey(db, params)

    @app.get("/api/lots/survey")
    def lot_survey():
        review = lots.payload(survey_query())
        query = request.query_string.decode("ascii")
        review["links"] = {
            "sheet": "/api/lots/survey.png?" + query,
            "csv": "/api/lots/survey.csv?" + query,
        }
        return jsonify(review)

    @app.get("/api/lots/survey.png")
    def lot_sheet():
        return binary(lots.contact_sheet(db, survey_query()), "image/png")

    @app.get("/api/lots/survey.csv")
    def lot_table():
        return binary(lots.to_csv(survey_query()), "text/csv", name="endgrain-lot-survey.csv")

    return app
