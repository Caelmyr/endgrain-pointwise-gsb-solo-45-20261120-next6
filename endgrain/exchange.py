from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile, ZipInfo
import json

from .captures import capture_payload, observation_fields
from .database import Database, encode
from .errors import Limits, WorkbenchError
from .geometry import normalize_calibration, normalize_region, pixel_point, plane_for
from .imaging import MAX_UPLOAD_BYTES, read_image
from .lighting import apply_flat_field
from .reports import export_run_csv, render_print
from .measurements import TOOLS
from .plates import report_plate
from .validation import document, fields, flag, list_of, number, string

ARCHIVE_MAX = 76 * 1024 * 1024
MAX_JSON = 800 * 1024
MEMBER_LIMIT = 24
INSPECTION_VERSION = "endgrain-face/1"


@dataclass(frozen=True)
class Bundle:
    name: str
    bytes: bytes


def zput(zipped: ZipFile, name: str, payload: bytes) -> None:
    member = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    member.compress_type = ZIP_DEFLATED if name.endswith((".json", ".csv", ".html")) else 0
    member.external_attr = 0o600 << 16
    zipped.writestr(member, payload)


def capture_entry(capture: dict, basename: str) -> dict:
    meta = capture["metadata"]
    return {
        "file": basename + "." + meta["extension"],
        "digest": capture["digest"],
        "title": capture["title"],
        "synthetic": meta["synthetic"],
        "filename": meta["filename"],
    }


def prepare(db: Database, inspection_id: str) -> Bundle:
    inspection = db.inspection(inspection_id)
    marker_list = db.notes(inspection_id, inspection["revision"])
    photo, source = db.capture_blob(inspection["capture_id"], "original")
    source_entry = capture_entry(source, "capture")
    lighting = inspection["state"]["lighting"]
    recall = None
    ref_image = None
    if lighting:
        ref_image, ref = db.capture_blob(lighting["reference_id"], "original")
        recall = {
            "entry": capture_entry(ref, "reference"),
            "adjustment": {key: val for key, val in lighting.items() if key != "reference_id"},
        }
    frame = {
        "format": INSPECTION_VERSION,
        "metadata": {key: inspection[key] for key in ("title", "material", "lot", "surface")},
        "capture": source_entry,
        "calibration": inspection["state"]["calibration"],
        "region": inspection["state"]["region"],
        "reference": recall,
        "markers": [{key: marker[key] for key in ("point", "label", "comment")} for marker in marker_list],
    }
    output = BytesIO()
    with ZipFile(output, "w") as zipped:
        zput(zipped, "inspection.json", encode(frame).encode("utf-8"))
        zput(zipped, source_entry["file"], photo)
        if recall:
            zput(zipped, recall["entry"]["file"], ref_image)
    return Bundle(f"endgrain-inspection-{inspection['id'][:10]}.zip", output.getvalue())


def allowed_members(zip_file: ZipFile) -> dict[str, ZipInfo]:
    keys = zip_file.infolist()
    if len(keys) > MEMBER_LIMIT:
        raise WorkbenchError("The archive has too many entries.", "file")
    output = {}
    accumulated = 0
    for item in keys:
        path = PurePosixPath(item.filename)
        if item.is_dir() or len(path.parts) != 1 or item.filename != path.name:
            raise WorkbenchError("Use an inspection archive with flat file names.", "file")
        if path.name in output or item.flag_bits & 1:
            raise WorkbenchError("The archive has a duplicate or password-protected entry.", "file")
        accumulated += item.file_size
        if accumulated > ARCHIVE_MAX:
            raise Limits("The expanded inspection archive exceeds 76 MiB.", "file")
        output[path.name] = item
    return output


def read_member(open_zip: ZipFile, table: dict[str, ZipInfo], name: str, budget: int) -> bytes:
    if name not in table:
        raise WorkbenchError(f"The archive is missing {name}.", "file")
    item = table[name]
    if item.file_size > budget or item.file_size < 1:
        raise Limits(f"{name} is empty or larger than the archive limit.", "file")
    with open_zip.open(item) as raw:
        payload = raw.read(budget + 1)
    if len(payload) != item.file_size or len(payload) > budget:
        raise Limits(f"{name} exceeds its recorded size.", "file")
    return payload


def decode_archive(open_zip: ZipFile, members: dict[str, ZipInfo]) -> dict:
    try:
        payload = read_member(open_zip, members, "inspection.json", MAX_JSON)
        data = document(json.loads(payload.decode("utf-8")), "inspection archive")
    except (ValueError, UnicodeError, RecursionError):
        raise WorkbenchError("The inspection manifest is unreadable.", "file") from None
    fields(data, {"format", "metadata", "capture", "calibration", "region", "reference", "markers"})
    if data.get("format") != INSPECTION_VERSION:
        raise WorkbenchError("Use an Endgrain Lab inspection exchange archive.", "file")
    return data


def read_capture(
    raw: dict, zipped: ZipFile, members: dict, source_type: str,
) -> tuple[dict, object, str]:
    values = document(raw, "source entry")
    fields(values, {"file", "digest", "title", "synthetic", "filename"}, "source entry")
    name = string(values.get("file"), "archive image entry", 45, required=True)
    digest = string(values.get("digest"), "source SHA-256", 64, required=True)
    title = string(values.get("title"), "original title", 100, required=True)
    filename = string(values.get("filename"), "source filename", 120, required=True)
    synthetic = flag(values.get("synthetic"), "synthetic source indicator")
    contents = read_member(zipped, members, name, MAX_UPLOAD_BYTES)
    if sha256(contents).hexdigest() != digest:
        raise WorkbenchError("An image SHA-256 does not match the manifest.", "file")
    photo = read_image(contents)
    return capture_payload(photo, title, filename, synthetic=synthetic, source_type=source_type), photo, name


def marker_values(raw, plane) -> list[dict]:
    result = []
    for item in list_of(raw, "markers", 0, 160):
        values = document(item, "marker")
        fields(values, {"point", "label", "comment"}, "marker")
        result.append({
            "point": pixel_point(values.get("point"), "marker position", plane.width, plane.height),
            "label": string(values.get("label"), "marker label", 42, required=True),
            "comment": string(values.get("comment"), "marker comment", 600, multiline=True),
        })
    return result


def restore(db: Database, raw: bytes) -> dict:
    if len(raw) > ARCHIVE_MAX:
        raise Limits("Choose an archive no larger than 76 MiB.", "file")
    try:
        with ZipFile(BytesIO(raw)) as zipped:
            return _restore_from_zip(db, zipped)
    except (BadZipFile, RuntimeError, NotImplementedError):
        raise WorkbenchError("The inspection exchange archive is damaged or uses unsupported compression.", "file") from None


def _restore_from_zip(db: Database, zipped: ZipFile) -> dict:
    members = allowed_members(zipped)
    manifest = decode_archive(zipped, members)
    capture, source, name = read_capture(manifest.get("capture"), zipped, members, "sample")
    fields_to_restore = document(manifest.get("metadata"), "inspection metadata")
    fields(fields_to_restore, {"title", "material", "lot", "surface"}, "inspection metadata")
    values = observation_fields(fields_to_restore, capture["title"])
    width, height = source.dimensions
    calibration = normalize_calibration(manifest.get("calibration"), width, height)
    plane = plane_for(width, height, calibration)
    region = normalize_region(manifest.get("region"), plane)
    notes = marker_values(manifest.get("markers", []), plane)
    reference_fields = manifest.get("reference")
    reference = controls = None
    files = {"inspection.json", name}
    if reference_fields is not None:
        reference, controls, ref_name = _read_reference(reference_fields, source, zipped, members)
        if ref_name == name:
            raise WorkbenchError("The source and reference must have separate entries.", "file")
        files.add(ref_name)
    if files != set(members):
        raise WorkbenchError("The archive contains an entry which the manifest does not describe.", "file")
    with db.transaction() as conn:
        source_id = db.put_capture(conn, capture)
        if reference:
            controls["reference_id"] = db.put_capture(conn, reference)
        values["state"] = {"calibration": calibration, "region": region, "lighting": controls}
        created = db.create_inspection(conn, source_id, values)
        _restore_notes_on(db, conn, created["id"], notes)
        db.event(conn, created["id"], created["revision"], "archive_restored", {"source_digest": capture["digest"]})
    return created


def _read_reference(raw, source, zipped: ZipFile, members: dict):
    described = document(raw, "lighting reference")
    fields(described, {"entry", "adjustment"}, "lighting reference")
    reference, ref, name = read_capture(described.get("entry"), zipped, members, "reference")
    if ref.dimensions != source.dimensions:
        raise WorkbenchError("Archived sample and reference sizes differ.", "reference_id")
    payload = document(described.get("adjustment"), "lighting settings")
    fields(payload, {"strength", "max_gain", "blur_px", "highlight_knee"}, "lighting settings")
    controls = {
        "strength": number(payload.get("strength"), "correction strength", 0, 1),
        "max_gain": number(payload.get("max_gain"), "maximum gain", 1, 4),
        "blur_px": number(payload.get("blur_px"), "reference smoothing", 2, 100),
        "highlight_knee": number(payload.get("highlight_knee", 0.9), "highlight protection", 0.6, 1.0),
    }
    apply_flat_field(source.pixels, source.available, ref.pixels, ref.available, controls)
    return reference, controls, name


def _restore_notes_on(db: Database, conn, inspection_id: str, notes: list[dict]) -> None:
    from .database import identifier, timestamp
    for note in notes:
        conn.execute(
            """INSERT INTO notes (id,inspection_id,revision,point,label,comment,created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (identifier(), inspection_id, 1, encode(note["point"]), note["label"], note["comment"], timestamp()),
        )
    if notes:
        db.event(conn, inspection_id, 1, "markers_restored", {"count": len(notes)})


def report_bundle(db: Database, inspection_id: str, report_id: str) -> Bundle:
    report = db.report(inspection_id, report_id)
    manifest = report["manifest"]
    output = BytesIO()
    with ZipFile(output, "w") as zipped:
        rows = _bundle_files(db, zipped, report, inspection_id)
        zput(zipped, "evidence.json", report_index_content(manifest, report["digest"], rows))
    return Bundle(f"endgrain-report-{report_id[:10]}.zip", output.getvalue())

REPORT_VERSION = "endgrain-evidence/1"
MAX_REPORT_PACKAGE = 138 * 1024 * 1024


def report_index_content(manifest: dict, digest: str, files: dict[str, dict]) -> bytes:
    evidence = {
        "format": REPORT_VERSION,
        "report_id": manifest["id"],
        "manifest_sha256": digest,
        "files": files,
    }
    return encode(evidence).encode("utf-8")


def member_check_info(payload: bytes) -> dict:
    return {"size": len(payload), "sha256": sha256(payload).hexdigest()}


def _bundle_files(
    db: Database, zipped: ZipFile, report: dict, inspection_id: str,
) -> dict[str, dict]:
    manifest = report["manifest"]
    seen = {}
    written = 0

    def add(name: str, data: bytes) -> None:
        nonlocal written
        written += len(data)
        if written > MAX_REPORT_PACKAGE:
            raise Limits("This evidence bundle is too large. Issue smaller reports with fewer measurements.")
        zput(zipped, name, data)
        seen[name] = member_check_info(data)

    add("report.html", render_print(report, offline=True).encode("utf-8"))
    add("manifest.json", encode(manifest).encode("utf-8"))
    add("manifest.sha256", (report["digest"] + "  manifest.json\n").encode("ascii"))
    inspected = manifest["inspection"]
    original, capture = db.capture_blob(inspected["capture_id"], "original")
    add("original." + capture["metadata"]["extension"], original)
    add("scope.png", report_plate(db, report))
    if inspected["state"]["lighting"]:
        ref, metadata = db.capture_blob(inspected["state"]["lighting"]["reference_id"], "original")
        add("reference." + metadata["metadata"]["extension"], ref)
    for index, run in enumerate(manifest["measurements"], 1):
        common = f"measurements/{index:02d}-{run['kind']}"
        add(common + ".png", db.run_attachment(inspection_id, run["id"], "preview"))
        add(common + ".csv", export_run_csv(db, inspection_id, run["id"]))
    return seen


def asset_inventory(zipped: ZipFile) -> dict[str, ZipInfo]:
    records = {}
    declared_size = 0
    names = zipped.infolist()
    if len(names) > 36:
        raise WorkbenchError("The evidence archive has too many files.", "file")
    for member in names:
        path = PurePosixPath(member.filename)
        if (
            member.is_dir() or path.is_absolute() or member.filename != path.as_posix()
            or ".." in path.parts or "\\" in member.filename
            or path.as_posix() in records or member.flag_bits & 1
        ):
            raise WorkbenchError("The evidence archive has an invalid entry.", "file")
        declared_size += member.file_size
        if declared_size > MAX_REPORT_PACKAGE:
            raise Limits("The evidence archive expands beyond 138 MiB.", "file")
        records[member.filename] = member
    return records


def verified_manifest(raw: bytes, index: dict) -> dict:
    if sha256(raw).hexdigest() != index.get("manifest_sha256"):
        raise WorkbenchError("The review manifest does not match its declared checksum.", "file")
    body = document(json.loads(raw), "report manifest")
    report_id = string(body.get("id"), "report ID", 40, required=True)
    if report_id != index.get("report_id") or body.get("version") != 1:
        raise WorkbenchError("The review manifest does not match the evidence index.", "file")
    inspect = document(body.get("inspection"), "reported inspection")
    review = document(body.get("review"), "review details")
    ledger = list_of(body.get("measurements"), "reported measurements", 1, 12)
    entries = standard_report_entries(inspect, ledger)
    return {
        "report_id": report_id,
        "report_title": string(review.get("title"), "report title", 120, required=True),
        "source_sha256": string(inspect.get("source_digest"), "source checksum", 64, required=True),
        "measurement_count": len(ledger),
        "entries": entries,
        "has_reference": bool(document(inspect.get("state"), "measuring state").get("lighting")),
    }


def standard_report_entries(inspection: dict, runs: list) -> set[str]:
    meta = document(inspection.get("capture_metadata"), "source metadata")
    ext = string(meta.get("extension"), "original extension", 8, required=True)
    if ext not in {"png", "jpg", "tiff", "webp"}:
        raise WorkbenchError("The manifest identifies an unsupported source format.", "file")
    body = {"report.html", "scope.png", "manifest.json", "manifest.sha256", "original." + ext}
    references = set()
    seen_ids = set()
    for count, item in enumerate(runs, 1):
        case = document(item, "reported measurement")
        tool = string(case.get("kind"), "measurement type", 25, required=True)
        run_id = string(case.get("id"), "measurement ID", 40, required=True)
        binding = document(case.get("snapshot"), "measurement source binding")
        if (
            tool not in TOOLS or run_id in seen_ids
            or case.get("revision") != inspection.get("revision")
            or case.get("inspection_id") != inspection.get("id")
            or binding.get("source_digest") != inspection.get("source_digest")
        ):
            raise WorkbenchError("A reported measurement does not match its inspection image and revision.", "file")
        references.add(binding.get("reference_digest"))
        seen_ids.add(run_id)
        root = f"measurements/{count:02d}-{tool}"
        body.update({root + ".png", root + ".csv"})
    if len(references) != 1:
        raise WorkbenchError("The report mixes measurement reference images.", "file")
    return body


def verify_reference_binding(manifest: bytes, references: list[str], choices: dict) -> None:
    body = json.loads(manifest)
    reading = body["measurements"][0]["snapshot"]
    expected = reading.get("reference_digest")
    if references:
        name = references[0]
        if name.split(".")[-1] not in {"jpg", "png", "tiff", "webp"}:
            raise WorkbenchError("The reference has an unknown image extension.", "file")
        entry = document(choices[name], "reference checksum")
        if entry.get("sha256") != expected:
            raise WorkbenchError("The reference checksum does not match the measurement snapshot.", "file")
    elif expected is not None:
        raise WorkbenchError("The reported measurements identify a missing reference capture.", "file")


def read_inventory(zipped: ZipFile, inventory: dict[str, ZipInfo]) -> tuple[dict, dict]:
    found = read_member(zipped, inventory, "evidence.json", 64 * 1024)
    index = document(json.loads(found), "evidence index")
    if index.get("format") != REPORT_VERSION:
        raise WorkbenchError("Choose an Endgrain Lab evidence ZIP.", "file")
    declared = document(index.get("files"), "evidence file checksums")
    if set(declared) | {"evidence.json"} != set(inventory) or "evidence.json" in declared:
        raise WorkbenchError("The evidence file set differs from the index.", "file")
    return index, declared


def record_digest(zipped: ZipFile, member: ZipInfo) -> str:
    digest = sha256()
    count = 0
    with zipped.open(member) as stream:
        while chunk := stream.read(128 * 1024):
            count += len(chunk)
            if count > member.file_size:
                raise WorkbenchError("An evidence file exceeds its declared size.", "file")
            digest.update(chunk)
    if count != member.file_size:
        raise WorkbenchError("An evidence file did not expand to the expected size.", "file")
    return digest.hexdigest()


def inspect_evidence(data: bytes) -> dict:
    if len(data) > MAX_REPORT_PACKAGE:
        raise Limits("Choose an evidence ZIP no larger than 138 MiB.", "file")
    try:
        with ZipFile(BytesIO(data)) as zipped:
            found = asset_inventory(zipped)
            index, choices = read_inventory(zipped, found)
            verified = read_member(zipped, found, "manifest.json", 8 * 1024 * 1024)
            outcome = verified_manifest(verified, index)
            checks = []
            source_records = [name for name in choices if name.startswith("original.")]
            if len(source_records) != 1:
                raise WorkbenchError("An evidence ZIP needs exactly one original capture.", "file")
            reference_names = [name for name in choices if name.startswith("reference.")]
            if len(reference_names) != int(outcome["has_reference"]):
                raise WorkbenchError("The reference file does not match the reported measuring state.", "file")
            expected_members = outcome.pop("entries")
            if expected_members | set(reference_names) != set(choices):
                raise WorkbenchError("The evidence index does not contain the files named by the report.", "file")
            verify_reference_binding(verified, reference_names, choices)
            checksum = read_member(zipped, found, "manifest.sha256", 128)
            expected_manifest_checksum = (index["manifest_sha256"] + "  manifest.json\n").encode("ascii")
            if checksum != expected_manifest_checksum:
                raise WorkbenchError("The separate manifest checksum does not match the report.", "file")
            for name, metadata in choices.items():
                declared = document(metadata, "file checksum")
                member = found[name]
                actual = record_digest(zipped, member)
                same = actual == declared.get("sha256") and member.file_size == declared.get("size")
                if name == source_records[0]:
                    same = same and actual == outcome["source_sha256"]
                checks.append({"name": name, "size": member.file_size, "verified": same})
    except (BadZipFile, RuntimeError, ValueError, RecursionError, NotImplementedError):
        raise WorkbenchError("The report evidence archive is unreadable.", "file") from None
    return {
        **outcome, "verified": all(item["verified"] for item in checks),
        "files": checks,
        "notice": "Checksums confirm agreement within this bundle. They do not authenticate who made the inspection.",
    }
