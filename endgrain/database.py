from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import sqlite3
from typing import Iterator
from uuid import uuid4

from .errors import Conflict, Missing

SCHEMA = """
CREATE TABLE IF NOT EXISTS captures (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    digest TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source BLOB NOT NULL,
    canonical BLOB NOT NULL,
    thumbnail BLOB NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(digest, title, source_type)
);
CREATE TABLE IF NOT EXISTS inspections (
    id TEXT PRIMARY KEY,
    capture_id TEXT NOT NULL REFERENCES captures(id),
    title TEXT NOT NULL,
    material TEXT NOT NULL,
    lot TEXT NOT NULL,
    surface TEXT NOT NULL,
    revision INTEGER NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    inspection_id TEXT NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    kind TEXT NOT NULL,
    params TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    result TEXT NOT NULL,
    preview BLOB NOT NULL,
    table_data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    inspection_id TEXT NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    point TEXT NOT NULL,
    label TEXT NOT NULL,
    comment TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    inspection_id TEXT NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    title TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    disposition TEXT NOT NULL,
    summary TEXT NOT NULL,
    manifest TEXT NOT NULL,
    digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    pk INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id TEXT NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recipes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL UNIQUE COLLATE NOCASE,
    description TEXT NOT NULL,
    inspection_id TEXT NOT NULL REFERENCES inspections(id),
    recipe TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stitches (
    id TEXT PRIMARY KEY,
    first_capture_id TEXT NOT NULL REFERENCES captures(id),
    second_capture_id TEXT NOT NULL REFERENCES captures(id),
    capture_id TEXT NOT NULL REFERENCES captures(id),
    inspection_id TEXT NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
    side TEXT NOT NULL,
    resolution TEXT NOT NULL,
    alignment TEXT NOT NULL,
    seam TEXT NOT NULL,
    seam_mask BLOB NOT NULL,
    seam_preview BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS inspection_order ON inspections(updated_at DESC, id);
CREATE INDEX IF NOT EXISTS run_order ON runs(inspection_id, revision DESC, created_at DESC, id);
CREATE INDEX IF NOT EXISTS note_order ON notes(inspection_id, revision, created_at DESC);
CREATE INDEX IF NOT EXISTS report_order ON reports(inspection_id, created_at DESC);
CREATE INDEX IF NOT EXISTS event_order ON events(inspection_id, pk DESC);
"""

INSPECTION_SELECT = """
SELECT i.*, c.title AS capture_title, c.digest AS source_digest, c.metadata AS capture_metadata
FROM inspections i JOIN captures c ON i.capture_id = c.id
"""

RUN_FIELDS = "id, inspection_id, revision, kind, params, snapshot, result, created_at"
CAPTURE_FIELDS = "id, title, digest, metadata, created_at"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def identifier() -> str:
    return uuid4().hex


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def as_dict(row: sqlite3.Row, *structured: str) -> dict:
    result = dict(row)
    for name in structured:
        result[name] = json.loads(result[name])
    return result


class Database:
    def __init__(self, folder: Path):
        self.folder = folder.resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.filename = self.folder / "endgrain.sqlite3"
        with self.connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.filename, timeout=12, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 12000")
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self, *, write: bool = True) -> Iterator[sqlite3.Connection]:
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield conn
            except BaseException:
                conn.rollback()
                raise
            else:
                conn.commit()

    def check(self) -> None:
        with self.connection() as conn:
            conn.execute("SELECT 1 FROM captures LIMIT 1").fetchone()

    def _capture_row(self, conn: sqlite3.Connection, capture_id: str) -> sqlite3.Row:
        row = conn.execute(
            f"SELECT {CAPTURE_FIELDS} FROM captures WHERE id = ?", (capture_id,)
        ).fetchone()
        if row is None:
            raise Missing("Capture")
        return row

    def capture(self, capture_id: str) -> dict:
        with self.connection() as conn:
            return as_dict(self._capture_row(conn, capture_id), "metadata")

    def capture_blob(self, capture_id: str, variant: str) -> tuple[bytes, dict]:
        columns = {"original": "source", "canonical": "canonical", "thumb": "thumbnail"}
        column = columns[variant]
        with self.connection() as conn:
            row = conn.execute(
                f"SELECT {CAPTURE_FIELDS}, {column} AS contents FROM captures WHERE id = ?",
                (capture_id,),
            ).fetchone()
            if row is None:
                raise Missing("Capture")
        result = as_dict(row, "metadata")
        return result.pop("contents"), result

    def captures(self) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT {CAPTURE_FIELDS} FROM captures ORDER BY created_at DESC, id LIMIT 120"
            ).fetchall()
        return [as_dict(row, "metadata") for row in rows]

    def put_capture(self, conn: sqlite3.Connection, payload: dict) -> str:
        existing = conn.execute(
            "SELECT id FROM captures WHERE digest = ? AND title = ? AND source_type = ?",
            (payload["digest"], payload["title"], payload["metadata"]["source_type"]),
        ).fetchone()
        if existing is not None:
            return existing["id"]
        new_id = identifier()
        conn.execute(
            """INSERT INTO captures
               (id,title,digest,source_type,source,canonical,thumbnail,metadata,created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                new_id,
                payload["title"],
                payload["digest"],
                payload["metadata"]["source_type"],
                payload["source"],
                payload["canonical"],
                payload["thumbnail"],
                encode(payload["metadata"]),
                timestamp(),
            ),
        )
        return new_id

    def create_inspection(self, conn: sqlite3.Connection, capture_id: str, values: dict) -> dict:
        self._capture_row(conn, capture_id)
        new_id = identifier()
        now = timestamp()
        conn.execute(
            """INSERT INTO inspections
               (id,capture_id,title,material,lot,surface,revision,state,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                new_id,
                capture_id,
                values["title"],
                values["material"],
                values["lot"],
                values["surface"],
                1,
                encode(values["state"]),
                now,
                now,
            ),
        )
        self.event(conn, new_id, 1, "inspection_created", {"title": values["title"]})
        return self.inspection_on(conn, new_id)

    def add_stitch(self, conn: sqlite3.Connection, values: dict) -> str:
        stitch_id = identifier()
        conn.execute(
            """INSERT INTO stitches
               (id,first_capture_id,second_capture_id,capture_id,inspection_id,side,resolution,
                alignment,seam,seam_mask,seam_preview,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                stitch_id,
                values["first_capture_id"],
                values["second_capture_id"],
                values["capture_id"],
                values["inspection_id"],
                values["side"],
                values["resolution"],
                encode(values["alignment"]),
                encode(values["seam"]),
                values["seam_mask"],
                values["seam_preview"],
                timestamp(),
            ),
        )
        return stitch_id

    def stitch_on(self, conn: sqlite3.Connection, stitch_id: str) -> dict:
        row = conn.execute("SELECT * FROM stitches WHERE id = ?", (stitch_id,)).fetchone()
        if row is None:
            raise Missing("Joined image")
        result = dict(row)
        del result["seam_mask"], result["seam_preview"]
        for name in ("alignment", "seam"):
            result[name] = json.loads(result[name])
        return result

    def stitch(self, stitch_id: str) -> dict:
        with self.connection() as conn:
            return self.stitch_on(conn, stitch_id)

    def stitch_for_inspection(self, inspection_id: str) -> dict:
        with self.connection() as conn:
            self.inspection_on(conn, inspection_id)
            row = conn.execute(
                "SELECT id FROM stitches WHERE inspection_id = ? ORDER BY created_at DESC, id LIMIT 1",
                (inspection_id,),
            ).fetchone()
            if row is None:
                raise Missing("Joined image")
            return self.stitch_on(conn, row["id"])

    def stitch_attachment(self, stitch_id: str, category: str) -> bytes:
        column = {"mask": "seam_mask", "preview": "seam_preview"}[category]
        with self.connection() as conn:
            row = conn.execute(f"SELECT {column} FROM stitches WHERE id = ?", (stitch_id,)).fetchone()
        if row is None:
            raise Missing("Joined image")
        return row[0]

    def import_inspection(self, capture: dict, values: dict) -> dict:
        with self.transaction() as conn:
            capture_id = self.put_capture(conn, capture)
            return self.create_inspection(conn, capture_id, values)

    def seed_inspections(self, items: list[tuple[dict, dict]]) -> list[dict]:
        with self.transaction() as conn:
            output = []
            for capture, values in items:
                capture_id = self.put_capture(conn, capture)
                existing = conn.execute(
                    """SELECT i.id FROM inspections i
                       JOIN events e ON e.inspection_id = i.id
                       WHERE i.capture_id = ? AND e.action = 'demo_created'
                       ORDER BY i.created_at LIMIT 1""",
                    (capture_id,),
                ).fetchone()
                if existing:
                    output.append(self.inspection_on(conn, existing["id"]))
                    continue
                created = self.create_inspection(conn, capture_id, values)
                self.event(conn, created["id"], 1, "demo_created", {"synthetic": True})
                output.append(created)
            return output

    def inspection_on(self, conn: sqlite3.Connection, inspection_id: str) -> dict:
        row = conn.execute(INSPECTION_SELECT + " WHERE i.id = ?", (inspection_id,)).fetchone()
        if row is None:
            raise Missing("Inspection")
        return as_dict(row, "state", "capture_metadata")

    def inspection(self, inspection_id: str) -> dict:
        with self.connection() as conn:
            return self.inspection_on(conn, inspection_id)

    def inspections(self, search: str, surface: str, limit: int) -> list[dict]:
        query = INSPECTION_SELECT
        conditions = []
        values = []
        if search:
            needle = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            conditions.append("(i.title LIKE ? ESCAPE '\\' OR i.lot LIKE ? ESCAPE '\\' OR i.material LIKE ? ESCAPE '\\')")
            values.extend((needle,) * 3)
        if surface:
            conditions.append("i.surface = ?")
            values.append(surface)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY i.updated_at DESC, i.id LIMIT ?"
        values.append(limit)
        with self.connection() as conn:
            rows = conn.execute(query, values).fetchall()
        return [as_dict(row, "state", "capture_metadata") for row in rows]

    def validate_revision(self, current: dict, expected: int) -> None:
        if current["revision"] != expected:
            raise Conflict(
                "The inspection changed. Reload it before applying new edits.",
                revision=current["revision"],
            )

    def update_state(self, inspection_id: str, expected: int, values: dict, changed: list[str]) -> dict:
        with self.transaction() as conn:
            current = self.inspection_on(conn, inspection_id)
            self.validate_revision(current, expected)
            if not changed:
                return current
            revision = expected + 1
            conn.execute(
                """UPDATE inspections SET title = ?, material = ?, lot = ?, surface = ?,
                   state = ?, revision = ?, updated_at = ? WHERE id = ?""",
                (
                    values["title"],
                    values["material"],
                    values["lot"],
                    values["surface"],
                    encode(values["state"]),
                    revision,
                    timestamp(),
                    inspection_id,
                ),
            )
            self.event(conn, inspection_id, revision, "inspection_revised", {"changes": changed})
            return self.inspection_on(conn, inspection_id)

    def event(self, conn: sqlite3.Connection, inspection_id: str, revision: int, action: str, detail: dict) -> None:
        conn.execute(
            "INSERT INTO events (inspection_id,revision,action,detail,created_at) VALUES (?,?,?,?,?)",
            (inspection_id, revision, action, encode(detail), timestamp()),
        )

    def events(self, inspection_id: str) -> list[dict]:
        with self.connection() as conn:
            self.inspection_on(conn, inspection_id)
            rows = conn.execute(
                """SELECT revision,action,detail,created_at FROM events
                   WHERE inspection_id = ? ORDER BY pk DESC LIMIT 90""",
                (inspection_id,),
            ).fetchall()
        return [as_dict(row, "detail") for row in rows]

    def put_run(self, inspection_id: str, revision: int, values: dict) -> dict:
        return self.put_runs(inspection_id, revision, [values])[0]

    def put_runs(
        self, inspection_id: str, revision: int, values: list[dict],
        *, recipe_id: str | None = None,
    ) -> list[dict]:
        saved = []
        with self.transaction() as conn:
            current = self.inspection_on(conn, inspection_id)
            self.validate_revision(current, revision)
            for outcome in values:
                saved.append(self.add_run_on(conn, inspection_id, revision, outcome))
            if recipe_id:
                self.event(
                    conn, inspection_id, revision, "recipe_applied",
                    {"recipe_id": recipe_id, "run_ids": [run["id"] for run in saved]},
                )
        return saved

    def add_run_on(self, conn: sqlite3.Connection, inspection_id: str, revision: int, values: dict) -> dict:
        run_id = identifier()
        conn.execute(
            """INSERT INTO runs
               (id,inspection_id,revision,kind,params,snapshot,result,preview,table_data,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                inspection_id,
                revision,
                values["kind"],
                encode(values["params"]),
                encode(values["snapshot"]),
                encode(values["result"]),
                values["preview"],
                encode(values["table_data"]),
                timestamp(),
            ),
        )
        self.event(conn, inspection_id, revision, "analysis_saved", {"kind": values["kind"], "run_id": run_id})
        return self.run_on(conn, inspection_id, run_id)

    def run_on(self, conn: sqlite3.Connection, inspection_id: str, run_id: str) -> dict:
        row = conn.execute(
            f"SELECT {RUN_FIELDS} FROM runs WHERE inspection_id = ? AND id = ?",
            (inspection_id, run_id),
        ).fetchone()
        if row is None:
            raise Missing("Measurement")
        return as_dict(row, "params", "snapshot", "result")

    def run(self, inspection_id: str, run_id: str) -> dict:
        with self.connection() as conn:
            return self.run_on(conn, inspection_id, run_id)

    def runs(self, inspection_id: str, limit: int = 60) -> list[dict]:
        with self.connection() as conn:
            self.inspection_on(conn, inspection_id)
            rows = conn.execute(
                f"""SELECT {RUN_FIELDS} FROM runs WHERE inspection_id = ?
                    ORDER BY revision DESC, created_at DESC, id LIMIT ?""",
                (inspection_id, limit),
            ).fetchall()
        return [as_dict(row, "params", "snapshot", "result") for row in rows]

    def run_attachment(self, inspection_id: str, run_id: str, category: str) -> bytes | dict:
        columns = {"preview": "preview", "table": "table_data"}
        with self.connection() as conn:
            row = conn.execute(
                f"SELECT {columns[category]} FROM runs WHERE inspection_id = ? AND id = ?",
                (inspection_id, run_id),
            ).fetchone()
        if row is None:
            raise Missing("Measurement")
        if category == "table":
            return json.loads(row[0])
        return row[0]

    def add_note(self, inspection_id: str, revision: int, payload: dict) -> dict:
        note_id = identifier()
        now = timestamp()
        with self.transaction() as conn:
            current = self.inspection_on(conn, inspection_id)
            self.validate_revision(current, revision)
            conn.execute(
                """INSERT INTO notes (id,inspection_id,revision,point,label,comment,created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (note_id, inspection_id, revision, encode(payload["point"]), payload["label"], payload["comment"], now),
            )
            self.event(conn, inspection_id, revision, "marker_added", {"note_id": note_id, "label": payload["label"]})
        return {"id": note_id, "revision": revision, "created_at": now, **payload}

    def notes_on(self, conn: sqlite3.Connection, inspection_id: str, revision: int) -> list[dict]:
        rows = conn.execute(
            """SELECT id,revision,point,label,comment,created_at FROM notes
               WHERE inspection_id = ? AND revision = ? ORDER BY created_at, id LIMIT 160""",
            (inspection_id, revision),
        ).fetchall()
        return [as_dict(row, "point") for row in rows]

    def notes(self, inspection_id: str, revision: int | None = None) -> list[dict]:
        with self.connection() as conn:
            current = self.inspection_on(conn, inspection_id)
            return self.notes_on(conn, inspection_id, revision or current["revision"])

    def drop_note(self, inspection_id: str, note_id: str, revision: int) -> None:
        with self.transaction() as conn:
            current = self.inspection_on(conn, inspection_id)
            self.validate_revision(current, revision)
            change = conn.execute(
                "DELETE FROM notes WHERE id = ? AND inspection_id = ? AND revision = ?",
                (note_id, inspection_id, revision),
            )
            if change.rowcount < 1:
                raise Missing("Marker")
            self.event(conn, inspection_id, revision, "marker_removed", {"note_id": note_id})

    def reports(self, inspection_id: str) -> list[dict]:
        with self.connection() as conn:
            self.inspection_on(conn, inspection_id)
            rows = conn.execute(
                """SELECT id,revision,title,reviewer,disposition,summary,digest,created_at FROM reports
                   WHERE inspection_id = ? ORDER BY created_at DESC, id LIMIT 50""",
                (inspection_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def report(self, inspection_id: str, report_id: str) -> dict:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM reports WHERE inspection_id = ? AND id = ?", (inspection_id, report_id)
            ).fetchone()
        if row is None:
            raise Missing("Report")
        return as_dict(row, "manifest")

    def create_report(self, inspection_id: str, expected: int, details: dict, run_ids: list[str]) -> dict:
        with self.transaction() as conn:
            inspection = self.inspection_on(conn, inspection_id)
            self.validate_revision(inspection, expected)
            measures = []
            for run_id in run_ids:
                run = self.run_on(conn, inspection_id, run_id)
                if run["revision"] != expected:
                    raise Conflict(
                        "Reports can only include measurements from the current inspection revision.",
                        run_id=run_id,
                        revision=expected,
                    )
                measures.append(run)
            notes = self.notes_on(conn, inspection_id, expected)
            report_id = identifier()
            created = timestamp()
            manifest = {
                "version": 1,
                "id": report_id,
                "created_at": created,
                "inspection": inspection,
                "review": details,
                "measurements": measures,
                "markers": notes,
            }
            canonical = encode(manifest)
            digest = sha256(canonical.encode("utf-8")).hexdigest()
            conn.execute(
                """INSERT INTO reports
                   (id,inspection_id,revision,title,reviewer,disposition,summary,manifest,digest,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    report_id,
                    inspection_id,
                    expected,
                    details["title"],
                    details["reviewer"],
                    details["disposition"],
                    details["summary"],
                    canonical,
                    digest,
                    created,
                ),
            )
            self.event(conn, inspection_id, expected, "report_issued", {"report_id": report_id, "digest": digest})
        return {"id": report_id, "manifest": manifest, "digest": digest, "revision": expected, "created_at": created}
