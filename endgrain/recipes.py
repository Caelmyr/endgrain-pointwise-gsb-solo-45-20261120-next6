from dataclasses import dataclass
from math import sqrt
from sqlite3 import IntegrityError
from typing import Any

from .database import Database, as_dict, encode, identifier, timestamp
from .errors import Conflict, Missing, WorkbenchError
from .geometry import Plane
from .measurements import MEASUREMENT_SLOTS, TOOLS, calculate, describe_run
from .validation import choice, document, expect_revision, fields, list_of, string
from .workspaces import describe_plane, frame_for

RECIPE_FIELDS = "id,title,description,inspection_id,recipe,created_at"
PIXEL_SCALED = {
    "quality": {},
    "rings": {"band": (1, 41), "minimum_gap": (2, 200)},
    "crossdating": {"band": (1, 41), "minimum_gap": (2, 200)},
    "cracks": {"scale": (2, 44), "minimum_length": (3, 600), "bridge": (0, 4)},
    "pores": {"min_area": (2, 6000), "max_area": (5, 6000)},
    "grain": {"window": (1, 32), "grid": (8, 200)},
    "finish": {"window": (32, 256), "min_period": (2.5, 40), "max_period": (2.6, 75)},
    "footprint": {"bridge": (0, 12), "minimum_hole_area": (1, 50000)},
    "colors": {"radius": (3, 60)},
    "edge_loss": {
        "depth": (4, 110), "station_spacing": (3, 100), "threshold": (0.4, 110), "bridge": (0, 12),
    },
}
PIXEL_SQUARE = {"min_area", "max_area", "minimum_hole_area"}
FRESH_GEOMETRY = {"rings": "endpoints", "footprint": "seed", "colors": "swatches", "edge_loss": "corners"}
MAX_REPLAY_PREVIEW_BYTES = 48 * 1024 * 1024


@dataclass(frozen=True)
class ReplayStep:
    kind: str
    parameters: dict
    notices: list[str]
    source_run: str

    def preview(self) -> dict:
        return {
            "kind": self.kind,
            "title": TOOLS[self.kind].title,
            "params": self.parameters,
            "notices": self.notices,
            "source_run": self.source_run,
        }


@dataclass(frozen=True)
class Replay:
    recipe: dict
    inspection: dict
    plane: Plane
    steps: list[ReplayStep]
    transfer: str


def serialized(record: dict) -> dict:
    content = record["recipe"]
    summary = [
        {"kind": step["kind"], "title": TOOLS[step["kind"]].title}
        for step in content["steps"]
    ]
    return {
        "id": record["id"],
        "title": record["title"],
        "description": record["description"],
        "inspection_id": record["inspection_id"],
        "created_at": record["created_at"],
        "source_title": content["inspection_title"],
        "source_revision": content["revision"],
        "unit": content["plane"]["unit"],
        "steps": summary,
    }


def catalog(db: Database) -> list[dict]:
    with db.connection() as conn:
        rows = conn.execute(
            f"SELECT {RECIPE_FIELDS} FROM recipes ORDER BY created_at DESC, id LIMIT 100"
        ).fetchall()
    return [serialized(as_dict(row, "recipe")) for row in rows]


def read_recipe(db: Database, recipe_id: str) -> dict:
    with db.connection() as conn:
        item = conn.execute(
            f"SELECT {RECIPE_FIELDS} FROM recipes WHERE id = ?", (recipe_id,)
        ).fetchone()
    if not item:
        raise Missing("Recipe")
    return as_dict(item, "recipe")


def create(db: Database, raw: Any) -> dict:
    body = document(raw, "measurement recipe")
    fields(body, {"title", "description", "inspection_id", "revision", "run_ids"}, "recipe")
    title = string(body.get("title"), "recipe name", 75, required=True)
    description = string(body.get("description"), "method notes", 600, multiline=True)
    inspection_id = string(body.get("inspection_id"), "source inspection", 40, required=True)
    revision = expect_revision(body)
    selected = [
        string(value, "measurement", 40, required=True)
        for value in list_of(body.get("run_ids"), "measurements", 1, 6)
    ]
    if len(set(selected)) != len(selected):
        raise WorkbenchError("Choose each measurement once.", "run_ids")
    recipe_id = identifier()
    at = timestamp()
    try:
        with db.transaction() as conn:
            inspection = db.inspection_on(conn, inspection_id)
            db.validate_revision(inspection, revision)
            steps = []
            kinds = set()
            for key in selected:
                outcome = db.run_on(conn, inspection_id, key)
                if outcome["revision"] != revision:
                    raise Conflict("Every recipe measurement must be on the current revision.", result_id=key)
                kind = outcome["kind"]
                if kind in kinds:
                    raise WorkbenchError("Choose one measurement for each kind of tool.", "run_ids")
                kinds.add(kind)
                steps.append({"kind": kind, "params": outcome["params"], "source_run": key})
            recipe = {
                "version": 1, "revision": revision, "inspection_title": inspection["title"],
                "plane": describe_plane(inspection).json(), "steps": steps,
            }
            conn.execute(
                "INSERT INTO recipes (id,title,description,inspection_id,recipe,created_at) VALUES (?,?,?,?,?,?)",
                (recipe_id, title, description, inspection_id, encode(recipe), at),
            )
            db.event(conn, inspection_id, revision, "recipe_saved", {"recipe_id": recipe_id, "steps": len(steps)})
    except IntegrityError as err:
        if "recipes.title" in str(err):
            raise Conflict("A recipe already uses that name.") from None
        raise
    return serialized({
        "id": recipe_id, "title": title, "description": description, "inspection_id": inspection_id,
        "recipe": recipe, "created_at": at,
    })


def transfer_ratio(source: dict, target: Plane) -> tuple[float, str | None]:
    if source["unit"] == "mm" and target.physical:
        source_area = source["mm_per_x"] * source["mm_per_y"]
        return sqrt(source_area / target.pixel_area), None
    area_source = source["width"] * source["height"]
    area_target = target.width * target.height
    return sqrt(area_target / area_source), "Physical scales cannot be compared. Pixel settings are scaled by the measuring-plane size."


def mapped_point(point: list, source: dict, target: Plane) -> list[float]:
    x, y = point
    return [
        round((x + 0.5) * target.width / source["width"] - 0.5, 4),
        round((y + 0.5) * target.height / source["height"] - 0.5, 4),
    ]


def geometry_parameters(kind: str, params: dict, source: dict, destination: Plane, mode: str) -> list[str]:
    warnings = []
    key = FRESH_GEOMETRY.get(kind)
    if key and key in params:
        values = params[key]
        if mode == "defaults":
            params.pop(key)
            warnings.append("The destination uses default measurement positions. Check the position on the target image.")
        elif values is not None:
            if key in {"endpoints", "corners"}:
                params[key] = [mapped_point(point, source, destination) for point in values]
            elif key == "swatches":
                params[key] = [
                    {"label": swatch["label"], "point": mapped_point(swatch["point"], source, destination)}
                    for swatch in values
                ]
            elif key == "seed":
                params[key] = mapped_point(values, source, destination)
            warnings.append("Marked positions use the same fractions of the target measuring image.")
    previous = params.pop("manual_crossings", [])
    if previous:
        warnings.append("Manual ring crossings are sample-specific. The target suggests new crossings from its pixels.")
    if kind == "crossdating":
        transects = params.get("transects")
        if isinstance(transects, dict):
            if mode == "defaults":
                for config in transects.values():
                    config.pop("endpoints", None)
                    config.pop("manual_crossings", None)
                warnings.append("The destination uses default transect positions. Check both lines on the target image.")
            else:
                for config in transects.values():
                    if "endpoints" in config:
                        config["endpoints"] = [
                            mapped_point(point, source, destination) for point in config["endpoints"]
                        ]
                    config.pop("manual_crossings", None)
                warnings.append("Both transect positions use the same fractions of the target measuring image.")
        params["decisions"] = []
        warnings.append("Ambiguity decisions are image-specific and are not replayed.")
    return warnings

def scale_controls(kind: str, params: dict, ratio: float, plane: Plane) -> list[str]:
    limited = []
    for key, bounds in PIXEL_SCALED[kind].items():
        if key not in params:
            continue
        low, high = bounds
        if kind == "cracks" and key == "minimum_length":
            high = max(13, min(high, max(plane.width, plane.height)))
        original = params[key]
        desired = original * (ratio ** 2 if key in PIXEL_SQUARE else ratio)
        bounded = max(low, min(high, desired))
        result = int(round(bounded)) if isinstance(original, int) else round(bounded, 5)
        params[key] = result
        if desired < low or desired > high:
            limited.append(key)
    if kind == "pores" and params["min_area"] > params["max_area"]:
        params["min_area"] = params["max_area"]
        limited.append("min_area")
    if kind == "finish" and params["max_period"] - params["min_period"] < 0.10001:
        params["max_period"] = min(75, params["min_period"] + 0.101)
        limited.append("max_period")
    if kind == "edge_loss" and params["threshold"] > params["depth"]:
        params["threshold"] = params["depth"]
        limited.append("threshold")
    return limited


def prepare(db: Database, recipe_id: str, raw: Any) -> Replay:
    payload = document(raw, "recipe run")
    fields(payload, {"inspection_id", "revision", "positions"}, "recipe run")
    inspection_id = string(payload.get("inspection_id"), "destination inspection", 40, required=True)
    expected = expect_revision(payload)
    mode = choice(payload.get("positions", "defaults"), "position transfer", ("defaults", "relative"))
    record = read_recipe(db, recipe_id)
    current = db.inspection(inspection_id)
    db.validate_revision(current, expected)
    target = describe_plane(current)
    source_plane = record["recipe"]["plane"]
    ratio, general_note = transfer_ratio(source_plane, target)
    steps = []
    for step in record["recipe"]["steps"]:
        kind = step["kind"]
        original = dict(step["params"])
        cautions = [general_note] if general_note else []
        cautions.extend(geometry_parameters(kind, original, source_plane, target, mode))
        clamped = scale_controls(kind, original, ratio, target)
        if clamped:
            cautions.append("Settings limited to the destination bounds: " + ", ".join(key.replace("_", " ") for key in clamped) + ".")
        try:
            params = TOOLS[kind].validator(original, target)
        except WorkbenchError as cause:
            raise WorkbenchError(TOOLS[kind].title + ": " + str(cause), cause.field) from None
        steps.append(ReplayStep(kind, params, cautions, step["source_run"]))
    return Replay(record, current, target, steps, mode)


def overview(value: Replay) -> dict:
    return {
        "recipe": serialized(value.recipe),
        "inspection_id": value.inspection["id"],
        "revision": value.inspection["revision"],
        "positions": value.transfer,
        "target_plane": value.plane.json(),
        "steps": [step.preview() for step in value.steps],
    }


def replay(db: Database, prepared: Replay) -> list[dict]:
    if not MEASUREMENT_SLOTS.acquire(blocking=False):
        raise Conflict("Two measurements are already processing. Retry when one finishes.")
    try:
        inspection = prepared.inspection
        frame = frame_for(db, inspection)
        if not frame.raster.support.any():
            raise WorkbenchError("The destination measuring region has no available source pixels.", "region")
        created = []
        total_bytes = 0
        recipe_id = prepared.recipe["id"]
        for step in prepared.steps:
            embedded = {
                "id": recipe_id,
                "source_run": step.source_run,
                "transfer": prepared.transfer,
                "notices": step.notices,
            }
            added = calculate(frame, TOOLS[step.kind], step.parameters, recipe=embedded)
            total_bytes += len(added["preview"])
            if total_bytes > MAX_REPLAY_PREVIEW_BYTES:
                raise WorkbenchError("The recipe produces too much image data in one run. Use fewer measurement steps.")
            created.append(added)
        ready = db.put_runs(inspection["id"], inspection["revision"], created, recipe_id=recipe_id)
        return [describe_run(item) for item in ready]
    finally:
        MEASUREMENT_SLOTS.release()
