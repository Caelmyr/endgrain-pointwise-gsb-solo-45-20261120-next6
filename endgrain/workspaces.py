from dataclasses import dataclass
from typing import Any

import numpy as np

from .captures import SURFACES, decoded
from .database import Database
from .errors import WorkbenchError
from .geometry import (
    Plane, Raster, mask_bounds, normalize_calibration, normalize_region,
    pixel_point, plane_for, rasterize,
)
from .imaging import safe_ratio, symmetric_overlay
from .lighting import LightingResult, field_for, normalize_lighting
from .validation import choice, document, expect_revision, fields, string


@dataclass(frozen=True)
class Frame:
    inspection: dict
    raster: Raster
    source: np.ndarray
    source_support: np.ndarray
    light_result: LightingResult | None
    reference: dict | None


def describe_plane(inspection: dict) -> Plane:
    meta = inspection["capture_metadata"]
    return plane_for(meta["width"], meta["height"], inspection["state"]["calibration"])


def project_metadata(inspection: dict) -> dict:
    result = inspection.copy()
    result["plane"] = describe_plane(inspection).json()
    root = "/api/inspections/" + inspection["id"]
    captures = "/api/captures/" + inspection["capture_id"]
    result["links"] = {
        "source": captures + "/canonical.png",
        "original": captures + "/original",
        "thumb": captures + "/thumb.png",
        "working": root + "/working.png",
        "clean": root + "/working.png?view=clean",
        "region": root + "/working.png?view=region",
        "runs": root + "/runs",
        "notes": root + "/notes",
        "reports": root + "/reports",
    }
    return result


def normalized_state(db: Database, state: Any, width: int, height: int) -> dict:
    body = document(state, "measuring state")
    fields(body, {"calibration", "region", "lighting"}, "measuring state")
    calibration = normalize_calibration(body.get("calibration"), width, height)
    plane = plane_for(width, height, calibration)
    region = normalize_region(body.get("region"), plane)
    lighting = normalize_lighting(body.get("lighting"), db, width, height)
    return {"calibration": calibration, "region": region, "lighting": lighting}


def update(db: Database, inspection_id: str, body: dict) -> dict:
    allowed = {"revision", "title", "material", "lot", "surface", "calibration", "region", "lighting"}
    fields(body, allowed)
    expected = expect_revision(body)
    current = db.inspection(inspection_id)
    db.validate_revision(current, expected)
    values = {key: current[key] for key in ("title", "material", "lot", "surface")}
    state = dict(current["state"])
    if "title" in body:
        values["title"] = string(body["title"], "inspection name", 100, required=True)
    if "material" in body:
        values["material"] = string(body["material"], "wood species", 80)
    if "lot" in body:
        values["lot"] = string(body["lot"], "lot identifier", 72)
    if "surface" in body:
        values["surface"] = choice(body["surface"], "surface", SURFACES)
    meta = current["capture_metadata"]
    width, height = meta["width"], meta["height"]
    if "calibration" in body:
        updated = normalize_calibration(body["calibration"], width, height)
        prior = state["calibration"]
        if updated != prior and (updated["mode"] == "quadrilateral" or prior["mode"] == "quadrilateral"):
            state["region"] = None
        state["calibration"] = updated
    if "region" in body:
        new_plane = plane_for(width, height, state["calibration"])
        state["region"] = normalize_region(body["region"], new_plane)
    if "lighting" in body:
        state["lighting"] = normalize_lighting(body["lighting"], db, width, height)
        if state["lighting"] is not None:
            rgb, support, _ = decoded(db, current["capture_id"])
            field_for(db, rgb, support, state["lighting"])
    changes = [name for name in values if values[name] != current[name]]
    changes.extend(name for name in state if current["state"][name] != state[name])
    values["state"] = state
    return project_metadata(db.update_state(inspection_id, expected, values, changes))


def frame_for(db: Database, inspection: dict) -> Frame:
    source, support, _ = decoded(db, inspection["capture_id"])
    plane = describe_plane(inspection)
    lighting = inspection["state"]["lighting"]
    correction, reference = None, None
    if lighting:
        correction, reference = field_for(db, source, support, lighting)
    base = correction.rgb if correction else source
    raster = rasterize(base, support, plane, inspection["state"]["region"])
    return Frame(inspection, raster, source, support, correction, reference)


def frame_stats(frame: Frame) -> dict:
    raster = frame.raster
    count = int(np.count_nonzero(raster.support))
    area_px = int(np.count_nonzero(raster.inspected))
    count_all = int(np.count_nonzero(raster.supported))
    return {
        "supported_pixels": count,
        "inspection_pixels": area_px,
        "available_pixels": count_all,
        "coverage": safe_ratio(count, area_px),
        "inspected_area": count * raster.plane.pixel_area,
        "unit": raster.plane.unit,
        "bounds": mask_bounds(raster.support),
        "lighting": frame.light_result.summary if frame.light_result else None,
        "reference_title": frame.reference["title"] if frame.reference else None,
    }


def lighting_view(frame: Frame, variant: str) -> np.ndarray:
    if frame.light_result is None or frame.reference is None:
        raise WorkbenchError("Attach an illumination reference first.", "lighting")
    corrected = frame.light_result
    if variant == "original":
        rgb = frame.source
    elif variant == "corrected":
        rgb = corrected.rgb
    else:
        gain = corrected.measured & (corrected.gain >= 1.06)
        dim = corrected.measured & (corrected.gain <= 0.94)
        rgb = symmetric_overlay(
            corrected.rgb,
            [
                # White-board measurement, gain direction.
                (gain, (105, 200, 175), 0.55),
                (dim, (124, 161, 218), 0.50),
                # Field estimated from nearby measured reference pixels.
                (corrected.estimated, (196, 150, 222), 0.62),
                # Locations kept on the original capture for lack of evidence.
                (corrected.retained, (228, 164, 89), 0.76),
                # Clipping drawn last so it is never covered by a class tint.
                (corrected.clipped, (228, 75, 83), 0.82),
            ],
            support=frame.source_support,
        )
    return rgb


def marker(db: Database, inspection_id: str, raw: Any) -> dict:
    request = document(raw, "marker")
    fields(request, {"revision", "point", "label", "comment"}, "marker")
    revision = expect_revision(request)
    current = db.inspection(inspection_id)
    db.validate_revision(current, revision)
    plane = describe_plane(current)
    point = pixel_point(request.get("point"), "marker position", plane.width, plane.height)
    frame = frame_for(db, current)
    x = int(np.clip(round(point[0]), 0, plane.width - 1))
    y = int(np.clip(round(point[1]), 0, plane.height - 1))
    if not frame.raster.support[y, x]:
        raise WorkbenchError("Place the marker on a supported pixel in the selected region.", "point")
    values = {
        "point": point,
        "label": string(request.get("label"), "marker label", 42, required=True),
        "comment": string(request.get("comment"), "marker comment", 600, multiline=True),
    }
    return db.add_note(inspection_id, revision, values)
