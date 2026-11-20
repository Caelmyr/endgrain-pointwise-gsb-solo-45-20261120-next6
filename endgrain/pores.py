from collections import Counter
from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .geometry import Plane, Raster, mask_bounds
from .imaging import luminance, masked_smooth, safe_ratio, symmetric_overlay
from .validation import document, fields, flag, integer, number

CONNECTIVITY = np.ones((3, 3), dtype=bool)
# How far a region-line fragment may sit from the contour seen outside the
# region and still be treated as the same pore. Masking a region slightly moves
# the local background estimate, so seed edges can shift by a pixel or two.
JOIN_REACH = 2

COUNTED = {"counted_inside", "counted_closed", "counted_partial"}


@dataclass(frozen=True)
class Contour:
    label: int
    bounds: tuple
    body: np.ndarray
    pixels: int
    shape: dict
    clipped: bool


@dataclass(frozen=True)
class Components:
    observed: np.ndarray
    retained: np.ndarray
    boundaries: np.ndarray
    excluded: np.ndarray
    continuations: np.ndarray
    elsewhere: np.ndarray
    limited: np.ndarray
    rows: list[dict]
    reasons: dict[str, int]
    summary: dict


def normalize_params(raw: Any, plane: Plane) -> dict:
    body = document(raw, "pore settings")
    fields(body, {"contrast", "min_area", "max_area", "compactness", "reject_edge"}, "pore")
    max_area = integer(body.get("max_area", 130), "largest pore area in square pixels", 5, 6000)
    return {
        "contrast": number(body.get("contrast", 0.095), "pore contrast", 0.01, 0.6),
        "min_area": integer(body.get("min_area", 5), "smallest pore area in square pixels", 2, max_area),
        "max_area": max_area,
        "compactness": number(body.get("compactness", 0.36), "minimum roundness", 0.05, 0.95),
        "reject_edge": flag(body.get("reject_edge", True), "exclude partial edge pores"),
    }


def perimeter(mask: np.ndarray, dx: float, dy: float) -> float:
    padded = np.pad(mask, 1, constant_values=False)
    h = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    v = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    return float(h * dy + v * dx)


def inertia(positions: tuple[np.ndarray, np.ndarray], plane: Plane) -> tuple[float, float]:
    ys, xs = positions
    dx, dy = plane.mm_x or 1, plane.mm_y or 1
    coords = np.stack([xs * dx, ys * dy], axis=1)
    vectors = coords - coords.mean(axis=0)
    moment = vectors.T @ vectors / max(1, len(xs))
    eigenvalues, axes = np.linalg.eigh(moment)
    direction = float(np.degrees(np.arctan2(axes[1, 1], axes[0, 1])) % 180)
    eccentricity = sqrt(max(0.0, 1 - eigenvalues[0] / max(eigenvalues[1], 1e-12)))
    return direction, eccentricity


def dark_seeds(raster: Raster, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    light = luminance(raster.rgb)
    support = raster.support
    radius = np.clip(sqrt(settings["max_area"] / pi), 3, 26)
    background, available = masked_smooth(light, support, radius)
    difference = background - light
    strongest = (difference >= settings["contrast"]) & support & (available >= 0.55)
    soft = (difference >= 0.68 * settings["contrast"]) & support & (available >= 0.55)
    # Two steps of growth let the body occupy a dim rim without treating every dark band as a seed.
    candidate = ndi.binary_dilation(strongest, structure=CONNECTIVITY, iterations=2, mask=soft)
    return candidate, difference


def support_rim(mask: np.ndarray, *, at_frame: bool = False) -> np.ndarray:
    """Boundary pixels of a mask.

    With ``at_frame`` the frame of the raster is treated as outside, so an image
    border is included in the rim. Otherwise the frame is treated as continuing,
    which leaves only the drawn outline as a boundary.
    """
    eroded = ndi.binary_erosion(mask, structure=CONNECTIVITY, border_value=0 if at_frame else 1)
    return mask & ~eroded


def pore_geometry(component: np.ndarray, plane: Plane, start: tuple[int, int]) -> dict:
    ys, xs = np.nonzero(component)
    pixel_count = len(xs)
    area = pixel_count * plane.pixel_area
    shape_area = perimeter(component, plane.mm_x or 1, plane.mm_y or 1)
    roundness = min(1.0, 4 * pi * area / max(shape_area ** 2, 1e-9))
    orientation, eccentricity = inertia((ys, xs), plane)
    bounds = mask_bounds(component)
    x, y = start
    return {
        "x": float(np.mean(xs) + x),
        "y": float(np.mean(ys) + y),
        "pixels": pixel_count,
        "area": area,
        "perimeter": shape_area,
        "diameter": sqrt(4 * area / pi),
        "roundness": roundness,
        "eccentricity": eccentricity,
        "orientation": orientation,
        "bbox": [bounds[0] + x, bounds[1] + y, bounds[2] + x, bounds[3] + y],
    }


def whole_contours(raster: Raster, settings: dict) -> tuple[np.ndarray, list[Contour]] | None:
    """Segment the available photograph with the drawn region ignored.

    This is the view in which a pore straddling the operator's line can still be
    seen whole. The available-pixel limit is kept, so a contour running into the
    image edge or a transparency gap is genuinely cut off and is never completed
    by assuming the unseen part is wood.
    """
    if np.array_equal(raster.support, raster.supported):
        return None
    context = Raster(raster.rgb, raster.supported, np.ones_like(raster.inspected), raster.plane)
    observed, _ = dark_seeds(context, settings)
    labels, count = ndi.label(observed, structure=CONNECTIVITY)
    edge = support_rim(raster.supported, at_frame=True)
    contours = []
    for label, bounds in enumerate(ndi.find_objects(labels), 1):
        if bounds is None:
            continue
        body = labels[bounds] == label
        contours.append(Contour(
            label=label,
            bounds=bounds,
            body=body,
            pixels=int(body.sum()),
            shape=pore_geometry(body, raster.plane, (bounds[1].start, bounds[0].start)),
            clipped=bool(np.any(body & edge[bounds])),
        ))
    return labels, contours


def associated_label(labels: np.ndarray | None, bounds: tuple, body: np.ndarray) -> int | None:
    """Find the whole contour a fragment cut by the region line belongs to.

    Exact overlap is preferred; a nearest dark contour within JOIN_REACH pixels
    covers seeds that shifted when the measuring support was reduced. A fragment
    with no such neighbour is left unattributed rather than guessed.
    """
    if labels is None:
        return None
    values = labels[bounds][body]
    ids, counts = np.unique(values[values > 0], return_counts=True)
    if len(ids):
        return int(ids[np.argmax(counts)])
    y0 = max(0, bounds[0].start - JOIN_REACH)
    y1 = min(labels.shape[0], bounds[0].stop + JOIN_REACH)
    x0 = max(0, bounds[1].start - JOIN_REACH)
    x1 = min(labels.shape[1], bounds[1].stop + JOIN_REACH)
    sub = labels[y0:y1, x0:x1]
    inner = np.zeros(sub.shape, dtype=bool)
    inner[bounds[0].start - y0:bounds[0].stop - y0, bounds[1].start - x0:bounds[1].stop - x0] = body
    dark = sub > 0
    if not dark.any() or not inner.any():
        return None
    distance, nearest = ndi.distance_transform_edt(~dark, return_indices=True)
    if float(distance[inner].min()) > JOIN_REACH:
        return None
    candidate = np.where(inner, distance, np.inf)
    y, x = np.unravel_index(np.argmin(candidate), candidate.shape)
    return int(sub[nearest[0][y, x], nearest[1][y, x]])


def owns_contour(contour: Contour, raster: Raster) -> bool:
    """Decide which region counts a contour that both sides can see.

    When the drawn region is not the whole available picture, the same visible
    contour may be a complete pore in one region and only a fragment in another.
    A contour is counted by the region holding its visible centre, so adjacent
    regions never both count it. A contour whose centre is in none of them is
    still counted by the region that sees it, rather than silently dropped.
    """
    full_picture = raster.supported & np.ones_like(raster.inspected)
    if np.array_equal(raster.inspected, full_picture):
        return True
    shape = contour.shape
    x = int(np.clip(round(shape["x"]), 0, raster.inspected.shape[1] - 1))
    y = int(np.clip(round(shape["y"]), 0, raster.inspected.shape[0] - 1))
    return bool(raster.inspected[y, x])


def limit_reason(pixels: int, roundness: float, settings: dict) -> str | None:
    if pixels < settings["min_area"]:
        return "too_small"
    if pixels > settings["max_area"]:
        return "too_big"
    if roundness < settings["compactness"]:
        return "shape"
    return None


def base_row(shape: dict, decision: str, contour: str, measured_pixels: int, strength: float) -> dict:
    measured = measured_pixels * shape["area"] / max(1, shape["pixels"])
    return {
        "id": None,
        "decision": decision,
        "contour": contour,
        "x": shape["x"],
        "y": shape["y"],
        "area": shape["area"],
        "measured_area": measured,
        "visible_fraction": measured / shape["area"] if shape["area"] else None,
        "diameter": shape["diameter"],
        "perimeter": shape["perimeter"],
        "roundness": shape["roundness"],
        "eccentricity": shape["eccentricity"],
        "orientation": shape["orientation"],
        "edge": contour != "inside",
        "contrast": strength,
    }


def detect_components(raster: Raster, settings: dict) -> Components:
    """Classify pore contours around the drawn region line and the capture edge.

    Contours are first found in the available photograph with the drawn region
    set aside (``whole_contours``). A pore touching the drawn line is then
    matched to that view:

    * A contour seen whole and crossing the line is counted once, from its
      complete contour, by the region that holds the contour's visible centre.
      Two adjacent regions therefore never both count it, and neither drops it.
      A region that sees a crossing contour whose centre is elsewhere lists it in
      blue as ``outside_region`` without counting it.
    * A contour running into the image border or a transparency gap is never
      completed. It is excluded, or kept as an explicit partial reading when
      edge pores are allowed. Its unseen part is never assumed to be wood.
    * A seed near the line whose visible contour does not cross it was not cut
      by the line at all, so it keeps its original interior reading.

    Pores that touch neither the drawn line nor the capture edge keep their
    original interior readings, pixel for pixel.
    """
    observed, strength = dark_seeds(raster, settings)
    labels, count = ndi.label(observed, structure=CONNECTIVITY)
    capture_line = support_rim(raster.supported, at_frame=True)
    measured = raster.support
    pair = whole_contours(raster, settings)
    whole_labels, contours = pair if pair else (None, [])
    contour_by_label = {entry.label: entry for entry in contours}
    line_distance = None
    if whole_labels is not None and not raster.inspected.all():
        rim = support_rim(raster.inspected)
        line_distance = ndi.distance_transform_edt(~rim) if rim.any() else None

    accepted = np.zeros(observed.shape, dtype=bool)
    outline = np.zeros(observed.shape, dtype=bool)
    excluded = np.zeros(observed.shape, dtype=bool)
    continuations = np.zeros(observed.shape, dtype=bool)
    elsewhere = np.zeros(observed.shape, dtype=bool)
    limited = np.zeros(observed.shape, dtype=bool)
    rows: list[dict] = []
    reasons: Counter = Counter()
    decided: set[int] = set()
    counted_pixels = 0

    def keep(shape: dict, bounds: tuple, body: np.ndarray, count_px: int, contrast: float, decision: str,
             contour: str, measured_px: int | None = None) -> None:
        nonlocal counted_pixels
        rows.append(base_row(shape, decision, contour, count_px if measured_px is None else measured_px, contrast))
        counted_pixels += count_px if measured_px is None else measured_px
        accepted[bounds] |= body
        outline[bounds] |= body & ~ndi.binary_erosion(body, structure=CONNECTIVITY, border_value=0)

    def drop(shape: dict, bounds: tuple, body: np.ndarray, decision: str, contour: str,
             count_px: int, contrast: float, measured_px: int | None = None) -> None:
        target = excluded if decision == "excluded_at_capture" else limited
        target[bounds] |= body
        rows.append(base_row(shape, decision, contour, count_px if measured_px is None else measured_px, contrast))

    pixels = np.bincount(labels.ravel(), minlength=count + 1)
    for label, bounds in enumerate(ndi.find_objects(labels), 1):
        if bounds is None:
            continue
        body = labels[bounds] == label
        count_px = int(pixels[label])
        shape = pore_geometry(body, raster.plane, (bounds[1].start, bounds[0].start))
        contrast = float(np.mean(strength[bounds][body]))
        limit = limit_reason(count_px, shape["roundness"], settings)
        touches_capture = bool(np.any(body & capture_line[bounds]))
        near_line = bool(line_distance is not None and float(line_distance[bounds][body].min()) <= JOIN_REACH)

        if not touches_capture and (line_distance is None or not near_line):
            # The drawn region is the whole frame, or this pore is clear of the
            # line and the capture edge. Keep the original reading unchanged.
            if limit:
                reasons[limit] += 1
            else:
                keep(shape, bounds, body, count_px, contrast, "counted_inside", "inside")
            continue

        match = associated_label(whole_labels, bounds, body) if whole_labels is not None else None
        contour = contour_by_label.get(match) if match is not None else None
        crosses = contour is not None and bool(np.any(contour.body & ~raster.inspected[contour.bounds]))
        if contour is not None and contour.label in decided:
            # This pore was decided from another fragment reaching the line
            # first. Never count or list the same contour twice.
            continue

        if touches_capture or (contour is not None and contour.clipped):
            # The visible contour runs into the image border or a transparency
            # gap (or the same edge cut the seed). Never complete it by assuming
            # the unseen part is wood.
            if contour is not None:
                decided.add(contour.label)
                if not owns_contour(contour, raster):
                    # The contour's centre is outside this region, so this region
                    # does not own it even though part of it is visible here.
                    elsewhere[contour.bounds] |= contour.body
                    rows.append(base_row(contour.shape, "outside_region", "cut",
                                         int(np.count_nonzero(contour.body & measured[contour.bounds])), contrast))
                    continue
            reasons["edge_at_capture"] += 1
            if settings["reject_edge"]:
                if contour is None:
                    drop(shape, bounds, body, "excluded_at_capture", "cut", count_px, contrast)
                else:
                    drop(contour.shape, contour.bounds, contour.body, "excluded_at_capture", "cut",
                         int(np.count_nonzero(contour.body & measured[contour.bounds])), contrast)
            else:
                source = shape if contour is None else contour.shape
                source_bounds = bounds if contour is None else contour.bounds
                source_body = body if contour is None else contour.body
                inside_px = int(np.count_nonzero(source_body & measured[source_bounds]))
                if not limit_reason(source["pixels"], source["roundness"], settings):
                    keep(source, source_bounds, source_body, inside_px, contrast, "counted_partial", "cut")
            continue

        if contour is None or not crosses:
            # Nothing outside the region completes this seed, so the line did not
            # cut a visible pore. Keep the original interior reading.
            if limit:
                reasons[limit] += 1
            else:
                keep(shape, bounds, body, count_px, contrast, "counted_inside", "inside")
            continue

        decided.add(contour.label)
        inside_px = int(np.count_nonzero(contour.body & measured[contour.bounds]))
        if not owns_contour(contour, raster):
            # The visible contour straddles the line, but its centre lies outside
            # this region. Show it here so the pore does not look lost, and so a
            # shifted region never counts it twice.
            elsewhere[contour.bounds] |= contour.body
            rows.append(base_row(contour.shape, "outside_region", "complete", inside_px, contrast))
            continue
        whole_limit = limit_reason(contour.pixels, contour.shape["roundness"], settings)
        if whole_limit:
            drop(contour.shape, contour.bounds, contour.body, "excluded_" + whole_limit, "complete", inside_px, contrast)
            reasons["edge_" + whole_limit] += 1
            continue
        keep(contour.shape, contour.bounds, contour.body, inside_px, contrast, "counted_closed", "complete")
        continuations[contour.bounds] |= contour.body

    counted = [entry for entry in rows if entry["decision"] in COUNTED]
    for index, entry in enumerate(counted, 1):
        entry["id"] = index
    rows.sort(key=lambda entry: (entry["id"] is None, entry["id"] or 0))
    reasons["too_small"] = reasons.get("too_small", 0)
    reasons["too_big"] = reasons.get("too_big", 0)
    reasons["shape"] = reasons.get("shape", 0)
    reasons["edge"] = reasons["edge_at_capture"]
    summary = {
        "survey_pixels": int(measured.sum()),
        "counted_pixels": counted_pixels,
        "counted": len(counted),
        "inside": sum(entry["decision"] == "counted_inside" for entry in counted),
        "closed": sum(entry["decision"] == "counted_closed" for entry in counted),
        "partial": sum(entry["decision"] == "counted_partial" for entry in counted),
        "outside": sum(entry["decision"] == "outside_region" for entry in rows),
        "at_capture": sum(entry["contour"] == "cut" for entry in rows),
        "excluded_at_capture": sum(entry["decision"] == "excluded_at_capture" for entry in rows),
        "excluded_by_limit": sum(entry["decision"] in {"excluded_too_small", "excluded_too_big", "excluded_shape"}
                                 for entry in rows),
        "region_groups": int(count),
        "whole_groups": len(contours),
    }
    return Components(
        observed, accepted, outline, excluded, continuations, elsewhere, limited, rows, dict(reasons), summary,
    )


def basis_lines(components: Components, plane: Plane) -> list[str]:
    data = components.summary
    area = data["survey_pixels"] * plane.pixel_area
    area_text = f"{area:,.2f} mm²" if plane.physical else f"{data['survey_pixels']:,} px²"
    lines = [
        f"Measurable area {area_text} = supported pixels inside the drawn region",
        f"Counted {data['counted']}: {data['inside']} whole inside, {data['closed']} closed across the line"
        + (f", {data['partial']} partial kept" if data["partial"] else ""),
        "Amber: counted once from the complete contour seen outside the drawn line",
        "Orange: not counted - contour runs to the capture edge or a transparency gap",
    ]
    if data["outside"]:
        lines.append(f"Blue: {data['outside']} visible contour(s) cut by the line have their centre outside it")
    return lines


def overlay(raster: Raster, records: Components) -> np.ndarray:
    base = symmetric_overlay(
        raster.rgb,
        [
            (records.retained, (68, 151, 142), 0.6),
            (records.continuations, (233, 196, 96), 0.52),
            (records.boundaries, (106, 224, 193), 0.86),
            (records.excluded, (251, 158, 93), 0.7),
            (records.elsewhere, (150, 168, 216), 0.55),
            (records.limited, (170, 158, 128), 0.45),
            (support_rim(raster.inspected), (247, 238, 184), 0.55),
        ],
        support=raster.support,
    )
    canvas = Image.fromarray(base)
    pen = ImageDraw.Draw(canvas)
    counted = sorted((item for item in records.rows if item["id"]), key=lambda item: item["area"], reverse=True)
    for record in counted[:22]:
        x, y = record["x"], record["y"]
        pen.text((x + 3, y - 9), str(record["id"]), fill=(221, 239, 212), stroke_width=2, stroke_fill=(29, 55, 55))
    for record in (item for item in records.rows if not item["id"]):
        x, y = record["x"], record["y"]
        pen.text((x + 3, y - 9), "x", fill=(250, 213, 170), stroke_width=2, stroke_fill=(46, 31, 24))
    lines = basis_lines(records, raster.plane)
    width = min(canvas.width - 10, max(len(line) for line in lines) * 6 + 14)
    height = len(lines) * 13 + 10
    pen.rectangle((5, 5, 5 + width, 5 + height), fill=(29, 38, 33))
    for index, line in enumerate(lines):
        pen.text((11, 10 + index * 13), line, fill=(231, 232, 201))
    return np.asarray(canvas)


def diameter_chart(rows: list[dict], unit: str) -> dict:
    sizes = np.array([item["diameter"] for item in rows])
    if not len(sizes):
        return {"kind": "bars", "rows": [], "x_label": f"Pore diameter ({unit})", "y_label": "Count"}
    start = float(sizes.min())
    stop = float(sizes.max())
    if stop <= start:
        stop = start * 1.01
    counts, edges = np.histogram(sizes, bins=10, range=(start, stop))
    return {
        "kind": "bars",
        "rows": [
            {"x": float((left + right) / 2), "value": int(count)}
            for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
        ],
        "x_label": f"Pore equivalent diameter ({unit})",
        "y_label": "Count",
    }


PORE_COLUMNS = [
    "id", "decision", "contour", "x", "y", "area", "measured_area", "visible_fraction",
    "diameter", "perimeter", "roundness", "eccentricity", "orientation", "edge", "contrast",
    "measurable_area", "measurable_area_unit", "density_basis",
]


def measure(raster: Raster, settings: dict) -> tuple[dict, np.ndarray, list[list]]:
    records = detect_components(raster, settings)
    data = records.summary
    plane = raster.plane
    supported = data["survey_pixels"]
    area = supported * plane.pixel_area
    counted = [item for item in records.rows if item["id"]]
    size = [item["diameter"] for item in counted]
    area_factor = 100 if plane.physical else 10000
    basis = "100 mm²" if plane.physical else "10,000 px²"
    pore_area = data["counted_pixels"] * plane.pixel_area
    result = {
        "metrics": {
            "pores": data["counted"],
            "whole_inside_pores": data["inside"],
            "closed_at_region_pores": data["closed"],
            "cut_at_capture_candidates": data["at_capture"],
            "contours_outside_region": data["outside"],
            "survey_area": area,
            "pore_area": pore_area,
            "pore_area_fraction": safe_ratio(data["counted_pixels"], supported),
            "density_per_basis": safe_ratio(data["counted"] * area_factor, area),
            "density_basis": basis,
            "median_diameter": float(np.median(size)) if size else None,
            "excluded_edge_candidates": records.reasons["edge"],
        },
        "notices": [
            "Dark round components are candidates for vessels. Compare the overlays before treating them as a species diagnostic.",
            "Measurable area is the supported pixels inside the drawn region; density and pore area use that same area.",
            "A pore crossing the drawn region line is counted once from its complete contour when that contour is visible.",
            "Pores reaching the image edge or a transparency gap are not completed; the unseen part is not assumed to be wood.",
        ],
        "rejected": records.reasons,
        "objects": counted[:100]
        + [item for item in records.rows if not item["id"] and item["decision"] == "outside_region"][:30]
        + [item for item in records.rows if not item["id"] and item["decision"] not in
           {"outside_region"}][:30],
        "chart": diameter_chart(counted, plane.unit),
        "unit": plane.unit,
        "basis": {
            "survey_area": area,
            "survey_pixels": supported,
            "pore_area": pore_area,
            "pore_pixels": data["counted_pixels"],
            "unit": plane.unit,
            "density_basis": basis,
            "contours": basis_lines(records, plane),
        },
    }
    table = [
        [item[name] for name in PORE_COLUMNS[:-3]] + [area, plane.unit, basis]
        for item in records.rows
    ]
    return result, overlay(raster, records), table