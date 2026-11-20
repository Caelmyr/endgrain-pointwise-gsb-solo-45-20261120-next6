from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from threading import BoundedSemaphore
from typing import Any

import numpy as np

from . import colorzones, cracks, crossdating, edge_lost, finish, footprint, grain, pores, quality, rings
from .database import Database
from .errors import Conflict, WorkbenchError
from .imaging import encode_png
from .validation import choice, document, expect_revision, fields, integer, number
from .workspaces import frame_for

MEASUREMENT_SLOTS = BoundedSemaphore(2)


@dataclass(frozen=True)
class Tool:
    code: str
    title: str
    description: str
    inputs: list[dict]
    metrics: dict
    columns: list[str]
    validator: Callable
    measure: Callable


def control(key: str, title: str, default: Any, *, step: float | None = None, values: tuple = ()) -> dict:
    return {"key": key, "title": title, "default": default, "step": step, "values": values}


def quality_params(raw, plane):
    body = document(raw, "capture quality settings")
    fields(body, {"grid", "softness_factor", "mask"}, "capture quality")
    return {
        "grid": integer(body.get("grid", 7), "quality grid divisions", 2, 16),
        "softness_factor": number(body.get("softness_factor", 0.60), "low detail factor", 0.15, 1.1),
        "mask": choice(body.get("mask", "all"), "display mask", ("all", "shadows", "glare", "softness", "plain")),
    }


def run_quality(raster, params):
    reading = quality.measure_quality(raster, params["grid"], params["softness_factor"])
    data = {
        "metrics": reading.metrics,
        "chart": {"kind": "histogram", "channels": reading.rgb_histogram, "x_label": "Display intensity", "y_label": "Pixels"},
        "notices": reading.notices,
        "unit": "px",
    }
    return data, quality.quality_preview(raster, reading, params["mask"]), reading.rows


TOOLS = {
    "quality": Tool(
        "quality", "Capture quality", "Inspect recoverable detail, glare and clipped pixels before measuring.",
        [
            control("grid", "Grid divisions", 7, step=1),
            control("softness_factor", "Low detail factor", 0.60, step=0.05),
            control("mask", "Overlay", "all", values=("all", "shadows", "glare", "softness", "plain")),
        ],
        {
            "coverage": ("Supported area", "%"),
            "glare_fraction": ("Glare", "%"),
            "shadow_fraction": ("Deep shadow", "%"),
            "low_detail_fraction": ("Low detail", "%"),
            "supported_pixels": ("Supported pixels", "px"),
            "highlight_fraction": ("Clipped channels", "%"),
            "luminance_p05": ("Luminance P05", "ratio"),
            "luminance_p95": ("Luminance P95", "ratio"),
        },
        quality.QUALITY_COLUMNS, quality_params, run_quality,
    ),
    "rings": Tool(
        "rings", "Annual ring transect", "Read candidate crossings and widths along one marked transect.",
        [
            control("band", "Band width (px)", 7, step=1),
            control("prominence", "Minimum prominence", 0.024, step=0.002),
            control("minimum_gap", "Minimum spacing (px)", 6, step=1),
            control("polarity", "Boundary tone", "dark", values=("dark", "light")),
        ],
        {
            "crossings": ("Ring candidates", "count"),
            "transect_length": ("Transect length", "linear"),
            "mean_width": ("Mean width", "linear"),
            "median_width": ("Median width", "linear"),
            "valid_fraction": ("Supported transect", "%"),
            "widths": ("Usable intervals", "count"),
        },
        rings.RING_COLUMNS, rings.normalize_params, rings.measure,
    ),
    "crossdating": Tool(
        "crossdating", "Dual-transect ring sequence",
        "Align the rings read along two inner-to-outer transects into one shared, reviewable annual sequence.",
        [
            control("width_tolerance", "Ring width tolerance", 0.35, step=0.05),
            control("band", "Band width (px)", 7, step=1),
            control("prominence", "Minimum prominence", 0.024, step=0.002),
            control("minimum_gap", "Minimum spacing (px)", 6, step=1),
            control("polarity", "Boundary tone", "dark", values=("dark", "light")),
        ],
        {
            "shared_rings": ("Shared annual rings", "count"),
            "paired_rings": ("Rings on both transects", "count"),
            "single_line_rings": ("Single-transect rings", "count"),
            "ambiguities": ("Ambiguities pending", "count"),
            "resolved_ambiguities": ("Ambiguities resolved", "count"),
            "evidence_gap_rings": ("Evidence-gap rings", "count"),
            "mean_width": ("Mean width", "linear"),
            "median_width": ("Median width", "linear"),
            "measured_widths": ("Measured widths", "count"),
            "supported_transect_a": ("Transect A support", "%"),
            "supported_transect_b": ("Transect B support", "%"),
        },
        crossdating.CROSSDATING_COLUMNS, crossdating.normalize_params, crossdating.measure,
    ),
    "cracks": Tool(
        "cracks", "Crack traces", "Trace connected dark checks and assess their width, branching and length.",
        [
            control("contrast", "Crack contrast", 0.14, step=0.01),
            control("scale", "Background radius (px)", 6, step=1),
            control("minimum_length", "Minimum length (px)", 13, step=1),
            control("bridge", "Gap closing radius (px)", 1, step=1),
        ],
        {
            "traces": ("Trace components", "count"),
            "total_length": ("Combined length", "linear"),
            "longest_trace": ("Longest trace", "linear"),
            "edge_traces": ("At support edge", "count"),
            "internal_traces": ("Interior traces", "count"),
            "trace_area_fraction": ("Trace area", "%"),
        },
        cracks.CRACK_COLUMNS, cracks.normalize_params, cracks.measure,
    ),
    "pores": Tool(
        "pores", "Vessel pores", "Estimate area, diameter and density of candidate round vessels.",
        [
            control("contrast", "Pore contrast", 0.095, step=0.005),
            control("min_area", "Smallest area (px²)", 5, step=1),
            control("max_area", "Largest area (px²)", 130, step=1),
            control("compactness", "Minimum roundness", 0.36, step=0.02),
            control("reject_edge", "Exclude partial edge pores", True),
        ],
        {
            "pores": ("Pore candidates", "count"),
            "pore_area_fraction": ("Pore area", "%"),
            "density_per_basis": ("Pore density", "density"),
            "median_diameter": ("Median diameter", "linear"),
            "survey_area": ("Survey area", "area"),
            "excluded_edge_candidates": ("Excluded edge candidates", "count"),
        },
        pores.PORE_COLUMNS, pores.normalize_params, pores.measure,
    ),
    "grain": Tool(
        "grain", "Grain orientation", "Map reliable local texture direction and compare it to an axis.",
        [
            control("window", "Integration radius (px)", 4, step=1),
            control("grid", "Tile width (px)", 36, step=1),
            control("minimum_coherence", "Minimum coherence", 0.38, step=0.02),
            control("minimum_energy", "Minimum gradient energy", 0.012, step=0.001),
            control("nominal_axis", "Comparison axis (°)", 0, step=1),
            control("tolerance", "Axis tolerance (°)", 20, step=1),
        ],
        {
            "principal_angle": ("Principal angle", "°"),
            "angle_concentration": ("Concentration", "%"),
            "classified_fraction": ("Reliable area", "%"),
            "within_axis_tolerance": ("Within tolerance", "%"),
            "median_coherence": ("Median coherence", "%"),
        },
        grain.GRAIN_COLUMNS, grain.normalize_params, grain.measure,
    ),
    "finish": Tool(
        "finish", "Saw chatter", "Locate periodic marks by their spacing and tool direction.",
        [
            control("window", "Window size (px)", 72, step=1),
            control("min_period", "Shortest period (px)", 3, step=0.5),
            control("max_period", "Longest period (px)", 18, step=0.5),
            control("axis", "Tool mark axis (°)", 0, step=1),
            control("angle_band", "Axis tolerance (°)", 20, step=1),
            control("flag_share", "Flagged power share", 0.28, step=0.01),
        ],
        {
            "flagged_window_fraction": ("Windows with chatter", "%"),
            "analyzable_windows": ("Analyzable windows", "count"),
            "peak_period_median": ("Median peak period", "linear"),
            "band_power_p90": ("Band power P90", "%"),
            "rms_p50": ("Median texture RMS", "ratio"),
            "total_windows": ("Total windows", "count"),
        },
        finish.FINISH_COLUMNS, finish.normalize_params, finish.measure,
    ),
    "footprint": Tool(
        "footprint", "Wood footprint", "Separate an end face from its backing and measure its silhouette.",
        [
            control("minimum_luma", "Minimum brightness", 0.24, step=0.01),
            control("warmth", "Red–blue separation", 0.10, step=0.01),
            control("bridge", "Edge smoothing radius (px)", 2, step=1),
            control("minimum_hole_area", "Smallest internal void (px²)", 110, step=1),
        ],
        {
            "wood_area": ("Wood area", "area"),
            "wood_fraction": ("Wood in region", "%"),
            "outer_extent_x": ("Horizontal span", "linear"),
            "outer_extent_y": ("Vertical span", "linear"),
            "trace_perimeter": ("Raster perimeter", "linear"),
            "equivalent_diameter": ("Equivalent diameter", "linear"),
            "boundary_truncated": ("Touches support edge", "bool"),
        },
        footprint.FOOTPRINT_COLUMNS, footprint.normalize_params, footprint.measure,
    ),
    "edge_loss": Tool(
        "edge_loss", "Edge recession", "Compare the visible face against four expected stock edges.",
        [
            control("depth", "Probe depth (px)", 28, step=1),
            control("station_spacing", "Station spacing (px)", 10, step=1),
            control("threshold", "Recession flag (px)", 4, step=0.5),
            control("minimum_luma", "Minimum brightness", 0.24, step=0.01),
            control("warmth", "Red–blue separation", 0.10, step=0.01),
            control("bridge", "Edge smoothing radius (px)", 2, step=1),
        ],
        {
            "expected_area": ("Reference area", "area"),
            "visible_loss": ("Exposed backing", "area"),
            "visible_loss_fraction": ("Visible loss", "%"),
            "unknown_fraction": ("Unknown area", "%"),
            "flagged_stations": ("Receded stations", "count"),
            "deepest_recession": ("Maximum recession", "linear"),
        },
        edge_lost.LOSS_COLUMNS, edge_lost.normalize_params, edge_lost.measure,
    ),
    "colors": Tool(
        "colors", "Color zones", "Measure area near two or more color swatches and flag ambiguous pixels.",
        [
            control("radius", "Swatch radius (px)", 14, step=1),
            control("limit", "Maximum color difference", 15, step=0.5),
            control("minimum_margin", "Closest-zone margin", 2, step=0.5),
        ],
        {
            "qualified_fraction": ("Classified area", "%"),
            "swatch_count": ("Reference swatches", "count"),
            "ambiguous_pixels": ("Ambiguous pixels", "px"),
            "distant_pixels": ("Color outliers", "px"),
        },
        colorzones.COLORS_COLUMNS, colorzones.normalize_params, colorzones.measure,
    ),
}


def catalog() -> list[dict]:
    return [
        {"id": tool.code, "title": tool.title, "description": tool.description, "inputs": tool.inputs, "metrics": tool.metrics}
        for tool in TOOLS.values()
    ]


def json_ready(val):
    if isinstance(val, np.generic):
        return json_ready(val.item())
    if isinstance(val, float):
        return val if isfinite(val) else None
    if isinstance(val, dict):
        return {key: json_ready(entry) for key, entry in val.items()}
    if isinstance(val, (list, tuple)):
        return [json_ready(entry) for entry in val]
    return val


def describe_run(run: dict) -> dict:
    item = dict(run)
    tool = TOOLS.get(run["kind"])
    item["title"] = tool.title if tool else run["kind"]
    url = "/api/inspections/" + run["inspection_id"] + "/runs/" + run["id"]
    item["links"] = {"preview": url + "/preview.png", "csv": url + "/table.csv", "data": url}
    return item


def calculate(frame, tool: Tool, params: dict, *, recipe: dict | None = None) -> dict:
    result, visual, rows = tool.measure(frame.raster, params)
    snapshot = {
        "source_digest": frame.inspection["source_digest"],
        "state": frame.inspection["state"],
        "plane": frame.raster.plane.json(),
        "reference_digest": frame.reference["digest"] if frame.reference else None,
    }
    if recipe:
        snapshot["recipe"] = recipe
    return {
        "kind": tool.code,
        "params": params,
        "snapshot": snapshot,
        "result": json_ready(result),
        "preview": encode_png(visual),
        "table_data": {"columns": tool.columns, "rows": json_ready(rows)},
    }


def execute(db: Database, inspection_id: str, raw: Any) -> dict:
    body = document(raw, "measurement")
    fields(body, {"kind", "revision", "params"}, "measurement")
    kind = choice(body.get("kind"), "measurement", tuple(TOOLS))
    revision = expect_revision(body)
    current = db.inspection(inspection_id)
    db.validate_revision(current, revision)
    if not MEASUREMENT_SLOTS.acquire(blocking=False):
        raise Conflict("Two measurements are already processing. Retry when one finishes.")
    try:
        frame = frame_for(db, current)
        if not frame.raster.support.any():
            raise WorkbenchError("The inspection region has no available source pixels.", "region")
        tool = TOOLS[kind]
        params = tool.validator(body.get("params", {}), frame.raster.plane)
        return describe_run(db.put_run(inspection_id, revision, calculate(frame, tool, params)))
    finally:
        MEASUREMENT_SLOTS.release()


def confirm(db: Database, inspection_id: str, run_id: str, raw: Any) -> dict:
    """Recompute a cross-dating run with the operator's ambiguity decisions.

    The run keeps its identity and its inspection revision; only its params,
    result, overlay and table are replaced so the history view and downloads
    reproduce the same correspondences and the same pending zones.
    """
    body = document(raw, "cross-dating confirmation")
    fields(body, {"revision", "decisions"}, "cross-dating confirmation")
    revision = expect_revision(body)
    current = db.inspection(inspection_id)
    db.validate_revision(current, revision)
    if not MEASUREMENT_SLOTS.acquire(blocking=False):
        raise Conflict("Two measurements are already processing. Retry when one finishes.")
    try:
        with db.transaction(write=False) as conn:
            saved = db.run_on(conn, inspection_id, run_id)
        if saved["kind"] != "crossdating":
            raise WorkbenchError("Ambiguity confirmation only applies to a dual-transect ring sequence.", "run_id")
        if saved["revision"] != current["revision"]:
            raise Conflict(
                "This reading belongs to an older revision. Re-run the dual transect on the current revision to edit its correspondences.",
                revision=current["revision"],
            )
        tool = TOOLS["crossdating"]
        frame = frame_for(db, current)
        params = tool.validator(
            {**saved["params"], "decisions": body.get("decisions", [])}, frame.raster.plane,
        )
        outcome = calculate(frame, tool, params)
        with db.transaction() as conn:
            updated = db.replace_run_on(conn, inspection_id, run_id, outcome)
        return describe_run(updated)
    finally:
        MEASUREMENT_SLOTS.release()
