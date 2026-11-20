from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .captures import decoded, reference_for
from .database import Database
from .errors import WorkbenchError
from .imaging import (
    linear_to_srgb, luminance, quantile, safe_ratio, srgb_to_linear,
)
from .validation import document, fields, number, string

# A reference pixel is trusted as a direct white-board measurement only inside
# these linear-light limits. Outside them it is glare, a transparent dead spot or
# an alpha hole and has to be estimated from nearby measured pixels.
REF_LO = 0.012
REF_HI = 0.97
# Confidence (Gaussian-weighted share of measured pixels inside the window) above
# which an estimated location is treated as well supported by its neighbourhood.
CONF_FULL = 0.75
CONF_NONE = 0.35
# Wider-window confidence decides whether a gap still has enough board around it
# to fill responsibly: a compact reflection passes, the middle of a large hole
# does not.
BROAD_FULL = 0.55
BROAD_NONE = 0.22
# Hard geometric guard, in coarse smoothing radii, beyond which even a plausible
# estimate is rejected because no measured board sits within useful reach.
REACH_FULL = 1.0
REACH_NONE = 1.75
# Locations whose final blend weight falls under this are reported as retained.
ESTIMATE_MIN_WEIGHT = 0.05
COARSE_SCALE = 2.0


@dataclass(frozen=True)
class LightingResult:
    rgb: np.ndarray
    valid: np.ndarray
    gain: np.ndarray
    clipped: np.ndarray
    summary: dict
    measured: np.ndarray
    estimated: np.ndarray
    retained: np.ndarray


def normalize_lighting(raw: Any, db: Database, width: int, height: int) -> dict | None:
    if raw is None:
        return None
    req = document(raw, "illumination")
    fields(req, {"reference_id", "strength", "max_gain", "blur_px", "highlight_knee"}, "illumination")
    reference_id = string(req.get("reference_id"), "reference image", 64, required=True)
    reference_for(db, reference_id, width, height)
    return {
        "reference_id": reference_id,
        "strength": number(req.get("strength", 1), "correction strength", 0, 1),
        "max_gain": number(req.get("max_gain", 2), "maximum gain", 1, 4),
        "blur_px": number(req.get("blur_px", 18), "reference smoothing", 2, 100),
        "highlight_knee": number(req.get("highlight_knee", 0.9), "highlight protection", 0.6, 1.0),
    }


def _ramp(value: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Smooth 0->1 ramp between lo and hi (value rises as the signal improves)."""
    t = np.clip((value - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _field_estimate(linear_reference: np.ndarray, observed: np.ndarray, sigma: float):
    """Normalized-convolution field plus its measured-pixel confidence.

    One Gaussian pass on the binary measured mask gives the window weight; the
    weighted numerator is a second pass, so each scale costs two filters rather
    than recomputing the denominator through the generic smoothing helper.
    """
    weights = ndi.gaussian_filter(observed.astype(np.float32), sigma, mode="constant")
    numerator = ndi.gaussian_filter(
        np.where(observed, linear_reference, 0.0).astype(np.float32), sigma, mode="constant"
    )
    field = np.divide(numerator, weights, out=np.zeros_like(numerator), where=weights > 1e-4)
    return field, weights


def _illumination_field(
    linear_reference: np.ndarray, observed: np.ndarray, sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the white-board field and classify every reference location.

    Returns the fine-scale field (identical to the legacy normalized convolution
    on measured pixels), a gap-filling field, the fine-window confidence, the
    wider-window confidence and the distance (in coarse smoothing radii) to the
    nearest directly measured reference pixel.
    """
    fine, confidence = _field_estimate(linear_reference, observed, sigma)
    coarse_sigma = max(sigma * COARSE_SCALE, sigma + 1.0)
    broad, broad_confidence = _field_estimate(linear_reference, observed, coarse_sigma)
    # Lean on the wider window only where the local window lacks evidence, so a
    # small gap is filled at full resolution while a broad gap degrades gently.
    trust_fine = _ramp(confidence, CONF_NONE, CONF_FULL)
    gap_field = np.where(trust_fine > 0.0, trust_fine * fine + (1.0 - trust_fine) * broad, broad)
    # Measured pixels always trust the fine field; gaps use the fused estimate.
    field = np.where(observed, fine, gap_field)
    # Distance to the nearest measured pixel, expressed in coarse-window radii.
    distance = ndi.distance_transform_edt(~observed).astype(np.float32)
    reach = distance / coarse_sigma
    return field, fine, confidence, broad_confidence, reach


def _linear_luma(sample_linear: np.ndarray) -> np.ndarray:
    return (
        sample_linear[..., 0] * np.float32(0.2126)
        + sample_linear[..., 1] * np.float32(0.7152)
        + sample_linear[..., 2] * np.float32(0.0722)
    )


def _highlight_factor(sample_linear: np.ndarray, knee: float) -> np.ndarray:
    """Attenuate brightening on already-bright sample pixels only.

    Darkening corrections pass through untouched, and below the knee the factor
    is exactly one, so ordinary mid-tone wood is corrected exactly as before.
    """
    if knee >= 1.0:
        return np.ones(sample_linear.shape[:2], dtype=np.float32)
    luma = _linear_luma(sample_linear)
    bright = _ramp(luma, knee, 1.0)
    return (1.0 - 0.85 * bright).astype(np.float32)


def apply_flat_field(
    sample: np.ndarray,
    sample_support: np.ndarray,
    reference: np.ndarray,
    reference_support: np.ndarray,
    params: dict,
) -> LightingResult:
    linear_reference = luminance(reference, linear=True)
    observed = reference_support & (linear_reference >= REF_LO) & (linear_reference <= REF_HI)
    field, fine, confidence, broad_confidence, reach = _illumination_field(
        linear_reference, observed, params["blur_px"]
    )

    # Direct white-board measurements always carry full authority. A gap earns
    # its blend weight from the amount of board still visible through the wider
    # window (a compact reflection keeps plenty; the centre of a large hole does
    # not) together with a hard neighbourhood-reach guard. The fine-window
    # confidence is intentionally not required here: the centre of even a small
    # gap has little local evidence, but a wide window around it still sees board.
    evidence_weight = _ramp(broad_confidence, BROAD_NONE, BROAD_FULL)
    reach_weight = 1.0 - _ramp(reach, REACH_FULL, REACH_NONE)
    weight = np.where(observed, 1.0, evidence_weight * reach_weight).astype(np.float32)

    # The anchor follows the original method exactly: the fine field at measured
    # reference pixels whose smoothing window is well supported. A complete,
    # clean white board is therefore corrected identically to the legacy path.
    anchor_pixels = sample_support & observed & (confidence >= CONF_FULL)
    paired = sample_support & (weight > ESTIMATE_MIN_WEIGHT)
    if np.count_nonzero(anchor_pixels) < 96 or np.count_nonzero(paired) < 96:
        raise WorkbenchError(
            "There are too few reliable reference pixels shared with the sample.",
            "reference_id",
        )

    anchor = float(np.median(fine[anchor_pixels]))
    raw_gain = anchor / np.clip(field, 0.005, 1.0)
    cap = params["max_gain"]
    capped = np.clip(raw_gain, 1 / cap, cap)
    full_gain = (1 + (capped - 1) * params["strength"]).astype(np.float32)

    # Protect recoverable highlights and fine dark lines from being pushed white:
    # only the brightening part of the gain is rolled off near the top of tone.
    sample_linear = srgb_to_linear(sample)
    highlight = _highlight_factor(sample_linear, params["highlight_knee"])
    protective = np.where(full_gain > 1.0, 1.0 + (full_gain - 1.0) * highlight, full_gain)
    effective_gain = 1.0 + (protective - 1.0) * weight

    adjusted = sample_linear * effective_gain[..., None]
    corrected = linear_to_srgb(adjusted)
    # Blend back toward the untouched sample as evidence thins out, which keeps
    # the transition seamless where a gap is too large to fill responsibly.
    blend = weight[..., None]
    rgb = np.rint(blend * corrected.astype(np.float32) + (1.0 - blend) * sample.astype(np.float32)).astype(np.uint8)

    measured = sample_support & observed
    estimated = sample_support & ~observed & (weight > ESTIMATE_MIN_WEIGHT)
    retained = sample_support & ~measured & ~estimated
    valid = measured | estimated
    clipped = np.any(adjusted > 1, axis=2) & valid

    numerator = int(np.count_nonzero(paired))
    support_pixels = int(np.count_nonzero(sample_support))
    measured_pixels = int(np.count_nonzero(measured))
    estimated_pixels = int(np.count_nonzero(estimated))
    retained_pixels = int(np.count_nonzero(retained))
    summary = {
        "shared_pixels": numerator,
        "sample_pixels": support_pixels,
        "reference_coverage": safe_ratio(numerator, support_pixels),
        "measured_pixels": measured_pixels,
        "estimated_pixels": estimated_pixels,
        "retained_pixels": retained_pixels,
        "measured_coverage": safe_ratio(measured_pixels, support_pixels),
        "estimated_coverage": safe_ratio(estimated_pixels, support_pixels),
        "retained_coverage": safe_ratio(retained_pixels, support_pixels),
        "reference_anchor_linear": anchor,
        "mean_gain": float(np.mean(effective_gain[paired])),
        "gain_p05": quantile(effective_gain[paired], 5),
        "gain_p95": quantile(effective_gain[paired], 95),
        "gain_cap_pixels": int(np.count_nonzero(((raw_gain > cap) | (raw_gain < 1 / cap)) & paired)),
        "would_clip_pixels": int(np.count_nonzero(clipped)),
    }
    return LightingResult(rgb, valid, effective_gain, clipped, summary, measured, estimated, retained)


def field_for(
    db: Database, sample: np.ndarray, support: np.ndarray, adjustment: dict,
) -> tuple[LightingResult, dict]:
    reference, reference_support, record = decoded(db, adjustment["reference_id"])
    result = apply_flat_field(sample, support, reference, reference_support, adjustment)
    return result, record
