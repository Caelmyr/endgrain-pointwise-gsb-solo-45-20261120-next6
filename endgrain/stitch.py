from dataclasses import dataclass
from io import BytesIO
from math import ceil

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy import signal

from .captures import capture_payload, decoded, observation_fields
from .database import Database, encode, timestamp
from .errors import Conflict, Limits, WorkbenchError
from .imaging import MAX_PIXELS, encode_png, luminance, read_image
from .validation import choice, document, fields, flag, number, string

SIDES = {
    "right": "the second frame sits to the right of the first",
    "left": "the second frame sits to the left of the first",
    "below": "the second frame sits below the first",
    "above": "the second frame sits above the first",
}
JOIN_FIELDS = {"first", "second", "side", "title", "material", "lot", "surface", "alignment", "seam"}
MIN_OVERLAP = 0.06
MAX_CANDIDATES = 4
AMBIGUITY_DELTA = 0.06
AMBIGUITY_RATIO = 0.92
AMBIGUITY_FLOOR = 0.45
TEXTURE_FLOOR = 2.0e-4
TIE_WEIGHT = 0.25
SPLIT_FIDELITY = 0.72
PREVIEW_SIDE = 760
SEAM_VERSION = "endgrain-seam/1"
EMPTY, FIRST, SECOND, UNSURE_FIRST, UNSURE_SECOND = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class Frame:
    capture_id: str
    title: str
    rgb: np.ndarray
    support: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.rgb.shape[:2]


@dataclass(frozen=True)
class Candidate:
    dx: int
    dy: int
    score: float
    overlap_width: int
    overlap_height: int
    ambiguous: bool
    reason: str | None

    @property
    def depth(self) -> int:
        return min(self.overlap_width, self.overlap_height)

    def json(self) -> dict:
        return {
            "dx": self.dx, "dy": self.dy, "score": round(self.score, 4),
            "overlap_width": self.overlap_width, "overlap_height": self.overlap_height,
            "ambiguous": self.ambiguous, "reason": self.reason,
        }


@dataclass(frozen=True)
class Composite:
    rgba: np.ndarray
    mask: np.ndarray
    seam: dict


@dataclass(frozen=True)
class Placement:
    first: Frame
    second: Frame
    side: str
    candidates: list[Candidate]

    @property
    def ambiguous(self) -> bool:
        return any(item.ambiguous for item in self.candidates)

    def payload(self) -> dict:
        return {
            "first": describe(self.first),
            "second": describe(self.second),
            "side": self.side,
            "side_hint": SIDES[self.side],
            "ambiguous": self.ambiguous,
            "recommended": self.candidates[0].json() if self.candidates else None,
            "candidates": [item.json() for item in self.candidates],
            "notice": notice(self),
        }


def describe(frame: Frame) -> dict:
    height, width = frame.shape
    return {
        "capture_id": frame.capture_id, "title": frame.title,
        "width": width, "height": height,
        "supported_fraction": round(float(frame.support.mean()), 4),
    }


def notice(placement: Placement) -> str:
    if not placement.candidates:
        return ("No usable overlap was matched. Check the chosen side, or include more shared area "
                "between the frames.")
    if placement.ambiguous:
        return ("More than one offset fits this overlap. Compare the candidates and confirm the "
                "intended alignment before saving; similar neighbouring rows of texture can produce "
                "two plausible matches.")
    return "One offset is clearly stronger. Review its preview, then save the joined working image."


def load_frame(db: Database, capture_id: str, label: str) -> Frame:
    record = db.capture(capture_id)
    meta = record["metadata"]
    if meta["source_type"] != "sample":
        raise WorkbenchError("Join two face photographs, not a flat-field reference.", label)
    if meta["width"] * meta["height"] > MAX_PIXELS:
        raise Limits("A source frame is larger than 16 megapixels.", label)
    rgb, support, _ = decoded(db, capture_id)
    return Frame(capture_id, record["title"], rgb, support)


def load_pair(db: Database, body: dict) -> tuple[Frame, Frame, str]:
    first_id = string(body.get("first"), "first frame", 64, required=True)
    second_id = string(body.get("second"), "second frame", 64, required=True)
    if first_id == second_id:
        raise WorkbenchError("Choose two different photographs to join.", "second")
    side = choice(body.get("side"), "second frame side", tuple(SIDES))
    return load_frame(db, first_id, "first"), load_frame(db, second_id, "second"), side


def downsample(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return values.astype(np.float32, copy=False)
    height, width = int(ceil(values.shape[0] / factor)), int(ceil(values.shape[1] / factor))
    padded = np.zeros((height * factor, width * factor), dtype=np.float32)
    padded[: values.shape[0], : values.shape[1]] = values
    return padded.reshape(height, factor, width, factor).mean(axis=(1, 3))


def downsample_rgb(rgb: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return rgb.astype(np.float32, copy=False)
    height, width = int(ceil(rgb.shape[0] / factor)), int(ceil(rgb.shape[1] / factor))
    padded = np.zeros((height * factor, width * factor, 3), dtype=np.float32)
    padded[: rgb.shape[0], : rgb.shape[1]] = rgb
    return padded.reshape(height, factor, width, factor, 3).mean(axis=(1, 3))


def reduced_support(support: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return support.copy()
    height, width = int(ceil(support.shape[0] / factor)), int(ceil(support.shape[1] / factor))
    padded = np.zeros((height * factor, width * factor), dtype=bool)
    padded[: support.shape[0], : support.shape[1]] = support
    return padded.reshape(height, factor, width, factor).all(axis=(1, 3))


def reduced_frame(frame: Frame, factor: int) -> Frame:
    if factor <= 1:
        return frame
    rgb = np.clip(np.rint(downsample_rgb(frame.rgb, factor)), 0, 255).astype(np.uint8)
    return Frame(frame.capture_id, frame.title, rgb, reduced_support(frame.support, factor))


def bandpass(gray: np.ndarray) -> np.ndarray:
    return gray - ndi.gaussian_filter(gray, 2.0, mode="nearest")


def ncc_surface(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Score the second frame at (dx, dy) in result[dy + hb - 1, dx + wb - 1]."""
    a, b = bandpass(first), bandpass(second)
    ones_a, ones_b = np.ones_like(a), np.ones_like(b)
    cross = signal.fftconvolve(a, b[::-1, ::-1], mode="full")
    own_a = signal.fftconvolve(a * a, ones_b[::-1, ::-1], mode="full")
    own_b = signal.fftconvolve(ones_a, (b * b)[::-1, ::-1], mode="full")
    return cross / np.sqrt(np.maximum(own_a * own_b, 1e-9))


def boxes(first: tuple[int, int], second: tuple[int, int]):
    rows, columns = np.mgrid[0: first[0] + second[0] - 1, 0: first[1] + second[1] - 1]
    dx, dy = columns - (second[1] - 1), rows - (second[0] - 1)
    width = np.minimum(first[1], dx + second[1]) - np.maximum(0, dx)
    height = np.minimum(first[0], dy + second[0]) - np.maximum(0, dy)
    return dx, dy, width, height


def allowed(first: tuple[int, int], second: tuple[int, int], side: str) -> np.ndarray:
    dx, dy, width, height = boxes(first, second)
    along = max(12, int(MIN_OVERLAP * min(first[1], second[1])))
    down = max(12, int(MIN_OVERLAP * min(first[0], second[0])))
    if side in {"right", "left"}:
        towards = dx >= along if side == "right" else -dx >= along
        return towards & (height >= 0.72 * min(first[0], second[0])) & (width >= along)
    towards = dy >= down if side == "below" else -dy >= down
    return towards & (width >= 0.72 * min(first[1], second[1])) & (height >= down)


def rectangle(first: tuple[int, int], second: tuple[int, int], dx: int, dy: int):
    x0, y0 = max(0, dx), max(0, dy)
    return x0, y0, min(first[1], dx + second[1]), min(first[0], dy + second[0])


def stride_score(first: np.ndarray, second: np.ndarray, dx: int, dy: int, budget: int = 160_000) -> float:
    x0, y0, x1, y1 = rectangle(first.shape, second.shape, dx, dy)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return float("nan")
    stride = max(1, int(np.sqrt((x1 - x0) * (y1 - y0) / budget)))
    xs, ys = np.meshgrid(np.arange(x0, x1, stride), np.arange(y0, y1, stride))
    a = first[ys, xs].astype(np.float64)
    b = second[ys - dy, xs - dx].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denominator) if denominator > 1e-9 else float("nan")


def masked_score(first: Frame, second: Frame, dx: int, dy: int) -> tuple[float, int, int]:
    x0, y0, x1, y1 = rectangle(first.shape, second.shape, dx, dy)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return float("nan"), max(0, x1 - x0), max(0, y1 - y0)
    shared = first.support[y0:y1, x0:x1] & second.support[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    if np.count_nonzero(shared) < 512:
        return float("nan"), x1 - x0, y1 - y0
    a = luminance(first.rgb[y0:y1, x0:x1])[shared].astype(np.float64)
    b = luminance(second.rgb[y0 - dy:y1 - dy, x0 - dx:x1 - dx])[shared].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denominator < 1e-9:
        return float("nan"), x1 - x0, y1 - y0
    return float(np.sum(a * b) / denominator), x1 - x0, y1 - y0


def coarse_seeds(first: np.ndarray, second: np.ndarray, side: str, factor: int) -> list[tuple[int, int]]:
    surface = ncc_surface(first, second)
    dx, dy, _, _ = boxes(first.shape, second.shape)
    region = allowed(first.shape, second.shape, side) & np.isfinite(surface)
    masked = np.where(region, surface, -2.0)
    reach = max(5, int(0.012 * max(first.shape + second.shape))) | 1
    peaks = (masked == ndi.maximum_filter(masked, size=(reach, reach), mode="nearest")) & region
    coordinates = np.argwhere(peaks)
    values = masked[peaks]
    return [
        (int(round(dx[row, column] * factor)), int(round(dy[row, column] * factor)))
        for row, column in coordinates[np.argsort(values)[::-1][: MAX_CANDIDATES * 4]]
    ]


def local_peak(first: np.ndarray, second: np.ndarray, dx: int, dy: int, reach: int) -> tuple[int, int]:
    best = (dx, dy, -2.0)
    for candidate_y in range(dy - reach, dy + reach + 1):
        for candidate_x in range(dx - reach, dx + reach + 1):
            score = stride_score(first, second, candidate_x, candidate_y)
            if np.isfinite(score) and score > best[2]:
                best = (candidate_x, candidate_y, score)
    return best[0], best[1]


def find_candidates(first: Frame, second: Frame, side: str) -> list[Candidate]:
    first_gray, second_gray = luminance(first.rgb), luminance(second.rgb)
    factor = max(1, ceil(max(first.shape + second.shape) / 850))
    coarse_first, coarse_second = downsample(first_gray, factor), downsample(second_gray, factor)
    found: list[Candidate] = []
    for seed_dx, seed_dy in coarse_seeds(coarse_first, coarse_second, side, factor):
        if any(abs(seed_dx - item.dx) <= 2 * factor and abs(seed_dy - item.dy) <= 2 * factor
               for item in found):
            continue
        coarse_x, coarse_y = local_peak(coarse_first, coarse_second,
                                        int(round(seed_dx / factor)), int(round(seed_dy / factor)), 2)
        full_x, full_y = local_peak(first_gray, second_gray, coarse_x * factor,
                                    coarse_y * factor, factor + 1)
        score, width, height = masked_score(first, second, full_x, full_y)
        if np.isfinite(score):
            found.append(Candidate(full_x, full_y, score, width, height, False, None))
        if len(found) >= MAX_CANDIDATES:
            break
    found.sort(key=lambda item: -item.score)
    merged: list[Candidate] = []
    for item in found:
        if not any(abs(item.dx - peer.dx) <= 3 and abs(item.dy - peer.dy) <= 3 for peer in merged):
            merged.append(item)
    if not merged:
        return []
    best = merged[0]
    tolerance = max(4, int(0.75 * best.depth))
    marked, ambiguous = [], False
    for item in merged:
        close = (best.score - item.score <= AMBIGUITY_DELTA
                 or item.score >= AMBIGUITY_RATIO * best.score) and item.score >= AMBIGUITY_FLOOR
        close = close and item is not best
        near = abs(item.dx - best.dx) <= tolerance and abs(item.dy - best.dy) <= tolerance
        ambiguous = ambiguous or close
        reason = ("a near-identical match about one texture period away" if near
                  else "a separate offset fits almost as well") if close else None
        marked.append(Candidate(item.dx, item.dy, item.score, item.overlap_width,
                                item.overlap_height, close, reason))
    marked[0] = Candidate(best.dx, best.dy, best.score, best.overlap_width, best.overlap_height,
                          ambiguous, "another offset lies within the ambiguity margin" if ambiguous else None)
    return marked


def analyze(db: Database, raw: dict) -> Placement:
    first, second, side = load_pair(db, raw)
    return Placement(first, second, side, find_candidates(first, second, side))


def resolve(placement: Placement, raw) -> Candidate:
    if not placement.candidates:
        raise WorkbenchError(
            "No usable overlap was found. Confirm which side joins the frames, or re-shoot with more overlap.",
            "side",
        )
    if raw is None:
        if placement.ambiguous:
            raise Conflict(
                "More than one alignment fits this overlap. Pick the intended offset after reviewing them.",
                candidates=[item.json() for item in placement.candidates],
            )
        return placement.candidates[0]
    body = document(raw, "alignment")
    fields(body, {"dx", "dy"}, "alignment")
    dx = number(body.get("dx"), "alignment x offset", -200000, 200000)
    dy = number(body.get("dy"), "alignment y offset", -200000, 200000)
    for item in placement.candidates:
        if abs(item.dx - dx) <= 2 and abs(item.dy - dy) <= 2:
            return item
    raise WorkbenchError("Confirm one of the reviewed offsets for this overlap.", "alignment")


def seam_settings(raw) -> dict:
    body = document(raw, "seam settings")
    fields(body, {"focus_margin", "match_brightness"}, "seam settings")
    return {
        "focus_margin": number(body.get("focus_margin", 0.12), "sharpness margin", 0.0, 2.0),
        "match_brightness": flag(body.get("match_brightness", True), "match frame brightness"),
    }


def focus_map(gray: np.ndarray) -> np.ndarray:
    detail = ndi.laplace(ndi.gaussian_filter(gray, 0.7, mode="nearest"))
    return ndi.gaussian_filter(detail * detail, 2.0, mode="nearest")


def clean_mask(mask: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """Drop isolated speckle so a review area is a contiguous place, not pixel noise."""
    opened = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    filled = ndi.binary_closing(opened, structure=np.ones((5, 5), dtype=bool))
    return filled & limit


def relative_focus(first: Frame, second: Frame, dx: int, dy: int, x0: int, y0: int, x1: int, y1: int):
    """Per-pixel sharpness preference: positive where the second frame is sharper."""
    first_focus = focus_map(luminance(first.rgb))[y0:y1, x0:x1]
    second_focus = focus_map(luminance(second.rgb))[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    relative = (second_focus - first_focus) / (first_focus + second_focus + 1e-6)
    textured = np.maximum(first_focus, second_focus) > TEXTURE_FLOOR
    return relative, textured


def seam_cuts(first: Frame, second: Frame, dx: int, dy: int, side: str, margin: float):
    """One cut per row (or column) where the sharper frame should take over."""
    x0, y0, x1, y1 = rectangle(first.shape, second.shape, dx, dy)
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise WorkbenchError("The chosen offsets leave too little overlap to form a seam.", "alignment")
    first_support = first.support[y0:y1, x0:x1]
    second_support = second.support[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    both = first_support & second_support
    if not np.any(both):
        raise WorkbenchError("The two frames share no supported pixels at this offset.", "alignment")
    relative, textured = relative_focus(first, second, dx, dy, x0, y0, x1, y1)
    textured &= both
    vertical = side in {"right", "left"}
    sigma = max(1.0, min(6.0, min(both.shape) / 24.0))
    # Normalised by the local sharpness sum, so the criterion is independent of how much
    # absolute detail each frame happens to carry.
    matrix = ndi.gaussian_filter(np.where(textured, relative, 0.0), sigma, mode="nearest")
    if not vertical:
        matrix = matrix.T
    support = both if vertical else both.T
    textured_matrix = textured if vertical else textured.T
    length = matrix.shape[1]
    middle = (length - 1) / 2
    positive = np.maximum(matrix, 0.0)
    negative = np.maximum(-matrix, 0.0)
    before_positive = np.cumsum(positive, axis=1) - positive
    before_negative = np.cumsum(negative, axis=1) - negative
    total_positive = positive.sum(axis=1, keepdims=True)
    total_negative = negative.sum(axis=1, keepdims=True)
    total = (total_positive + total_negative)[:, 0]
    if side in {"right", "below"}:
        # Second frame owns the later part: overruled evidence is second-sharper before the
        # cut plus first-sharper after it.
        cost = before_positive + (total_negative - before_negative)
    else:
        # Second frame owns the earlier part.
        cost = before_negative + (total_positive - before_positive)
    fidelity = 1.0 - cost.min(axis=1) / np.maximum(total, 1e-12)
    share = total / np.maximum(textured_matrix.sum(axis=1), 1)
    # A row without a clearly sharper side is a tie: the midline is the honest cut, and the
    # seam there is reported for review instead of being invented from noise.
    meaningful = (share > margin) & (fidelity >= SPLIT_FIDELITY)
    cuts = np.full(matrix.shape[0], int(round(middle)), dtype=int)
    if np.any(meaningful):
        bias = TIE_WEIGHT * (total_positive + total_negative) * (
            np.abs(np.arange(length)[None, :] - middle) / max(middle, 1))
        cuts[meaningful] = np.argmin((cost + bias)[meaningful], axis=1)
    if len(cuts) >= 5:
        cuts = ndi.median_filter(cuts, size=7, mode="nearest")
    tie_rows = ~meaningful
    # Review band: a narrow strip along the seam where no clear winner existed.
    reach = max(6, int(0.06 * length))
    around = np.abs(np.arange(length)[None, :] - cuts[:, None]) <= reach
    band = (around & tie_rows[:, None]) if vertical else (around & tie_rows[:, None]).T
    uncertain = clean_mask(band, both)
    tie_fraction = float(np.count_nonzero(tie_rows)) / max(1, len(tie_rows))
    return (x0, y0, x1, y1, cuts, uncertain, tie_fraction, spans_of(tie_rows),
            both, first_support, second_support, matrix)


def spans_of(flags: np.ndarray) -> list[list[int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(np.int8), 0]) != 0)
    return [[int(start), int(stop)] for start, stop in zip(edges[::2], edges[1::2])]


def second_wins(first: Frame, second: Frame, dx: int, dy: int, side: str, margin: float) -> tuple:
    (x0, y0, x1, y1, cuts, uncertain, tie_fraction, tie_spans,
     both, first_support, second_support, _) = seam_cuts(first, second, dx, dy, side, margin)
    height, width = y1 - y0, x1 - x0
    vertical = side in {"right", "left"}
    wins = np.zeros((height, width), dtype=bool)
    for index in range(height if vertical else width):
        cut = int(cuts[index])
        if vertical:
            wins[index, cut:] = side == "right"
            wins[index, :cut] = side == "left"
        else:
            wins[cut:, index] = side == "below"
            wins[:cut, index] = side == "above"
    if vertical:
        split = (np.arange(width)[None, :] >= width // 2 if side == "right"
                 else np.arange(width)[None, :] < width // 2)
        split = np.broadcast_to(split, (height, width))
    else:
        split = (np.arange(height)[:, None] >= height // 2 if side == "below"
                 else np.arange(height)[:, None] < height // 2)
        split = np.broadcast_to(split, (height, width))
    # Where neither frame is clearly sharper, the frame nearer its own side keeps the pixel.
    wins = np.where(uncertain, split, wins)
    wins &= both
    wins[second_support & ~first_support] = True
    wins[first_support & ~second_support] = False
    return (x0, y0, x1, y1, wins, uncertain & both, tie_fraction, tie_spans)


def brightness_offset(first: Frame, second: Frame, dx: int, dy: int) -> float:
    x0, y0, x1, y1 = rectangle(first.shape, second.shape, dx, dy)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    both = first.support[y0:y1, x0:x1] & second.support[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    if np.count_nonzero(both) < 256:
        return 0.0
    a = luminance(first.rgb[y0:y1, x0:x1])[both]
    b = luminance(second.rgb[y0 - dy:y1 - dy, x0 - dx:x1 - dx])[both]
    delta = float(np.median(a) - np.median(b))
    return delta if 0.004 < abs(delta) < 0.25 else 0.0


def compose(first: Frame, second: Frame, dx: int, dy: int, side: str, *, focus_margin: float,
            match_brightness: bool) -> Composite:
    left, top = min(0, dx), min(0, dy)
    width = max(first.shape[1], dx + second.shape[1]) - left
    height = max(first.shape[0], dy + second.shape[0]) - top
    if width * height > MAX_PIXELS:
        raise Limits(
            "The joined image would exceed 16 megapixels. Crop each frame nearer the overlap and join again.",
            "first",
        )
    delta = brightness_offset(first, second, dx, dy) if match_brightness else 0.0
    second_rgb = second.rgb
    if delta:
        second_rgb = np.clip(np.rint(second.rgb.astype(np.float32) + delta * 255.0), 0, 255).astype(np.uint8)
    ax, ay, bx, by = -left, -top, dx - left, dy - top
    support_first = np.zeros((height, width), dtype=bool)
    support_second = np.zeros((height, width), dtype=bool)
    support_first[ay:ay + first.shape[0], ax:ax + first.shape[1]] = first.support
    support_second[by:by + second.shape[0], bx:bx + second.shape[1]] = second.support
    x0, y0, x1, y1, wins, uncertain, tie_fraction, tie_spans = second_wins(
        first, second, dx, dy, side, focus_margin)
    wins_canvas = np.zeros((height, width), dtype=bool)
    wins_canvas[y0 - top:y1 - top, x0 - left:x1 - left] = wins
    uncertain_canvas = np.zeros((height, width), dtype=bool)
    uncertain_canvas[y0 - top:y1 - top, x0 - left:x1 - left] = uncertain
    owned_second = (support_second & ~support_first) | (support_second & support_first & wins_canvas)
    owned_first = support_first & ~owned_second
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[owned_first] = FIRST
    mask[owned_second] = SECOND
    mask[uncertain_canvas & owned_first] = UNSURE_FIRST
    mask[uncertain_canvas & owned_second] = UNSURE_SECOND
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[ay:ay + first.shape[0], ax:ax + first.shape[1], :3] = first.rgb
    rgba[by:by + second.shape[0], bx:bx + second.shape[1], :3] = second_rgb
    if np.any(owned_second):
        ys, xs = np.nonzero(owned_second)
        rgba[ys, xs, :3] = second_rgb[ys - by, xs - bx]
    rgba[..., 3][owned_first | owned_second] = 255
    seam = seam_report(first, second, dx, dy, side, focus_margin, delta, mask, uncertain_canvas,
                       x0, y0, x1, y1, tie_fraction, tie_spans)
    return Composite(rgba, mask, seam)


def row_runs(row: np.ndarray, offset: int) -> list[list]:
    codes = np.where((row == FIRST) | (row == UNSURE_FIRST), 1,
                     np.where((row == SECOND) | (row == UNSURE_SECOND), 2, 0))
    edges = np.flatnonzero(np.diff(np.r_[0, codes, 0]) != 0)
    return [
        [int(start + offset), int(stop + offset), "first" if codes[start] == 1 else "second"]
        for start, stop in zip(edges[::2], edges[1::2]) if codes[start]
    ]


def uncertain_regions(uncertain: np.ndarray, x0: int, y0: int) -> list[dict]:
    labels, count = ndi.label(uncertain, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return []
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    floor = max(64, int(0.0004 * np.count_nonzero(uncertain)))
    regions = []
    for label, bounds in enumerate(ndi.find_objects(labels), 1):
        if bounds is None or sizes[label] < floor:
            continue
        regions.append({
            "x": int(bounds[1].start + x0), "y": int(bounds[0].start + y0),
            "width": int(bounds[1].stop - bounds[1].start),
            "height": int(bounds[0].stop - bounds[0].start), "pixels": int(sizes[label]),
        })
    regions.sort(key=lambda item: -item["pixels"])
    return regions[:24]


def seam_report(first, second, dx, dy, side, margin, delta, mask, uncertain, x0, y0, x1, y1,
                tie_fraction: float = 0.0, tie_spans: list | None = None) -> dict:
    height, width = mask.shape
    groups = []
    for row in range(y0, y1):
        clipped = [
            [max(start, x0), min(stop, x1), owner]
            for start, stop, owner in row_runs(mask[row], 0)
            if min(stop, x1) > max(start, x0)
        ]
        if groups and groups[-1]["runs"] == clipped:
            groups[-1]["last_row"] = row
        else:
            groups.append({"first_row": row, "last_row": row, "runs": clipped})
    total = int(np.count_nonzero(uncertain))
    area = max(1, (x1 - x0) * (y1 - y0))
    return {
        "version": SEAM_VERSION,
        "side": side,
        "orientation": "vertical" if side in {"right", "left"} else "horizontal",
        "canvas": {"width": width, "height": height},
        "overlap": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
        "placement": {"first": [0, 0], "second": [dx, dy]},
        "sources": {
            "first": {"capture_id": first.capture_id, "title": first.title},
            "second": {"capture_id": second.capture_id, "title": second.title},
        },
        "uncertain_pixels": total,
        "uncertain_fraction": round(total / area, 4),
        "tie_rows_fraction": round(tie_fraction, 4),
        "tie_spans": (tie_spans or [])[:40],
        "uncertain_regions": uncertain_regions(uncertain, x0, y0),
        "focus_margin": margin,
        "brightness_offset": round(delta, 5),
        "row_groups": groups[:4000],
        "row_groups_truncated": len(groups) > 4000,
        "notice": (
            "Overlap pixels come whole from one frame and are never averaged, so a vessel edge that is "
            "sharp in one frame is kept instead of becoming a ghost edge. Magenta areas had no clear "
            "sharpness winner and follow the frame nearer that side."
        ),
    }


def scaled(composite: Composite, longest: int) -> tuple[np.ndarray, np.ndarray]:
    rgba, mask = composite.rgba, composite.mask
    factor = max(1, int(ceil(max(rgba.shape[:2]) / longest)))
    image, coded = Image.fromarray(rgba, "RGBA"), Image.fromarray(mask, "L")
    if factor > 1:
        size = (max(1, rgba.shape[1] // factor), max(1, rgba.shape[0] // factor))
        image = image.resize(size, Image.Resampling.LANCZOS)
        coded = coded.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image), np.asarray(coded)


def preview_image(composite: Composite) -> np.ndarray:
    return scaled(composite, PREVIEW_SIDE)[0]


def seam_visual(composite: Composite, longest: int = PREVIEW_SIDE, *, scale_note: str = "") -> np.ndarray:
    rgba, mask = scaled(composite, longest)
    base = rgba[..., :3].astype(np.float32) * 0.55
    first = (mask == FIRST) | (mask == UNSURE_FIRST)
    second = (mask == SECOND) | (mask == UNSURE_SECOND)
    uncertain = (mask == UNSURE_FIRST) | (mask == UNSURE_SECOND)
    base[first] += np.array([24, 74, 86], dtype=np.float32)
    base[second] += np.array([118, 82, 38], dtype=np.float32)
    base[uncertain] = base[uncertain] * 0.30 + np.array([228, 72, 150], dtype=np.float32) * 0.70
    base[rgba[..., 3] == 0] = 30
    visual = np.clip(np.rint(base), 0, 255).astype(np.uint8)
    image = Image.fromarray(visual)
    painter = ImageDraw.Draw(image)
    sources = composite.seam["sources"]
    lines = [
        "SEAM SOURCE MAP",
        "blue   " + sources["first"]["title"][:54],
        "amber  " + sources["second"]["title"][:54],
        "pink   uncertain " + f"{composite.seam['uncertain_fraction'] * 100:.1f}% of overlap",
    ]
    if scale_note:
        lines.append(scale_note)
    for index, line in enumerate(lines):
        painter.text((10, 8 + index * 16), line, fill=(248, 240, 207),
                     stroke_width=1, stroke_fill=(30, 35, 29))
    return visual


def preview_query(placement: Placement, candidate: Candidate, seam, *, view: str = "image") -> str:
    from urllib.parse import urlencode
    query = {
        "first": placement.first.capture_id,
        "second": placement.second.capture_id,
        "side": placement.side,
        "dx": candidate.dx,
        "dy": candidate.dy,
        "view": view,
    }
    if seam:
        settings = seam_settings(seam)
        query["focus_margin"] = settings["focus_margin"]
        query["match_brightness"] = "1" if settings["match_brightness"] else "0"
    return "/api/stitches/preview.png?" + urlencode(query)


def preview_request(args) -> dict:
    """Read a join preview from a GET query so the browser can show it in an image tag."""
    body = {
        "first": args.get("first"),
        "second": args.get("second"),
        "side": args.get("side"),
    }
    if args.get("dx") is not None and args.get("dy") is not None:
        body["alignment"] = {"dx": args.get("dx", type=float), "dy": args.get("dy", type=float)}
    if args.get("focus_margin") is not None or args.get("match_brightness") is not None:
        body["seam"] = {
            "focus_margin": args.get("focus_margin", 0.12, type=float),
            "match_brightness": args.get("match_brightness", "1") == "1",
        }
    return body


def preview(db: Database, body: dict) -> tuple[Placement, Candidate, Composite]:
    """Build a small preview. The saved image always recomputes at the full size."""
    placement = analyze(db, body)
    chosen = resolve(placement, body.get("alignment"))
    settings = seam_settings(body.get("seam", {}))
    factor = max(1, int(ceil(max(placement.first.shape + placement.second.shape) / 900)))
    first = reduced_frame(placement.first, factor)
    second = reduced_frame(placement.second, factor)
    reduced = Candidate(int(round(chosen.dx / factor)), int(round(chosen.dy / factor)),
                        chosen.score, 0, 0, chosen.ambiguous, chosen.reason)
    return placement, chosen, compose(first, second, reduced.dx, reduced.dy, placement.side, **settings)


def create(db: Database, raw: dict) -> dict:
    placement = analyze(db, raw)
    chosen = resolve(placement, raw.get("alignment"))
    settings = seam_settings(raw.get("seam", {}))
    composite = compose(placement.first, placement.second, chosen.dx, chosen.dy,
                        placement.side, **settings)
    title = string(raw.get("title"), "joined image name", 90) or join_name(placement)
    png = encode_png(composite.rgba)
    image = read_image(png)
    payload = capture_payload(image, title, title + ".png")
    payload["metadata"]["joined"] = {
        "side": placement.side,
        "offset": [chosen.dx, chosen.dy],
        "sources": [placement.first.capture_id, placement.second.capture_id],
        "seam_version": SEAM_VERSION,
        "ambiguous_at_save": chosen.ambiguous,
        "uncertain_fraction": composite.seam["uncertain_fraction"],
    }
    values = observation_fields(
        {key: raw.get(key) for key in ("title", "material", "lot", "surface")}, title,
    )
    with db.transaction() as conn:
        capture_id = db.put_capture(conn, payload)
        inspection = db.create_inspection(conn, capture_id, values)
        stitch_id = db.add_stitch(conn, {
            "first_capture_id": placement.first.capture_id,
            "second_capture_id": placement.second.capture_id,
            "capture_id": capture_id,
            "inspection_id": inspection["id"],
            "side": placement.side,
            "resolution": "confirmed" if chosen.ambiguous else "automatic",
            "alignment": {
                **chosen.json(),
                "candidates": [item.json() for item in placement.candidates],
            },
            "seam": composite.seam,
            "seam_mask": encode_png(composite.mask),
            "seam_preview": encode_png(seam_visual(composite)),
        })
        state = dict(inspection["state"])
        state["join"] = {
            "stitch": stitch_id,
            "side": placement.side,
            "offset": [chosen.dx, chosen.dy],
            "ambiguous_at_save": chosen.ambiguous,
            "sources": [
                {"capture_id": placement.first.capture_id, "title": placement.first.title},
                {"capture_id": placement.second.capture_id, "title": placement.second.title},
            ],
            "uncertain_fraction": composite.seam["uncertain_fraction"],
            "uncertain_regions": len(composite.seam["uncertain_regions"]),
        }
        conn.execute(
            "UPDATE inspections SET state = ?, revision = 2, updated_at = ? WHERE id = ?",
            (encode(state), timestamp(), inspection["id"]),
        )
        db.event(conn, inspection["id"], 2, "stitch_saved", {
            "stitch_id": stitch_id,
            "sources": [placement.first.capture_id, placement.second.capture_id],
            "offset": [chosen.dx, chosen.dy],
            "resolution": "confirmed" if chosen.ambiguous else "automatic",
        })
    return describe_stitch(db.stitch(stitch_id))


def join_name(placement: Placement) -> str:
    return (placement.first.title[:38] + " + " + placement.second.title[:38])[:90]


def working_image(db: Database, stitch_id: str, *, view: str) -> tuple[str, bytes]:
    """The saved working raster with a scale bar and the provenance printed on it."""
    from .quality import prepare_download_image
    from .calibration_preview import positive_tick
    from .workspaces import frame_for

    record = db.stitch(stitch_id)
    inspection = db.inspection(record["inspection_id"])
    frame = frame_for(db, inspection)
    raster = frame.raster
    image = prepare_download_image(raster, view).convert("RGB")
    painter = ImageDraw.Draw(image)
    plane = raster.plane
    tick = positive_tick(plane.width * (plane.mm_x or 1) * 0.16)
    step = tick / (plane.mm_x or 1)
    length = max(8, int(round(step)))
    x, y = 14, image.height - 26
    label = f"{tick:g} {plane.unit}"
    painter.rectangle((x - 5, y - 24, x + max(length, len(label) * 8) + 6, y + 8), fill=(38, 43, 36))
    painter.text((x + 1, y - 21), label, fill=(247, 232, 184))
    painter.line((x, y, x + length, y), fill=(247, 232, 184), width=2)
    painter.line((x, y - 5, x, y + 4), fill=(247, 232, 184), width=2)
    painter.line((x + length, y - 5, x + length, y + 4), fill=(247, 232, 184), width=2)
    seam = record["seam"]
    detail = (
        f"{seam['sources']['first']['title'][:34]} + {seam['sources']['second']['title'][:34]}"
        f"  ·  offset {tuple(seam['placement']['second'])}"
        f"  ·  seam uncertain {seam['uncertain_fraction'] * 100:.1f}%"
        f"  ·  revision {inspection['revision']}"
    )
    painter.rectangle((8, 6, min(image.width - 8, 8 + len(detail) * 6 + 12), 26), fill=(38, 43, 36))
    painter.text((13, 10), detail, fill=(247, 232, 184))
    buffer = BytesIO()
    image.save(buffer, format="PNG", compress_level=4)
    safe = inspection["title"].replace("/", "-").replace("\\", "-")[:60] or "joined"
    return f"endgrain-joined-{safe}-r{inspection['revision']}.png", buffer.getvalue()


def describe_stitch(record: dict) -> dict:
    item = dict(record)
    root = "/api/stitches/" + record["id"]
    inspection = "/api/inspections/" + record["inspection_id"]
    item["links"] = {
        "seam": root + "/seam.png",
        "seam_json": root + "/seam.json",
        "working": inspection + "/working.png",
        "download": root + "/working.png?download=1",
        "inspection": inspection,
        "first_original": "/api/captures/" + record["first_capture_id"] + "/original",
        "second_original": "/api/captures/" + record["second_capture_id"] + "/original",
        "first_thumb": "/api/captures/" + record["first_capture_id"] + "/thumb.png",
        "second_thumb": "/api/captures/" + record["second_capture_id"] + "/thumb.png",
    }
    return item
