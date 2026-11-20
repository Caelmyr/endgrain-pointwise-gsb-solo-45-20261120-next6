from dataclasses import dataclass
from functools import lru_cache
from math import cos, pi, sin

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi

from .captures import capture_payload, observation_fields
from .geometry import plane_for
from .imaging import encode_png, read_image


@dataclass(frozen=True)
class Example:
    slug: str
    title: str
    species: str
    surface: str
    seed: int
    color: tuple[int, int, int]
    width: int
    height: int
    ring_gap: float
    pores: int
    saw_marks: float
    shape: str
    hint: str


MATERIALS = (
    Example("oak-17", "White oak · lot 17", "White oak", "sanded", 137, (192, 141, 93), 960, 700, 19, 670, 0.0, "disc",
            "End-grain disc with distinct rings, scattered pores and a branching check."),
    Example("walnut-08", "Black walnut · lot 08", "Black walnut", "polished", 448, (130, 93, 73), 940, 670, 18, 340, 0.0, "board",
            "Planed board with a dark heartwood band, pores and a bright reflection."),
    Example("ash-04", "European ash · lot 04", "European ash", "sawn", 804, (202, 161, 116), 930, 690, 21, 590, 8.0, "board",
            "Sawn block with localized periodic chatter and an edge check."),
)


@dataclass(frozen=True)
class Landmarks:
    x: np.ndarray
    y: np.ndarray
    radial: np.ndarray
    direction: np.ndarray
    body: np.ndarray
    edge: np.ndarray


def landmarks(example: Example) -> Landmarks:
    h, w = example.height, example.width
    y, x = np.mgrid[:h, :w].astype(np.float32)
    center_x, center_y = w * 0.47, h * 0.55
    a = (x - center_x) + 0.14 * (y - center_y)
    b = (y - center_y) * 1.125 + 0.05 * (x - center_x)
    radial = np.hypot(a, b)
    direction = np.arctan2(b, a)
    if example.shape == "disc":
        raw = np.hypot((x - w * 0.50) * 0.89, (y - h * 0.50) * 1.03)
        radius = h * (0.395 + 0.016 * np.cos(direction * 5 + 0.6) + 0.008 * np.sin(direction * 11))
        edge = radius - raw
    else:
        left = 76 + 0.075 * (y - h / 2)
        right = w - 63 + 0.045 * (y - h / 2)
        top = 90 - 0.04 * (x - w / 2)
        bottom = h - 85 + 0.035 * (x - w / 2)
        edge = np.minimum(np.minimum(x - left, right - x), np.minimum(y - top, bottom - y))
    return Landmarks(x, y, radial, direction, edge > 0, edge)


def microroughness(rng: np.random.Generator, shape: tuple[int, int], breadth: float) -> np.ndarray:
    rows, cols = shape
    seed = rng.normal(0, 1, shape).astype(np.float32)
    fine = ndi.gaussian_filter(seed, sigma=(0.5, 0.5))
    narrow = ndi.gaussian_filter(seed, sigma=(1.5 + breadth, 0.4))
    coarse = ndi.gaussian_filter(seed, sigma=(7.0, 9 + 4 * breadth))
    norm = lambda row: row / max(float(row.std()), 0.00001)
    return 1.30 * norm(fine) + 1.85 * norm(narrow) + 2.1 * norm(coarse)


def ring_signal(example: Example, where: Landmarks) -> tuple[np.ndarray, np.ndarray]:
    x, y, theta, radial = where.x, where.y, where.direction, where.radial
    phase = (
        radial
        + 6.5 * np.sin(theta * 3.0 + 0.013 * radial)
        + 2.4 * np.sin(theta * 11.0 - 0.01 * radial)
        + 4.0 * np.sin(0.017 * radial + 0.32)
        + 2.1 * np.sin(x / 52 + y / 66)
    ) / example.ring_gap
    phase += np.maximum(radial - 142, 0) * 0.0019
    phase += np.maximum(radial - 225, 0) * 0.0024
    index = phase % 1
    dark = -23 * np.exp(-np.minimum(index, 1 - index) ** 2 / 0.006)
    broad = 9 * np.sin((index - 0.29) * 2 * pi)
    if example.slug.startswith("walnut"):
        dark *= 0.72
        broad *= 0.53
    ray = (np.maximum(np.cos(49 * theta + 0.005 * radial), 0.0) ** 22)
    smaller = (np.maximum(np.cos(102 * theta - 0.011 * radial + 0.6), 0.0) ** 28)
    ray_fade = np.clip(radial / 80, 0, 1) * (0.85 + 0.15 * np.sin(radial / 40))
    anatomy = dark + broad + ray_fade * (5 * ray + 4 * smaller)
    return anatomy, index


def render_vessels(
    rng: np.random.Generator, where: Landmarks, ring_index: np.ndarray, count: int,
) -> np.ndarray:
    h, w = ring_index.shape
    ready = where.body & (where.edge >= 12)
    early = ready & ((ring_index < 0.33) | (ring_index > 0.92))
    spots = np.argwhere(early)
    diffuse = np.argwhere(ready)
    pores = Image.new("L", (w, h))
    brush = ImageDraw.Draw(pores)
    groups = ((spots, int(count * 0.76)), (diffuse, count - int(count * 0.76)))
    for distribution, quantity in groups:
        indices = rng.integers(0, len(distribution), size=quantity)
        for y, x in distribution[indices]:
            center = float(rng.uniform(1.4, 3.8))
            stretch = float(rng.uniform(0.60, 1.31))
            darkness = int(rng.uniform(70, 205))
            brush.ellipse((x - center, y - center * stretch, x + center, y + center * stretch), fill=darkness)
            if center > 2.3:
                brush.point((x + 1, y - 1), fill=223)
    return np.asarray(pores.filter(ImageFilter.GaussianBlur(0.60)), dtype=np.float32) * 0.48


def broken_path(
    rng: np.random.Generator,
    start: tuple[float, float],
    finish: tuple[float, float],
    *,
    points: int,
    wander: float,
) -> list[tuple[float, float]]:
    x, y = np.asarray(start)
    u, v = np.asarray(finish)
    sideways = np.array([y - v, u - x])
    sideways /= np.linalg.norm(sideways)
    result = []
    for index, p in enumerate(np.linspace(0, 1, points)):
        across = sin(p * pi) * rng.normal(0, wander)
        approach = (x + p * (u - x), y + p * (v - y))
        result.append((float(approach[0] + sideways[0] * across), float(approach[1] + sideways[1] * across)))
    return result


def render_checks(rng: np.random.Generator, example: Example, where: Landmarks) -> np.ndarray:
    h, w = where.body.shape
    crack_map = Image.new("L", (w, h))
    brush = ImageDraw.Draw(crack_map)
    if example.slug.startswith("oak"):
        main = broken_path(rng, (w * 0.58, h * 0.51), (w * 0.826, h * 0.738), points=17, wander=5.7)
        branches = [
            broken_path(rng, main[7], (w * 0.665, h * 0.77), points=10, wander=3.5),
            broken_path(rng, main[10], (w * 0.798, h * 0.574), points=9, wander=3.0),
        ]
    elif example.slug.startswith("walnut"):
        main = broken_path(rng, (w * 0.411, h * 0.53), (w * 0.50, h * 0.72), points=14, wander=3.4)
        branches = []
    else:
        main = broken_path(rng, (w * 0.63, h * 0.39), (w * 0.985, h * 0.46), points=18, wander=4.4)
        branches = [broken_path(rng, tuple(main[8]), (w * 0.82, h * 0.52), points=9, wander=3.5)]
    for start, end in zip(main, main[1:]):
        brush.line([start, end], width=int(rng.integers(3, 7)), fill=int(rng.integers(185, 238)))
    for curve in branches:
        for index, (start, end) in enumerate(zip(curve, curve[1:])):
            brush.line([start, end], width=max(2, 5 - index // 3), fill=190 - 3 * index)
    blurred = np.asarray(crack_map.filter(ImageFilter.GaussianBlur(1.6)), dtype=np.float32)
    detailed = np.asarray(crack_map.filter(ImageFilter.GaussianBlur(0.48)), dtype=np.float32)
    return np.where(where.body, detailed * 0.48 + blurred * 0.12, 0)


def renderer(example: Example) -> np.ndarray:
    rng = np.random.default_rng(example.seed)
    where = landmarks(example)
    x, y, theta = where.x, where.y, where.direction
    w, h = example.width, example.height
    grain, phase = ring_signal(example, where)
    field = 10.0 * (1 - x / w) + 4.0 * np.cos(y * 2 * pi / (h * 1.68))
    glow = np.exp(-((x - 0.67 * w) / (0.21 * w)) ** 2 - ((y - 0.28 * h) / (0.33 * h)) ** 2)
    hard_edge = np.clip((19 - where.edge) / 16, 0, 1)
    rough = microroughness(rng, where.body.shape, 1.0 if example.shape == "disc" else 2.6)
    anatomy = grain + field * 1.4 + rough * 1.65
    anatomy -= 37 * hard_edge ** 2
    anatomy -= render_vessels(rng, where, phase, example.pores)
    anatomy -= render_checks(rng, example, where)
    if example.saw_marks:
        sway = 0.13 * x + 7 * np.sin(x / 133)
        stripe = np.sin((y + sway) * 2 * pi / 8.5) + 0.31 * np.sin((y + sway) * 2 * pi / 4.2)
        domain = 0.3 + 0.7 * np.exp(-((x - w * 0.72) / (w * 0.24)) ** 2)
        anatomy += example.saw_marks * stripe * domain
    if example.slug.startswith("walnut"):
        heart = 13 * np.tanh((where.radial - 184 - 8 * np.sin(theta * 3)) / 32)
        anatomy += heart + 11 * glow
    else:
        anatomy += 11 * glow
    wood = np.stack([
        example.color[0] + anatomy,
        example.color[1] + 0.88 * anatomy,
        example.color[2] + 0.71 * anatomy,
    ], axis=-1)
    if example.slug.startswith("walnut"):
        specular = 127 * np.exp(-((x - w * 0.695) / (w * 0.057)) ** 2 - ((y - h * 0.305) / (h * 0.077)) ** 2)
        wood += specular[..., None]
    if example.shape == "disc":
        bark = -17 + 13 * np.sin(theta * 65) + 5 * np.sin(theta * 134.0 + where.radial / 3.9)
        wood += np.where((where.edge > 0) & (where.edge < 9), bark, 0)[..., None]
    table_shadow = ndi.gaussian_filter(where.body.astype(np.float32), 13) * 12
    background = 52 + 7 * np.cos(x / 243 + 0.2) + microroughness(rng, where.body.shape, 1) * 0.52 - table_shadow
    table = np.stack([background * 1.04, background * 1.045, background * 1.00], axis=-1)
    output = np.where(where.body[..., None], wood, table)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def references(example: Example) -> np.ndarray:
    w, h = example.width, example.height
    y, x = np.mgrid[:h, :w].astype(np.float32)
    soft = 26 * (1 - x / w) - 15 * ((y / h - 0.42) ** 2) + 5 * np.cos(y / 149)
    spot = 25 * np.exp(-((x - w * 0.77) / (w * 0.17)) ** 2 - ((y - h * 0.22) / (h * 0.23)) ** 2)
    gray = np.clip(np.rint(168 + soft + spot), 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    alpha = np.full((h, w), 255, dtype=np.uint8)
    # A localized specular reflection on the board: blown-out, unmeasurable.
    glare = ((x - w * 0.70) / (w * 0.045)) ** 2 + ((y - h * 0.26) / (h * 0.05)) ** 2 <= 1
    rgb[glare] = 255
    # A handful of small transparent dead spots: small enough to interpolate.
    for cx, cy, radius in ((0.30, 0.34, 7), (0.345, 0.39, 5), (0.47, 0.20, 4)):
        hole = (x - w * cx) ** 2 + (y - h * cy) ** 2 <= radius ** 2
        alpha[hole] = 0
        rgb[hole] = 0
    # One broad damaged region with too little board around its centre to trust
    # a fill: the rims blend smoothly while the middle stays on the original.
    damaged = ((x - w * 0.60) / (w * 0.105)) ** 2 + ((y - h * 0.70) / (h * 0.135)) ** 2 <= 1
    alpha[damaged] = 0
    rgb[damaged] = 0
    return np.dstack((rgb, alpha))


def make_region(example: Example, width: int, height: int) -> list[list[float]]:
    if example.shape == "disc":
        return [
            [
                float(round(width * 0.50 + width * 0.315 * cos(index * 2 * pi / 24), 1)),
                float(round(height * 0.50 + height * 0.354 * sin(index * 2 * pi / 24), 1)),
            ]
            for index in range(24)
        ]
    return [
        [round(width * sx, 1), round(height * sy, 1)]
        for sx, sy in ((0.115, 0.17), (0.91, 0.147), (0.93, 0.84), (0.13, 0.82))
    ]


def values_for(example: Example) -> dict:
    form = {
        "title": example.title,
        "material": example.species,
        "surface": example.surface,
        "lot": example.slug[-2:].upper(),
    }
    data = observation_fields(form, example.title)
    cal = {
        "mode": "ruler", "a": [float(example.width * 0.1), float(example.height * 0.9)],
        "b": [float(example.width * 0.35), float(example.height * 0.9)], "length_mm": 40.0,
    }
    plane = plane_for(example.width, example.height, cal)
    data["state"] = {"calibration": cal, "region": make_region(example, plane.width, plane.height), "lighting": None}
    return data


@lru_cache(maxsize=1)
def demo_payloads() -> tuple[list[tuple[dict, dict]], list[dict]]:
    samples, flatfields = [], []
    for example in MATERIALS:
        name = example.slug + "-synthetic.png"
        photo = read_image(encode_png(renderer(example)))
        record = capture_payload(photo, example.title, name, synthetic=True)
        record["metadata"]["demo_hint"] = example.hint
        samples.append((record, values_for(example)))
        if example.slug != "oak-17":
            continue
        rgb = references(example)
        flat = read_image(encode_png(rgb))
        control = capture_payload(flat, "White oak · calibration flat", "oak-flat-synthetic.png", synthetic=True, source_type="reference")
        flatfields.append(control)
    return samples, flatfields
