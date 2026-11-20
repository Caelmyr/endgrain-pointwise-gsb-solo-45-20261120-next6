from dataclasses import dataclass
from io import BytesIO
from math import hypot

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi

from .calibration_preview import positive_tick, readings
from .database import Database
from .geometry import mask_bounds
from .imaging import encode_png, symmetric_overlay
from .workspaces import Frame, frame_for


@dataclass(frozen=True)
class Plate:
    rgb: Image.Image
    offset: tuple[int, int]
    scale: float
    original: tuple[int, int]


def contain(
    squares: np.ndarray, size: tuple[int, int], background: tuple[int, int, int] = (44, 47, 43),
) -> Plate:
    photo = Image.fromarray(squares)
    resized = ImageOps.contain(photo, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    position = ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2)
    canvas.paste(resized, position)
    return Plate(canvas, position, resized.width / photo.width, photo.size)


def canvas_position(point: list[float], layout: Plate) -> tuple[float, float]:
    x, y = point
    return (
        layout.offset[0] + (x + 0.5) * layout.scale,
        layout.offset[1] + (y + 0.5) * layout.scale,
    )


def sized_label_rgb(
    painting: ImageDraw.ImageDraw,
    pos: tuple[float, float],
    label: str,
    colors: tuple[tuple[int, int, int], tuple[int, int, int]],
    available: int,
) -> None:
    x, y = pos
    label_width = painting.textlength(label)
    if label_width > available:
        while label and painting.textlength(label + "…") > available:
            label = label[:-1]
        label += "…"
        label_width = painting.textlength(label)
    fill, text = colors
    painting.rectangle((x, y, x + label_width + 8, y + 15), fill=fill)
    painting.text((x + 4, y + 1), label, fill=text)


def calibration_positions(working: Frame) -> tuple[list, str]:
    cal = working.raster.plane.calibration
    if cal["mode"] == "ruler":
        return [cal["a"], cal["b"]], f"Ruler · {cal['length_mm']:g} mm"
    if cal["mode"] == "quadrilateral":
        return cal["corners"], f"Rectangle · {cal['width_mm']:g} × {cal['height_mm']:g} mm"
    return [], "Pixel scale"


def working_anchor_points(frame: Frame) -> tuple[list[list[float]], str]:
    """Calibration anchors expressed in measuring-plane coordinates."""
    plane = frame.raster.plane
    calibration = plane.calibration
    if calibration["mode"] == "ruler":
        known = np.asarray([calibration["a"], calibration["b"]], dtype=np.float64)
        return [list(point) for point in plane.map_plane(known)], f"Ruler · {calibration['length_mm']:g} mm"
    if calibration["mode"] == "quadrilateral":
        width, height = plane.width - 0.5, plane.height - 0.5
        return (
            [[-0.5, -0.5], [float(width), -0.5], [float(width), float(height)], [-0.5, float(height)]],
            f"Rectangle · {calibration['width_mm']:g} × {calibration['height_mm']:g} mm",
        )
    return [], "Pixel scale"


def source_panel(frame: Frame, layout_size: tuple[int, int]) -> Image.Image:
    source = symmetric_overlay(frame.source, [(~frame.source_support, (125, 89, 82), 0.72)])
    plate = contain(source, layout_size)
    painter = ImageDraw.Draw(plate.rgb)
    anchors, label = calibration_positions(frame)
    points = [canvas_position(value, plate) for value in anchors]
    accent = (248, 217, 154)
    if points:
        path = points + points[:1] if len(points) == 4 else points
        painter.line(path, fill=accent, width=2)
    for index, (x, y) in enumerate(points, 1):
        painter.ellipse((x - 4, y - 4, x + 4, y + 4), fill=accent, outline=(34, 39, 33))
        pos = (min(x + 5, plate.rgb.width - 18), min(y + 5, plate.rgb.height - 16))
        sized_label_rgb(painter, pos, str(index), ((42, 49, 40), (239, 223, 177)), 22)
    sized_label_rgb(painter, (7, 6), label, ((42, 49, 40), (239, 223, 177)), plate.rgb.width - 18)
    return plate.rgb


def draw_region(plate: Plate, frame: Frame) -> None:
    points = frame.inspection["state"]["region"]
    if not points:
        return
    color = (221, 211, 156)
    entries = [canvas_position(point, plate) for point in points]
    painter = ImageDraw.Draw(plate.rgb)
    dashed_loop(painter, entries, color)


def dashed_loop(painter: ImageDraw.ImageDraw, entries: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
    for start, stop in zip(entries, entries[1:] + entries[:1], strict=True):
        dx, dy = stop[0] - start[0], stop[1] - start[1]
        span = hypot(dx, dy)
        if not span:
            continue
        for piece in np.arange(0, span, 10.0):
            a, b = piece / span, min(span, piece + 5.0) / span
            painter.line(
                ((start[0] + a * dx, start[1] + a * dy), (start[0] + b * dx, start[1] + b * dy)),
                fill=color, width=2,
            )


def bounded_crop(
    rgb: np.ndarray, mask: np.ndarray, longest: int = 1600, padding: int = 2,
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    height, width = mask.shape
    bounds = mask_bounds(mask)
    if bounds is None:
        return rgb[:2, :2], (0, 0, width, height), 1.0
    x0, y0, x1, y1 = bounds
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1, y1 = min(width, x1 + padding), min(height, y1 + padding)
    piece = rgb[y0:y1, x0:x1]
    scale = min(1.0, longest / max(piece.shape[0], piece.shape[1]))
    if scale < 1.0:
        size = (max(1, round(piece.shape[1] * scale)), max(1, round(piece.shape[0] * scale)))
        piece = np.asarray(Image.fromarray(piece).resize(size, Image.Resampling.LANCZOS))
    return piece, (x0, y0, x1, y1), scale


def loaded_cutout(cut: bytes) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(BytesIO(cut)) as source:
        rgba = np.asarray(source.convert("RGBA"))
    alpha = rgba[..., 3]
    colour = np.zeros_like(rgba[..., :3])
    showing = alpha > 0
    colour[showing] = np.rint(
        rgba[..., :3][showing].astype(np.float32) * (255.0 / alpha[showing, None])
    ).astype(np.uint8)
    return colour, alpha


def external_plate(cut: bytes | None, working: Image.Image, caption: str) -> bytes:
    layout_size = (637, 471)
    original = external_source_panel(cut, layout_size, caption)
    projected = ImageOps.contain(working.convert("RGB"), layout_size, Image.Resampling.LANCZOS)
    layout = Image.new("RGB", (1308, 557), (240, 237, 223))
    painter = ImageDraw.Draw(layout)
    tiles = (("SOURCE / REGION ONLY", original), ("WORKING / MEASURING SUPPORT", projected))
    for index, (title, image) in enumerate(tiles):
        start_x = 11 + index * (layout_size[0] + 12)
        layout.paste(image, (start_x, 55))
        painter.text((start_x, 32), title, fill=(66, 79, 53))
    return encode_png(np.asarray(layout))


def external_source_panel(cut: bytes | None, layout_size: tuple[int, int], caption: str) -> Image.Image:
    canvas = Image.new("RGB", layout_size, (44, 47, 43))
    painter = ImageDraw.Draw(canvas)
    if cut is None:
        painter.text((12, 40), "No supported pixels inside the reported region.", fill=(239, 223, 177))
        return canvas
    colour, alpha = loaded_cutout(cut)
    photo = Image.fromarray(colour)
    resized = ImageOps.contain(photo, (layout_size[0] - 20, layout_size[1] - 46), Image.Resampling.LANCZOS)
    mask = Image.fromarray(alpha).resize(resized.size, Image.Resampling.LANCZOS)
    position = ((layout_size[0] - resized.width) // 2, 28 + (layout_size[1] - 46 - resized.height) // 2)
    canvas.paste(resized, position, mask)
    if caption:
        sized_label_rgb(painter, (7, layout_size[1] - 19), caption, ((42, 49, 40), (239, 223, 177)), layout_size[0] - 18)
    return canvas


def external_working_panel(frame: Frame, marks: list[dict], layout_size: tuple[int, int]) -> Image.Image:
    raster = frame.raster
    rendered = symmetric_overlay(
        raster.rgb, [(~raster.supported, (135, 88, 76), 0.69)], support=raster.inspected,
    )
    unsupported = ~raster.support
    clean = rendered.copy()
    for channel in range(3):
        # Fill withheld pixels before the Lanczos scale so edge colours are drawn
        # only from the supported part of the measuring region.
        fill = ndi.grey_dilation(np.where(~unsupported, clean[..., channel], 0).astype(np.uint8), size=(3, 3))
        clean[..., channel] = np.where(unsupported, fill, clean[..., channel])
    picture, bounds, scale = bounded_crop(clean, raster.support, longest=max(layout_size))
    alpha, _, _ = bounded_crop(
        np.where(raster.support, np.uint8(255), np.uint8(0)), raster.support, longest=max(layout_size),
    )
    canvas = Image.new("RGB", layout_size, (44, 47, 43))
    shown = Image.fromarray(picture)
    position = ((layout_size[0] - shown.width) // 2, (layout_size[1] - shown.height) // 2)
    canvas.paste(shown, position, Image.fromarray(alpha.astype(np.uint8)))
    plate = Plate(
        canvas,
        (position[0] - bounds[0] * scale, position[1] - bounds[1] * scale),
        scale,
        (raster.plane.width, raster.plane.height),
    )
    draw_anchors(plate, frame, (position, shown.size))
    visible = [
        index for index, mark in enumerate(marks, 1)
        if point_visible(mark["point"], plate, position, shown.size) and raster_supported(frame, mark["point"])
    ]
    draw_markers(plate, [mark for index, mark in enumerate(marks, 1) if index in visible], visible)
    draw_scale(plate, frame)
    return canvas


def raster_supported(frame: Frame, point) -> bool:
    support = frame.raster.support
    height, width = support.shape
    x, y = point
    if not (-0.5 <= x <= width - 0.5 and -0.5 <= y <= height - 0.5):
        return False
    return bool(support[min(height - 1, max(0, round(y))), min(width - 1, max(0, round(x)))])


def point_visible(point, plate: Plate, position: tuple[int, int], size: tuple[int, int]) -> bool:
    x, y = canvas_position(point, plate)
    return position[0] <= x <= position[0] + size[0] and position[1] <= y <= position[1] + size[1]


def draw_anchors(
    plate: Plate, frame: Frame, window: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> None:
    painter = ImageDraw.Draw(plate.rgb)
    anchors, label = working_anchor_points(frame)
    shown = [
        (index, point) for index, point in enumerate(anchors, 1)
        if (window is None or point_visible(point, plate, *window)) and raster_supported(frame, point)
    ]
    drawn = [canvas_position(point, plate) for _, point in shown]
    if drawn:
        painter.line(drawn + drawn[:1] if len(drawn) == 4 else drawn, fill=(248, 217, 154), width=2)
    for index, point in shown:
        x, y = canvas_position(point, plate)
        painter.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(248, 217, 154), outline=(34, 39, 33))
        pos = (min(x + 5, plate.rgb.width - 18), min(y + 5, plate.rgb.height - 16))
        sized_label_rgb(painter, pos, str(index), ((42, 49, 40), (239, 223, 177)), 22)
    sized_label_rgb(painter, (7, 6), label, ((42, 49, 40), (239, 223, 177)), plate.rgb.width - 18)
    if window is not None and len(shown) < len(anchors):
        sized_label_rgb(
            painter, (7, 24), "Calibration marks outside the region are withheld",
            ((70, 58, 42), (247, 226, 183)), plate.rgb.width - 18,
        )


def draw_scale(plate: Plate, frame: Frame) -> None:
    data = readings(frame.raster)
    plane = data.plane
    tick = positive_tick(data.extents[0] * 0.15)
    proportion = tick / data.extents[0]
    length = max(6, round(plate.original[0] * plate.scale * proportion))
    x = max(12.0, plate.offset[0] + 12)
    y = min(plate.rgb.height - 12, plate.offset[1] + plate.original[1] * plate.scale - 12)
    x = min(x, plate.rgb.width - 20.0)
    y = max(y, 30.0)
    painter = ImageDraw.Draw(plate.rgb)
    backing = (44, 50, 39)
    tint = (247, 224, 166)
    label = f"{tick:g} {plane.unit}"
    if len(label) * 8 > length:
        length_for_box = len(label) * 8
    else:
        length_for_box = length
    painter.rectangle((x - 5, y - 25, x + length_for_box + 5, y + 7), fill=backing)
    painter.text((x + 1, y - 22), label, fill=tint)
    painter.line((x, y, x + length, y), fill=tint, width=2)
    painter.line((x, y - 4, x, y + 3), fill=tint, width=2)
    painter.line((x + length, y - 4, x + length, y + 3), fill=tint, width=2)


def draw_markers(
    plate: Plate, marks: list[dict], numbers: list[int] | None = None,
) -> None:
    painter = ImageDraw.Draw(plate.rgb)
    color = (255, 200, 134)
    labels = numbers if numbers is not None else range(1, len(marks) + 1)
    for number, mark in zip(labels, marks, strict=True):
        x, y = canvas_position(mark["point"], plate)
        painter.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(51, 41, 30), width=1)
        label = str(number) + "  " + mark["label"]
        tx = max(7, min(x + 7, plate.rgb.width - 130))
        ty = max(5, min(y - 16, plate.rgb.height - 17))
        sized_label_rgb(painter, (tx, ty), label, ((60, 50, 39), (252, 226, 183)), int(plate.rgb.width - tx - 8))


def plane_panel(frame: Frame, marks: list[dict], layout_size: tuple[int, int]) -> Image.Image:
    raster = frame.raster
    rendered = symmetric_overlay(
        raster.rgb, [(~raster.supported, (135, 88, 76), 0.69)], support=raster.inspected,
    )
    plate = contain(rendered, layout_size)
    draw_region(plate, frame)
    draw_markers(plate, marks)
    draw_scale(plate, frame)
    return plate.rgb


def report_plate(db: Database, report: dict) -> bytes:
    manifest = report["manifest"]
    current = frame_for(db, manifest["inspection"])
    marks = manifest["markers"]
    layout_size = (637, 471)
    original = source_panel(current, layout_size)
    projected = plane_panel(current, marks, layout_size)
    layout = Image.new("RGB", (1308, 557), (240, 237, 223))
    painter = ImageDraw.Draw(layout)
    base_labels = (
        ("SOURCE / RAW CAPTURE", original),
        ("WORKING / MEASURING SUPPORT", projected),
    )
    for index, (title, image) in enumerate(base_labels):
        start_x = 11 + index * (layout_size[0] + 12)
        layout.paste(image, (start_x, 55))
        painter.text((start_x, 32), title, fill=(66, 79, 53))
    spec = manifest["inspection"]
    name = ("[Synthetic] " if spec["capture_metadata"]["synthetic"] else "") + spec["title"]
    painter.text((10, 10), name, fill=(61, 68, 49))
    revision = spec["revision"]
    detail = f"Revision {revision}   /   {len(marks)} markers   /   Source {spec['source_digest'][:16]}"
    painter.text((10, 535), detail, fill=(61, 68, 49))
    return encode_png(np.asarray(layout))
