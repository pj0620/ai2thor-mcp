"""Set-of-marks annotation: draw numbered destination markers on the RGB frame."""

from typing import Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ai2thor_mcp import config
from ai2thor_mcp.describe import DestinationRecord


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=config.FONT_SIZE)
    except TypeError:  # older Pillow: no size kwarg
        return ImageFont.load_default()


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    center: Tuple[int, int],
    number: int,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
    frame_size: Tuple[int, int],
) -> None:
    r = config.MARKER_RADIUS_PX
    cx = int(np.clip(center[0], r, frame_size[0] - r))
    cy = int(np.clip(center[1], r, frame_size[1] - r))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=(255, 255, 255))
    draw.text((cx, cy), str(number), fill=(255, 255, 255), font=font, anchor="mm")


def annotate_view(frame: np.ndarray, records: Iterable[DestinationRecord]) -> Image.Image:
    img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font()
    size = img.size

    for rec in records:
        if rec.marker is None:
            continue
        if rec.kind == "object" and rec.bbox is not None:
            x1, y1, x2, y2 = rec.bbox
            draw.rectangle((x1, y1, x2, y2), outline=config.OBJECT_COLOR, width=2)
            _draw_marker(draw, (x1, y1), rec.marker, config.OBJECT_COLOR, font, size)
        elif rec.kind == "passage":
            if rec.region_px is not None:
                draw.rectangle(rec.region_px, outline=config.PASSAGE_COLOR, width=2)
            anchor: Optional[Tuple[int, int]] = rec.marker_px
            if anchor is None and rec.region_px is not None:
                anchor = (
                    (rec.region_px[0] + rec.region_px[2]) // 2,
                    (rec.region_px[1] + rec.region_px[3]) // 2,
                )
            if anchor is not None:
                _draw_marker(draw, anchor, rec.marker, config.PASSAGE_COLOR, font, size)
    return img
