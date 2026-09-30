"""Occupancy-grid mapping from the robot's depth camera at camera height.

This is the simulator's stand-in for the real robot's lidar SLAM map. Each rendered depth frame
becomes a planar range scan: pixels whose world height lies within a thin band around the camera
height are unprojected with the known camera pose and, per image column, the nearest such point
is the beam's hit. Beams are ray-cast into a log-odds grid (free along the ray, occupied at the
hit), exactly as a 2D lidar scan would be. The pose comes from the simulator, so this is the
mapping half of SLAM; on the real robot slam_toolbox does both halves, and its /map uses the same
convention: unknown grey, free white, obstacles black.
"""
from __future__ import annotations

import math
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ai2thor_mcp import config
from ai2thor_mcp.perception import Intrinsics, intrinsics_from_metadata, rotation_world_from_camera
from ai2thor_mcp.utils import PX_PER_M, MapOverlays, _draw_map_overlays

RESOLUTION_M = 0.05
BAND_HALF_HEIGHT_M = 0.08  # points within this of the camera height belong to the scan plane
BAND_ROWS = 24  # image rows above/below the horizon row that are unprojected
COLUMN_STRIDE = 2
MIN_RANGE_M = 0.15
MAX_RANGE_M = config.MAX_RANGE_M
# One noise-free observation is enough to call a cell free or occupied (|log-odds| > 0.405 at
# the 0.4 / 0.6 thresholds); repeated observations only firm the belief up to the clamp.
LOG_ODDS_FREE = -0.6
LOG_ODDS_OCCUPIED = 0.9
LOG_ODDS_CLAMP = 4.0
OCCUPIED_PROB = 0.6
FREE_PROB = 0.4
MARGIN_M = 0.5
UNKNOWN_GREY = 205
FREE_WHITE = 254
OCCUPIED_BLACK = 0


def planar_scan(
    depth: np.ndarray,
    intr: Intrinsics,
    camera_pos: Dict[str, float],
    yaw_deg: float,
    horizon_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One lidar-like scan from a depth frame.

    Returns (angles, ranges, hits): world bearing per beam (radians, 0 = +z, clockwise to +x,
    matching AI2-THOR yaw), horizontal range in metres (capped at MAX_RANGE_M), and whether the
    beam ended on an obstacle. Columns with no usable depth are dropped.
    """
    height, width = depth.shape[:2]
    horizon_row = intr.cy - intr.fy * math.tan(math.radians(horizon_deg))
    row0 = max(0, int(round(horizon_row)) - BAND_ROWS)
    row1 = min(height, int(round(horizon_row)) + BAND_ROWS + 1)
    rows = np.arange(row0, row1)
    cols = np.arange(0, width, COLUMN_STRIDE)
    if rows.size == 0 or cols.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool)

    vv, uu = np.meshgrid(rows, cols, indexing="ij")
    d = depth[vv, uu].astype(np.float64)
    valid = np.isfinite(d) & (d > 1e-3)

    x_c = d * (uu - intr.cx) / intr.fx
    y_c = -d * (vv - intr.cy) / intr.fy
    rot = rotation_world_from_camera(yaw_deg, horizon_deg)
    cam = np.array([camera_pos["x"], camera_pos["y"], camera_pos["z"]])
    pts = np.stack([x_c, y_c, d], axis=-1) @ rot.T + cam  # (rows, cols, 3)

    in_band = valid & (np.abs(pts[..., 1] - cam[1]) <= BAND_HALF_HEIGHT_M)
    dx = pts[..., 0] - cam[0]
    dz = pts[..., 2] - cam[2]
    horizontal = np.hypot(dx, dz)
    horizontal_band = np.where(in_band, horizontal, np.inf)
    nearest = np.argmin(horizontal_band, axis=0)  # per column
    col_index = np.arange(cols.size)
    r = horizontal_band[nearest, col_index]
    hit = np.isfinite(r) & (r < MAX_RANGE_M)

    # beam bearing: from the hit point when there is one, else from the column's ray
    bearing_hit = np.arctan2(dx[nearest, col_index], dz[nearest, col_index])
    bearing_col = math.radians(yaw_deg) + np.arctan((cols - intr.cx) / intr.fx)
    angles = np.where(hit, bearing_hit, bearing_col)

    # a column with no point in the band but with valid depth is treated as a clear ray
    usable = hit | valid.any(axis=0)
    ranges = np.where(hit, r, MAX_RANGE_M)
    keep = usable & (ranges >= MIN_RANGE_M)
    return angles[keep], ranges[keep], hit[keep]


class OccupancyMapper:
    """A log-odds occupancy grid over the scene bounds, updated from every rendered frame."""

    def __init__(self, resolution: float = RESOLUTION_M) -> None:
        self.resolution = resolution
        self._lock = threading.Lock()
        self._grid: Optional[np.ndarray] = None  # [row = z, col = x] log-odds
        self._origin: Tuple[float, float] = (0.0, 0.0)  # world (x, z) of cell (0, 0)
        self.scans = 0

    # --- grid geometry -------------------------------------------------------------------------

    def _ensure_grid_locked(self, meta: Dict[str, Any]) -> np.ndarray:
        if self._grid is not None:
            return self._grid
        corners = meta["sceneBounds"]["cornerPoints"]
        xs = [c[0] for c in corners]
        zs = [c[2] for c in corners]
        min_x, min_z = min(xs) - MARGIN_M, min(zs) - MARGIN_M
        cols = int(math.ceil((max(xs) - min(xs) + 2 * MARGIN_M) / self.resolution))
        rows = int(math.ceil((max(zs) - min(zs) + 2 * MARGIN_M) / self.resolution))
        self._origin = (min_x, min_z)
        self._grid = np.zeros((rows, cols), dtype=np.float32)
        return self._grid

    def cell(self, x: float, z: float) -> Tuple[int, int]:
        """(row, col) of a world point; may be out of bounds."""
        ox, oz = self._origin
        return int(math.floor((z - oz) / self.resolution)), int(math.floor((x - ox) / self.resolution))

    @property
    def origin(self) -> Tuple[float, float]:
        return self._origin

    @property
    def grid(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._grid is None else self._grid.copy()

    def reset(self) -> None:
        with self._lock:
            self._grid = None
            self.scans = 0

    # --- updates -------------------------------------------------------------------------------

    def on_step(self, event: Any) -> None:
        """Sim hook: integrate the frame's depth image. Never raises."""
        depth = getattr(event, "depth_frame", None)
        meta = getattr(event, "metadata", None)
        if depth is None or not meta or "sceneBounds" not in meta:
            return
        self.integrate(depth, meta)

    def integrate(self, depth: np.ndarray, meta: Dict[str, Any]) -> int:
        """Ray-cast one frame into the grid. Returns the number of beams used."""
        intr = intrinsics_from_metadata(meta)
        agent = meta["agent"]
        camera_pos = meta.get("cameraPosition") or agent["position"]
        yaw = float(agent["rotation"]["y"])
        horizon = float(agent.get("cameraHorizon") or 0.0)
        angles, ranges, hits = planar_scan(depth, intr, camera_pos, yaw, horizon)
        if angles.size == 0:
            return 0
        with self._lock:
            grid = self._ensure_grid_locked(meta)
            self._raycast_locked(grid, camera_pos["x"], camera_pos["z"], angles, ranges, hits)
            self.scans += 1
        return int(angles.size)

    def _raycast_locked(
        self, grid: np.ndarray, cx: float, cz: float, angles: np.ndarray, ranges: np.ndarray, hits: np.ndarray
    ) -> None:
        rows, cols = grid.shape
        ox, oz = self._origin
        res = self.resolution
        step = res * 0.5
        sin_a, cos_a = np.sin(angles), np.cos(angles)

        # free cells: samples along each beam, stopping one cell short of the hit
        counts = np.where(
            hits,
            np.maximum(0, np.floor((ranges - res) / step).astype(int)),
            np.floor(ranges / step).astype(int),
        )
        beam_idx = np.repeat(np.arange(angles.size), counts)
        t = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)) * step
        fx = cx + sin_a[beam_idx] * t
        fz = cz + cos_a[beam_idx] * t
        free_r = np.floor((fz - oz) / res).astype(int)
        free_c = np.floor((fx - ox) / res).astype(int)
        inside = (free_r >= 0) & (free_r < rows) & (free_c >= 0) & (free_c < cols)
        free_lin = np.unique(free_r[inside] * cols + free_c[inside])

        # occupied cells: beam end points
        hx = cx + sin_a[hits] * ranges[hits]
        hz = cz + cos_a[hits] * ranges[hits]
        occ_r = np.floor((hz - oz) / res).astype(int)
        occ_c = np.floor((hx - ox) / res).astype(int)
        inside_o = (occ_r >= 0) & (occ_r < rows) & (occ_c >= 0) & (occ_c < cols)
        occ_lin = np.unique(occ_r[inside_o] * cols + occ_c[inside_o])

        flat = grid.reshape(-1)
        free_lin = np.setdiff1d(free_lin, occ_lin, assume_unique=True)
        flat[free_lin] += LOG_ODDS_FREE
        flat[occ_lin] += LOG_ODDS_OCCUPIED
        np.clip(flat, -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP, out=flat)

    # --- rendering -----------------------------------------------------------------------------

    def classify(self) -> Optional[np.ndarray]:
        """uint8 image array: unknown grey, free white, occupied black."""
        grid = self.grid
        if grid is None:
            return None
        prob = 1.0 / (1.0 + np.exp(-grid))
        image = np.full(grid.shape, UNKNOWN_GREY, dtype=np.uint8)
        image[prob < FREE_PROB] = FREE_WHITE
        image[prob > OCCUPIED_PROB] = OCCUPIED_BLACK
        return image

    def render(self, overlays: Optional[MapOverlays] = None, meta: Optional[Dict[str, Any]] = None) -> Image.Image:
        """The map as an RGB image at PX_PER_M, with the usual overlays."""
        with self._lock:
            if self._grid is None and meta is not None:
                self._ensure_grid_locked(meta)
        image = self.classify()
        if image is None:
            return Image.new("RGB", (100, 100), (UNKNOWN_GREY,) * 3)
        scale = max(1, int(round(PX_PER_M * self.resolution)))
        img = Image.fromarray(image, mode="L").convert("RGB")
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        if overlays is not None:
            _draw_map_overlays(ImageDraw.Draw(img), overlays, self._origin[0], self._origin[1])
        return img
