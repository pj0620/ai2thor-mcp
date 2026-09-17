"""Depth-based perception: backprojection and passage detection.

Passage detection works on a polar free-space profile around the agent:
depth pixels are backprojected to world points, split into obstacle points
(heights that block the robot) and floor points, and binned by azimuth over
the camera FOV. A passage is a run of azimuth bins that is clear well beyond
its flanking obstacles ("the opening is deeper than its frame"), has floor
visible through it (rejects windows/mirrors, which show far depth but no
drivable floor), and is wide enough for the robot.

Coordinate conventions (AI2-THOR / Unity, left-handed): world +y up, yaw is
degrees clockwise from +z, so the agent's forward vector is (sin yaw, 0,
cos yaw); cameraHorizon > 0 means the camera pitches down. depth_frame holds
planar depth in meters (distance along the camera z axis, not ray length).
The exact sign conventions here are pinned by the synthetic-floor regression
test, which requires backprojected floor pixels to land at world y ~= 0.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai2thor_mcp import config


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def half_hfov_deg(self) -> float:
        return float(np.degrees(np.arctan((self.width * 0.5) / self.fx)))


def intrinsics_from_metadata(meta: Dict[str, Any]) -> Intrinsics:
    width = int(meta["screenWidth"])
    height = int(meta["screenHeight"])
    vfov = np.deg2rad(float(meta["fov"]))
    fy = (height * 0.5) / np.tan(vfov * 0.5)
    return Intrinsics(width=width, height=height, fx=fy, fy=fy, cx=width * 0.5, cy=height * 0.5)


def rotation_world_from_camera(yaw_deg: float, horizon_deg: float) -> np.ndarray:
    """3x3 rotation mapping camera-frame (x right, y up, z forward) to world."""
    pitch = np.deg2rad(horizon_deg)
    yaw = np.deg2rad(yaw_deg)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    # pitch down by `horizon` about camera x, then yaw clockwise-from-+z about world y
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return ry @ rx


def depth_to_world(
    depth: np.ndarray,
    intr: Intrinsics,
    camera_pos: Dict[str, float],
    yaw_deg: float,
    horizon_deg: float,
    stride: int = config.DEPTH_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backproject a (sub-sampled) depth image.

    Returns (points, pixels): points is (N, 3) world xyz, pixels is (N, 2)
    integer source (u, v) for each point.
    """
    us = np.arange(0, intr.width, stride)
    vs = np.arange(0, intr.height, stride)
    uu, vv = np.meshgrid(us, vs)
    uu = uu.ravel()
    vv = vv.ravel()
    d = depth[vv, uu].astype(np.float64)

    valid = np.isfinite(d) & (d > 1e-3)
    uu, vv, d = uu[valid], vv[valid], d[valid]

    # camera frame, y up: image v grows downward
    x_c = d * (uu - intr.cx) / intr.fx
    y_c = -d * (vv - intr.cy) / intr.fy
    z_c = d
    pts_cam = np.stack([x_c, y_c, z_c], axis=1)

    rot = rotation_world_from_camera(yaw_deg, horizon_deg)
    cam = np.array([camera_pos["x"], camera_pos["y"], camera_pos["z"]])
    pts_world = pts_cam @ rot.T + cam
    pixels = np.stack([uu, vv], axis=1)
    return pts_world, pixels


def world_to_pixel(
    pts_world: np.ndarray,
    intr: Intrinsics,
    camera_pos: Dict[str, float],
    yaw_deg: float,
    horizon_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) world points to pixel (u, v). Returns (pixels, valid)."""
    pts_world = np.atleast_2d(np.asarray(pts_world, dtype=np.float64))
    rot = rotation_world_from_camera(yaw_deg, horizon_deg)
    cam = np.array([camera_pos["x"], camera_pos["y"], camera_pos["z"]])
    pts_cam = (pts_world - cam) @ rot
    x_c, y_c, z_c = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid = z_c > 0.05
    z_safe = np.where(valid, z_c, 1.0)
    u = intr.cx + intr.fx * x_c / z_safe
    v = intr.cy + intr.fy * (-y_c) / z_safe
    return np.stack([u, v], axis=1), valid


def wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def snap_to_reachable(
    x: float,
    z: float,
    reachable_xz: Optional[np.ndarray],
    max_snap: float = config.SNAP_MAX_M,
) -> Optional[Tuple[float, float]]:
    """Nearest reachable position within max_snap, or None."""
    if reachable_xz is None:
        return (x, z)
    if len(reachable_xz) == 0:
        return None
    d2 = (reachable_xz[:, 0] - x) ** 2 + (reachable_xz[:, 1] - z) ** 2
    i = int(np.argmin(d2))
    if d2[i] > max_snap**2:
        return None
    return (float(reachable_xz[i, 0]), float(reachable_xz[i, 1]))


@dataclass
class PassageCandidate:
    kind: str  # "doorway" | "hallway" | "opening"
    width_m: float
    width_is_lower_bound: bool
    free_depth_m: float
    bearing_deg: float  # relative to agent yaw, clockwise positive
    waypoint: Dict[str, float]  # world nav target, y=0
    region_px: Tuple[int, int, int, int]
    marker_px: Tuple[int, int]


def detect_passages(
    depth: np.ndarray,
    camera_pos: Dict[str, float],
    agent_pos: Dict[str, float],
    yaw_deg: float,
    horizon_deg: float,
    intr: Intrinsics,
    reachable_xz: Optional[np.ndarray] = None,
    min_width: float = config.MIN_PASSAGE_WIDTH_M,
) -> List[PassageCandidate]:
    pts, pixels = depth_to_world(depth, intr, camera_pos, yaw_deg, horizon_deg)
    ax, az = agent_pos["x"], agent_pos["z"]
    dx = pts[:, 0] - ax
    dz = pts[:, 2] - az
    rng = np.hypot(dx, dz)
    y_rel = pts[:, 1] - config.FLOOR_Y_M

    in_range = (rng > 0.05) & (rng <= config.MAX_RANGE_M)
    azim = (np.degrees(np.arctan2(dx, dz)) - yaw_deg + 180.0) % 360.0 - 180.0

    half_fov = intr.half_hfov_deg
    n_bins = int(np.ceil(2 * half_fov / config.AZ_BIN_DEG))
    bin_idx = np.floor((azim + half_fov) / config.AZ_BIN_DEG).astype(int)
    in_fov = (bin_idx >= 0) & (bin_idx < n_bins)

    lo, hi = config.OBSTACLE_BAND_M
    obstacle = in_range & in_fov & (y_rel >= lo) & (y_rel <= hi)
    floor = in_fov & (np.abs(y_rel) <= config.FLOOR_TOL_M) & (rng > 0.05)

    d_obs = np.full(n_bins, np.inf)
    np.minimum.at(d_obs, bin_idx[obstacle], rng[obstacle])
    d_floor = np.zeros(n_bins)
    np.maximum.at(d_floor, bin_idx[floor], rng[floor])
    d_prof = np.minimum(d_obs, config.MAX_RANGE_M)

    # cache obstacle point arrays for flank lookups
    ob_bin = bin_idx[obstacle]
    ob_rng = rng[obstacle]
    ob_x = pts[obstacle, 0]
    ob_z = pts[obstacle, 2]

    def nearest_flank(bins: range, max_range: float) -> Optional[Tuple[float, float, float]]:
        """(range, x, z) of the closest obstacle point within max_range in the bins."""
        mask = np.isin(ob_bin, list(bins)) & (ob_rng < max_range)
        if not mask.any():
            return None
        i = int(np.argmin(np.where(mask, ob_rng, np.inf)))
        return (float(ob_rng[i]), float(ob_x[i]), float(ob_z[i]))

    # level-set gap finding: at each scan depth L, a passage shows up as a run
    # of bins clear beyond L whose neighboring bins are blocked nearer than L.
    # Scanning several levels catches both sharp doorframes and corridors whose
    # flanking walls recede gradually; near-duplicates are deduped at the end.
    candidates: List[PassageCandidate] = []
    for level in config.GAP_LEVELS_M:
        open_mask = d_prof >= level
        runs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for i in range(n_bins):
            if open_mask[i] and start is None:
                start = i
            elif not open_mask[i] and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, n_bins - 1))

        for i0, i1 in runs:
            if i1 - i0 + 1 < config.MIN_SECTOR_BINS:
                continue
            run_min = float(np.min(d_prof[i0 : i1 + 1]))
            sector_angle_rad = np.deg2rad((i1 - i0 + 1) * config.AZ_BIN_DEG)

            left = None
            if i0 > 0:
                left = nearest_flank(
                    range(max(0, i0 - config.FLANK_BINS), i0), max_range=level
                )
            right = None
            if i1 < n_bins - 1:
                right = nearest_flank(
                    range(i1 + 1, min(n_bins, i1 + 1 + config.FLANK_BINS)),
                    max_range=level,
                )

            if left and right:
                d_open = min(left[0], right[0])
                if run_min - d_open < config.GAP_BEYOND_M:
                    continue  # not deeper than its frame
                width = float(np.hypot(left[1] - right[1], left[2] - right[2]))
                lower_bound = False
            elif left or right:
                d_open = (left or right)[0]  # type: ignore[index]
                if run_min - d_open < config.GAP_BEYOND_M:
                    continue
                width = float(d_open * sector_angle_rad)
                lower_bound = True
            else:
                # no nearer frame on either side: only genuinely open space counts
                if run_min < config.OPEN_SPACE_MIN_M:
                    continue
                d_open = config.MIN_FREE_DEPTH_M
                width = float(2.0 * sector_angle_rad)  # chord at a nominal 2 m
                lower_bound = True

            if width < min_width:
                continue

            # drivable floor must be visible past the opening plane in most of
            # the run — rejects windows/glass, which show far depth but no
            # floor beyond
            floor_beyond = d_floor[i0 : i1 + 1] >= d_open + config.FLOOR_BEYOND_M
            if float(np.mean(floor_beyond)) < config.FLOOR_SUPPORT_FRAC:
                continue

            free_depth = max(run_min - d_open, 0.0)

            if width <= config.DOORWAY_MAX_WIDTH_M:
                kind = "doorway"
            elif (
                width <= config.HALLWAY_MAX_WIDTH_M
                and free_depth >= config.HALLWAY_MIN_DEPTH_M
            ):
                kind = "hallway"
            else:
                kind = "opening"

            bearing = -half_fov + (i0 + i1 + 1) * 0.5 * config.AZ_BIN_DEG
            abs_azim = np.deg2rad(yaw_deg + bearing)

            def waypoint_at(r: float, azim_rad: float = abs_azim) -> Dict[str, float]:
                return {
                    "x": ax + r * float(np.sin(azim_rad)),
                    "y": 0.0,
                    "z": az + r * float(np.cos(azim_rad)),
                }

            wp_range = d_open + min(config.WAYPOINT_BEYOND_M, max(free_depth - 0.3, 0.1))
            snapped = None
            for r in (wp_range, max(d_open - 0.3, 0.5)):
                wp = waypoint_at(r)
                xz = snap_to_reachable(wp["x"], wp["z"], reachable_xz)
                if xz is not None:
                    snapped = {"x": xz[0], "y": 0.0, "z": xz[1]}
                    break
            if snapped is None:
                continue  # nothing drivable there: likely a window/mirror phantom

            # image region: sector points around the opening, below ceiling height
            in_sector = in_fov & (bin_idx >= i0) & (bin_idx <= i1)
            region_mask = (
                in_sector
                & (rng >= max(0.5, d_open - 0.2))
                & (rng <= d_open + max(free_depth, 0.5))
                & (y_rel <= config.MAX_CEILING_Y_M)
            )
            marker_wp = np.array([[snapped["x"], 0.3, snapped["z"]]])
            marker_uv, _ = world_to_pixel(
                marker_wp, intr, camera_pos, yaw_deg, horizon_deg
            )
            mu = int(np.clip(marker_uv[0, 0], 0, intr.width - 1))
            mv = int(np.clip(marker_uv[0, 1], 0, intr.height - 1))
            if region_mask.any():
                reg_px = pixels[region_mask]
                region = (
                    int(reg_px[:, 0].min()),
                    int(reg_px[:, 1].min()),
                    int(reg_px[:, 0].max()),
                    int(reg_px[:, 1].max()),
                )
            else:
                r_pad = config.MARKER_RADIUS_PX * 2
                region = (
                    max(0, mu - r_pad),
                    max(0, mv - r_pad),
                    min(intr.width - 1, mu + r_pad),
                    min(intr.height - 1, mv + r_pad),
                )

            candidates.append(
                PassageCandidate(
                    kind=kind,
                    width_m=round(width, 2),
                    width_is_lower_bound=lower_bound,
                    free_depth_m=round(free_depth, 2),
                    bearing_deg=round(float(bearing), 1),
                    waypoint=snapped,
                    region_px=region,
                    marker_px=(mu, mv),
                )
            )

    # dedupe by waypoint proximity, widest first
    candidates.sort(key=lambda c: -c.width_m)
    kept: List[PassageCandidate] = []
    for cand in candidates:
        if all(
            np.hypot(
                cand.waypoint["x"] - k.waypoint["x"], cand.waypoint["z"] - k.waypoint["z"]
            )
            >= config.PASSAGE_DEDUPE_M
            for k in kept
        ):
            kept.append(cand)
    kept.sort(key=lambda c: abs(c.bearing_deg))
    return kept
