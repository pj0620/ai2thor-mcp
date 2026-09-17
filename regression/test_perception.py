"""Offline geometry tests: analytic depth scenes pin the sign conventions
and the passage detector's behavior — no Unity required."""

import json
from pathlib import Path

import numpy as np
import pytest

from ai2thor_mcp import config
from ai2thor_mcp.perception import (
    depth_to_world,
    detect_passages,
    intrinsics_from_metadata,
    rotation_world_from_camera,
    world_to_pixel,
)

W = H = 500
FOV = 60.0
CAM_H = 0.9
ORIGIN = {"x": 0.0, "y": CAM_H, "z": 0.0}


def make_intrinsics():
    return intrinsics_from_metadata({"screenWidth": W, "screenHeight": H, "fov": FOV})


def ray_slopes(intr):
    """Per-pixel camera-frame ray slopes (x right, y up) per unit forward z."""
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    dir_x = (uu - intr.cx) / intr.fx
    dir_y = -(vv - intr.cy) / intr.fy
    return dir_x, dir_y


def floor_hit_depth(dir_y, fill):
    """Planar depth where each downward ray meets the y=0 floor, else fill."""
    down = dir_y < -1e-6
    with np.errstate(divide="ignore"):
        return np.where(down, -CAM_H / np.where(down, dir_y, -1.0), fill)


def make_scene_depth(intr, hole_halfwidth, wall_z=2.0, far_z=6.0, wall_top=2.5):
    """Planar depth for: flat floor at y=0, a wall at z=wall_z with a full-height
    hole |x| <= hole_halfwidth, and a far wall at z=far_z."""
    dir_x, dir_y = ray_slopes(intr)
    t_floor = floor_hit_depth(dir_y, np.inf)
    x_wall = wall_z * dir_x
    y_wall = CAM_H + wall_z * dir_y
    wall_hit = (np.abs(x_wall) > hole_halfwidth) & (y_wall > 0.0) & (y_wall < wall_top)
    t_wall = np.where(wall_hit, wall_z, np.inf)
    depth = np.minimum(np.minimum(t_floor, t_wall), far_z)
    return depth.astype(np.float32)


def test_floor_backprojects_to_zero_height():
    intr = make_intrinsics()
    _, dir_y = ray_slopes(intr)
    depth = floor_hit_depth(dir_y, 0.0).astype(np.float32)
    pts, pixels = depth_to_world(depth, intr, ORIGIN, yaw_deg=0.0, horizon_deg=0.0)
    assert len(pts) > 1000
    assert float(np.max(np.abs(pts[:, 1]))) < 0.02
    right = pixels[:, 0] > intr.cx
    assert np.all(pts[right, 0] > -1e-6)
    assert np.all(pts[:, 2] > 0)


def test_world_pixel_round_trip():
    intr = make_intrinsics()
    cam = {"x": 1.0, "y": 0.9, "z": -2.0}
    rng = np.random.default_rng(0)
    pts_cam = np.stack(
        [rng.uniform(-2, 2, 50), rng.uniform(-2, 2, 50), rng.uniform(0.5, 5, 50)], axis=1
    )
    for yaw, horizon in [(0.0, 0.0), (90.0, 0.0), (37.0, 25.0), (-120.0, -10.0)]:
        rot = rotation_world_from_camera(yaw, horizon)
        pts_world = pts_cam @ rot.T + np.array([cam["x"], cam["y"], cam["z"]])
        uv, ok = world_to_pixel(pts_world, intr, cam, yaw, horizon)
        assert ok.all()
        exp_u = intr.cx + intr.fx * pts_cam[:, 0] / pts_cam[:, 2]
        exp_v = intr.cy + intr.fy * (-pts_cam[:, 1]) / pts_cam[:, 2]
        np.testing.assert_allclose(uv[:, 0], exp_u, atol=1e-6)
        np.testing.assert_allclose(uv[:, 1], exp_v, atol=1e-6)


def test_doorway_detected():
    intr = make_intrinsics()
    depth = make_scene_depth(intr, hole_halfwidth=0.4)  # 0.8 m doorway dead ahead
    cands = detect_passages(depth, ORIGIN, ORIGIN, 0.0, 0.0, intr, reachable_xz=None)
    assert len(cands) == 1
    door = cands[0]
    assert door.kind == "doorway"
    assert not door.width_is_lower_bound
    assert 0.6 <= door.width_m <= 1.1
    assert abs(door.bearing_deg) <= 4.0
    assert abs(door.waypoint["x"]) <= 0.35
    assert 2.1 <= door.waypoint["z"] <= 3.1
    assert door.free_depth_m >= 1.0
    x1, y1, x2, y2 = door.region_px
    assert 0 <= x1 < x2 < W and 0 <= y1 < y2 < H


def test_narrow_gap_rejected():
    intr = make_intrinsics()
    depth = make_scene_depth(intr, hole_halfwidth=0.2)  # 0.4 m gap: too narrow
    cands = detect_passages(depth, ORIGIN, ORIGIN, 0.0, 0.0, intr, reachable_xz=None)
    assert cands == []


def test_plain_wall_yields_nothing():
    intr = make_intrinsics()
    depth = make_scene_depth(intr, hole_halfwidth=0.0)  # wall, no hole
    cands = detect_passages(depth, ORIGIN, ORIGIN, 0.0, 0.0, intr, reachable_xz=None)
    assert cands == []


def test_open_space_detected():
    intr = make_intrinsics()
    _, dir_y = ray_slopes(intr)
    # floor stretching away, nothing else within range
    depth = floor_hit_depth(dir_y, 8.0).astype(np.float32)
    cands = detect_passages(depth, ORIGIN, ORIGIN, 0.0, 0.0, intr, reachable_xz=None)
    assert len(cands) >= 1
    assert cands[0].kind == "opening"
    assert cands[0].width_is_lower_bound


def test_unreachable_waypoint_discarded():
    intr = make_intrinsics()
    depth = make_scene_depth(intr, hole_halfwidth=0.4)
    # reachable positions exist but nowhere near the doorway: phantom filter fires
    reachable = np.array([[5.0, -5.0]])
    cands = detect_passages(depth, ORIGIN, ORIGIN, 0.0, 0.0, intr, reachable_xz=reachable)
    assert cands == []


FIXTURE = Path("data/example_frame.npz")


@pytest.mark.skipif(
    not FIXTURE.exists(), reason="no captured frame; run scripts/capture_fixture.py"
)
def test_captured_fixture_frame():
    data = np.load(FIXTURE)
    meta = json.loads(Path("data/example_frame.json").read_text())
    intr = intrinsics_from_metadata(meta)
    agent = meta["agent"]
    cands = detect_passages(
        data["depth"],
        meta["cameraPosition"],
        agent["position"],
        agent["rotation"]["y"],
        agent["cameraHorizon"],
        intr,
        reachable_xz=None,
    )
    for cand in cands:
        assert cand.width_m >= config.MIN_PASSAGE_WIDTH_M
        assert -35.0 <= cand.bearing_deg <= 35.0
    # the FloorPlan2 spawn view looks straight down the corridor between the
    # kitchen island and the counters: it must be found as a ~1 m passage ahead
    assert any(
        abs(c.bearing_deg) <= 10.0 and 0.7 <= c.width_m <= 1.3 for c in cands
    ), f"corridor not detected: {cands}"
