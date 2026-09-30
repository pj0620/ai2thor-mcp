"""Offline: occupancy mapping from depth frames (synthetic scenes and the captured fixture)."""

import json
import math

import numpy as np
import pytest

from ai2thor_mcp.mapping import (
    FREE_WHITE,
    MAX_RANGE_M,
    OCCUPIED_BLACK,
    UNKNOWN_GREY,
    OccupancyMapper,
    planar_scan,
)
from ai2thor_mcp.perception import intrinsics_from_metadata

W = H = 200
FOV = 60.0
CAM_Y = 0.9


def metadata(x=0.0, z=0.0, yaw=0.0, horizon=0.0):
    half = 5.0
    corners = [[sx * half, y, sz * half] for sx in (-1, 1) for y in (0.0, 2.5) for sz in (-1, 1)]
    return {
        "screenWidth": W,
        "screenHeight": H,
        "fov": FOV,
        "agent": {"position": {"x": x, "y": CAM_Y, "z": z}, "rotation": {"x": 0.0, "y": yaw, "z": 0.0}, "cameraHorizon": horizon},
        "cameraPosition": {"x": x, "y": CAM_Y, "z": z},
        "sceneBounds": {"cornerPoints": corners, "center": {"x": 0, "y": 1.25, "z": 0}, "size": {"x": 10, "y": 2.5, "z": 10}},
    }


def wall_depth(distance):
    """A flat wall perpendicular to the view, `distance` metres ahead (z-depth is constant)."""
    return np.full((H, W), distance, dtype=np.float32)


def test_wall_ahead_becomes_a_scan_at_the_wall_distance():
    meta = metadata()
    angles, ranges, hits = planar_scan(wall_depth(2.0), intrinsics_from_metadata(meta), meta["cameraPosition"], 0.0, 0.0)
    assert angles.size > 50 and hits.all()
    # a perpendicular wall at z-depth 2 m: horizontal range = 2 / cos(bearing)
    np.testing.assert_allclose(ranges * np.cos(angles), 2.0, atol=0.02)
    assert abs(angles).max() < math.radians(FOV / 2 + 1)


def test_grid_marks_the_wall_occupied_the_approach_free_and_the_rest_unknown():
    mapper = OccupancyMapper()
    meta = metadata()
    beams = mapper.integrate(wall_depth(2.0), meta)
    assert beams > 50 and mapper.scans == 1
    image = mapper.classify()
    assert image is not None
    r, c = mapper.cell(0.0, 2.0)
    assert OCCUPIED_BLACK in image[r - 1 : r + 2, c]  # wall straight ahead
    assert image[mapper.cell(0.0, 1.0)] == FREE_WHITE  # halfway there
    assert image[mapper.cell(0.0, -1.0)] == UNKNOWN_GREY  # behind the robot
    assert image[mapper.cell(3.0, 0.0)] == UNKNOWN_GREY  # outside the field of view
    assert image[mapper.cell(0.0, 3.0)] == UNKNOWN_GREY  # beyond the wall


def test_rotating_fills_in_more_of_the_room():
    mapper = OccupancyMapper()
    mapper.integrate(wall_depth(2.0), metadata(yaw=0.0))
    known_before = int((mapper.classify() != UNKNOWN_GREY).sum())
    mapper.integrate(wall_depth(2.0), metadata(yaw=90.0))  # now facing +x
    image = mapper.classify()
    r, c = mapper.cell(2.0, 0.0)
    assert OCCUPIED_BLACK in image[r, c - 1 : c + 2]
    assert int((image != UNKNOWN_GREY).sum()) > known_before * 1.5


def test_far_depth_clears_free_space_up_to_max_range():
    mapper = OccupancyMapper()
    mapper.integrate(wall_depth(50.0), metadata())
    image = mapper.classify()
    assert image[mapper.cell(0.0, MAX_RANGE_M - 0.3)] == FREE_WHITE
    assert OCCUPIED_BLACK not in image


def test_render_scales_cells_and_draws_the_robot():
    mapper = OccupancyMapper()
    mapper.integrate(wall_depth(2.0), metadata())
    from ai2thor_mcp.utils import MapOverlays

    img = mapper.render(MapOverlays(agent_position={"x": 0.0, "y": CAM_Y, "z": 0.0}, agent_yaw_deg=0.0))
    grid = mapper.grid
    assert img.size == (grid.shape[1] * 5, grid.shape[0] * 5)
    pixels = np.asarray(img)
    assert (pixels == [230, 50, 50]).all(axis=-1).any()  # the red robot triangle


def test_captured_frame_maps_the_kitchen():
    frame = np.load("data/example_frame.npz")
    with open("data/example_metadata.json") as fh:
        meta = json.load(fh)
    mapper = OccupancyMapper()
    beams = mapper.integrate(frame["depth"], meta)
    assert beams > 100
    image = mapper.classify()
    occupied = int((image == OCCUPIED_BLACK).sum())
    free = int((image == FREE_WHITE).sum())
    assert occupied > 5 and free > occupied
    # every obstacle lies within sensor range of the camera
    cam = meta["cameraPosition"]
    rows, cols = np.nonzero(image == OCCUPIED_BLACK)
    ox, oz = mapper.origin
    dist = np.hypot(cols * mapper.resolution + ox - cam["x"], rows * mapper.resolution + oz - cam["z"])
    assert dist.max() <= MAX_RANGE_M + 0.1
    rendered = np.asarray(mapper.render(meta=meta).convert("L"))
    assert {UNKNOWN_GREY, FREE_WHITE, OCCUPIED_BLACK} <= set(np.unique(rendered).tolist())
