"""Offline: the free camera's pose math (no Unity)."""

import math

import numpy as np
import pytest

from ai2thor_mcp.freecam import (
    FOV_MAX_DEG,
    FOV_MIN_DEG,
    HEIGHT_MAX_M,
    CameraPose,
    forward_vector,
    look_at_pose,
    move_pose,
    preset_pose,
    right_vector,
)


def test_unity_axes_yaw_zero_looks_along_plus_z():
    np.testing.assert_allclose(forward_vector(0, 0), [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(forward_vector(90, 0), [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(right_vector(0), [1, 0, 0], atol=1e-9)
    # positive pitch tilts the view down
    assert forward_vector(0, 45)[1] < 0


def test_move_is_camera_relative_and_keeps_height_flat():
    pose = CameraPose(x=0, y=1, z=0, yaw=90, pitch=30)
    moved = move_pose(pose, forward=2, right=1, up=0.5)
    assert moved.x == pytest.approx(2)  # forward at yaw 90 is +x
    assert moved.z == pytest.approx(-1)  # right at yaw 90 is -z
    assert moved.y == pytest.approx(1.5)
    assert moved.pitch == 30 and moved.yaw == 90


def test_look_and_zoom_are_clamped_and_wrap():
    pose = CameraPose(x=0, y=1, z=0, yaw=350, pitch=80, fov=100)
    turned = move_pose(pose, yaw=20, pitch=30, zoom=-50)
    assert turned.yaw == pytest.approx(10)
    assert turned.pitch == 89
    assert turned.fov == FOV_MAX_DEG
    assert move_pose(pose, zoom=200).fov == FOV_MIN_DEG
    assert move_pose(pose, up=100).y == HEIGHT_MAX_M


def test_look_at_faces_the_target():
    pose = look_at_pose(0, 2, -2, {"x": 0, "y": 0.3, "z": 0})
    assert pose.yaw == pytest.approx(0)  # target is along +z
    assert pose.pitch == pytest.approx(math.degrees(math.atan2(1.7, 2)))
    behind = look_at_pose(2, 1, 0, {"x": 0, "y": 1, "z": 0})
    assert behind.yaw == pytest.approx(270)  # looking along -x
    assert behind.pitch == pytest.approx(0)


def test_presets_are_relative_to_the_robot():
    agent = {"x": 1.0, "y": 0.9, "z": 2.0}
    chase = preset_pose("chase", agent, agent_yaw_deg=0)
    assert chase.z < agent["z"] and chase.y > agent["y"]  # behind and above a +z-facing robot
    assert 0 < chase.pitch < 90 and chase.yaw == pytest.approx(0)
    front = preset_pose("front", agent, agent_yaw_deg=0)
    assert front.z > agent["z"] and front.yaw == pytest.approx(180)
    top = preset_pose("top", agent, agent_yaw_deg=45)
    assert top.pitch == 89 and top.x == agent["x"] and top.z == agent["z"]
    with pytest.raises(ValueError):
        preset_pose("sideways", agent, 0)


def test_action_payload_matches_ai2thor_fields():
    action = CameraPose(x=1, y=2, z=3, yaw=45, pitch=10, fov=60).as_action()
    assert action["position"] == {"x": 1, "y": 2, "z": 3}
    assert action["rotation"] == {"x": 10, "y": 45, "z": 0.0}
    assert action["fieldOfView"] == 60 and action["orthographic"] is False
