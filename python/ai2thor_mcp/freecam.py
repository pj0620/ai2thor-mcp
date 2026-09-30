"""A user-controlled "free" third-party camera for the simulator (sim-only feature).

The portal drives it: fly around the room, look around, zoom. It is a second AI2-THOR
third-party camera (the overhead recorder owns another), tracked by index and re-added if the
scene loses it. Pose math is pure so it can be unit-tested; only `FreeCamera` touches the Sim.

Unity conventions: position {x, y, z} in metres (y up), rotation {x: pitch (down is positive),
y: yaw (0 = +z, 90 = +x), z: roll}. Field of view is vertical, in degrees.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import numpy as np

FOV_MIN_DEG = 15.0
FOV_MAX_DEG = 120.0
PITCH_MIN_DEG = -89.0
PITCH_MAX_DEG = 89.0
HEIGHT_MIN_M = 0.05
HEIGHT_MAX_M = 6.0
CHASE_DISTANCE_M = 1.6
CHASE_HEIGHT_M = 1.2
TOP_HEIGHT_M = 2.3
FRONT_DISTANCE_M = 1.5
DEFAULT_FOV_DEG = 70.0
SKYBOX = "black"


@dataclass(frozen=True)
class CameraPose:
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    fov: float = DEFAULT_FOV_DEG

    def as_action(self) -> Dict[str, Any]:
        return {
            "position": {"x": self.x, "y": self.y, "z": self.z},
            "rotation": {"x": self.pitch, "y": self.yaw, "z": 0.0},
            "fieldOfView": self.fov,
            "orthographic": False,
            "skyboxColor": SKYBOX,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "position": {"x": round(self.x, 3), "y": round(self.y, 3), "z": round(self.z, 3)},
            "yaw": round(self.yaw % 360.0, 2),
            "pitch": round(self.pitch, 2),
            "fov": round(self.fov, 1),
        }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def forward_vector(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit vector the camera looks along (Unity left-handed, y up)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    return np.array([math.cos(pitch) * math.sin(yaw), -math.sin(pitch), math.cos(pitch) * math.cos(yaw)])


def right_vector(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    return np.array([math.cos(yaw), 0.0, -math.sin(yaw)])


def move_pose(
    pose: CameraPose,
    *,
    forward: float = 0.0,
    right: float = 0.0,
    up: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    zoom: float = 0.0,
) -> CameraPose:
    """Apply a relative move (metres, camera-relative), look (degrees) and zoom (degrees of FOV;
    positive zooms in). Height, pitch and FOV are clamped to sane ranges."""
    new_yaw = (pose.yaw + yaw) % 360.0
    new_pitch = _clamp(pose.pitch + pitch, PITCH_MIN_DEG, PITCH_MAX_DEG)
    # move along the *horizontal* forward so W/S never changes height; up/down is world-vertical
    flat_forward = forward_vector(new_yaw, 0.0)
    delta = flat_forward * forward + right_vector(new_yaw) * right + np.array([0.0, up, 0.0])
    return replace(
        pose,
        x=pose.x + float(delta[0]),
        y=_clamp(pose.y + float(delta[1]), HEIGHT_MIN_M, HEIGHT_MAX_M),
        z=pose.z + float(delta[2]),
        yaw=new_yaw,
        pitch=new_pitch,
        fov=_clamp(pose.fov - zoom, FOV_MIN_DEG, FOV_MAX_DEG),
    )


def look_at_pose(x: float, y: float, z: float, target: Dict[str, float], fov: float = DEFAULT_FOV_DEG) -> CameraPose:
    """A pose at (x, y, z) looking at `target` {x, y, z}."""
    dx, dy, dz = target["x"] - x, target["y"] - y, target["z"] - z
    yaw = math.degrees(math.atan2(dx, dz)) % 360.0
    flat = math.hypot(dx, dz)
    pitch = _clamp(math.degrees(math.atan2(-dy, flat)), PITCH_MIN_DEG, PITCH_MAX_DEG)
    return CameraPose(x=x, y=y, z=z, yaw=yaw, pitch=pitch, fov=fov)


def preset_pose(mode: str, agent_position: Dict[str, float], agent_yaw_deg: float) -> CameraPose:
    """Named viewpoints relative to the robot: `chase` (behind and above, looking at it),
    `front` (in front, looking back at it), `top` (straight down from above)."""
    ax, ay, az = agent_position["x"], agent_position["y"], agent_position["z"]
    target = {"x": ax, "y": ay + 0.3, "z": az}
    back = forward_vector(agent_yaw_deg, 0.0)
    if mode == "chase":
        return look_at_pose(ax - float(back[0]) * CHASE_DISTANCE_M, ay + CHASE_HEIGHT_M, az - float(back[2]) * CHASE_DISTANCE_M, target)
    if mode == "front":
        return look_at_pose(ax + float(back[0]) * FRONT_DISTANCE_M, ay + CHASE_HEIGHT_M, az + float(back[2]) * FRONT_DISTANCE_M, target)
    if mode == "top":
        return CameraPose(x=ax, y=ay + TOP_HEIGHT_M, z=az, yaw=agent_yaw_deg % 360.0, pitch=89.0, fov=DEFAULT_FOV_DEG)
    raise ValueError(f"unknown preset {mode!r}: use chase, front or top")


class FreeCameraError(RuntimeError):
    pass


class FreeCamera:
    """Owns one third-party camera in the Sim. Every method takes the Sim; calls are serialised."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index: Optional[int] = None
        self._pose: Optional[CameraPose] = None

    @property
    def pose(self) -> Optional[CameraPose]:
        return self._pose

    @property
    def index(self) -> Optional[int]:
        return self._index

    # --- internals (call with self._lock held) ---

    def _apply_locked(self, sim, pose: CameraPose):
        """Add or update the camera to `pose`; returns the event (with fresh frames)."""
        event = sim.last_event()
        cameras = event.metadata.get("thirdPartyCameras") or []
        if self._index is None or self._index >= len(cameras):
            event = sim.step({"action": "AddThirdPartyCamera", **pose.as_action()})
            if not event.metadata["lastActionSuccess"]:
                raise FreeCameraError(f"AddThirdPartyCamera failed: {event.metadata.get('errorMessage')}")
            self._index = len(event.metadata["thirdPartyCameras"]) - 1
        else:
            event = sim.step({"action": "UpdateThirdPartyCamera", "thirdPartyCameraId": self._index, **pose.as_action()})
            if not event.metadata["lastActionSuccess"]:
                raise FreeCameraError(f"UpdateThirdPartyCamera failed: {event.metadata.get('errorMessage')}")
        self._pose = pose
        return event

    def _frame_from(self, event) -> np.ndarray:
        frames = getattr(event, "third_party_camera_frames", None)
        if frames is None or self._index is None or len(frames) <= self._index:
            raise FreeCameraError("no free-camera frame in the event")
        return np.ascontiguousarray(frames[self._index])

    # --- public API ---

    def ensure(self, sim, mode: str = "chase") -> CameraPose:
        """Create the camera at a preset if it does not exist yet."""
        with self._lock:
            if self._pose is None:
                position, yaw, _ = sim.agent_pose()
                self._apply_locked(sim, preset_pose(mode, position, yaw))
            assert self._pose is not None
            return self._pose

    def frame(self, sim) -> np.ndarray:
        """The latest rendered frame: free when the last step already rendered it, else a refresh."""
        with self._lock:
            if self._pose is None:
                position, yaw, _ = sim.agent_pose()
                event = self._apply_locked(sim, preset_pose("chase", position, yaw))
                return self._frame_from(event)
            event = sim.last_event()
            cameras = event.metadata.get("thirdPartyCameras") or []
            frames = getattr(event, "third_party_camera_frames", None)
            if self._index is None or self._index >= len(cameras) or frames is None or len(frames) <= self._index:
                event = self._apply_locked(sim, self._pose)
            return self._frame_from(event)

    def move(self, sim, **deltas: float) -> tuple:
        """Relative move/look/zoom; returns (pose, frame)."""
        with self._lock:
            if self._pose is None:
                position, yaw, _ = sim.agent_pose()
                self._apply_locked(sim, preset_pose("chase", position, yaw))
            assert self._pose is not None
            event = self._apply_locked(sim, move_pose(self._pose, **deltas))
            return self._pose, self._frame_from(event)

    def reset(self, sim, mode: str = "chase") -> tuple:
        """Jump to a preset relative to the robot's current pose; returns (pose, frame)."""
        with self._lock:
            position, yaw, _ = sim.agent_pose()
            event = self._apply_locked(sim, preset_pose(mode, position, yaw))
            return self._pose, self._frame_from(event)

    def set_pose(self, sim, pose: CameraPose) -> tuple:
        with self._lock:
            event = self._apply_locked(sim, pose)
            return self._pose, self._frame_from(event)
