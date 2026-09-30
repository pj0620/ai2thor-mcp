"""Single owner of the AI2-THOR Controller.

The Controller talks to Unity over one blocking FIFO pipe and is not
thread-safe, while FastMCP runs sync tools in a threadpool and navigation
runs in its own thread — so every controller.step in this process must go
through Sim while its lock is held. plan_path holds the lock for its whole
duration because ai2thor.util.metrics drives controller.step internally;
the plain (non-reentrant) Lock also enforces that nothing calls back into
Sim while planning.

An attached OverheadRecorder's on_step hook runs inside step() with the Sim
lock held; the recorder's own lock may nest inside the Sim lock, never the
other way around. plan_path and get_reachable_positions bypass the hook by
design (metadata-only queries; the recorder's real-time writer holds the
last frame through them).
"""

import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
from ai2thor.controller import Controller
from ai2thor.util import metrics
from loguru import logger


class Sim:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._controller: Optional[Controller] = None
        self._last_event = None
        self.scene_override: Optional[str] = None
        self._reachable_xz: Optional[np.ndarray] = None
        self._recorder = None
        self._mapper = None

    def attach_recorder(self, recorder) -> None:
        """Wire an OverheadRecorder; set once at startup, not hot-swapped."""
        self._recorder = recorder

    def attach_mapper(self, mapper) -> None:
        """Wire an OccupancyMapper whose on_step hook integrates every rendered depth frame."""
        self._mapper = mapper

    def _controller_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "agentMode": "locobot",
            "visibilityDistance": 1.5,
            "scene": self.scene_override or "FloorPlan2",
            "gridSize": 0.25,
            # follower issues non-grid-multiple moveMagnitudes
            "snapToGrid": False,
            "renderDepthImage": True,
            "renderInstanceSegmentation": True,
            "width": 500,
            "height": 500,
            "fieldOfView": 60,
        }
        return kwargs

    def _ensure_controller_locked(self) -> Controller:
        if self._controller is None:
            kwargs = self._controller_kwargs()
            logger.info(
                "Creating AI2-THOR controller scene={} grid={}m",
                kwargs["scene"],
                kwargs["gridSize"],
            )
            self._controller = Controller(**kwargs)
            self._last_event = self._controller.step(action="MoveAhead", moveMagnitude=0.0)
            self._reachable_xz = None
            logger.debug(
                "Simulator ready (bootstrap success={})",
                self._last_event.metadata.get("lastActionSuccess"),
            )
        return self._controller

    def step(self, action: Dict[str, Any]) -> Any:
        with self._lock:
            controller = self._ensure_controller_locked()
            event = controller.step(**action)
            if self._recorder is not None:
                try:
                    event = self._recorder.on_step(controller, event)
                except Exception:
                    logger.exception("Overhead recorder hook failed; continuing")
            if self._mapper is not None:
                try:
                    self._mapper.on_step(event)
                except Exception:
                    logger.exception("Occupancy mapper hook failed; continuing")
            self._last_event = event
            return event

    def refresh(self) -> Any:
        """Re-render without moving (MoveAhead with zero magnitude)."""
        return self.step({"action": "MoveAhead", "moveMagnitude": 0.0})

    def last_event(self) -> Any:
        event = self._last_event
        if event is None:
            return self.refresh()
        return event

    def peek_pose(self) -> Optional[Tuple[Any, float, float]]:
        """(position, yaw_deg, horizon_deg) if the sim has booted, else None."""
        event = self._last_event
        if event is None:
            return None
        agent = event.metadata["agent"]
        return agent["position"], float(agent["rotation"]["y"]), float(agent["cameraHorizon"])

    def agent_pose(self) -> Tuple[Any, float, float]:
        pose = self.peek_pose()
        if pose is None:
            self.refresh()
            pose = self.peek_pose()
            assert pose is not None
        return pose

    def get_reachable_positions(self) -> np.ndarray:
        cached = self._reachable_xz
        if cached is not None:
            return cached
        with self._lock:
            controller = self._ensure_controller_locked()
            # metadata-only query: don't overwrite _last_event (frame is unchanged)
            event = controller.step(action="GetReachablePositions")
            positions = event.metadata["actionReturn"] or []
            self._reachable_xz = np.array(
                [[p["x"], p["z"]] for p in positions], dtype=np.float64
            )
            logger.debug("Cached {} reachable positions", len(self._reachable_xz))
            return self._reachable_xz

    def _path_to_point_locked(
        self, start: Dict[str, float], target: Dict[str, float]
    ) -> list:
        # the ai2thor 5.0 build takes Vector3 position/target; the bundled
        # metrics.get_shortest_path_to_point still sends the pre-5.0 scalar
        # x/y/z form, which the build rejects — so call the action directly
        event = self._controller.step(
            action="GetShortestPathToPoint",
            position={k: float(start[k]) for k in ("x", "y", "z")},
            target={k: float(target[k]) for k in ("x", "y", "z")},
        )
        if not event.metadata["lastActionSuccess"]:
            raise ValueError(event.metadata["errorMessage"] or "no path found")
        return event.metadata["actionReturn"]["corners"]

    def plan_path(
        self, target: Dict[str, float], object_id: Optional[str] = None
    ) -> list:
        """Navmesh shortest path to a world point (or object) as waypoint corners."""
        with self._lock:
            self._ensure_controller_locked()
            agent_pos = self._last_event.metadata["agent"]["position"]
            if object_id is not None:
                try:
                    return metrics.get_shortest_path_to_object(
                        self._controller, object_id, initial_position=agent_pos
                    )
                except ValueError as exc:
                    logger.warning(
                        "Object path planning failed for {} ({}); falling back to point goal",
                        object_id,
                        exc,
                    )
            try:
                return self._path_to_point_locked(
                    agent_pos,
                    {"x": target["x"], "y": agent_pos["y"], "z": target["z"]},
                )
            except ValueError:
                return self._path_to_point_locked(
                    agent_pos, {"x": target["x"], "y": 0.0, "z": target["z"]}
                )
