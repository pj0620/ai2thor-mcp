"""nav2-style asynchronous goal following.

NavManager mirrors the semantics of a nav2 NavigateToPose action server so
the real robot (ROS2 nav2 + SLAM) can implement the same MCP contract:
set_goal returns immediately and a background thread walks the planned
corners; status is polled; cancel preempts. States:
idle | planning | following | succeeded | canceled | failed.

Locking: _state_lock guards only NavManager's own fields and is never held
across a sim call; the sim's internal lock serializes Unity steps. The two
locks never nest. Pacing sleeps use cancel.wait so a cancel interrupts them
immediately, and they happen outside any lock.

The sim dependency is duck-typed (step/agent_pose/peek_pose/plan_path) so
tests can drive the manager with a kinematic fake.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

from ai2thor_mcp import config
from ai2thor_mcp.perception import wrap_deg


def _dist_xz(a: Dict[str, float], b: Dict[str, float]) -> float:
    return float(np.hypot(a["x"] - b["x"], a["z"] - b["z"]))


def _cum_from(corners: List[Dict[str, float]]) -> List[float]:
    """cum[i] = path length from corner i through the last corner."""
    cum = [0.0] * len(corners)
    for i in range(len(corners) - 2, -1, -1):
        cum[i] = cum[i + 1] + _dist_xz(corners[i], corners[i + 1])
    return cum


@dataclass
class NavGoal:
    goal_id: str
    dest_id: str
    label: str
    target: Dict[str, float]
    speed: float
    corners: List[Dict[str, float]] = field(default_factory=list)
    cum_from: List[float] = field(default_factory=list)


class NavManager:
    def __init__(self, sim) -> None:
        self._sim = sim
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._phase: str = "idle"
        self._goal: Optional[NavGoal] = None
        self._reason: Optional[str] = None
        self._wp_index = 0
        self._traveled = 0.0
        self._collisions = 0

    # --- public API ---

    def set_goal(
        self,
        dest_id: str,
        target: Dict[str, float],
        label: str,
        speed: float,
        object_id: Optional[str] = None,
    ) -> dict:
        preempted: Optional[str] = None
        with self._state_lock:
            if self._phase in ("planning", "following") and self._goal is not None:
                preempted = self._goal.goal_id
        self.cancel(wait=True)

        with self._state_lock:
            self._phase = "planning"
            self._reason = None
        try:
            corners = self._sim.plan_path(target, object_id)
        except ValueError as exc:
            with self._state_lock:
                self._phase = "failed"
                self._reason = f"planning failed: {exc}"
            raise
        if not corners:
            with self._state_lock:
                self._phase = "failed"
                self._reason = "planning returned an empty path"
            raise ValueError("planner returned an empty path")

        goal = NavGoal(
            goal_id=uuid.uuid4().hex[:8],
            dest_id=dest_id,
            label=label,
            target=dict(target),
            speed=speed,
            corners=[dict(c) for c in corners],
            cum_from=_cum_from(corners),
        )
        pos, _, _ = self._sim.agent_pose()
        planned_len = _dist_xz(pos, goal.corners[0]) + goal.cum_from[0]

        cancel = threading.Event()
        with self._state_lock:
            self._goal = goal
            self._phase = "following"
            self._wp_index = 0
            self._traveled = 0.0
            self._collisions = 0
            self._cancel = cancel
        self._thread = threading.Thread(
            target=self._follow, args=(goal, cancel), name=f"nav-{goal.goal_id}", daemon=True
        )
        self._thread.start()
        logger.info(
            "Nav goal {} -> {} ({:.2f} m, {} waypoints, {:.2f} m/s)",
            goal.goal_id,
            dest_id,
            planned_len,
            len(goal.corners),
            speed,
        )
        return {
            "goal_id": goal.goal_id,
            "destination_id": dest_id,
            "label": label,
            "state": "following",
            "path": {
                "waypoints": [[round(c["x"], 2), round(c["z"], 2)] for c in goal.corners],
                "length_m": round(planned_len, 2),
            },
            "speed_mps": speed,
            "eta_s": round(planned_len / speed, 1),
            "preempted_goal_id": preempted,
        }

    def cancel(self, wait: bool = True, timeout: float = 15.0) -> bool:
        thread = self._thread
        if thread is None or not thread.is_alive():
            return False
        self._cancel.set()
        if wait:
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("Nav thread did not stop within {}s", timeout)
        return True

    def preempt_for_manual(self) -> bool:
        return self.cancel(wait=True)

    def path_snapshot(self) -> Optional[List[Dict[str, float]]]:
        """Remaining corners of the active path, for map overlays."""
        with self._state_lock:
            if self._phase != "following" or self._goal is None:
                return None
            return [dict(c) for c in self._goal.corners[self._wp_index :]]

    def goal_target(self) -> Optional[Dict[str, float]]:
        with self._state_lock:
            if self._goal is None or self._phase not in ("planning", "following"):
                return None
            return dict(self._goal.target)

    def status(self) -> dict:
        with self._state_lock:
            phase = self._phase
            goal = self._goal
            wp_index = self._wp_index
            traveled = self._traveled
            collisions = self._collisions
            reason = self._reason
            corners = [dict(c) for c in goal.corners] if goal else []
            cum = list(goal.cum_from) if goal else []

        agent = None
        pose = self._sim.peek_pose()
        if pose is not None:
            pos, yaw, _ = pose
            agent = {"x": round(pos["x"], 2), "z": round(pos["z"], 2), "yaw": round(yaw, 1)}

        out = {
            "state": phase,
            "goal": None,
            "progress": None,
            "agent": agent,
            "collisions": collisions,
            "reason": reason,
        }
        if goal is None:
            return out
        out["goal"] = {
            "goal_id": goal.goal_id,
            "destination_id": goal.dest_id,
            "label": goal.label,
        }
        remaining = 0.0
        if phase == "succeeded" or not corners:
            remaining = 0.0
        elif pose is not None:
            i = min(wp_index, len(corners) - 1)
            remaining = _dist_xz(pose[0], corners[i]) + cum[i]
        total = traveled + remaining
        out["progress"] = {
            "distance_traveled_m": round(traveled, 2),
            "distance_remaining_m": round(remaining, 2),
            "percent": 100 if phase == "succeeded" else int(100 * traveled / total) if total > 1e-6 else 0,
            "eta_s": round(remaining / goal.speed, 1),
            "waypoint": min(wp_index + 1, len(corners)) if corners else 0,
            "waypoints_total": len(corners),
        }
        return out

    # --- follower thread ---

    def _finish(self, phase: str, reason: Optional[str] = None) -> None:
        with self._state_lock:
            self._phase = phase
            self._reason = reason
        if reason:
            logger.info("Nav finished: {} ({})", phase, reason)
        else:
            logger.info("Nav finished: {}", phase)

    def _follow(self, goal: NavGoal, cancel: threading.Event) -> None:
        try:
            self._run_follow(goal, cancel)
        except Exception as exc:  # Unity pipe hiccups etc. — never kill silently
            logger.exception("Nav goal {} crashed", goal.goal_id)
            self._finish("failed", f"internal error: {exc!r}")

    def _run_follow(self, goal: NavGoal, cancel: threading.Event) -> None:
        retries = 0
        replanned = False
        nudge_right = True
        skip_aim = False  # after a nudge, try the move once without re-aiming
        i = 0
        while i < len(goal.corners):
            with self._state_lock:
                self._wp_index = i
            corner = goal.corners[i]
            restart = False
            while True:
                if cancel.is_set():
                    self._finish("canceled")
                    return
                pos, yaw, _ = self._sim.agent_pose()
                dist = _dist_xz(pos, corner)
                if dist < config.WAYPOINT_TOL_M:
                    break

                bearing = float(
                    np.degrees(np.arctan2(corner["x"] - pos["x"], corner["z"] - pos["z"]))
                )
                err = wrap_deg(bearing - yaw)
                if abs(err) > config.ROT_TOL_DEG and not skip_aim:
                    deg = min(abs(err), config.MAX_ROT_STEP_DEG)
                    started = time.monotonic()
                    self._sim.step(
                        {
                            "action": "RotateRight" if err > 0 else "RotateLeft",
                            "degrees": deg,
                        }
                    )
                    cancel.wait(
                        max(0.0, deg / config.ROT_SPEED_DPS - (time.monotonic() - started))
                    )
                    continue

                step_len = max(
                    config.MOVE_STEP_MIN_M,
                    min(goal.speed * config.STEP_PERIOD_S, dist, config.MOVE_STEP_MAX_M),
                )
                started = time.monotonic()
                event = self._sim.step({"action": "MoveAhead", "moveMagnitude": step_len})
                skip_aim = False
                new_pos = event.metadata["agent"]["position"]
                if not event.metadata["lastActionSuccess"]:
                    with self._state_lock:
                        self._collisions += 1
                    retries += 1
                    if retries <= config.BLOCK_RETRIES:
                        self._sim.step(
                            {
                                "action": "RotateRight" if nudge_right else "RotateLeft",
                                "degrees": config.BLOCK_NUDGE_DEG,
                            }
                        )
                        nudge_right = not nudge_right
                        skip_aim = True
                        continue
                    if not replanned:
                        replanned = True
                        retries = 0
                        logger.info("Nav blocked; replanning once")
                        try:
                            corners = self._sim.plan_path(goal.target, None)
                        except ValueError as exc:
                            self._finish("failed", f"blocked, replan failed: {exc}")
                            return
                        with self._state_lock:
                            goal.corners = [dict(c) for c in corners]
                            goal.cum_from = _cum_from(goal.corners)
                            self._wp_index = 0
                        i = 0
                        restart = True
                        break
                    self._finish(
                        "failed", f"blocked near ({pos['x']:.2f}, {pos['z']:.2f})"
                    )
                    return
                retries = 0
                moved = _dist_xz(pos, new_pos)
                with self._state_lock:
                    self._traveled += moved
                cancel.wait(
                    max(0.0, step_len / goal.speed - (time.monotonic() - started))
                )
            if restart:
                continue
            i += 1

        pos, _, _ = self._sim.agent_pose()
        end = goal.corners[-1]
        d_end = _dist_xz(pos, end)
        # object goals end at the nearest navigable point, not the object itself,
        # so arrival is measured against the path end rather than the raw target
        if d_end < config.GOAL_TOL_M:
            self._finish("succeeded")
        else:
            self._finish("failed", f"stopped {d_end:.2f} m short of the path end")
