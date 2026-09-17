"""NavManager state-machine tests against a kinematic fake sim (no Unity)."""

import time

import numpy as np
import pytest

from ai2thor_mcp.nav import NavManager


class FakeEvent:
    def __init__(self, agent, success=True, error=""):
        self.metadata = {
            "agent": agent,
            "lastActionSuccess": success,
            "errorMessage": error,
        }


class FakeSim:
    """Duck-typed Sim: teleports on MoveAhead, tracks yaw, scripted blockers."""

    def __init__(self, corners, blocked=None):
        self.pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.yaw = 0.0
        self.corners = corners
        self.blocked = blocked or (lambda p: False)
        self.plans = 0

    def _agent(self):
        return {
            "position": dict(self.pos),
            "rotation": {"x": 0.0, "y": self.yaw, "z": 0.0},
            "cameraHorizon": 0.0,
        }

    def peek_pose(self):
        return dict(self.pos), self.yaw, 0.0

    def agent_pose(self):
        return dict(self.pos), self.yaw, 0.0

    def step(self, action):
        name = action["action"]
        if name == "RotateRight":
            self.yaw = (self.yaw + action["degrees"]) % 360.0
        elif name == "RotateLeft":
            self.yaw = (self.yaw - action["degrees"]) % 360.0
        elif name == "MoveAhead":
            rad = np.deg2rad(self.yaw)
            new = {
                "x": self.pos["x"] + action["moveMagnitude"] * float(np.sin(rad)),
                "y": 0.0,
                "z": self.pos["z"] + action["moveMagnitude"] * float(np.cos(rad)),
            }
            if self.blocked(new):
                return FakeEvent(self._agent(), success=False, error="blocked")
            self.pos = new
        return FakeEvent(self._agent())

    def plan_path(self, target, object_id=None):
        self.plans += 1
        return [dict(c) for c in self.corners]


def wait_terminal(nav, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = nav.status()
        if status["state"] in ("succeeded", "canceled", "failed"):
            return status
        time.sleep(0.05)
    pytest.fail("nav did not reach a terminal state in time")


def test_follow_succeeds():
    sim = FakeSim([{"x": 0.0, "y": 0.0, "z": 0.5}, {"x": 0.0, "y": 0.0, "z": 1.5}])
    nav = NavManager(sim)
    summary = nav.set_goal("psg:1", {"x": 0.0, "y": 0.0, "z": 1.5}, "doorway", speed=1.0)
    assert summary["state"] == "following"
    assert summary["eta_s"] > 0
    assert summary["path"]["length_m"] == pytest.approx(1.5, abs=0.05)

    status = wait_terminal(nav)
    assert status["state"] == "succeeded"
    assert status["progress"]["percent"] == 100
    assert abs(sim.pos["z"] - 1.5) < 0.35


def test_turn_then_follow():
    sim = FakeSim([{"x": -1.0, "y": 0.0, "z": 0.0}])
    nav = NavManager(sim)
    nav.set_goal("psg:1", {"x": -1.0, "y": 0.0, "z": 0.0}, "opening", speed=1.0)
    status = wait_terminal(nav)
    assert status["state"] == "succeeded"
    assert abs(sim.pos["x"] + 1.0) < 0.35


def test_cancel_mid_follow():
    sim = FakeSim([{"x": 0.0, "y": 0.0, "z": 5.0}])
    nav = NavManager(sim)
    nav.set_goal("psg:1", {"x": 0.0, "y": 0.0, "z": 5.0}, "hallway", speed=0.3)
    time.sleep(0.6)
    assert nav.status()["state"] == "following"
    assert nav.cancel(wait=True)
    assert nav.status()["state"] == "canceled"
    assert 0.0 < sim.pos["z"] < 5.0
    assert nav.cancel(wait=True) is False  # nothing left to cancel


def test_preemption_replaces_goal():
    sim = FakeSim([{"x": 0.0, "y": 0.0, "z": 5.0}])
    nav = NavManager(sim)
    first = nav.set_goal("psg:1", {"x": 0.0, "y": 0.0, "z": 5.0}, "hallway", speed=0.3)
    time.sleep(0.3)
    second = nav.set_goal("psg:2", {"x": 0.0, "y": 0.0, "z": 5.0}, "hallway", speed=1.0)
    assert second["preempted_goal_id"] == first["goal_id"]
    status = wait_terminal(nav)
    assert status["state"] == "succeeded"
    assert status["goal"]["goal_id"] == second["goal_id"]


def test_blocked_fails_after_replan():
    sim = FakeSim(
        [{"x": 0.0, "y": 0.0, "z": 3.0}], blocked=lambda p: p["z"] > 1.0
    )
    nav = NavManager(sim)
    nav.set_goal("psg:1", {"x": 0.0, "y": 0.0, "z": 3.0}, "hallway", speed=1.0)
    status = wait_terminal(nav)
    assert status["state"] == "failed"
    assert "blocked" in status["reason"]
    assert status["collisions"] >= 3
    assert sim.plans >= 2  # initial plan + one replan


def test_status_idle_before_any_goal():
    nav = NavManager(FakeSim([]))
    status = nav.status()
    assert status["state"] == "idle"
    assert status["goal"] is None
    assert status["progress"] is None
