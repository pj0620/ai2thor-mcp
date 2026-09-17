"""Offline OverheadRecorder tests: stub sim, injected clock, real ffmpeg.

Flat gray frames survive CRF encoding nearly losslessly, so a cv2 probe can
identify which source frame occupies each video frame.
"""

import threading

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ai2thor_mcp.recorder import OverheadRecorder, RecorderError, RecorderStateError

FPS = 10


def flat(value):
    return np.full((64, 64, 3), value, np.uint8)


class FakeRecEvent:
    def __init__(self, metadata, frames, action_name=""):
        self.metadata = metadata
        self.third_party_camera_frames = frames
        self.action_name = action_name


class FakeRecSim:
    """Duck-typed Sim: tracks third-party cameras and an agent position."""

    def __init__(self):
        self.agent_xz = (0.0, 0.0)
        self.cameras = []
        self.log = []
        self.frame = flat(40)
        self.fail_actions = set()

    def make_event(self, action_name="", action_return=None, success=True, error=""):
        metadata = {
            "agent": {
                "position": {"x": self.agent_xz[0], "y": 0.9, "z": self.agent_xz[1]},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "cameraHorizon": 0.0,
            },
            "lastActionSuccess": success,
            "errorMessage": error,
            "thirdPartyCameras": [dict(c) for c in self.cameras],
            "actionReturn": action_return,
        }
        frames = [np.array(self.frame) for _ in self.cameras]
        return FakeRecEvent(metadata, frames, action_name)

    def step(self, action):
        name = action["action"]
        self.log.append(dict(action))
        if name in self.fail_actions:
            return self.make_event(name, success=False, error=f"{name} failed")
        if name == "AddThirdPartyCamera":
            self.cameras.append({k: v for k, v in action.items() if k != "action"})
        elif name == "UpdateThirdPartyCamera":
            cam = self.cameras[action["thirdPartyCameraId"]]
            cam.update(
                {k: v for k, v in action.items() if k not in ("action", "thirdPartyCameraId")}
            )
        elif name == "GetMapViewCameraProperties":
            return self.make_event(
                name,
                action_return={
                    "position": {"x": 1.0, "y": 2.6, "z": 1.0},
                    "rotation": {"x": 90.0, "y": 0.0, "z": 0.0},
                    "orthographicSize": 3.2,
                    "orthographic": True,
                },
            )
        return self.make_event(name)

    def refresh(self):
        return self.step({"action": "MoveAhead", "moveMagnitude": 0.0})


class FakeController:
    def __init__(self, sim):
        self.sim = sim

    def step(self, **kwargs):
        return self.sim.step(kwargs)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def probe_means(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"cv2 could not open {path}"
    means = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        means.append(float(img.mean()))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little")
    cap.release()
    return means, fps, fourcc.decode(errors="ignore").lower()


@pytest.fixture
def rig(tmp_path):
    sim = FakeRecSim()
    clock = FakeClock()
    recorder = OverheadRecorder(fps=FPS, out_dir=str(tmp_path), clock=clock, paced=False)
    return recorder, sim, clock


def test_state_errors(rig):
    recorder, sim, clock = rig
    with pytest.raises(RecorderStateError):
        recorder.end(sim)
    with pytest.raises(RecorderError):
        recorder.start(sim, "sideways")
    assert recorder.status()["state"] == "idle"

    recorder.start(sim, "follow")
    with pytest.raises(RecorderStateError):
        recorder.start(sim, "follow")
    stats = recorder.end(sim)
    assert stats["frames_written"] >= 1
    with pytest.raises(RecorderStateError):
        recorder.end(sim)


def test_start_failure_resets_idle(rig):
    recorder, sim, clock = rig
    sim.fail_actions = {"AddThirdPartyCamera"}
    with pytest.raises(RecorderError):
        recorder.start(sim, "follow")
    assert recorder.status()["state"] == "idle"
    sim.fail_actions = set()
    recorder.start(sim, "follow")
    recorder.end(sim)


def test_concurrent_start_one_winner(rig):
    recorder, sim, clock = rig
    barrier = threading.Barrier(2)
    results = []

    def attempt():
        barrier.wait()
        try:
            results.append(recorder.start(sim, "follow"))
        except RecorderStateError:
            results.append("rejected")

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("rejected") == 1
    recorder.end(sim)


def test_on_step_inert_while_idle(rig):
    recorder, sim, clock = rig
    event = sim.make_event()
    assert recorder.on_step(FakeController(sim), event) is event
    assert recorder.status()["frames_captured"] == 0


def test_on_step_never_raises_on_broken_events(rig):
    recorder, sim, clock = rig
    recorder.start(sim, "follow")
    no_frames = sim.make_event()
    no_frames.third_party_camera_frames = None
    recorder.on_step(FakeController(sim), no_frames)  # missing frames: hold, no failure
    assert recorder.status()["failed"] is False

    class Junk:
        metadata = {}

    junk = Junk()
    assert recorder.on_step(FakeController(sim), junk) is junk  # must not raise
    assert recorder.status()["failed"] is True  # ...but the recording is marked failed
    with pytest.raises(RecorderError):
        recorder.end(sim)  # surfaces at finalize; partial file + log kept
    assert recorder.status()["state"] == "idle"


def test_follow_repose_threshold(rig):
    recorder, sim, clock = rig
    recorder.start(sim, "follow")
    controller = FakeController(sim)

    sim.agent_xz = (0.05, 0.0)
    event = sim.make_event()
    returned = recorder.on_step(controller, event)
    assert returned is event
    assert not any(a["action"] == "UpdateThirdPartyCamera" for a in sim.log)

    sim.agent_xz = (0.30, 0.0)
    returned = recorder.on_step(controller, sim.make_event())
    updates = [a for a in sim.log if a["action"] == "UpdateThirdPartyCamera"]
    assert len(updates) == 1
    assert updates[0]["position"]["x"] == pytest.approx(0.30)
    assert returned.action_name == "UpdateThirdPartyCamera"
    recorder.end(sim)


def test_scene_mode_uses_map_view_properties(rig):
    recorder, sim, clock = rig
    recorder.start(sim, "scene")
    add = next(a for a in sim.log if a["action"] == "AddThirdPartyCamera")
    assert add["orthographic"] is True
    assert add["orthographicSize"] == pytest.approx(3.2)
    assert any(a["action"] == "GetMapViewCameraProperties" for a in sim.log)
    recorder.end(sim)


def test_camera_reused_across_recordings(rig):
    recorder, sim, clock = rig
    recorder.start(sim, "follow")
    recorder.end(sim)
    clock.t = 1.0
    recorder.start(sim, "follow")
    recorder.end(sim)
    adds = [a for a in sim.log if a["action"] == "AddThirdPartyCamera"]
    updates = [a for a in sim.log if a["action"] == "UpdateThirdPartyCamera"]
    assert len(adds) == 1
    assert len(updates) >= 1  # second start re-posed the existing camera
    assert len(sim.cameras) == 1


def test_hold_semantics_exact(rig):
    recorder, sim, clock = rig
    sim.frame = flat(40)  # frame A
    recorder.start(sim, "follow")

    clock.t = 0.30
    recorder._write_due_frames(clock.t)  # writer would have run by now

    clock.t = 0.35
    sim.frame = flat(200)  # frame B
    recorder.on_step(FakeController(sim), sim.make_event())

    clock.t = 0.50
    stats = recorder.end(sim)
    assert stats["frames_written"] == 5
    assert stats["video_duration_s"] == pytest.approx(0.5)

    means, fps, fourcc = probe_means(stats["path"])
    assert fps == pytest.approx(FPS, abs=0.5)
    assert fourcc in ("avc1", "h264")
    assert len(means) == 5
    assert all(abs(m - 40) < 10 for m in means[:3]), means
    assert all(abs(m - 200) < 10 for m in means[3:]), means


def test_minimum_one_frame(rig):
    recorder, sim, clock = rig
    recorder.start(sim, "follow")
    stats = recorder.end(sim)  # zero elapsed time
    assert stats["frames_written"] == 1
    means, _, _ = probe_means(stats["path"])
    assert len(means) == 1


def test_truncation_clamps_catchup(tmp_path):
    sim = FakeRecSim()
    clock = FakeClock()
    recorder = OverheadRecorder(
        fps=FPS, max_duration_s=0.3, out_dir=str(tmp_path), clock=clock, paced=False
    )
    recorder.start(sim, "follow")
    clock.t = 1.0
    stats = recorder.end(sim)
    assert stats["frames_written"] == 3
    assert stats["truncated"] is True


def test_paced_writer_smoke(tmp_path):
    import time

    sim = FakeRecSim()
    recorder = OverheadRecorder(fps=FPS, out_dir=str(tmp_path), paced=True)
    recorder.start(sim, "follow")
    time.sleep(0.55)
    recorder.on_step(FakeController(sim), sim.make_event())
    stats = recorder.end(sim)
    assert 3 <= stats["frames_written"] <= 10
    means, _, fourcc = probe_means(stats["path"])
    assert fourcc in ("avc1", "h264")
    assert len(means) == stats["frames_written"]
