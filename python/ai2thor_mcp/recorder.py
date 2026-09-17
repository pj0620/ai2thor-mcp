"""Overhead mp4 recording of the agent via an AI2-THOR third-party camera.

One recording at a time, real-time semantics: the mp4 duration matches the
wall clock between start and end, with the last frame held through idle gaps.
Frames are captured passively from every Sim.step (the build has no way to
attach a camera to the agent, so follow mode re-poses the camera whenever the
agent has moved); a paced writer thread streams exactly one frame per 1/fps
tick into an ffmpeg pipe, so pipe I/O never happens under the Sim lock and
idle gaps never cause burst writes.

States: IDLE -> STARTING -> ACTIVE -> STOPPING -> IDLE, with `truncated` and
`failed` as flags within ACTIVE.

Locking invariants:
- The recorder lock is strictly innermost. The only permitted nesting is
  Sim._lock -> recorder._lock (the on_step path). Never call a Sim method,
  write to the pipe, spawn/wait/kill a process, or join a thread while
  holding the recorder lock — field reads/writes only.
- Inside on_step (Sim lock held by the caller), camera re-posing uses
  controller.step(...) directly, never Sim.step (non-reentrant Sim lock).
- start()/end() take their atomicity from the state gates: flip the state
  under the lock in microseconds, then do all real work with no locks held.
- The pipe has a single writer at any time: the writer thread while ACTIVE,
  then exclusively end() after the thread is joined.
"""

import enum
import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import imageio_ffmpeg
import numpy as np
from loguru import logger

from ai2thor_mcp import config


class RecorderError(RuntimeError):
    """Recording failed or could not be started."""


class RecorderStateError(RecorderError):
    """start while a recording is active, or end while none is."""


class _State(enum.Enum):
    IDLE = "idle"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"


class OverheadRecorder:
    def __init__(
        self,
        *,
        fps: int = config.RECORD_FPS,
        max_duration_s: float = config.MAX_RECORDING_S,
        out_dir: str = config.RECORDINGS_DIR,
        clock: Callable[[], float] = time.monotonic,
        paced: bool = True,
        ffmpeg_exe: Optional[str] = None,
    ) -> None:
        self._fps = fps
        self._max_frames = max(1, int(max_duration_s * fps))
        self._max_duration_s = max_duration_s
        self._out_dir = out_dir
        self._clock = clock
        self._paced = paced
        self._ffmpeg_exe = ffmpeg_exe

        self._lock = threading.Lock()
        self._state = _State.IDLE
        self._camera_index: Optional[int] = None  # persists across recordings

        # per-recording fields (reset by _clear_recording_locked)
        self._mode = ""
        self._proc: Optional[subprocess.Popen] = None
        self._log_fh = None
        self._path = ""
        self._t0 = 0.0
        self._latest_frame: Optional[np.ndarray] = None
        self._frames_captured = 0
        self._frames_written = 0
        self._truncated = False
        self._failed = False
        self._fail_reason: Optional[str] = None
        self._stop_writer = threading.Event()
        self._writer_thread: Optional[threading.Thread] = None
        self._cam_xz: Optional[Tuple[float, float]] = None
        self._warned_repose = False

    # --- public API (sim is duck-typed: .step(dict) -> event, .refresh() -> event) ---

    def start(self, sim, mode: str) -> dict:
        if mode not in ("follow", "scene"):
            raise RecorderError(f"Unknown mode '{mode}': use 'follow' or 'scene'.")
        with self._lock:
            if self._state is not _State.IDLE:
                elapsed = self._clock() - self._t0
                if self._state is _State.ACTIVE:
                    raise RecorderStateError(
                        f"A recording is already active (started {elapsed:.0f}s ago)."
                        " Call end_overhead_recording first."
                    )
                raise RecorderStateError(
                    f"A recording is currently {self._state.value}; retry in a moment."
                )
            self._state = _State.STARTING

        proc = None
        log_fh = None
        path = ""
        try:
            event = sim.refresh()
            pose = self._pose_kwargs(mode, event, sim)

            cameras = event.metadata.get("thirdPartyCameras") or []
            if self._camera_index is None or self._camera_index >= len(cameras):
                event = sim.step({"action": "AddThirdPartyCamera", **pose})
                if not event.metadata["lastActionSuccess"]:
                    raise RecorderError(
                        f"AddThirdPartyCamera failed: {event.metadata.get('errorMessage')}"
                    )
                index = len(event.metadata["thirdPartyCameras"]) - 1
            else:
                index = self._camera_index
                event = sim.step(
                    {"action": "UpdateThirdPartyCamera", "thirdPartyCameraId": index, **pose}
                )
                if not event.metadata["lastActionSuccess"]:
                    raise RecorderError(
                        f"UpdateThirdPartyCamera failed: {event.metadata.get('errorMessage')}"
                    )

            frames = getattr(event, "third_party_camera_frames", None)
            if frames is None or len(frames) <= index:
                raise RecorderError("no third-party camera frame in the event")
            first = np.ascontiguousarray(frames[index])
            height, width = first.shape[:2]

            path = self._new_output_path(mode)
            log_fh = open(path + ".log", "wb")
            proc = self._spawn_ffmpeg(width, height, path, log_fh)
            t0 = self._clock()

            agent = event.metadata["agent"]["position"]
            with self._lock:
                self._camera_index = index
                self._mode = mode
                self._proc = proc
                self._log_fh = log_fh
                self._path = path
                self._t0 = t0
                self._latest_frame = first
                self._frames_captured = 1
                self._frames_written = 0
                self._truncated = False
                self._failed = False
                self._fail_reason = None
                self._stop_writer = threading.Event()
                self._cam_xz = (agent["x"], agent["z"]) if mode == "follow" else None
                self._warned_repose = False
                self._state = _State.ACTIVE
            if self._paced:
                self._writer_thread = threading.Thread(
                    target=self._writer_loop, name="overhead-writer", daemon=True
                )
                self._writer_thread.start()
            logger.info("Overhead recording started ({}) -> {}", mode, path)
            return {
                "recording": True,
                "mode": mode,
                "fps": self._fps,
                "max_duration_s": self._max_duration_s,
                "path": os.path.abspath(path),
            }
        except BaseException:
            with self._lock:
                self._state = _State.IDLE
            if proc is not None:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
            if log_fh is not None:
                try:
                    log_fh.close()
                    os.remove(path + ".log")
                except OSError:
                    pass
            if path:
                try:
                    if os.path.exists(path) and os.path.getsize(path) == 0:
                        os.remove(path)
                except OSError:
                    pass
            raise

    def end(self, sim) -> dict:
        with self._lock:
            if self._state is _State.IDLE:
                raise RecorderStateError(
                    "No recording is active. Call start_overhead_recording first."
                )
            if self._state is not _State.ACTIVE:
                raise RecorderStateError(
                    f"Recording is currently {self._state.value}; retry in a moment."
                )
        # one last fresh frame via the hook — best effort, Unity may be dead
        try:
            sim.refresh()
        except Exception as exc:
            logger.warning("Final refresh before finalizing failed: {!r}", exc)

        with self._lock:
            if self._state is not _State.ACTIVE:
                raise RecorderStateError("No recording is active (another end() won).")
            self._state = _State.STOPPING
            proc = self._proc
            log_fh = self._log_fh
            path = self._path
            mode = self._mode
            t0 = self._t0
            stop = self._stop_writer
            writer = self._writer_thread
            frames_captured = self._frames_captured
        t_end = self._clock()

        stop.set()
        if writer is not None and writer.is_alive():
            writer.join(10)
            if writer.is_alive():
                # a wedged pipe write only unblocks when ffmpeg dies
                logger.warning("Writer thread stuck; killing ffmpeg to unblock it")
                try:
                    proc.kill()
                except Exception:
                    pass
                writer.join(5)
                with self._lock:
                    self._failed = True
                    self._fail_reason = self._fail_reason or "writer thread stuck"

        self._write_due_frames(t_end)
        with self._lock:
            if self._frames_written == 0 and not self._failed:
                frame = self._latest_frame
            else:
                frame = None
        if frame is not None:  # >=1-frame guard for instant start/end
            try:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                with self._lock:
                    self._frames_written = 1
            except OSError as exc:
                with self._lock:
                    self._failed = True
                    self._fail_reason = f"ffmpeg pipe: {exc!r}"

        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            returncode = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            returncode = proc.wait(timeout=5)
            with self._lock:
                self._failed = True
                self._fail_reason = self._fail_reason or "ffmpeg did not exit"
        try:
            log_fh.close()
        except OSError:
            pass

        with self._lock:
            frames_written = self._frames_written
            truncated = self._truncated
            failed = self._failed
            reason = self._fail_reason
            self._clear_recording_locked()

        size = os.path.getsize(path) if os.path.exists(path) else 0
        if failed or returncode != 0 or size == 0:
            raise RecorderError(
                f"recording failed ({reason or f'ffmpeg exit {returncode}'});"
                f" partial file and ffmpeg log kept at {os.path.abspath(path)}[.log]"
            )
        try:
            os.remove(path + ".log")
        except OSError:
            pass
        duration = t_end - t0
        stats = {
            "path": os.path.abspath(path),
            "mode": mode,
            "fps": self._fps,
            "duration_s": round(duration, 2),
            "video_duration_s": round(frames_written / self._fps, 2),
            "frames_captured": frames_captured,
            "frames_written": frames_written,
            "size_mb": round(size / 1e6, 2),
            "truncated": truncated,
            "note": "H.264 mp4 — play with: open <path>",
        }
        logger.info(
            "Overhead recording finished: {} ({}s, {} frames)",
            path,
            stats["duration_s"],
            frames_written,
        )
        return stats

    def on_step(self, controller, event) -> Any:
        """Capture hook, called by Sim.step with the Sim lock held. Never raises."""
        if self._state is not _State.ACTIVE:  # unlocked fast path
            return event
        try:
            with self._lock:
                if self._state is not _State.ACTIVE or self._failed or self._truncated:
                    return event
                index = self._camera_index
                mode = self._mode
                cam_xz = self._cam_xz

            if mode == "follow" and cam_xz is not None:
                agent = event.metadata["agent"]["position"]
                dx = agent["x"] - cam_xz[0]
                dz = agent["z"] - cam_xz[1]
                if (dx * dx + dz * dz) ** 0.5 > config.OVERHEAD_REPOSITION_M:
                    repose = controller.step(
                        action="UpdateThirdPartyCamera",
                        thirdPartyCameraId=index,
                        position={
                            "x": agent["x"],
                            "y": config.OVERHEAD_HEIGHT_M,
                            "z": agent["z"],
                        },
                    )
                    if repose.metadata["lastActionSuccess"]:
                        event = repose
                        with self._lock:
                            self._cam_xz = (agent["x"], agent["z"])
                    else:
                        with self._lock:
                            warned = self._warned_repose
                            self._warned_repose = True
                        if not warned:
                            logger.warning(
                                "UpdateThirdPartyCamera failed: {}",
                                repose.metadata.get("errorMessage"),
                            )

            frames = getattr(event, "third_party_camera_frames", None)
            if frames is not None and len(frames) > index:
                frame = np.ascontiguousarray(frames[index])
                with self._lock:
                    if self._state is _State.ACTIVE:
                        self._latest_frame = frame
                        self._frames_captured += 1
        except Exception as exc:
            with self._lock:
                self._failed = True
                self._fail_reason = self._fail_reason or f"capture error: {exc!r}"
            logger.exception("Overhead capture failed; recording marked failed")
        return event

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self._state.value,
                "mode": self._mode,
                "path": os.path.abspath(self._path) if self._path else None,
                "elapsed_s": round(self._clock() - self._t0, 2)
                if self._state is _State.ACTIVE
                else None,
                "frames_captured": self._frames_captured,
                "frames_written": self._frames_written,
                "truncated": self._truncated,
                "failed": self._failed,
            }

    # --- internals ---

    def _clear_recording_locked(self) -> None:
        self._state = _State.IDLE
        self._mode = ""
        self._proc = None
        self._log_fh = None
        self._path = ""
        self._latest_frame = None
        self._writer_thread = None
        self._cam_xz = None

    def _pose_kwargs(self, mode: str, event, sim) -> Dict[str, Any]:
        if mode == "follow":
            agent = event.metadata["agent"]["position"]
            return {
                "position": {
                    "x": agent["x"],
                    "y": config.OVERHEAD_HEIGHT_M,
                    "z": agent["z"],
                },
                "rotation": {"x": 90.0, "y": 0.0, "z": 0.0},
                "fieldOfView": config.OVERHEAD_FOV_DEG,
                "orthographic": False,
                "skyboxColor": "black",
            }
        props_event = sim.step({"action": "GetMapViewCameraProperties"})
        props = props_event.metadata.get("actionReturn")
        if not props_event.metadata["lastActionSuccess"] or not props:
            raise RecorderError(
                f"GetMapViewCameraProperties failed: {props_event.metadata.get('errorMessage')}"
            )
        return {
            "position": props["position"],
            "rotation": props["rotation"],
            "orthographic": True,
            "orthographicSize": props["orthographicSize"],
            "skyboxColor": "black",
        }

    def _new_output_path(self, mode: str) -> str:
        os.makedirs(self._out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = os.path.join(self._out_dir, f"overhead_{mode}_{stamp}")
        path = base + ".mp4"
        n = 2
        while os.path.exists(path):
            path = f"{base}_{n}.mp4"
            n += 1
        return path

    def _spawn_ffmpeg(self, width: int, height: int, path: str, log_fh) -> subprocess.Popen:
        exe = self._ffmpeg_exe or imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            exe,
            "-hide_banner", "-loglevel", "error", "-nostats",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(self._fps),
            "-i", "pipe:0",
            "-an", "-c:v", "libx264",
            "-preset", config.FFMPEG_PRESET,
            "-crf", str(config.FFMPEG_CRF),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        if width % 2 or height % 2:
            cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        cmd += ["-y", path]
        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log_fh
        )

    def _writer_loop(self) -> None:
        period = 1.0 / self._fps
        stop = self._stop_writer
        while not stop.is_set():
            with self._lock:
                if self._truncated or self._failed or self._state is not _State.ACTIVE:
                    return
                k = self._frames_written
                t0 = self._t0
            delay = (t0 + (k + 1) * period) - self._clock()
            if delay > 0 and stop.wait(delay):
                return
            if stop.is_set():
                return
            self._write_due_frames(self._clock())

    def _write_due_frames(self, now: float) -> None:
        """Write every frame due by `now` (hold semantics), clamped to the cap.

        Single-writer discipline: called from the writer thread while ACTIVE,
        or from end() after that thread is joined — never both at once.
        """
        with self._lock:
            t0 = self._t0
            proc = self._proc
        if proc is None:
            return
        due = min(int((now - t0) * self._fps), self._max_frames)
        while True:
            with self._lock:
                if self._failed or self._frames_written >= due:
                    return
                frame = self._latest_frame
            if frame is None:
                return
            data = frame.tobytes()
            try:
                proc.stdin.write(data)  # may block on the pipe; no locks held
            except OSError as exc:
                with self._lock:
                    self._failed = True
                    self._fail_reason = f"ffmpeg pipe: {exc!r}"
                return
            with self._lock:
                self._frames_written += 1
                if self._frames_written >= self._max_frames:
                    self._truncated = True
                    logger.warning(
                        "Recording hit the {}s cap; no further footage is added",
                        self._max_duration_s,
                    )
                    return
