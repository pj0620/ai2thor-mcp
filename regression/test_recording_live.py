"""Live overhead-recording tests: boots the real AI2-THOR controller.

Run with: pytest regression/test_recording_live.py -m live
"""

import asyncio
import json
import os
import time

import pytest
from fastmcp import Client

from ai2thor_mcp import main as server

cv2 = pytest.importorskip("cv2")

pytestmark = pytest.mark.live


def tool_json(result):
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue
    pytest.fail(f"no JSON payload in tool result: {result.content!r}")


def probe(path):
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), f"cv2 could not open {path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="ignore")
    ok, first = cap.read()
    cap.release()
    assert ok, "could not decode the first frame"
    return fps, frames, width, height, fourcc.lower()


@pytest.mark.asyncio
async def test_follow_recording_and_state_errors():
    async with Client(server.mcp) as client:
        start = tool_json(await client.call_tool("start_overhead_recording", {}))
        assert start["recording"] is True
        assert start["mode"] == "follow"

        with pytest.raises(Exception) as excinfo:
            await client.call_tool("start_overhead_recording", {"mode": "scene"})
        assert "already active" in str(excinfo.value)

        await client.call_tool("do_rotate", {"action": "RotateRight", "degrees": 90})
        await client.call_tool("do_move", {"action": "MoveAhead", "moveMagnitude": 0.5})
        await asyncio.sleep(1.2)  # idle gap: held frames must keep the clock running

        stats = tool_json(await client.call_tool("end_overhead_recording", {}))
        assert stats["mode"] == "follow"
        assert stats["duration_s"] > 1.0
        assert os.path.exists(stats["path"])
        assert abs(stats["frames_written"] - stats["duration_s"] * stats["fps"]) <= 3

        fps, frames, width, height, fourcc = probe(stats["path"])
        assert fps == pytest.approx(10, abs=0.5)
        assert (width, height) == (500, 500)
        assert fourcc in ("avc1", "h264")
        assert abs(frames - stats["frames_written"]) <= 2

        # the follow camera actually tracked the agent
        event = server.SIM.last_event()
        cam = event.metadata["thirdPartyCameras"][0]["position"]
        agent = event.metadata["agent"]["position"]
        assert abs(cam["x"] - agent["x"]) < 0.5
        assert abs(cam["z"] - agent["z"]) < 0.5

        with pytest.raises(Exception) as excinfo2:
            await client.call_tool("end_overhead_recording", {})
        assert "No recording" in str(excinfo2.value)


@pytest.mark.asyncio
async def test_scene_recording_reuses_camera_with_projection_switch():
    async with Client(server.mcp) as client:
        start = tool_json(
            await client.call_tool("start_overhead_recording", {"mode": "scene"})
        )
        assert start["mode"] == "scene"
        await client.call_tool("do_rotate", {"action": "RotateLeft", "degrees": 45})
        await asyncio.sleep(0.6)
        stats = tool_json(await client.call_tool("end_overhead_recording", {}))
        assert stats["mode"] == "scene"
        assert os.path.exists(stats["path"])
        fps, frames, width, height, fourcc = probe(stats["path"])
        assert frames >= 5
        assert fourcc in ("avc1", "h264")
        # still exactly one third-party camera: reused, not re-added
        event = server.SIM.last_event()
        assert len(event.metadata["thirdPartyCameras"]) == 1
