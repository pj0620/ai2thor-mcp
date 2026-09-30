"""Live free-camera tests: boots the real AI2-THOR controller.

Run with: pytest regression/test_freecam_live.py -m live
"""

import json

import pytest
from fastmcp import Client

from ai2thor_mcp import main as server

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


def image_blocks(result):
    return [block for block in result.content if getattr(block, "type", "") == "image"]


@pytest.mark.asyncio
async def test_free_camera_view_move_and_reset():
    async with Client(server.mcp) as client:
        view = await client.call_tool("sim_camera_view", {})
        assert image_blocks(view) and image_blocks(view)[0].mimeType == "image/jpeg"
        pose = tool_json(view)
        assert set(pose) >= {"position", "yaw", "pitch", "fov"}

        moved = tool_json(await client.call_tool("sim_camera_move", {"forward": 0.5, "yaw": 30, "zoom": 10}))
        assert moved["yaw"] == pytest.approx((pose["yaw"] + 30) % 360, abs=0.05)
        assert moved["fov"] == pytest.approx(pose["fov"] - 10, abs=0.05)
        assert moved["position"] != pose["position"]

        top = tool_json(await client.call_tool("sim_camera_reset", {"mode": "top"}))
        assert top["pitch"] == pytest.approx(89)

        # the robot's own tools keep working and the free camera survives a robot step
        await client.call_tool("do_rotate", {"action": "RotateLeft", "degrees": 30})
        again = await client.call_tool("sim_camera_view", {})
        assert image_blocks(again) and tool_json(again)["pitch"] == pytest.approx(89)

        with pytest.raises(Exception):
            await client.call_tool("sim_camera_reset", {"mode": "sideways"})
