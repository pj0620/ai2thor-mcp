"""Live end-to-end tests: boots the real AI2-THOR controller (opens Unity).

Run with: pytest regression -m live
"""

import asyncio
import json
import time

import pytest
from fastmcp import Client

from ai2thor_mcp import main as server

pytestmark = pytest.mark.live


def tool_json(result):
    """Structured dict from a fastmcp call_tool result, tolerating either
    structured output or a JSON text block."""
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


@pytest.mark.asyncio
async def test_view_navigate_and_cancel():
    async with Client(server.mcp) as client:
        view = tool_json(await client.call_tool("get_current_view", {}))
        destinations = view["destinations"]
        assert destinations, "expected at least one destination in the view"
        assert any(d["kind"] == "object" for d in destinations)
        image_blocks = [
            b
            for b in (await client.call_tool("get_current_view", {})).content
            if getattr(b, "mimeType", "").startswith("image/")
        ]
        assert image_blocks, "get_current_view should include an annotated image"

        # drive to the nearest object destination
        with_distance = [d for d in destinations if "distance_m" in d]
        dest = min(with_distance, key=lambda d: d["distance_m"])
        plan = tool_json(
            await client.call_tool(
                "set_nav_goal", {"destination_id": dest["id"], "speed_mps": 1.0}
            )
        )
        assert plan["state"] == "following"
        assert plan["path"]["length_m"] >= 0

        deadline = time.monotonic() + 90
        status = None
        while time.monotonic() < deadline:
            status = tool_json(await client.call_tool("get_nav_status", {}))
            if status["state"] in ("succeeded", "canceled", "failed"):
                break
            await asyncio.sleep(1.0)
        assert status is not None
        assert status["state"] == "succeeded", f"navigation ended as {status}"

        # cancel semantics on a fresh goal
        far = max(with_distance, key=lambda d: d["distance_m"])
        tool_json(
            await client.call_tool(
                "set_nav_goal", {"destination_id": far["id"], "speed_mps": 0.2}
            )
        )
        await asyncio.sleep(1.0)
        cancel = tool_json(await client.call_tool("cancel_nav", {}))
        assert cancel["canceled"] is True
        final = tool_json(await client.call_tool("get_nav_status", {}))
        assert final["state"] == "canceled"


@pytest.mark.asyncio
async def test_unknown_destination_rejected():
    async with Client(server.mcp) as client:
        with pytest.raises(Exception) as excinfo:
            await client.call_tool("set_nav_goal", {"destination_id": "psg:999"})
        assert "Unknown destination" in str(excinfo.value)
