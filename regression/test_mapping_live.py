"""Live: the occupancy map fills in as the real robot looks around (boots Unity).

Run with: pytest regression/test_mapping_live.py -m live
"""

import base64
import io

import numpy as np
import pytest
from fastmcp import Client
from PIL import Image

from ai2thor_mcp import main as server
from ai2thor_mcp.mapping import FREE_WHITE, OCCUPIED_BLACK, UNKNOWN_GREY

pytestmark = pytest.mark.live


def map_pixels(result) -> np.ndarray:
    block = next(b for b in result.content if getattr(b, "type", "") == "image")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(block.data))).convert("L"))


@pytest.mark.asyncio
async def test_map_has_walls_free_space_and_grows_with_rotation():
    async with Client(server.mcp) as client:
        await client.call_tool("get_current_view", {})
        first = map_pixels(await client.call_tool("get_map", {}))
        assert {UNKNOWN_GREY, FREE_WHITE, OCCUPIED_BLACK} <= set(np.unique(first).tolist())
        known_before = int((first != UNKNOWN_GREY).sum())
        for _ in range(3):
            await client.call_tool("do_rotate", {"action": "RotateRight", "degrees": 90})
        after = map_pixels(await client.call_tool("get_map", {}))
        assert int((after != UNKNOWN_GREY).sum()) > known_before * 1.3
