from contextlib import asynccontextmanager
import io
from typing import Any, AsyncIterator, Dict
from fastmcp import FastMCP
from ai2thor_mcp.model import MOVE_ACTION
from PIL import Image as PILImage
from fastmcp.utilities.types import Image
from loguru import logger

from ai2thor_mcp.controller import create_sim, do_action, get_last_event


mcp = FastMCP(
    name="Ai2thorMcpServer",
    instructions="Use the MCP to control an agent in the Ai2thor environment.",
    # lifespan=server_lifespan
)


@mcp.resource(
    uri="view://rgb",
    name="RGB View",
    description="Provides view from robots RGB camera",
    mime_type="image/png"
)
def rgb_view():
    logger.debug("Fetching RGB view from last event")
    last_event = get_last_event()
    rgb_arr = last_event.frame
    rgb_image = PILImage.fromarray(rgb_arr)
    img_byte_arr = io.BytesIO()
    rgb_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


@mcp.tool()
def do_move(
    action: MOVE_ACTION,
    moveMagnitude: float = 1.
):
    action_dict = dict(action=action, moveMagnitude=moveMagnitude)
    logger.info(
        "Executing move action {} with magnitude {}",
        action,
        moveMagnitude,
    )
    last_event = do_action(action_dict)

    rgb_arr = last_event.frame
    rgb_image = PILImage.fromarray(rgb_arr)
    img_byte_arr = io.BytesIO()
    rgb_image.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    logger.debug("Move action {} complete; returning JPEG frame", action)
    return last_event.metadata


@mcp.tool()
def get_current_view():
    """gets current view of robot from rgb camera

    Returns:
        Image: rgb image from robots camera
    """
    rgb_arr = get_last_event().frame
    rgb_image = PILImage.fromarray(rgb_arr)
    img_byte_arr = io.BytesIO()
    rgb_image.save(img_byte_arr, format='jpeg')
    img_byte_arr = img_byte_arr.getvalue()
    logger.debug("Move action {} complete; returning JPEG frame", action)
    return Image(
        data=img_byte_arr,
        format="jpeg"
    )


if __name__ == "__main__":
    mcp.run()
