from contextlib import asynccontextmanager
import io
from typing import Any, AsyncIterator, Dict
from fastmcp import FastMCP
from ai2thor_mcp.model import MOVE_ACTION
from PIL import Image as PILImage
from fastmcp.utilities.types import Image
from loguru import logger
from typing import Any, Dict, Union
from ai2thor.controller import Controller
from loguru import logger
from fastmcp.server.server import Transport


# initialize last_event with no-op action
_last_event: None | dict = None
_controller: None | Controller = None


def _describe_action(action: Union[str, Dict[str, Any]]) -> str:
    """Compact string for logging distinct actions."""
    if isinstance(action, str):
        return action
    action_name = action.get("action", "<custom>")
    params = {k: action[k] for k in action if k != "action"}
    return f"{action_name}({params})"


def _normalize_action(action: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    return action if isinstance(action, dict) else {"action": action}


def create_ai2thor_controller() -> Controller:
    controller_kwargs = {
        "agentMode": "locobot",
        "visibilityDistance": 1.5,
        # "scene": "FloorPlan_Train1_3",
        "scene": "FloorPlan2",
        "gridSize": 0.25,
        # "movementGaussianSigma": 0.005,
        # "rotateStepDegrees": 90,
        # "rotateGaussianSigma": 0.5,
        "renderDepthImage": True,
        "renderInstanceSegmentation": True,
        "width": 500,
        "height": 500,
        "fieldOfView": 60,
        # "allowHorizontalMovement": True
    }
    logger.info(
        "Creating AI2-THOR controller scene={} grid={}m",
        controller_kwargs["scene"],
        controller_kwargs["gridSize"],
    )
    return Controller(**controller_kwargs)


def create_sim():
    global _controller, _last_event
    logger.info("Bootstrapping AI2-THOR simulator")
    _controller = create_ai2thor_controller()
    _last_event = _controller.step(action="MoveAhead", moveMagnitude=0.0)
    bootstrap_status = _last_event.metadata.get("lastActionStatus") if _last_event else None
    logger.debug("Simulator ready (bootstrap status={})", bootstrap_status)


def get_last_event():
    global _last_event
    if _last_event is None:
        logger.warning("No cached event available; initializing simulator on demand")
        create_sim()
    return _last_event


def do_action(
    action: Union[str, Dict[str, Any]]
):
    global _controller, _last_event
    if _controller is None:
        logger.warning("Controller missing; creating simulator before action dispatch")
        create_sim()
    assert _controller
    normalized_action = _normalize_action(action)
    logger.info("Dispatching action {}", _describe_action(normalized_action))
    _last_event = _controller.step(**normalized_action)
    # status = _last_event.metadata.get("lastActionStatus") if _last_event else None
    logger.debug("Action {} completed {}", _describe_action(normalized_action), _last_event)
    return _last_event


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
    mcp.run(transport="streamable-http", host="0.0.0.0", stateless_http=True)
