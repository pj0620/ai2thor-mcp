
from typing import Any, Dict, Union
from ai2thor.controller import Controller
from loguru import logger


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


def create_ai2thor_controller():
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
