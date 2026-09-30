import argparse
import io
from typing import Any, Dict, Literal, Union

import numpy as np
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.utilities.types import Image
from loguru import logger
from PIL import Image as PILImage

from ai2thor_mcp import config
from ai2thor_mcp.annotate import annotate_view
from ai2thor_mcp.freecam import FreeCamera, FreeCameraError
from ai2thor_mcp.mapping import OccupancyMapper
from ai2thor_mcp.describe import (
    DestinationRegistry,
    describe_object,
    describe_passage,
)
from ai2thor_mcp.model import MOVE_ACTION, ROTATE_ACTION
from ai2thor_mcp.nav import NavManager
from ai2thor_mcp.perception import detect_passages, intrinsics_from_metadata
from ai2thor_mcp.recorder import OverheadRecorder, RecorderError
from ai2thor_mcp.sim import Sim
from ai2thor_mcp.utils import MapOverlays

SIM = Sim()
REGISTRY = DestinationRegistry()
NAV = NavManager(SIM)
RECORDER = OverheadRecorder()
SIM.attach_recorder(RECORDER)
FREECAM = FreeCamera()
MAPPER = OccupancyMapper()
SIM.attach_mapper(MAPPER)

CAMERA_PRESET = Literal["chase", "front", "top"]


def _describe_action(action: Union[str, Dict[str, Any]]) -> str:
    """Compact string for logging distinct actions."""
    if isinstance(action, str):
        return action
    action_name = action.get("action", "<custom>")
    params = {k: action[k] for k in action if k != "action"}
    return f"{action_name}({params})"


def _normalize_action(action: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    return action if isinstance(action, dict) else {"action": action}


def create_sim():
    logger.info("Bootstrapping AI2-THOR simulator")
    SIM.refresh()


def get_last_event():
    return SIM.last_event()


def do_action(action: Union[str, Dict[str, Any]]):
    normalized_action = _normalize_action(action)
    logger.info("Dispatching action {}", _describe_action(normalized_action))
    return SIM.step(normalized_action)


def _image_from(pil_image: PILImage.Image, fmt: str) -> Image:
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return Image(data=buf.getvalue(), format=fmt.lower())


mcp = FastMCP(
    name="Ai2thorMcpServer",
    instructions=(
        "Control a camera-only home robot in the AI2-THOR simulator. "
        "Call get_current_view to see the world and its selectable destinations, "
        "then set_nav_goal with a destination id to drive there; poll get_nav_status "
        "and use cancel_nav to stop. do_move/do_rotate give manual control."
    ),
)


@mcp.resource(
    uri="view://rgb",
    name="RGB View",
    description="Provides view from robots RGB camera",
    mime_type="image/png",
)
def rgb_view():
    logger.debug("Fetching RGB view from last event")
    last_event = get_last_event()
    rgb_image = PILImage.fromarray(last_event.frame)
    img_byte_arr = io.BytesIO()
    rgb_image.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


@mcp.tool(output_schema=None)
def get_current_view():
    """Gets the robot's current camera view, annotated with selectable destinations.

    Returns the RGB frame with numbered markers plus a JSON list of destinations:
    visible objects (green boxes) and drivable passages detected from depth
    (cyan boxes: doorways, hallways, wide openings that fit the robot). Pass any
    destination "id" to set_nav_goal to drive there. Marker numbers restart each
    view; ids are stable.
    """
    event = SIM.refresh()
    meta = event.metadata
    agent = meta["agent"]
    agent_pos = agent["position"]
    yaw = float(agent["rotation"]["y"])
    horizon = float(agent["cameraHorizon"])
    intr = intrinsics_from_metadata(meta)

    # objects with a usable 2D detection in this frame (the metadata "visible"
    # flag only means within interaction range, so it is not used here)
    detections = event.instance_detections2D
    object_records = []
    for obj in meta["objects"]:
        if obj["objectType"] in config.EXCLUDED_OBJECT_TYPES:
            continue
        bbox = detections.get(obj["objectId"]) if detections is not None else None
        if bbox is None:
            continue
        if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) < config.MIN_OBJECT_BBOX_AREA_PX:
            continue
        rec = REGISTRY.upsert_object(
            obj, describe_object(obj, agent_pos, yaw), bbox, obj["distance"]
        )
        object_records.append(rec)
    object_records.sort(key=lambda r: r.distance_m or 0.0)
    object_records = object_records[: config.MAX_OBJECT_DESTINATIONS]

    # passages: from depth, validated against reachable positions
    reachable = SIM.get_reachable_positions()
    passages = detect_passages(
        event.depth_frame,
        meta["cameraPosition"],
        agent_pos,
        yaw,
        horizon,
        intr,
        reachable_xz=reachable,
    )
    passage_records = []
    for cand in passages:
        distance = float(
            np.hypot(
                cand.waypoint["x"] - agent_pos["x"], cand.waypoint["z"] - agent_pos["z"]
            )
        )
        passage_records.append(
            REGISTRY.upsert_passage(cand, describe_passage(cand), distance)
        )

    # marker numbers are per-view: clear all, then number this view's records
    for rec in REGISTRY.list():
        rec.marker = None
    view_records = object_records + passage_records
    for n, rec in enumerate(view_records, start=1):
        rec.marker = n

    annotated = annotate_view(event.frame, view_records)

    destinations = []
    for rec in view_records:
        entry: Dict[str, Any] = {
            "marker": rec.marker,
            "id": rec.id,
            "kind": rec.kind,
            "label": rec.label,
            "description": rec.description,
        }
        if rec.distance_m is not None:
            entry["distance_m"] = round(rec.distance_m, 1)
        if rec.kind == "object" and rec.bbox is not None:
            entry["bbox"] = list(rec.bbox)
        if rec.kind == "passage" and rec.region_px is not None:
            entry["region"] = list(rec.region_px)
        destinations.append(entry)

    payload = {
        "agent": {
            "x": round(agent_pos["x"], 2),
            "z": round(agent_pos["z"], 2),
            "yaw": round(yaw, 1),
        },
        "destinations": destinations,
        "note": "Pass a destination id to set_nav_goal to drive there.",
    }
    logger.info(
        "View: {} object and {} passage destinations",
        len(object_records),
        len(passage_records),
    )
    return [_image_from(annotated, "JPEG"), payload]


@mcp.tool()
def set_nav_goal(
    destination_id: str, speed_mps: float = config.DEFAULT_SPEED_MPS
) -> dict:
    """Plans a path to a destination and starts driving there in the background.

    destination_id comes from get_current_view / list_destinations (e.g.
    "obj:Fridge|-01.50|+00.00|-00.70" or "psg:1"). Returns immediately with the
    planned path and ETA; poll get_nav_status for progress. An active goal is
    preempted (canceled and replaced), like nav2.

    Args:
        destination_id (str): id of the destination to drive to
        speed_mps (float, optional): driving speed in meters/second. Defaults to 0.35.
    """
    speed = float(min(max(speed_mps, config.MIN_SPEED_MPS), config.MAX_SPEED_MPS))
    try:
        rec = REGISTRY.resolve(destination_id)
    except KeyError:
        raise ToolError(
            f"Unknown destination id '{destination_id}'. "
            "Call get_current_view or list_destinations first."
        )

    target = rec.target
    object_id = None
    if rec.kind == "object":
        event = SIM.last_event()
        obj = next(
            (o for o in event.metadata["objects"] if o["objectId"] == rec.object_id),
            None,
        )
        if obj is None:
            raise ToolError(
                f"Destination '{destination_id}' is stale: the object is no longer "
                "present. Call get_current_view for fresh destinations."
            )
        target = obj["position"]
        object_id = rec.object_id

    try:
        return NAV.set_goal(destination_id, target, rec.label, speed, object_id)
    except ValueError as exc:
        raise ToolError(f"Could not plan a path to '{destination_id}': {exc}")


@mcp.tool()
def get_nav_status() -> dict:
    """Gets the state of the current/last navigation goal.

    state is one of idle | planning | following | succeeded | canceled | failed;
    progress holds distance traveled/remaining, percent, eta_s, and waypoint
    counts while a goal exists. Poll this after set_nav_goal until a terminal
    state (succeeded/canceled/failed).
    """
    return NAV.status()


@mcp.tool()
def cancel_nav() -> dict:
    """Cancels the active navigation goal, stopping the robot within one step."""
    canceled = NAV.cancel(wait=True)
    return {"canceled": canceled, "state": NAV.status()["state"]}


@mcp.tool()
def list_destinations() -> dict:
    """Lists every destination the robot has seen so far (across all views).

    Ids remain valid for set_nav_goal: objects are re-resolved to their current
    position; passages are fixed world waypoints.
    """
    return {
        "destinations": [
            {
                "id": rec.id,
                "kind": rec.kind,
                "label": rec.label,
                "description": rec.description,
                "target": [round(rec.target["x"], 2), round(rec.target["z"], 2)],
            }
            for rec in REGISTRY.list()
        ]
    }


@mcp.tool()
def get_map() -> Image:
    """Gets the top-down occupancy map the robot has built from its depth camera at camera
    height, like a lidar SLAM map: unexplored grey, free space white, obstacles black. Overlaid:
    robot pose (red triangle), remaining planned path (yellow), known destinations (dots), and
    the active goal (star). Unexplored areas fill in as the robot looks around and drives."""
    event = SIM.last_event()
    pose = SIM.peek_pose()
    overlays = MapOverlays(
        agent_position=pose[0] if pose else None,
        agent_yaw_deg=pose[1] if pose else None,
        path=NAV.path_snapshot(),
        destinations=REGISTRY.list(),
        goal_target=NAV.goal_target(),
    )
    top_view = MAPPER.render(overlays, meta=event.metadata)
    return _image_from(top_view, "PNG")


@mcp.tool()
def start_overhead_recording(mode: str = "follow") -> dict:
    """Starts recording an overhead video of the robot.

    Exactly one recording can be active at a time; starting while one is
    active is an error. mode="follow" (default) keeps a top-down camera
    hovering above the robot as it drives; mode="scene" fixes an orthographic
    camera over the whole floor plan. The video is real time (idle gaps show
    as held frames) and capped at 10 minutes. Call end_overhead_recording to
    finalize and get the mp4 path (saved under data/recordings/ on the server
    machine). Note: the overhead camera keeps rendering after the first
    recording, adding a small cost to every subsequent step.

    Args:
        mode (str, optional): "follow" or "scene". Defaults to "follow".
    """
    try:
        return RECORDER.start(SIM, mode)
    except RecorderError as exc:
        raise ToolError(str(exc))


@mcp.tool()
def end_overhead_recording() -> dict:
    """Stops the active overhead recording and finalizes the video.

    Returns the absolute path of a browser-playable H.264 mp4 on the server's
    filesystem plus duration and frame stats. Errors if no recording is
    active.
    """
    try:
        return RECORDER.end(SIM)
    except RecorderError as exc:
        raise ToolError(str(exc))


@mcp.tool()
def do_move(action: MOVE_ACTION, moveMagnitude: float = 1.0):
    """Moves the agent ahead("MoveAhead"), back("MoveBack"), left("MoveLeft"), or right("MoveRight"). Returns a summary of result of moving.

    Manual moves preempt (cancel) any active navigation goal, like nav2 teleop.

    Example: do_move(action="MoveAhead", moveMagnitude=2) -> moves agent forward by 2 meters

    Args:
        action (MOVE_ACTION): direction to move
        moveMagnitude (float, optional): how much to move by in meters. Defaults to 1..
    """
    preempted = NAV.preempt_for_manual()
    logger.info("Executing move action {} with magnitude {}", action, moveMagnitude)
    event = do_action(dict(action=action, moveMagnitude=moveMagnitude))
    meta = event.metadata
    if meta["lastActionSuccess"]:
        message = "Successfully moved"
    else:
        message = f"Move blocked: {meta.get('errorMessage') or 'movement failed'}"
    if preempted:
        message += " (preempted active navigation)"
    return message


@mcp.tool()
def do_rotate(action: ROTATE_ACTION, degrees: float = 90.0):
    """Rotates the agent left or right. Returns a summary of the result of rotating.

    Manual rotations preempt (cancel) any active navigation goal, like nav2 teleop.

    Example: do_rotate(action="RotateRight", degrees=45) -> rotates agent 45 degrees to the right

    Args:
        action (ROTATE_ACTION): rotation direction/action (e.g. "RotateLeft" or "RotateRight")
        degrees (float, optional): how many degrees to rotate. Defaults to 90..
    """
    preempted = NAV.preempt_for_manual()
    logger.info("Executing rotate action {} with degrees = {}", action, degrees)
    event = do_action(dict(action=action, degrees=degrees))
    meta = event.metadata
    if meta["lastActionSuccess"]:
        message = "Successfully rotated"
    else:
        message = f"Rotate blocked: {meta.get('errorMessage') or 'rotation failed'}"
    if preempted:
        message += " (preempted active navigation)"
    return message


# --- simulator-only free camera (not part of the robot contract) -----------------------------


def _freecam_result(pose, frame):
    return [_image_from(PILImage.fromarray(frame), "JPEG"), pose.as_dict()]


@mcp.tool(output_schema=None)
def sim_camera_view():
    """Simulator only: the free camera's current frame (JPEG) and pose.

    The free camera is a user-controlled third-party camera for watching the robot from
    anywhere in the room. It is created at a chase viewpoint on first use. Not part of the
    robot contract; robot brains must not use it.
    """
    try:
        FREECAM.ensure(SIM)
        frame = FREECAM.frame(SIM)
    except FreeCameraError as exc:
        raise ToolError(str(exc))
    return _freecam_result(FREECAM.pose, frame)


@mcp.tool(output_schema=None)
def sim_camera_move(
    forward: float = 0.0,
    right: float = 0.0,
    up: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    zoom: float = 0.0,
):
    """Simulator only: fly the free camera and return its new frame (JPEG) and pose.

    Args:
        forward, right, up: metres, relative to where the camera looks (forward/right stay
            level; up is world-vertical). Height is clamped to the room.
        yaw, pitch: degrees to turn (yaw) and tilt (pitch, positive looks down).
        zoom: degrees of field of view to remove (positive zooms in).
    """
    try:
        pose, frame = FREECAM.move(SIM, forward=forward, right=right, up=up, yaw=yaw, pitch=pitch, zoom=zoom)
    except FreeCameraError as exc:
        raise ToolError(str(exc))
    return _freecam_result(pose, frame)


@mcp.tool(output_schema=None)
def sim_camera_reset(mode: CAMERA_PRESET = "chase"):
    """Simulator only: jump the free camera to a preset relative to the robot and return the
    frame (JPEG) and pose. Presets: "chase" (behind and above, looking at the robot), "front"
    (in front, looking back at it), "top" (straight down)."""
    try:
        pose, frame = FREECAM.reset(SIM, mode)
    except (FreeCameraError, ValueError) as exc:
        raise ToolError(str(exc))
    return _freecam_result(pose, frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="use streamable-http transport")
    parser.add_argument(
        "--scene",
        default=None,
        help="AI2-THOR scene override (e.g. FloorPlan_Train1_3 for a RoboTHOR apartment)",
    )

    args = parser.parse_args()
    if args.scene:
        SIM.scene_override = args.scene
        logger.info("Scene override: {}", args.scene)
    if args.http:
        mcp.run(transport="streamable-http", host="0.0.0.0", stateless_http=True)
    else:
        mcp.run()
