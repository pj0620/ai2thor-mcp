from __future__ import annotations
from typing import TypedDict, Literal, NotRequired, Optional, List

import numpy as np


MOVE_ACTION = Literal["MoveAhead", "MoveBack", "MoveLeft", "MoveRight"]
ROTATE_ACTION = Literal["RotateRight", "RotateLeft"]

# --- small reusable shapes ---

class Vector3(TypedDict):
  x: float
  y: float
  z: float


class RGB(TypedDict):
  color: List[int]   # [r, g, b]
  name: str


class AABBox(TypedDict):
  cornerPoints: List[List[float]]  # 8 corners, each [x, y, z]
  center: Vector3
  size: Vector3


class OBB(TypedDict):
  cornerPoints: List[List[float]]  # 8 corners, each [x, y, z]


class HeldObjectPose(TypedDict):
  position: Vector3
  rotation: Vector3
  localPosition: Vector3
  localRotation: Vector3


class SceneBounds(TypedDict):
  cornerPoints: List[List[float]]  # 8 corners, each [x, y, z]
  center: Vector3
  size: Vector3


# --- object / agent ---

Temperature = Literal["Cold", "RoomTemp", "Hot"]


class SceneObject(TypedDict):
  name: str
  position: Vector3
  rotation: Vector3

  visible: bool
  isInteractable: bool
  receptacle: bool
  toggleable: bool
  isToggled: bool
  breakable: bool
  isBroken: bool

  canFillWithLiquid: bool
  isFilledWithLiquid: bool
  fillLiquid: Optional[str]

  dirtyable: bool
  isDirty: bool
  canBeUsedUp: bool
  isUsedUp: bool
  cookable: bool
  isCooked: bool
  temperature: Temperature

  isHeatSource: bool
  isColdSource: bool

  sliceable: bool
  isSliced: bool

  openable: bool
  isOpen: bool
  openness: float

  pickupable: bool
  isPickedUp: bool
  moveable: bool
  mass: float

  salientMaterials: Optional[List[str]]
  receptacleObjectIds: Optional[List[str]]

  distance: float
  objectType: str
  objectId: str
  assetId: str

  parentReceptacles: Optional[List[str]]
  controlledObjects: Optional[List[str]]

  isMoving: bool

  axisAlignedBoundingBox: AABBox
  objectOrientedBoundingBox: Optional[OBB]


class AgentState(TypedDict):
  name: str
  position: Vector3
  rotation: Vector3
  cameraHorizon: float
  isStanding: Optional[bool]
  inHighFrictionArea: bool


# --- the top-level TypedDict you asked for ---

class EventMetadata(TypedDict):
  objects: List[SceneObject]
  isSceneAtRest: bool
  agent: AgentState
  heldObjectPose: HeldObjectPose
  arm: Optional[object]

  fov: float
  cameraPosition: Vector3
  cameraOrthSize: float
  thirdPartyCameras: List[object]

  collided: bool
  collidedObjects: List[object]
  inventoryObjects: List[object]

  sceneName: str
  lastAction: str
  errorMessage: str
  errorCode: Optional[object]
  lastActionSuccess: bool

  screenWidth: int
  screenHeight: int
  agentId: int

  depthFormat: str
  colors: List[RGB]

  flatSurfacesOnGrid: List[object]
  distances: List[object]
  normals: List[object]
  isOpenableGrid: List[object]
  segmentedObjectIds: List[object]
  objectIdsInBox: List[object]

  actionIntReturn: int
  actionFloatReturn: float
  actionStringsReturn: List[str]
  actionFloatsReturn: List[float]
  actionVector3sReturn: List[Vector3]

  visibleRange: Optional[object]
  currentTime: float

  sceneBounds: SceneBounds
  actionReturn: Optional[object]

class Ai2thorEvent(TypedDict):
    cv2img: np.ndarray
    depth_frame: np.ndarray


# --- navigation / destination wire shapes ---

NAV_STATE = Literal["idle", "planning", "following", "succeeded", "canceled", "failed"]

DESTINATION_KIND = Literal["object", "passage"]

PASSAGE_KIND = Literal["doorway", "hallway", "opening"]


class AgentPose2D(TypedDict):
  x: float
  z: float
  yaw: float


class NavGoalInfo(TypedDict):
  goal_id: str
  destination_id: str
  label: str


class NavProgress(TypedDict):
  distance_traveled_m: float
  distance_remaining_m: float
  percent: int
  eta_s: float
  waypoint: int
  waypoints_total: int


class NavStatus(TypedDict):
  state: NAV_STATE
  goal: Optional[NavGoalInfo]
  progress: Optional[NavProgress]
  agent: Optional[AgentPose2D]
  collisions: int
  reason: Optional[str]


class PlannedPath(TypedDict):
  waypoints: List[List[float]]  # [x, z] pairs
  length_m: float


class PlanSummary(TypedDict):
  goal_id: str
  destination_id: str
  label: str
  state: NAV_STATE
  path: PlannedPath
  speed_mps: float
  eta_s: float
  preempted_goal_id: Optional[str]


class DestinationPayload(TypedDict, total=False):
  marker: int
  id: str
  kind: DESTINATION_KIND
  label: str
  description: str
  distance_m: float
  bbox: List[int]    # objects: [x1, y1, x2, y2] in the current frame
  region: List[int]  # passages: [x1, y1, x2, y2] in the current frame
