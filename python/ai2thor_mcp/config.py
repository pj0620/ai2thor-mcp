"""Tunable constants for perception, navigation, and annotation."""

# --- robot / navigation ---
ROBOT_WIDTH_M = 0.35
MIN_PASSAGE_WIDTH_M = 0.65
DEFAULT_SPEED_MPS = 0.35
MAX_SPEED_MPS = 1.0
MIN_SPEED_MPS = 0.05
WAYPOINT_TOL_M = 0.20
GOAL_TOL_M = 0.30
ROT_TOL_DEG = 5.0
MAX_ROT_STEP_DEG = 45.0
ROT_SPEED_DPS = 90.0
MOVE_STEP_MAX_M = 0.25
MOVE_STEP_MIN_M = 0.05
STEP_PERIOD_S = 0.5
BLOCK_RETRIES = 2
BLOCK_NUDGE_DEG = 15.0

# --- passage detection ---
DEPTH_STRIDE = 4
OBSTACLE_BAND_M = (0.05, 1.0)  # heights above floor that block the robot
FLOOR_Y_M = 0.0
FLOOR_TOL_M = 0.06
MAX_RANGE_M = 5.0
MIN_FREE_DEPTH_M = 1.5
GAP_LEVELS_M = (1.5, 2.0, 2.5, 3.0, 4.0)  # level-set scan depths for gap finding
FLOOR_BEYOND_M = 0.3  # floor must be visible this far past the opening plane
FLOOR_SUPPORT_FRAC = 0.5  # ...in at least this fraction of the run's bins
AZ_BIN_DEG = 1.5
MIN_SECTOR_BINS = 2
FLANK_BINS = 2
GAP_BEYOND_M = 0.8  # opening must be at least this much deeper than its frame
OPEN_SPACE_MIN_M = 3.0  # flankless sectors must be clear at least this far
DOORWAY_MAX_WIDTH_M = 1.2
HALLWAY_MAX_WIDTH_M = 2.0
HALLWAY_MIN_DEPTH_M = 3.0
WAYPOINT_BEYOND_M = 0.5
SNAP_MAX_M = 0.35
PASSAGE_DEDUPE_M = 0.5
MAX_CEILING_Y_M = 2.2  # ignore points above this when boxing a passage region

# --- overhead recording ---
OVERHEAD_HEIGHT_M = 2.3  # follow camera height: below iTHOR ceilings, above doorframes
OVERHEAD_FOV_DEG = 75.0  # ~3.5 m square of floor visible from 2.3 m
OVERHEAD_REPOSITION_M = 0.15  # re-pose the follow camera when the agent moves this far
RECORD_FPS = 10
MAX_RECORDING_S = 600.0
RECORDINGS_DIR = "data/recordings"  # relative to server CWD (repo root)
FFMPEG_PRESET = "veryfast"
FFMPEG_CRF = 23

# --- annotation / map ---
MARKER_RADIUS_PX = 14
OBJECT_COLOR = (40, 200, 40)
PASSAGE_COLOR = (0, 190, 255)
GOAL_COLOR = (255, 60, 200)
PATH_COLOR = (255, 210, 40)
AGENT_COLOR = (230, 50, 50)
FONT_SIZE = 18
MIN_OBJECT_BBOX_AREA_PX = 400  # skip labels for sub-20x20px detections
MAX_OBJECT_DESTINATIONS = 20  # nearest N objects per view, to bound clutter
EXCLUDED_OBJECT_TYPES = {"Floor"}  # structural pseudo-objects, useless as destinations
