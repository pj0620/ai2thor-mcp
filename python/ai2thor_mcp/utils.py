from dataclasses import dataclass
from math import cos, radians, sin
from typing import Any, List, Literal, Optional
from ai2thor_mcp import config
from ai2thor_mcp.model import EventMetadata, Vector3
from PIL import Image, ImageDraw, ImageFont
import open3d as o3d
import numpy as np

SCAN_HEIGHT = 0.9
PX_PER_M = 100
FLOAT_SLACK = 0.01
DEPTH_SCALE = 1024


@dataclass
class MapOverlays:
    agent_position: Optional[Vector3] = None
    agent_yaw_deg: Optional[float] = None
    path: Optional[List[dict]] = None  # remaining waypoint corners, world {x,y,z}
    destinations: Optional[list] = None  # DestinationRecord-like: .target/.kind/.marker
    goal_target: Optional[dict] = None

def m2px(dist: float) -> float:
    return PX_PER_M * dist

def get_bounds(center: Vector3, size: Vector3, axis: Literal["x", "y", "z"], px: bool = False) -> tuple[float, float]:
    lb, ub = center[axis] - size[axis] / 2, center[axis] + size[axis] / 2
    if px:
        lb, ub = m2px(lb), m2px(ub)
    return lb, ub

def generate_top_view(
    event_metadata: EventMetadata, overlays: Optional[MapOverlays] = None
) -> Image.Image:
    objects_metadata = event_metadata['objects']
    
    size_m = event_metadata['sceneBounds']['size']
    width_m, length_m = size_m['x'], size_m['z']
    width_px, length_px = int(m2px(width_m)) + 1, int(m2px(length_m)) + 1
    
    bounds = event_metadata['sceneBounds']['cornerPoints']
    bound_min_x = min(v[0] for v in bounds)
    bound_min_z = min(v[2] for v in bounds)
    
    img = Image.new("RGB", (width_px, length_px), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    for obj in objects_metadata:
        axisAlignedBoundingBox = obj['axisAlignedBoundingBox']
        center = axisAlignedBoundingBox['center']
        size = axisAlignedBoundingBox['size']
        
        # must intersect scan height
        y_min, y_max = get_bounds(center, size, "y")
        if not (y_min <= SCAN_HEIGHT <= y_max):
            continue
        
        x_min, x_max = get_bounds(center, size, "x")
        z_min, z_max = get_bounds(center, size, "z")
        x_min -= bound_min_x
        x_max -= bound_min_x
        z_min -= bound_min_z
        z_max -= bound_min_z
        
        x_min, x_max = m2px(x_min), m2px(x_max)
        z_min, z_max = m2px(z_min), m2px(z_max)
        
        assert -FLOAT_SLACK <= x_min <= width_px + FLOAT_SLACK
        assert -FLOAT_SLACK <= x_max <= width_px + FLOAT_SLACK
        assert -FLOAT_SLACK <= z_min <= length_px + FLOAT_SLACK
        assert -FLOAT_SLACK <= z_max <= length_px + FLOAT_SLACK
        
        draw.rectangle(
            xy = (x_min, z_min, x_max, z_max),
            fill = (0, 255, 0),
            # outline = (255, 255, 255),
            # width = 5
        )

    if overlays is not None:
        _draw_map_overlays(draw, overlays, bound_min_x, bound_min_z)

    return img


def _draw_map_overlays(
    draw: ImageDraw.ImageDraw, overlays: MapOverlays, bound_min_x: float, bound_min_z: float
) -> None:
    def to_px(x: float, z: float) -> tuple[float, float]:
        return m2px(x - bound_min_x), m2px(z - bound_min_z)

    try:
        font = ImageFont.load_default(size=12)
    except TypeError:
        font = ImageFont.load_default()

    if overlays.path and len(overlays.path) >= 1:
        points = [to_px(c["x"], c["z"]) for c in overlays.path]
        if overlays.agent_position is not None:
            points.insert(0, to_px(overlays.agent_position["x"], overlays.agent_position["z"]))
        if len(points) >= 2:
            draw.line(points, fill=config.PATH_COLOR, width=3)

    if overlays.destinations:
        for rec in overlays.destinations:
            target = getattr(rec, "target", None)
            if not target:
                continue
            px, pz = to_px(target["x"], target["z"])
            color = (
                config.PASSAGE_COLOR
                if getattr(rec, "kind", "") == "passage"
                else config.OBJECT_COLOR
            )
            r = 8
            draw.ellipse((px - r, pz - r, px + r, pz + r), fill=color, outline=(255, 255, 255))
            marker = getattr(rec, "marker", None)
            if marker is not None:
                draw.text((px, pz), str(marker), fill=(0, 0, 0), font=font, anchor="mm")

    if overlays.goal_target is not None:
        gx, gz = to_px(overlays.goal_target["x"], overlays.goal_target["z"])
        star = []
        for i in range(10):
            r = 13 if i % 2 == 0 else 5
            angle = radians(i * 36 - 90)
            star.append((gx + r * cos(angle), gz + r * sin(angle)))
        draw.polygon(star, fill=config.GOAL_COLOR, outline=(255, 255, 255))

    if overlays.agent_position is not None:
        ax, az = overlays.agent_position["x"], overlays.agent_position["z"]
        yaw = radians(overlays.agent_yaw_deg or 0.0)
        tri = []
        for off_deg, r_m in ((0.0, 0.22), (140.0, 0.14), (-140.0, 0.14)):
            theta = yaw + radians(off_deg)
            tri.append(to_px(ax + r_m * sin(theta), az + r_m * cos(theta)))
        draw.polygon(tri, fill=config.AGENT_COLOR, outline=(255, 255, 255))


def create_point_cloud_from_rgbd(event: Any):
    """Create a point cloud from an AI2-THOR event's RGB + metric depth.

    Notes:
    - AI2-THOR `event.depth_frame` is already metric depth in meters (float).
    - Camera intrinsics are derived from `event.metadata` (screen size + FOV).
    """

    if not hasattr(event, "metadata"):
        raise ValueError("event is missing `metadata`; expected an AI2-THOR Event")

    meta = event.metadata
    if "screenWidth" not in meta or "screenHeight" not in meta or "fov" not in meta:
        raise ValueError(
            "event.metadata must include screenWidth, screenHeight, and cameraFieldOfView to compute intrinsics"
        )

    width = int(meta["screenWidth"])
    height = int(meta["screenHeight"])
    vfov_deg = float(meta["fov"])
    vfov = np.deg2rad(vfov_deg)
    fy = (height * 0.5) / np.tan(vfov * 0.5)
    fx = fy
    cx = width * 0.5
    cy = height * 0.5
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # 1. Read the color and depth images (Open3D format)
    color_o3d = o3d.geometry.Image(event.frame.astype(np.uint8))

    depth_frame = event.depth_frame
    depth_o3d = o3d.geometry.Image(depth_frame.astype(np.float32))

    # 2. Create RGBD
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,  # AI2-THOR depth is already in meters
        depth_trunc=10.0,
        convert_rgb_to_intensity=False,
    )

    # 3. Create the point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)

    # Optional: visualize if the caller has asked for it
    o3d.visualization.draw_geometries([pcd])

    return np.asarray(pcd.points)