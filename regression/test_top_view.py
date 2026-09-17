from ai2thor_mcp.model import EventMetadata
from ai2thor_mcp.utils import generate_top_view, create_point_cloud_from_rgbd
import pytest
import argparse
from contextlib import asynccontextmanager
import io
from typing import Any, AsyncIterator, Dict, cast
from fastmcp import FastMCP
from ai2thor_mcp.model import MOVE_ACTION, ROTATE_ACTION
from PIL import Image as PILImage
from fastmcp.utilities.types import Image
from loguru import logger
from typing import Any, Dict, Union
from ai2thor.controller import Controller
from loguru import logger
from fastmcp.server.server import Transport
from loguru import logger
import open3d as o3d

@pytest.fixture
def controller():
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

def test_generate_top_view(example_metadata: EventMetadata):
    top_view = generate_top_view(example_metadata)
    
    top_view.show()
    
@pytest.mark.live
def test_create_point_cloud_from_rgbd(controller: Controller):
    event = controller.step(action="MoveAhead", moveMagnitude=0.01)
    event = cast(Any, event)
    point_cloud = create_point_cloud_from_rgbd(event)
    