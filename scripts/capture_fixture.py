"""Capture a depth/RGB frame + metadata as an offline perception fixture.

Usage (from the repo root):
    uv run scripts/capture_fixture.py [--scene FloorPlan_Train1_3] [--rotate 90]

Writes data/example_frame.npz (depth float32 meters, rgb uint8) and
data/example_frame.json (agent pose, camera position, intrinsics inputs).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ai2thor.controller import Controller


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="FloorPlan2")
    parser.add_argument("--rotate", type=float, default=0.0, help="initial RotateRight degrees")
    parser.add_argument("--out", default="data/example_frame")
    args = parser.parse_args()

    controller = Controller(
        agentMode="locobot",
        visibilityDistance=1.5,
        scene=args.scene,
        gridSize=0.25,
        snapToGrid=False,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        width=500,
        height=500,
        fieldOfView=60,
    )
    try:
        if args.rotate:
            controller.step(action="RotateRight", degrees=args.rotate)
        event = controller.step(action="MoveAhead", moveMagnitude=0.0)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out.with_suffix(".npz"),
            depth=event.depth_frame.astype(np.float32),
            rgb=event.frame.astype(np.uint8),
        )
        meta = {
            key: event.metadata[key]
            for key in (
                "agent",
                "cameraPosition",
                "screenWidth",
                "screenHeight",
                "fov",
                "sceneName",
            )
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        print(f"wrote {out.with_suffix('.npz')} and {out.with_suffix('.json')}")
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
