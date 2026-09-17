# ai2thor-mcp

An MCP server that gives an LLM agent control of a simulated home robot in
[AI2-THOR](https://ai2thor.allenai.org/). The robot is camera-only (no arms):
it looks around, discovers **destinations** — visible objects and drivable
passages detected from depth — and drives to them with nav2-style asynchronous
navigation. An overhead camera can record mp4 videos of what it does.

Built as the simulation backend for GPTPet; the tool contract mirrors ROS2
nav2's `NavigateToPose` semantics (async goal, status polling, cancel,
preemption) so a real robot running nav2+SLAM can implement the same MCP
interface later.

## Tools

| Tool | What it does |
|---|---|
| `get_current_view()` | Camera frame annotated with numbered destination markers, plus a JSON list: objects (green boxes, e.g. `"Fridge, 2.3 m ahead, closed, contains 2 items"`) and passages (cyan boxes: doorways/hallways/openings wide enough for the robot). Ids are stable; markers restart each view. |
| `set_nav_goal(destination_id, speed_mps=0.35)` | Plans a navmesh path to a destination and starts driving in the background. Returns waypoints + ETA immediately. An active goal is preempted. |
| `get_nav_status()` | `idle \| planning \| following \| succeeded \| canceled \| failed(reason)` plus distance traveled/remaining, percent, ETA, waypoint counts. |
| `cancel_nav()` | Stops the robot within one step. |
| `list_destinations()` | Everything seen so far, across views. Object ids re-resolve to current positions; passage ids are fixed world waypoints. |
| `get_map()` | Top-down map: objects, robot pose, remaining path, destinations, goal. |
| `do_move(action, moveMagnitude)` / `do_rotate(action, degrees)` | Manual control; preempts active navigation (nav2 teleop behavior). |
| `start_overhead_recording(mode)` | `mode="follow"` (default): camera hovers ~2.3 m above the robot and tracks it. `mode="scene"`: fixed orthographic view of the whole floor plan. One recording at a time. |
| `end_overhead_recording()` | Finalizes and returns the path of a browser-playable H.264 mp4 (real-time: idle gaps are held frames) under `data/recordings/`. |

Plus an MCP resource `view://rgb` with the raw camera frame.

## Setup

Requires Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/). The first run
downloads the AI2-THOR Unity build (~0.5 GB) and opens a Unity window; the
simulator boots lazily on the first tool call.

```bash
uv sync            # add --extra dev for the test suite
```

## Running

```bash
# streamable HTTP on 0.0.0.0:8000 (endpoint: /mcp, stateless)
uv run python/ai2thor_mcp/main.py --http

# stdio transport
uv run python/ai2thor_mcp/main.py

# pick a scene (default FloorPlan2, a single kitchen; RoboTHOR apartments
# like FloorPlan_Train1_3 have real doorways and corridors)
uv run python/ai2thor_mcp/main.py --http --scene FloorPlan_Train1_3
```

HTTP is stateless, so multiple MCP clients share the same simulator instance.

### Connecting clients

Claude Code:

```bash
claude mcp add --transport http ai2thor http://localhost:8000/mcp
```

Claude Desktop only speaks stdio in `claude_desktop_config.json`, so bridge it
with [mcp-remote](https://www.npmjs.com/package/mcp-remote):

```json
"ai2thor": {
  "command": "npx",
  "args": ["-y", "mcp-remote", "http://localhost:8000/mcp"]
}
```

## How it works

```
python/ai2thor_mcp/
├── main.py        MCP tools and wiring
├── sim.py         single lock-guarded owner of the AI2-THOR Controller
├── perception.py  depth → world backprojection; passage detection
├── describe.py    template descriptions + destination registry
├── annotate.py    numbered set-of-marks drawing on the frame
├── nav.py         nav2-style background goal follower
├── recorder.py    overhead mp4 recording (paced ffmpeg writer thread)
├── utils.py       top-down map rendering
└── config.py      all tunables (robot width, speeds, detection thresholds…)
```

- **Passages** come from the depth image: points are backprojected to world
  space, binned into a polar free-space profile, and gaps are found by
  level-set scanning (a passage is a run of bins clear well beyond its
  flanking obstacles). Floor-visibility and navmesh-reachability checks
  reject windows/mirrors, and a minimum-width filter
  (`MIN_PASSAGE_WIDTH_M = 0.65`) keeps only gaps the robot fits through.
- **Path planning** uses AI2-THOR's navmesh (`GetShortestPath`); a follower
  thread walks the corners at the requested speed with collision retries and
  one replan, so `get_nav_status` shows honest live progress.
- **Recording** captures the third-party camera frame on every sim step; a
  paced writer streams one frame per tick to a bundled ffmpeg
  (`imageio-ffmpeg`), so the mp4 tracks wall-clock time without ever blocking
  the simulator. Note: AI2-THOR 5.0 can't remove a camera once added, so
  after the first recording every step renders it (small cost).

## Testing

```bash
uv run --extra dev pytest regression -m "not live"   # offline, no Unity
uv run --extra dev pytest regression -m live         # boots Unity, drives the robot
```

Offline tests cover the geometry with analytic depth scenes (synthetic
doorways must be found, narrow gaps and plain walls must not), the nav state
machine against a kinematic fake sim, and the recorder's hold-to-wall-clock
writer frame-by-frame. `scripts/capture_fixture.py` regenerates the captured
real-frame fixture in `data/`.

## Notes

- Agent mode is `locobot` (differential-drive, no manipulation).
- `data/recordings/` is git-ignored output; `data/example_*` are test fixtures.
- With Claude Code, the project skill `/restart-ai2thor-mcp` restarts the
  server on port 8000 and health-checks it.
