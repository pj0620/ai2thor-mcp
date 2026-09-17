"""Template descriptions for destinations and the destination registry.

Registry id scheme: objects use "obj:{ai2thor objectId}" so a stale id can be
re-resolved (or rejected) against current metadata at set_nav_goal time;
passages use "psg:{n}" and resolve to a world-fixed waypoint forever. Marker
numbers are per-view presentation only — the JSON payload maps marker to the
stable id.
"""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai2thor_mcp import config
from ai2thor_mcp.perception import PassageCandidate, wrap_deg


def relative_bearing(
    agent_pos: Dict[str, float], yaw_deg: float, target_pos: Dict[str, float]
) -> float:
    dx = target_pos["x"] - agent_pos["x"]
    dz = target_pos["z"] - agent_pos["z"]
    return wrap_deg(float(np.degrees(np.arctan2(dx, dz))) - yaw_deg)


def direction_word(bearing_deg: float) -> str:
    b = wrap_deg(bearing_deg)
    a = abs(b)
    if a <= 22.5:
        return "ahead"
    if a >= 157.5:
        return "behind"
    side = "right" if b > 0 else "left"
    if a <= 67.5:
        return f"ahead-{side}"
    if a <= 112.5:
        return side
    return f"behind-{side}"


def describe_object(obj: Dict[str, Any], agent_pos: Dict[str, float], yaw_deg: float) -> str:
    bearing = relative_bearing(agent_pos, yaw_deg, obj["position"])
    parts = [f"{obj['objectType']}, {obj['distance']:.1f} m {direction_word(bearing)}"]
    if obj.get("openable"):
        parts.append("open" if obj.get("isOpen") else "closed")
    contents = obj.get("receptacleObjectIds")
    if obj.get("receptacle") and contents:
        parts.append(f"contains {len(contents)} item{'s' if len(contents) != 1 else ''}")
    if obj.get("toggleable") and obj.get("isToggled"):
        parts.append("switched on")
    if obj.get("isBroken"):
        parts.append("broken")
    return ", ".join(parts)


def describe_passage(p: PassageCandidate) -> str:
    width = f">={p.width_m:.1f}" if p.width_is_lower_bound else f"~{p.width_m:.1f}"
    return (
        f"passage: {p.kind} {direction_word(p.bearing_deg)}, "
        f"{width} m wide, opens {p.free_depth_m:.1f} m deep"
    )


@dataclass
class DestinationRecord:
    id: str
    kind: str  # "object" | "passage"
    label: str
    description: str
    target: Dict[str, float]  # world nav target
    object_id: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # objects, current-view px
    region_px: Optional[Tuple[int, int, int, int]] = None  # passages, current-view px
    marker_px: Optional[Tuple[int, int]] = None
    distance_m: Optional[float] = None
    marker: Optional[int] = None  # number drawn in the most recent view


class DestinationRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: Dict[str, DestinationRecord] = {}
        self._psg_counter = 0

    def upsert_object(
        self,
        obj: Dict[str, Any],
        description: str,
        bbox: Tuple[int, int, int, int],
        distance_m: float,
    ) -> DestinationRecord:
        dest_id = f"obj:{obj['objectId']}"
        with self._lock:
            rec = self._records.get(dest_id)
            if rec is None:
                rec = DestinationRecord(
                    id=dest_id,
                    kind="object",
                    label=obj["objectType"],
                    description=description,
                    target=dict(obj["position"]),
                    object_id=obj["objectId"],
                )
                self._records[dest_id] = rec
            rec.description = description
            rec.target = dict(obj["position"])
            rec.bbox = tuple(int(v) for v in bbox)  # type: ignore[assignment]
            rec.distance_m = round(float(distance_m), 2)
            return rec

    def upsert_passage(
        self, cand: PassageCandidate, description: str, distance_m: Optional[float] = None
    ) -> DestinationRecord:
        with self._lock:
            rec = None
            for existing in self._records.values():
                if existing.kind != "passage":
                    continue
                d = np.hypot(
                    existing.target["x"] - cand.waypoint["x"],
                    existing.target["z"] - cand.waypoint["z"],
                )
                if d < config.PASSAGE_DEDUPE_M:
                    rec = existing
                    break
            if rec is None:
                self._psg_counter += 1
                rec = DestinationRecord(
                    id=f"psg:{self._psg_counter}",
                    kind="passage",
                    label=cand.kind,
                    description=description,
                    target=dict(cand.waypoint),
                )
                self._records[rec.id] = rec
            rec.label = cand.kind
            rec.description = description
            rec.target = dict(cand.waypoint)
            rec.region_px = cand.region_px
            rec.marker_px = cand.marker_px
            rec.distance_m = round(float(distance_m), 2) if distance_m is not None else None
            return rec

    def resolve(self, dest_id: str) -> DestinationRecord:
        with self._lock:
            return self._records[dest_id]  # KeyError surfaces to the tool layer

    def list(self) -> List[DestinationRecord]:
        with self._lock:
            return list(self._records.values())

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._psg_counter = 0
