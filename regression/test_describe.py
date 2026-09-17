import pytest

from ai2thor_mcp.describe import (
    DestinationRegistry,
    describe_object,
    describe_passage,
    direction_word,
    relative_bearing,
)
from ai2thor_mcp.perception import PassageCandidate


def make_candidate(x=0.0, z=2.5, kind="doorway", width=0.9, lower=False):
    return PassageCandidate(
        kind=kind,
        width_m=width,
        width_is_lower_bound=lower,
        free_depth_m=3.0,
        bearing_deg=10.0,
        waypoint={"x": x, "y": 0.0, "z": z},
        region_px=(0, 0, 10, 10),
        marker_px=(5, 5),
    )


def test_direction_words():
    assert direction_word(0) == "ahead"
    assert direction_word(45) == "ahead-right"
    assert direction_word(-45) == "ahead-left"
    assert direction_word(90) == "right"
    assert direction_word(-90) == "left"
    assert direction_word(135) == "behind-right"
    assert direction_word(180) == "behind"
    assert direction_word(-170) == "behind"


def test_relative_bearing():
    origin = {"x": 0.0, "y": 0.0, "z": 0.0}
    assert relative_bearing(origin, 0.0, {"x": 0.0, "z": 2.0}) == pytest.approx(0.0)
    assert relative_bearing(origin, 0.0, {"x": 2.0, "z": 0.0}) == pytest.approx(90.0)
    assert relative_bearing(origin, 90.0, {"x": 2.0, "z": 0.0}) == pytest.approx(0.0)
    assert relative_bearing(origin, 0.0, {"x": -2.0, "z": 0.0}) == pytest.approx(-90.0)


def test_describe_object(example_metadata):
    agent = example_metadata["agent"]
    obj = next(o for o in example_metadata["objects"] if o["visible"])
    desc = describe_object(obj, agent["position"], agent["rotation"]["y"])
    assert obj["objectType"] in desc
    assert " m " in desc


def test_describe_passage():
    desc = describe_passage(make_candidate())
    assert desc.startswith("passage: doorway")
    assert "~0.9 m wide" in desc
    assert "opens 3.0 m deep" in desc
    lower = describe_passage(make_candidate(lower=True))
    assert ">=0.9 m wide" in lower


def test_registry_object_upsert(example_metadata):
    registry = DestinationRegistry()
    obj = next(o for o in example_metadata["objects"] if o["visible"])
    rec1 = registry.upsert_object(obj, "first", (0, 0, 10, 10), obj["distance"])
    rec2 = registry.upsert_object(obj, "second", (1, 1, 11, 11), obj["distance"])
    assert rec1.id == rec2.id == f"obj:{obj['objectId']}"
    assert registry.resolve(rec1.id).description == "second"
    assert len(registry.list()) == 1


def test_registry_passage_dedupe():
    registry = DestinationRegistry()
    rec1 = registry.upsert_passage(make_candidate(x=0.0), "a", 2.5)
    rec2 = registry.upsert_passage(make_candidate(x=0.2), "b", 2.5)  # < 0.5 m away
    rec3 = registry.upsert_passage(make_candidate(x=2.0), "c", 3.0)
    assert rec1.id == rec2.id == "psg:1"
    assert rec3.id == "psg:2"
    assert registry.resolve("psg:1").description == "b"

    with pytest.raises(KeyError):
        registry.resolve("psg:99")

    registry.clear()
    assert registry.list() == []
    assert registry.upsert_passage(make_candidate(), "d", 2.5).id == "psg:1"
