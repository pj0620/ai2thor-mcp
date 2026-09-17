import pytest
import json
from ai2thor_mcp.model import EventMetadata


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: test drives a real AI2-THOR controller (opens Unity)"
    )


@pytest.fixture(scope="module")
def example_metadata() -> EventMetadata:
    with open("data/example_metadata.json") as f:
        return json.load(f)
