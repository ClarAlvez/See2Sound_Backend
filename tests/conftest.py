"""Keep real media/model integration tests explicitly opt-in."""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-media-tests", action="store_true", default=False,
                     help="Run real media integrations (may load/download models).")


def pytest_configure(config):
    config.addinivalue_line("markers", "media: real video/audio/model integration; opt-in")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-media-tests"):
        return
    skip = pytest.mark.skip(reason="Real media integration requires --run-media-tests")
    for item in items:
        if "media" in item.keywords:
            item.add_marker(skip)
