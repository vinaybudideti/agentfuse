"""
Version and public API tests.
"""

import agentfuse


def test_version_string():
    assert agentfuse.__version__ == "0.2.1"


def test_version_info_tuple():
    assert agentfuse.__version_info__ == (0, 2, 1)


def test_all_exports_exist():
    """Every name in __all__ must be importable."""
    for name in agentfuse.__all__:
        obj = getattr(agentfuse, name, None)
        assert obj is not None, f"Export '{name}' is None"


def test_minimum_export_count():
    """Public API must have at least 20 exports."""
    assert len(agentfuse.__all__) >= 20
