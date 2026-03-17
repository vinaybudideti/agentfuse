"""
Tests for the dashboard API (without FastAPI running).
"""

import pytest


def test_create_app_requires_fastapi():
    """create_app must work if FastAPI is installed."""
    from agentfuse.dashboard.server import create_app
    try:
        app = create_app()
        assert app is not None
        assert app.title == "AgentFuse Cost Dashboard"
    except ImportError:
        pytest.skip("FastAPI not installed")


def test_app_has_health_route():
    """App must have /api/health route."""
    from agentfuse.dashboard.server import create_app
    try:
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/api/health" in routes
    except ImportError:
        pytest.skip("FastAPI not installed")


def test_app_has_spend_route():
    """App must have /api/spend route."""
    from agentfuse.dashboard.server import create_app
    try:
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/api/spend" in routes
    except ImportError:
        pytest.skip("FastAPI not installed")


def test_app_has_forecast_route():
    """App must have /api/forecast route."""
    from agentfuse.dashboard.server import create_app
    try:
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/api/forecast" in routes
    except ImportError:
        pytest.skip("FastAPI not installed")


def test_app_has_analytics_route():
    """App must have /api/analytics route."""
    from agentfuse.dashboard.server import create_app
    try:
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/api/analytics" in routes
    except ImportError:
        pytest.skip("FastAPI not installed")


def test_app_has_models_route():
    """App must have /api/models route."""
    from agentfuse.dashboard.server import create_app
    try:
        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/api/models" in routes
    except ImportError:
        pytest.skip("FastAPI not installed")
