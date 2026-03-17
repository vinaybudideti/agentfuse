"""
Minimal cost dashboard API — serves JSON data from SpendLedger.

Requires: pip install fastapi uvicorn

Usage:
    from agentfuse.dashboard.server import create_app
    app = create_app()
    # Run with: uvicorn agentfuse.dashboard.server:app
"""

import logging

logger = logging.getLogger(__name__)


def create_app():
    """Create a FastAPI app for the cost dashboard."""
    try:
        from fastapi import FastAPI
    except ImportError:
        raise ImportError("FastAPI required for dashboard: pip install fastapi uvicorn")

    app = FastAPI(title="AgentFuse Cost Dashboard", version="0.2.1")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "0.2.1"}

    @app.get("/api/spend")
    def spend():
        from agentfuse.gateway import get_spend_report
        return get_spend_report()

    @app.get("/api/forecast")
    def forecast():
        from agentfuse.gateway import get_spend_report
        from agentfuse.core.cost_forecast import CostForecast
        report = get_spend_report()
        return CostForecast(report, days_of_data=1.0).predict_monthly()

    @app.get("/api/analytics")
    def analytics():
        from agentfuse.gateway import get_spend_report
        from agentfuse.core.analytics import UsageAnalytics
        return UsageAnalytics(get_spend_report()).get_insights()

    @app.get("/api/models")
    def models():
        from agentfuse.providers.registry import ModelRegistry
        reg = ModelRegistry(refresh_hours=0)
        return {"models": reg.list_models(), "count": len(reg.list_models())}

    return app
