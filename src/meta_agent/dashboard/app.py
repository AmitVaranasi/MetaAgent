"""Flask application factory for the dashboard."""

from __future__ import annotations

from flask import Flask

from ..agent_manager import AgentManager


def create_app(manager: AgentManager) -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.config["manager"] = manager

    from .routes import bp

    app.register_blueprint(bp)
    return app
