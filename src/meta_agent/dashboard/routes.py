"""Dashboard routes — JSON API and page routes."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from ..agent_manager import AgentManager
from ..models import AgentConfig, AgentStatus

bp = Blueprint("dashboard", __name__)


def _mgr() -> AgentManager:
    return current_app.config["manager"]


# --- Page routes ---

@bp.route("/")
def index():
    return render_template("index.html")


# --- JSON API ---

@bp.route("/api/agents")
def api_list_agents():
    agents = _mgr().list_agents()
    return jsonify([
        {
            "id": a.config.id,
            "name": a.config.name,
            "status": a.status.value,
            "description": a.config.description,
            "model": a.config.model,
            "current_task_id": a.current_task_id,
            "error": a.error,
        }
        for a in agents
    ])


@bp.route("/api/agents/<agent_id>")
def api_get_agent(agent_id: str):
    state = _mgr().get_agent(agent_id)
    if state is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "id": state.config.id,
        "name": state.config.name,
        "status": state.status.value,
        "description": state.config.description,
        "current_task_id": state.current_task_id,
        "error": state.error,
    })


@bp.route("/api/agents", methods=["POST"])
def api_create_agent():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    tools = data.get("allowed_tools", [])
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",") if t.strip()]
    config = AgentConfig(
        name=data["name"],
        system_prompt=data.get("system_prompt", "You are a helpful assistant."),
        allowed_tools=tools,
        model=data.get("model", "claude-sonnet-4-5-20250929"),
        description=data.get("description", ""),
    )
    state = _mgr().register_agent(config)
    return jsonify({"id": state.config.id, "name": state.config.name}), 201


@bp.route("/api/agents/<agent_id>", methods=["DELETE"])
def api_delete_agent(agent_id: str):
    if _mgr().unregister_agent(agent_id):
        return jsonify({"deleted": True})
    return jsonify({"error": "not found"}), 404


@bp.route("/api/agents/<agent_id>/start", methods=["POST"])
def api_start_agent(agent_id: str):
    state = _mgr().get_agent(agent_id)
    if state is None:
        return jsonify({"error": "not found"}), 404
    state.status = AgentStatus.IDLE
    return jsonify({"id": agent_id, "status": state.status.value})


@bp.route("/api/agents/<agent_id>/stop", methods=["POST"])
def api_stop_agent(agent_id: str):
    state = _mgr().get_agent(agent_id)
    if state is None:
        return jsonify({"error": "not found"}), 404
    state.status = AgentStatus.STOPPED
    return jsonify({"id": agent_id, "status": state.status.value})


@bp.route("/api/agents/<agent_id>/logs")
def api_agent_logs(agent_id: str):
    lines = request.args.get("lines", 100, type=int)
    log_text = _mgr().get_logs(agent_id, lines=lines)
    return jsonify({"agent_id": agent_id, "logs": log_text})


@bp.route("/api/tasks")
def api_list_tasks():
    agent_id = request.args.get("agent_id")
    tasks = _mgr().list_tasks(agent_id)
    return jsonify([
        {
            "id": t.id,
            "agent_id": t.agent_id,
            "status": t.status,
            "prompt": t.prompt[:100],
            "result": t.result[:200] if t.result else None,
            "error": t.error,
            "created_at": str(t.created_at),
            "completed_at": str(t.completed_at) if t.completed_at else None,
        }
        for t in tasks
    ])


@bp.route("/api/tasks", methods=["POST"])
def api_submit_task():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400
    try:
        task = _mgr().submit_task(data["agent_id"], data["prompt"])
        return jsonify({"task_id": task.id, "status": task.status}), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/api/tasks/<task_id>")
def api_task_status(task_id: str):
    task = _mgr().get_task(task_id)
    if task is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "id": task.id,
        "agent_id": task.agent_id,
        "status": task.status,
        "prompt": task.prompt,
        "result": task.result,
        "error": task.error,
        "created_at": str(task.created_at),
        "completed_at": str(task.completed_at) if task.completed_at else None,
    })
