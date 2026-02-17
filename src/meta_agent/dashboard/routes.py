"""Dashboard routes — JSON API and page routes."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

from ..agent_manager import AgentManager
from ..models import AgentConfig, AgentStatus, Workflow

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


# --- Workflow API ---

@bp.route("/api/workflows")
def api_list_workflows():
    workflows = _mgr().db.list_workflows()
    return jsonify([
        {
            "id": w.id,
            "prompt": w.prompt[:100],
            "status": w.status.value,
            "subtask_count": len(w.subtask_ids),
            "created_at": str(w.created_at),
            "completed_at": str(w.completed_at) if w.completed_at else None,
        }
        for w in workflows
    ])


@bp.route("/api/workflows", methods=["POST"])
def api_create_workflow():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "JSON body with 'prompt' required"}), 400

    from ..brain import BRAIN_AGENT_ID, get_brain_config

    mgr = _mgr()

    # Ensure brain agent exists
    if mgr.get_agent(BRAIN_AGENT_ID) is None:
        brain_config = get_brain_config(["meta-agent", "mcp-server"])
        mgr.register_agent(brain_config)

    workflow = Workflow(prompt=data["prompt"], brain_agent_id=BRAIN_AGENT_ID)
    mgr.db.save_workflow(workflow)

    brain_prompt = (
        f"Workflow ID: {workflow.id}\n\n"
        f"User Request: {data['prompt']}\n\n"
        "Please analyze this task, create a workflow plan, decompose into subtasks "
        "if needed, and execute. Use the workflow tools to track progress."
    )
    try:
        task = mgr.submit_task(BRAIN_AGENT_ID, brain_prompt, workflow_id=workflow.id)
        return jsonify({
            "workflow_id": workflow.id,
            "task_id": task.id,
            "status": workflow.status.value,
        }), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@bp.route("/kanban")
def kanban_board():
    return render_template("kanban.html")


@bp.route("/api/kanban")
def api_kanban():
    """Return tasks grouped by lifecycle stage with agent info for Kanban display."""
    workflow_id = request.args.get("workflow_id")
    mgr = _mgr()
    tasks = mgr.list_tasks()
    agents_list = mgr.list_agents()
    agents_map = {a.config.id: a for a in agents_list}

    # Optionally filter by workflow
    if workflow_id:
        tasks = [t for t in tasks if t.workflow_id == workflow_id]

    columns = {
        "pending": {"label": "Pending", "icon": "⏳", "tasks": []},
        "running": {"label": "Running", "icon": "🔄", "tasks": []},
        "completed": {"label": "Completed", "icon": "✅", "tasks": []},
        "failed": {"label": "Failed", "icon": "❌", "tasks": []},
    }

    for t in tasks:
        agent = agents_map.get(t.agent_id)
        agent_info = {
            "id": agent.config.id,
            "name": agent.config.name,
            "model": agent.config.model,
            "status": agent.status.value,
        } if agent else {"id": t.agent_id, "name": "Unknown", "model": "N/A", "status": "unknown"}

        task_data = {
            "id": t.id,
            "agent": agent_info,
            "status": t.status,
            "prompt": t.prompt[:150],
            "full_prompt": t.prompt,
            "result": t.result[:300] if t.result else None,
            "error": t.error,
            "created_at": str(t.created_at),
            "completed_at": str(t.completed_at) if t.completed_at else None,
            "workflow_id": t.workflow_id,
            "parent_task_id": t.parent_task_id,
        }

        col = t.status if t.status in columns else "pending"
        columns[col]["tasks"].append(task_data)

    # Sort: running first by created_at, pending by created_at, completed/failed by completed_at desc
    for key in columns:
        columns[key]["tasks"].sort(
            key=lambda x: x.get("completed_at") or x.get("created_at") or "",
            reverse=(key in ("completed", "failed")),
        )

    # Also gather workflow info if a workflow_id is specified
    workflow_info = None
    if workflow_id:
        wf = mgr.db.get_workflow(workflow_id)
        if wf:
            workflow_info = {
                "id": wf.id,
                "prompt": wf.prompt,
                "plan": wf.plan,
                "status": wf.status.value,
                "result": wf.result,
                "error": wf.error,
            }

    return jsonify({
        "columns": columns,
        "workflow": workflow_info,
        "total_tasks": len(tasks),
        "summary": {
            "pending": len(columns["pending"]["tasks"]),
            "running": len(columns["running"]["tasks"]),
            "completed": len(columns["completed"]["tasks"]),
            "failed": len(columns["failed"]["tasks"]),
        },
    })


@bp.route("/api/workflows/<workflow_id>")
def api_get_workflow(workflow_id: str):
    workflow = _mgr().db.get_workflow(workflow_id)
    if workflow is None:
        return jsonify({"error": "not found"}), 404
    subtasks = []
    for tid in workflow.subtask_ids:
        task = _mgr().get_task(tid)
        if task:
            subtasks.append({
                "id": task.id,
                "agent_id": task.agent_id,
                "status": task.status,
                "prompt": task.prompt[:100],
                "result": task.result[:200] if task.result else None,
                "error": task.error,
            })
    return jsonify({
        "id": workflow.id,
        "prompt": workflow.prompt,
        "plan": workflow.plan,
        "status": workflow.status.value,
        "brain_agent_id": workflow.brain_agent_id,
        "brain_task_id": workflow.brain_task_id,
        "subtasks": subtasks,
        "result": workflow.result,
        "error": workflow.error,
        "created_at": str(workflow.created_at),
        "completed_at": str(workflow.completed_at) if workflow.completed_at else None,
    })
