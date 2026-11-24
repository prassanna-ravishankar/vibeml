"""FastMCP server for VibeML - natural language interface to AI training.

This module is vibeml's socket: it exposes MCP tools and translates between
tool arguments and vibeml's own layers. Task construction lives in
``tasks``, pricing in ``cloud.cost_estimator``, and SkyPilot in
``cloud.skypilot``.
"""

import logging
import uuid
from typing import Any

from fastmcp import FastMCP
from pydantic import ValidationError as PydanticValidationError

from .cloud import CostEstimator, skypilot
from .exceptions import SkyPilotError, ValidationError
from .models import JobStatus, TrainingRequest
from .tasks import WORKFLOWS, build_task

logger = logging.getLogger(__name__)

CLUSTER_PREFIX = "vibeml-"
ACTIVE_STATUSES = {JobStatus.PENDING, JobStatus.RUNNING}

mcp = FastMCP(
    "vibeml",
    instructions="Natural language interface for AI model training on Nebius Cloud",
    version="0.0.1",
)


def _error(kind: str, message: str) -> dict[str, Any]:
    """Build an error response.

    Args:
        kind: Machine-readable error kind.
        message: Human-readable explanation.

    Returns:
        The error response.
    """
    return {"status": "error", "error": kind, "message": message}


@mcp.tool()
async def launch_training(
    workflow: str = "unsloth",
    model: str = "mistralai/Mistral-7B-v0.1",
    dataset: str = "tatsu-lab/alpaca",
    gpu_type: str | None = None,
    max_steps: int | None = 60,
    max_cost: float | None = None,
    learning_rate: float | None = None,
    lora_r: int | None = None,
    max_seq_length: int | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Launch a model training job on Nebius Cloud.

    The launch is submitted and this tool returns immediately; provisioning
    takes several minutes. Follow it with ``get_training_status``.

    Args:
        workflow: Training workflow type (unsloth, gpt-oss-lora, gpt-oss-full)
        model: HuggingFace model ID, or size for GPT-OSS ("20b", "120b")
        dataset: HuggingFace dataset ID
        gpu_type: GPU type (L40S, RTX4090, H100, A100) for Unsloth
        max_steps: Maximum training steps
        max_cost: Maximum cost limit in USD
        learning_rate: Learning rate for training
        lora_r: LoRA rank parameter
        max_seq_length: Maximum sequence length
        output_dir: Output directory for results

    Returns:
        Dict with cluster_id, the launch request ID, and how to follow it
    """
    hyperparameters = {
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "max_seq_length": max_seq_length,
        "output_dir": output_dir,
    }
    try:
        request = TrainingRequest(
            model=model,
            dataset=dataset,
            workflow=workflow,
            gpu_type=gpu_type,
            max_cost=max_cost,
            hyperparameters={k: v for k, v in hyperparameters.items() if v is not None},
        )
        task = build_task(workflow, model, dataset, gpu_type, **hyperparameters)
    except (PydanticValidationError, ValidationError) as e:
        return _error("validation_error", str(e))

    cluster_id = f"{CLUSTER_PREFIX}{workflow}-{uuid.uuid4().hex[:8]}"
    try:
        request_id = await skypilot.launch(task, cluster_name=cluster_id)
    except SkyPilotError as e:
        return _error("skypilot_error", str(e))

    logger.info(
        "Submitted %s launch for %s (request %s)", workflow, cluster_id, request_id
    )
    return {
        "status": "submitted",
        "cluster_id": cluster_id,
        "launch_request_id": request_id,
        "workflow": request.workflow,
        "model": request.model,
        "dataset": request.dataset,
        "logs_command": f"sky api logs {request_id}",
    }


@mcp.tool()
async def get_training_status(cluster_id: str) -> dict[str, Any]:
    """Get the status of a training job.

    Args:
        cluster_id: The cluster ID returned from launch_training

    Returns:
        Dict with the job status and SkyPilot's cluster status
    """
    try:
        cluster = await skypilot.get_cluster(cluster_id)
    except SkyPilotError as e:
        return _error("skypilot_error", str(e))

    if cluster is None:
        return _error(
            "not_found",
            f"Cluster {cluster_id} is not known to SkyPilot. If it was just launched, "
            "follow provisioning with `sky api logs <launch_request_id>`.",
        )
    return {
        "cluster_id": cluster.name,
        "status": cluster.status.value,
        "sky_status": cluster.sky_status,
        "launched_at": cluster.launched_at.isoformat() if cluster.launched_at else None,
    }


@mcp.tool()
async def stop_training(
    cluster_id: str, download_results: bool = True
) -> dict[str, Any]:
    """Stop a training job, optionally downloading its logs first.

    Args:
        cluster_id: The cluster ID to stop
        download_results: Whether to download job logs before tearing down

    Returns:
        Dict with stop status and the local log directories, if downloaded
    """
    try:
        if await skypilot.get_cluster(cluster_id) is None:
            return _error("not_found", f"Cluster {cluster_id} not found")
        log_paths = await skypilot.download_logs(cluster_id) if download_results else {}
        await skypilot.down(cluster_id)
    except SkyPilotError as e:
        return _error("skypilot_error", str(e))

    return {"status": "stopped", "cluster_id": cluster_id, "log_paths": log_paths}


@mcp.tool()
async def list_available_workflows() -> dict[str, Any]:
    """List all available training workflows and their descriptions.

    Returns:
        Dict with available workflows and their configurations
    """
    workflows_info = {
        "unsloth": {
            "description": (
                "Efficient fine-tuning for 7B-13B models using 4-bit quantization + LoRA"
            ),
            "gpu_types": ["L40S", "RTX4090", "H100", "A100"],
            "best_for": "Quick experiments, cost-effective training",
        },
        "gpt-oss-lora": {
            "description": "LoRA fine-tuning for large GPT models (20B-120B)",
            "gpu_types": ["H100"],
            "best_for": "Production quality with lower resource usage",
        },
        "gpt-oss-full": {
            "description": "Full parameter training for maximum quality",
            "gpu_types": ["H100"],
            "best_for": "Maximum quality, large-scale production training",
        },
    }
    return {
        "workflows": workflows_info,
        "available": list(WORKFLOWS),
        "message": "Use estimate_training_cost for pricing before launching",
    }


@mcp.tool()
async def estimate_training_cost(
    workflow: str = "unsloth",
    model: str = "mistralai/Mistral-7B-v0.1",
    dataset: str = "tatsu-lab/alpaca",
    gpu_type: str | None = None,
    estimated_hours: float = 4.0,
) -> dict[str, Any]:
    """Estimate the cost of a training job before launching.

    Prices exactly the resources the workflow would request.

    Args:
        workflow: Training workflow type
        model: Model ID, or size for GPT-OSS ("20b", "120b")
        dataset: HuggingFace dataset ID
        gpu_type: GPU type for Unsloth (auto-selected if not specified)
        estimated_hours: Estimated training hours

    Returns:
        Dict with the priced resources and the cost range
    """
    try:
        task = build_task(workflow, model, dataset, gpu_type)
    except ValidationError as e:
        return _error("validation_error", str(e))

    (resources,) = task.resources
    ((gpu, per_node),) = resources.accelerators.items()
    gpu_count = int(per_node) * task.num_nodes
    estimate = CostEstimator().estimate_cost(
        gpu_type=gpu,
        estimated_hours=estimated_hours,
        gpu_count=gpu_count,
        use_spot=resources.use_spot,
    )
    return {
        "workflow": workflow,
        "gpu_type": gpu,
        "gpu_count": gpu_count,
        "use_spot": resources.use_spot,
        "estimated_hours": estimated_hours,
        "hourly_cost": estimate.hourly_rate,
        "expected_cost": estimate.expected_cost,
        "min_cost": estimate.min_cost,
        "max_cost": estimate.max_cost,
        "currency": estimate.currency,
    }


@mcp.tool()
async def list_active_training_jobs() -> dict[str, Any]:
    """List all active (pending or running) training jobs.

    Returns:
        Dict with the active training clusters
    """
    try:
        clusters = await skypilot.list_clusters(prefix=CLUSTER_PREFIX)
    except SkyPilotError as e:
        return _error("skypilot_error", str(e))

    jobs = [
        {
            "cluster_id": c.name,
            "status": c.status.value,
            "sky_status": c.sky_status,
            "launched_at": c.launched_at.isoformat() if c.launched_at else None,
        }
        for c in clusters
        if c.status in ACTIVE_STATUSES
    ]
    return {"active_jobs": jobs, "count": len(jobs)}


def run_server() -> None:
    """Run the MCP server over stdio.

    Stdout carries the MCP protocol, so diagnostics go to stderr via logging.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Available workflows: %s", ", ".join(WORKFLOWS))
    mcp.run()
