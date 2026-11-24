"""FastMCP server for VibeML - natural language interface to AI training.

This module is vibeml's socket: it exposes MCP tools and translates between
tool arguments and vibeml's own layers. Task construction lives in
``tasks``, pricing in ``cloud.cost_estimator``, and SkyPilot in
``cloud.skypilot``.
"""

import logging
import os
import sys
import uuid
from typing import Annotated, Any, TextIO

from fastmcp import FastMCP
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from pydantic import Field
from pydantic import ValidationError as PydanticValidationError

from .cloud import CostEstimator, skypilot
from .exceptions import SkyPilotError, TaskGenerationError, ValidationError
from .models import JobStatus, TrainingRequest
from .tasks import WORKFLOWS, build_task

logger = logging.getLogger(__name__)

CLUSTER_PREFIX = "vibeml-"
Hours = Annotated[float, Field(gt=0, allow_inf_nan=False)]
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
    max_cost: Annotated[float | None, Field(ge=0, allow_inf_nan=False)] = None,
    learning_rate: float | None = None,
    lora_r: int | None = None,
    max_seq_length: int | None = None,
    output_dir: str | None = None,
    cloud_bucket: str | None = None,
    estimated_hours: Hours = 4.0,
) -> dict[str, Any]:
    """Launch a model training job on Nebius Cloud.

    The launch is submitted and this tool returns immediately; provisioning
    takes several minutes. Follow it with ``get_training_status``.

    With ``max_cost``, the launch is refused if the cost estimate for
    ``estimated_hours`` exceeds it. Otherwise the cluster is torn down once
    the budget's worth of runtime has elapsed, even if training is still
    running.

    Args:
        workflow: Training workflow type (unsloth, gpt-oss-lora, gpt-oss-full)
        model: HuggingFace model ID, or size for GPT-OSS ("20b", "120b")
        dataset: HuggingFace dataset ID
        gpu_type: GPU type (L40S, RTX4090, H100, A100) for Unsloth
        max_steps: Maximum training steps
        max_cost: Budget in USD, enforced before and during the run
        learning_rate: Learning rate for training
        lora_r: LoRA rank parameter
        max_seq_length: Maximum sequence length
        output_dir: Absolute output directory (default /outputs)
        cloud_bucket: Bucket to mount at the output directory, so checkpoints persist
        estimated_hours: Expected training time, used to check max_cost

    Returns:
        Dict with cluster_id, the launch request ID, how to follow it, and
        the cost guard if max_cost was given
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
        task = build_task(
            workflow,
            model,
            dataset,
            gpu_type,
            cloud_bucket=cloud_bucket,
            **hyperparameters,
        )
    except (PydanticValidationError, ValidationError) as e:
        return _error("validation_error", str(e))
    except TaskGenerationError as e:
        return _error("task_generation_error", str(e))

    max_minutes = None
    if max_cost is not None:
        estimator = CostEstimator()
        expected = estimator.estimate_task(task, estimated_hours).estimate.expected_cost
        if expected > max_cost:
            return {
                **_error(
                    "over_budget",
                    f"Estimated ${expected:.2f} for {estimated_hours}h exceeds "
                    f"max_cost ${max_cost:.2f}; not launched.",
                ),
                "estimated_cost": expected,
            }
        max_minutes = estimator.affordable_minutes(task, max_cost)
        if max_minutes < 1:
            # SkyPilot's autostop granularity is one minute; less cannot be enforced.
            return _error(
                "over_budget",
                f"max_cost ${max_cost:.2f} buys less than one minute; not launched.",
            )

    cluster_id = f"{CLUSTER_PREFIX}{workflow}-{uuid.uuid4().hex[:8]}"
    try:
        request_id = await skypilot.launch(
            task, cluster_name=cluster_id, max_minutes=max_minutes
        )
    except SkyPilotError as e:
        return _error("skypilot_error", str(e))

    logger.info(
        "Submitted %s launch for %s (request %s)", workflow, cluster_id, request_id
    )
    result: dict[str, Any] = {
        "status": "submitted",
        "cluster_id": cluster_id,
        "launch_request_id": request_id,
        "workflow": request.workflow,
        "model": request.model,
        "dataset": request.dataset,
        "logs_command": f"sky api logs {request_id}",
    }
    if max_minutes is not None:
        result["cost_guard"] = {
            "max_cost": max_cost,
            "teardown_after_minutes": max_minutes,
        }
    return result


@mcp.tool()
async def get_training_status(cluster_id: str) -> dict[str, Any]:
    """Get the status of a training job's cluster.

    This is cluster status: "running" means the cluster is up, not that the
    training script is still running. Use ``sky queue <cluster_id>`` for the
    job itself.

    Args:
        cluster_id: The cluster ID returned from launch_training

    Returns:
        Dict with the cluster's job status and SkyPilot's cluster status
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
    """Tear a training cluster down, optionally downloading its logs first.

    Log download is best effort: a cluster that is not up (for example
    stopped) has no logs to fetch, and a running cluster costs money, so
    teardown always proceeds.

    Args:
        cluster_id: The cluster ID to stop
        download_results: Whether to download job logs before tearing down

    Returns:
        Dict with stop status, the local log directories, and any log error
    """
    try:
        if await skypilot.get_cluster(cluster_id) is None:
            return _error("not_found", f"Cluster {cluster_id} not found")
    except SkyPilotError as e:
        return _error("skypilot_error", str(e))

    result: dict[str, Any] = {
        "status": "stopped",
        "cluster_id": cluster_id,
        "log_paths": {},
    }
    if download_results:
        try:
            result["log_paths"] = await skypilot.download_logs(cluster_id)
        except SkyPilotError as e:
            logger.warning("Log download for %s failed: %s", cluster_id, e)
            result["log_error"] = str(e)

    try:
        await skypilot.down(cluster_id)
    except SkyPilotError as e:
        return _error(
            "skypilot_error",
            f"Teardown of {cluster_id} failed; it may still be running: {e}",
        )
    return result


@mcp.tool()
async def list_available_workflows() -> dict[str, Any]:
    """List all available training workflows and their descriptions.

    Returns:
        Dict with available workflows and their configurations
    """
    workflows_info = {
        "unsloth": {
            "description": (
                "Efficient fine-tuning for 7B-13B models with 4-bit quantization + LoRA"
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
    estimated_hours: Hours = 4.0,
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
    except TaskGenerationError as e:
        return _error("task_generation_error", str(e))

    cost = CostEstimator().estimate_task(task, estimated_hours)
    estimate = cost.estimate
    return {
        "workflow": workflow,
        "gpu_type": cost.gpu_type,
        "gpu_count": cost.gpu_count,
        "use_spot": cost.use_spot,
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


class _ReleaseStdout(Middleware):
    """Point ``sys.stdout`` back at the console once the transport is bound.

    Initialize is the first message of every MCP session, so no tool (and no
    SkyPilot call) can run before this.
    """

    def __init__(self, console: TextIO) -> None:
        """Remember the console stream to restore.

        Args:
            console: Stream that call-time ``print()`` should use.
        """
        self._console = console

    async def on_initialize(
        self, context: MiddlewareContext[Any], call_next: CallNext[Any, Any]
    ) -> Any:
        """Restore the console stream, then continue initialization.

        Args:
            context: Middleware context.
            call_next: Next handler in the chain.

        Returns:
            The initialize result.
        """
        sys.stdout = self._console
        return await call_next(context)


def run_server() -> None:
    """Run the MCP server over stdio.

    Stdout belongs to the MCP protocol, but SkyPilot writes to it: it prints
    streamed request logs and its logging handler targets ``sys.stdout``. So
    file descriptor 1 is pointed at stderr for the whole process, and the
    transport gets a private duplicate of the real stdout.

    This takes over the process: stdout stays redirected and the middleware
    stays registered until exit, so call it once, as the CLI does, and do not
    embed it in a process that needs its own stdout.
    """
    logging.basicConfig(level=logging.INFO)
    console = sys.stdout
    console.flush()
    protocol = os.fdopen(os.dup(console.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), console.fileno())

    # The stdio transport binds sys.stdout when it starts; afterwards
    # _ReleaseStdout hands sys.stdout back to the (now stderr-backed) console.
    sys.stdout = protocol
    mcp.add_middleware(_ReleaseStdout(console))
    logger.info("Available workflows: %s", ", ".join(WORKFLOWS))
    mcp.run()
