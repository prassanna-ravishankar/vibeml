"""FastMCP server for VibeML - natural language interface to AI training."""

import asyncio
import json
import uuid
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastmcp import FastMCP
import sky
from pydantic import ValidationError as PydanticValidationError

from .tasks import get_workflow, WORKFLOWS
from .exceptions import SkyPilotError, ValidationError, ResourceError
from .models import (
    TrainingRequest,
    JobHandle,
    JobStatus,
    CostEstimate,
    WorkflowMetadata,
)
from .persistence import JobsRepository


# Initialize FastMCP server
mcp = FastMCP("vibeml", version="0.0.1")
mcp.description = "Natural language interface for AI model training on Nebius Cloud"


# Initialize job persistence repository
_repository: Optional[JobsRepository] = None

# Store active clusters for tracking (synced with database)
ACTIVE_CLUSTERS: Dict[str, JobHandle] = {}


async def _get_repository() -> JobsRepository:
    """Get or initialize the jobs repository.

    Lazy initialization to ensure it's created in the async context.
    """
    global _repository
    if _repository is None:
        _repository = JobsRepository()
        # Hydrate ACTIVE_CLUSTERS from persisted jobs
        active_jobs = await _repository.list_active()
        for job in active_jobs:
            ACTIVE_CLUSTERS[job.job_id] = job
        print(f"Loaded {len(active_jobs)} active jobs from persistence layer")
    return _repository


@mcp.tool()
async def launch_training(
    workflow: str = "unsloth",
    model: str = "mistralai/Mistral-7B-v0.1",
    dataset: str = "tatsu-lab/alpaca",
    gpu_type: Optional[str] = None,
    max_steps: Optional[int] = 60,
    max_cost: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Launch a model training job on Nebius Cloud.

    Args:
        workflow: Training workflow type (unsloth, gpt-oss-lora, gpt-oss-full)
        model: HuggingFace model ID or size for GPT-OSS (e.g., "20b")
        dataset: HuggingFace dataset ID
        gpu_type: GPU type (L40S, RTX4090, H100, A100) - auto-selected if not specified
        max_steps: Maximum training steps
        max_cost: Maximum cost limit in USD
        **kwargs: Additional workflow-specific parameters

    Returns:
        Dict with cluster_id, status, and launch details

    Examples:
        - "Train Mistral 7B on Alpaca dataset using cheapest GPU"
        - "Fine-tune Llama model with LoRA on my custom dataset"
        - "Launch GPT-OSS 20B LoRA training on H100 GPUs"
    """
    try:
        # Validate request using Pydantic model
        try:
            request = TrainingRequest(
                model=model,
                dataset=dataset,
                workflow=workflow,
                gpu_type=gpu_type,
                max_cost=max_cost,
                hyperparameters=kwargs,
            )
        except PydanticValidationError as e:
            return {
                "status": "error",
                "error": "validation_error",
                "message": str(e),
                "details": e.errors(),
            }
        # Get the workflow function
        workflow_func = get_workflow(workflow)

        # Prepare arguments based on workflow type
        task_args = {}

        if workflow == "unsloth":
            task_args = {
                "model": model,
                "dataset": dataset,
                "gpu_type": gpu_type or "L40S",  # Default to L40S for cost-effectiveness
                "max_steps": max_steps,
            }
            # Add any additional kwargs that the workflow supports
            for key in ["learning_rate", "lora_r", "max_seq_length", "output_dir"]:
                if key in kwargs:
                    task_args[key] = kwargs[key]

        elif workflow in ["gpt-oss-lora", "gpt-oss-full"]:
            # For GPT-OSS, model is the size (20b, 120b)
            task_args = {
                "model_size": model if model in ["20b", "120b"] else "20b",
                "dataset": dataset,
            }
            if "output_dir" in kwargs:
                task_args["output_dir"] = kwargs["output_dir"]

        # Generate the SkyPilot task
        task = workflow_func(**task_args)

        # Generate unique cluster name
        cluster_id = f"vibeml-{workflow}-{uuid.uuid4().hex[:8]}"

        # Launch the task asynchronously
        print(f"Launching {workflow} training job: {cluster_id}")

        # Use asyncio to run SkyPilot's sync launch in background
        loop = asyncio.get_event_loop()
        cluster_handle = await loop.run_in_executor(
            None,
            lambda: sky.launch(task, cluster_name=cluster_id, detach_run=True)
        )

        # Create JobHandle for tracking
        job_handle = JobHandle(
            job_id=cluster_id,
            cluster_name=cluster_id,
            status=JobStatus.RUNNING,
            model=request.model,
            dataset=request.dataset,
            workflow=request.workflow,
            gpu_type=request.gpu_type or task_args.get("gpu_type", "auto"),
            created_at=datetime.now(UTC),
            metadata={"handle": cluster_handle, "task_args": task_args},
        )

        # Store cluster info in memory and persist to database
        ACTIVE_CLUSTERS[cluster_id] = job_handle

        # Persist job to database
        repository = await _get_repository()
        await repository.save_job(job_handle)

        return {
            "cluster_id": cluster_id,
            "status": "launched",
            "workflow": request.workflow,
            "model": request.model,
            "dataset": request.dataset,
            "message": f"Training job launched successfully on Nebius Cloud",
            "monitor_command": f"sky status {cluster_id}",
            "logs_command": f"sky logs {cluster_id}",
        }

    except ValidationError as e:
        return {
            "status": "error",
            "error": "validation_error",
            "message": str(e),
        }
    except Exception as e:
        raise SkyPilotError(f"Failed to launch training: {str(e)}")


@mcp.tool()
async def get_training_status(cluster_id: str) -> Dict[str, Any]:
    """Get the status of a running training job.

    Args:
        cluster_id: The cluster ID returned from launch_training

    Returns:
        Dict with cluster status, logs preview, and resource usage
    """
    try:
        repository = await _get_repository()

        # Try to get from memory first, then from database
        job_handle = ACTIVE_CLUSTERS.get(cluster_id)
        if not job_handle:
            job_handle = await repository.get_job(cluster_id)
            if not job_handle:
                return {
                    "status": "error",
                    "message": f"Cluster {cluster_id} not found",
                }
            # Restore to ACTIVE_CLUSTERS if it's still active
            if job_handle.status in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.REVIEW]:
                ACTIVE_CLUSTERS[cluster_id] = job_handle

        # Get cluster status using SkyPilot
        loop = asyncio.get_event_loop()
        status_output = await loop.run_in_executor(
            None,
            lambda: sky.status(cluster_names=[cluster_id], refresh=True)
        )

        # Parse status (this is simplified - actual implementation would parse the output)
        # In production, we'd parse status_output to update job_handle.status

        return {
            "cluster_id": cluster_id,
            "workflow": job_handle.workflow,
            "model": job_handle.model,
            "dataset": job_handle.dataset,
            "gpu_type": job_handle.gpu_type,
            "status": job_handle.status.value,
            "created_at": job_handle.created_at.isoformat(),
            "message": "Training in progress",
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to get status: {str(e)}",
        }


@mcp.tool()
async def stop_training(cluster_id: str, download_results: bool = True) -> Dict[str, Any]:
    """Stop a running training job and optionally download results.

    Args:
        cluster_id: The cluster ID to stop
        download_results: Whether to download results before stopping

    Returns:
        Dict with stop status and download location if applicable
    """
    try:
        if cluster_id not in ACTIVE_CLUSTERS:
            return {
                "status": "error",
                "message": f"Cluster {cluster_id} not found",
            }

        # Download results if requested
        download_path = None
        if download_results:
            download_path = f"./vibeml_outputs/{cluster_id}"
            Path(download_path).mkdir(parents=True, exist_ok=True)

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: sky.download_logs(cluster_id, download_path)
            )

        # Stop the cluster
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: sky.down(cluster_id)
        )

        # Update status in database
        repository = await _get_repository()
        await repository.update_status(
            job_id=cluster_id,
            status=JobStatus.TERMINATED,
            metadata_updates={"stopped_at": datetime.now(UTC).isoformat()} if download_results else {}
        )

        # Remove from active clusters (still in DB for history)
        if cluster_id in ACTIVE_CLUSTERS:
            del ACTIVE_CLUSTERS[cluster_id]

        return {
            "status": "stopped",
            "cluster_id": cluster_id,
            "message": "Training job stopped successfully",
            "download_path": download_path if download_results else None,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to stop training: {str(e)}",
        }


@mcp.tool()
async def list_available_workflows() -> Dict[str, Any]:
    """List all available training workflows and their descriptions.

    Returns:
        Dict with available workflows and their configurations
    """
    workflows_info = {
        "unsloth": {
            "description": "Efficient fine-tuning for 7B-13B models using 4-bit quantization + LoRA",
            "gpu_types": ["L40S", "RTX4090", "H100", "A100"],
            "estimated_cost": "$0.50-2/hour",
            "training_time": "2-8 hours",
            "best_for": "Quick experiments, cost-effective training",
        },
        "gpt-oss-lora": {
            "description": "LoRA fine-tuning for large GPT models (20B-120B)",
            "gpu_types": ["H100"],
            "estimated_cost": "$6-24/hour",
            "training_time": "4-24 hours",
            "best_for": "Production quality with lower resource usage",
        },
        "gpt-oss-full": {
            "description": "Full parameter training for maximum quality",
            "gpu_types": ["H100"],
            "estimated_cost": "$24-200/hour",
            "training_time": "8-48 hours",
            "best_for": "Maximum quality, large-scale production training",
        },
    }

    return {
        "workflows": workflows_info,
        "available": list(WORKFLOWS.keys()),
        "message": "Choose a workflow based on your model size and quality requirements",
    }


@mcp.tool()
async def estimate_training_cost(
    workflow: str = "unsloth",
    model: str = "mistralai/Mistral-7B-v0.1",
    gpu_type: Optional[str] = None,
    estimated_hours: float = 4.0,
) -> Dict[str, Any]:
    """Estimate the cost of a training job before launching.

    Args:
        workflow: Training workflow type
        model: Model ID or size
        gpu_type: GPU type (auto-selected if not specified)
        estimated_hours: Estimated training hours

    Returns:
        Dict with cost estimation and recommendations
    """
    # Nebius GPU pricing (approximate - would fetch real prices in production)
    gpu_prices = {
        "L40S": 1.2,  # $/hour
        "RTX4090": 0.8,
        "H100": 3.5,
        "A100": 2.4,
    }

    # Determine GPU based on workflow if not specified
    if not gpu_type:
        if workflow == "unsloth":
            gpu_type = "L40S"
        elif workflow in ["gpt-oss-lora", "gpt-oss-full"]:
            gpu_type = "H100"

    if gpu_type not in gpu_prices:
        return {
            "status": "error",
            "message": f"Unknown GPU type: {gpu_type}",
        }

    # Calculate costs
    hourly_cost = gpu_prices[gpu_type]

    # Adjust for multi-GPU setups
    if workflow == "gpt-oss-lora" and "120b" in str(model):
        hourly_cost *= 8  # 8x H100 for 120B model
    elif workflow == "gpt-oss-lora" and "20b" in str(model):
        hourly_cost *= 2  # 2x H100 for 20B model
    elif workflow == "gpt-oss-full":
        hourly_cost *= 8  # Minimum 8x H100 for full training

    total_cost = hourly_cost * estimated_hours

    # Add spot instance discount
    spot_discount = 0.7  # 30% cheaper on average
    spot_cost = total_cost * spot_discount

    return {
        "workflow": workflow,
        "gpu_type": gpu_type,
        "gpu_count": 8 if "full" in workflow or "120b" in str(model) else (2 if "20b" in str(model) else 1),
        "hourly_cost": f"${hourly_cost:.2f}",
        "estimated_hours": estimated_hours,
        "on_demand_cost": f"${total_cost:.2f}",
        "spot_cost": f"${spot_cost:.2f}",
        "recommended": "spot",
        "savings": f"${total_cost - spot_cost:.2f}",
        "message": f"Estimated cost for {estimated_hours}h training on {gpu_type} GPU(s)",
    }


@mcp.tool()
async def list_active_training_jobs() -> Dict[str, Any]:
    """List all active training jobs.

    Returns:
        Dict with list of active training clusters
    """
    repository = await _get_repository()

    # Get active jobs from database (authoritative source)
    active_jobs = await repository.list_active()

    # Sync ACTIVE_CLUSTERS with database
    ACTIVE_CLUSTERS.clear()
    for job in active_jobs:
        ACTIVE_CLUSTERS[job.job_id] = job

    if not active_jobs:
        return {
            "active_jobs": [],
            "count": 0,
            "message": "No active training jobs",
        }

    jobs = []
    for job_handle in active_jobs:
        jobs.append({
            "cluster_id": job_handle.job_id,
            "workflow": job_handle.workflow,
            "model": job_handle.model,
            "dataset": job_handle.dataset,
            "gpu_type": job_handle.gpu_type,
            "status": job_handle.status.value,
            "created_at": job_handle.created_at.isoformat(),
        })

    return {
        "active_jobs": jobs,
        "count": len(jobs),
        "message": f"Found {len(jobs)} active training job(s)",
    }


def run_server():
    """Run the FastMCP server."""
    import asyncio
    from fastmcp.server import run_stdio

    print("Starting VibeML MCP Server...")
    print(f"Available workflows: {', '.join(WORKFLOWS.keys())}")

    # Run the server
    asyncio.run(run_stdio(mcp))