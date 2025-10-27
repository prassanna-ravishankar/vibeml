"""FastMCP server for VibeML - natural language interface to AI training."""

import asyncio
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


# Initialize FastMCP server
mcp = FastMCP("vibeml", version="0.0.1")
mcp.description = "Natural language interface for AI model training on Nebius Cloud"


def _map_sky_status_to_job_status(sky_status: str) -> JobStatus:
    """Map SkyPilot cluster status to VibeML JobStatus.

    SkyPilot statuses: INIT, UP, STOPPED, TERMINATED
    """
    status_map = {
        "INIT": JobStatus.PENDING,
        "UP": JobStatus.RUNNING,
        "STOPPED": JobStatus.TERMINATED,
        "TERMINATED": JobStatus.TERMINATED,
    }
    return status_map.get(sky_status, JobStatus.UNKNOWN)


async def _get_cluster_from_skypilot(cluster_id: str) -> Optional[Dict[str, Any]]:
    """Get cluster information from SkyPilot's state.

    Args:
        cluster_id: Cluster name to query

    Returns:
        Dict with cluster info or None if not found
    """
    try:
        loop = asyncio.get_event_loop()
        clusters = await loop.run_in_executor(
            None,
            lambda: sky.status(cluster_names=[cluster_id], refresh=True)
        )

        if not clusters:
            return None

        # SkyPilot returns list of cluster records
        # Each has: name, status, launched_at, resources, handle
        cluster = clusters[0]
        return {
            "cluster_name": cluster.name,
            "status": str(cluster.status),
            "launched_at": cluster.launched_at,
            "resources": cluster.resources,
            "handle": cluster.handle,
        }
    except Exception as e:
        print(f"Failed to get cluster {cluster_id} from SkyPilot: {e}")
        return None


@mcp.tool()
async def launch_training(
    workflow: str = "unsloth",
    model: str = "mistralai/Mistral-7B-v0.1",
    dataset: str = "tatsu-lab/alpaca",
    gpu_type: Optional[str] = None,
    max_steps: Optional[int] = 60,
    max_cost: Optional[float] = None,
    learning_rate: Optional[float] = None,
    lora_r: Optional[int] = None,
    max_seq_length: Optional[int] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Launch a model training job on Nebius Cloud.

    Args:
        workflow: Training workflow type (unsloth, gpt-oss-lora, gpt-oss-full)
        model: HuggingFace model ID or size for GPT-OSS (e.g., "20b")
        dataset: HuggingFace dataset ID
        gpu_type: GPU type (L40S, RTX4090, H100, A100) - auto-selected if not specified
        max_steps: Maximum training steps
        max_cost: Maximum cost limit in USD
        learning_rate: Learning rate for training
        lora_r: LoRA rank parameter
        max_seq_length: Maximum sequence length
        output_dir: Output directory for results

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
                hyperparameters={
                    k: v for k, v in {
                        "learning_rate": learning_rate,
                        "lora_r": lora_r,
                        "max_seq_length": max_seq_length,
                        "output_dir": output_dir,
                        "max_steps": max_steps,
                    }.items() if v is not None
                },
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
            # Add optional workflow-specific parameters
            if learning_rate is not None:
                task_args["learning_rate"] = learning_rate
            if lora_r is not None:
                task_args["lora_r"] = lora_r
            if max_seq_length is not None:
                task_args["max_seq_length"] = max_seq_length
            if output_dir is not None:
                task_args["output_dir"] = output_dir

        elif workflow in ["gpt-oss-lora", "gpt-oss-full"]:
            # For GPT-OSS, model is the size (20b, 120b)
            task_args = {
                "model_size": model if model in ["20b", "120b"] else "20b",
                "dataset": dataset,
            }
            if output_dir is not None:
                task_args["output_dir"] = output_dir

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
        # Get cluster info directly from SkyPilot
        cluster_info = await _get_cluster_from_skypilot(cluster_id)

        if not cluster_info:
            return {
                "status": "error",
                "message": f"Cluster {cluster_id} not found in SkyPilot state",
            }

        # Map SkyPilot status to our JobStatus
        job_status = _map_sky_status_to_job_status(cluster_info["status"])

        return {
            "cluster_id": cluster_id,
            "status": job_status.value,
            "sky_status": cluster_info["status"],
            "launched_at": cluster_info["launched_at"].isoformat() if cluster_info["launched_at"] else None,
            "message": f"Cluster is {cluster_info['status'].lower()}",
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
        # Verify cluster exists in SkyPilot
        cluster_info = await _get_cluster_from_skypilot(cluster_id)
        if not cluster_info:
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
    try:
        # Get all clusters from SkyPilot
        loop = asyncio.get_event_loop()
        all_clusters = await loop.run_in_executor(
            None,
            lambda: sky.status(refresh=True)
        )

        if not all_clusters:
            return {
                "active_jobs": [],
                "count": 0,
                "message": "No active training jobs",
            }

        # Filter for VibeML clusters (start with "vibeml-")
        # and are in active states (INIT, UP)
        vibeml_clusters = [
            c for c in all_clusters
            if c.name.startswith("vibeml-") and str(c.status) in ["INIT", "UP"]
        ]

        jobs = []
        for cluster in vibeml_clusters:
            jobs.append({
                "cluster_id": cluster.name,
                "status": _map_sky_status_to_job_status(str(cluster.status)).value,
                "sky_status": str(cluster.status),
                "launched_at": cluster.launched_at.isoformat() if cluster.launched_at else None,
            })

        return {
            "active_jobs": jobs,
            "count": len(jobs),
            "message": f"Found {len(jobs)} active training job(s)",
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list jobs: {str(e)}",
        }


def run_server():
    """Run the FastMCP server."""
    import asyncio
    from fastmcp.server import run_stdio

    print("Starting VibeML MCP Server...")
    print(f"Available workflows: {', '.join(WORKFLOWS.keys())}")
    print("Note: Job state is managed by SkyPilot (~/.sky/state.json)")

    # Run the server
    asyncio.run(run_stdio(mcp))
