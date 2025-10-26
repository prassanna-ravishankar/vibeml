"""Pydantic models for VibeML data validation and type safety."""

from datetime import datetime, UTC
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class JobStatus(str, Enum):
    """Status of a training job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class CostEstimate(BaseModel):
    """Cost estimation for a training job."""

    model_config = ConfigDict(frozen=True)

    hourly_rate: float = Field(gt=0, description="Cost per hour in USD")
    estimated_duration_hours: float = Field(
        gt=0, description="Estimated training duration in hours"
    )
    min_cost: float = Field(ge=0, description="Minimum expected cost in USD")
    max_cost: float = Field(ge=0, description="Maximum expected cost in USD")
    expected_cost: float = Field(ge=0, description="Expected cost in USD")
    currency: str = Field(default="USD", description="Currency code")

    @field_validator("max_cost")
    @classmethod
    def max_cost_greater_than_min(cls, v: float, info: Any) -> float:
        """Validate that max_cost >= min_cost."""
        if "min_cost" in info.data and v < info.data["min_cost"]:
            raise ValueError("max_cost must be greater than or equal to min_cost")
        return v

    @field_validator("expected_cost")
    @classmethod
    def expected_cost_in_range(cls, v: float, info: Any) -> float:
        """Validate that expected_cost is between min and max."""
        if "min_cost" in info.data and "max_cost" in info.data:
            if not (info.data["min_cost"] <= v <= info.data["max_cost"]):
                raise ValueError(
                    "expected_cost must be between min_cost and max_cost"
                )
        return v


class WorkflowMetadata(BaseModel):
    """Metadata about a training workflow."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="Workflow name")
    description: str = Field(min_length=1, description="Workflow description")
    gpu_requirements: list[str] = Field(
        min_length=1, description="List of supported GPU types"
    )
    typical_duration_hours: tuple[float, float] = Field(
        description="Typical duration range (min, max) in hours"
    )
    cost_range_usd: tuple[float, float] = Field(
        description="Typical cost range (min, max) in USD"
    )
    supported_model_sizes: list[str] = Field(
        default_factory=list, description="Supported model sizes or types"
    )

    @field_validator("typical_duration_hours", "cost_range_usd")
    @classmethod
    def validate_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate that range tuples have min <= max."""
        if len(v) != 2:
            raise ValueError("Range must be a tuple of (min, max)")
        if v[0] > v[1]:
            raise ValueError("Min value must be less than or equal to max value")
        if v[0] < 0 or v[1] < 0:
            raise ValueError("Range values must be non-negative")
        return v


class TrainingRequest(BaseModel):
    """Request to launch a training job."""

    model: str = Field(min_length=1, description="HuggingFace model ID or size")
    dataset: str = Field(min_length=1, description="HuggingFace dataset ID")
    workflow: str = Field(
        default="unsloth", description="Training workflow type"
    )
    gpu_type: Optional[str] = Field(
        default=None, description="GPU type (auto-selected if not specified)"
    )
    cloud: str = Field(default="nebius", description="Cloud provider")
    max_cost: Optional[float] = Field(
        default=None, ge=0, description="Maximum cost limit in USD"
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional hyperparameters"
    )

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model format."""
        # Check for HuggingFace format (org/model) or size format (20b, 120b)
        if v in ["20b", "120b"]:
            return v
        if "/" not in v:
            raise ValueError(
                "Model must be in format 'org/model' or a size like '20b', '120b'"
            )
        return v

    @field_validator("gpu_type")
    @classmethod
    def validate_gpu_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate GPU type if specified."""
        if v is None:
            return v
        valid_gpus = {"L40S", "RTX4090", "H100", "A100"}
        if v not in valid_gpus:
            raise ValueError(
                f"GPU type must be one of {valid_gpus}, got '{v}'"
            )
        return v

    @field_validator("workflow")
    @classmethod
    def validate_workflow(cls, v: str) -> str:
        """Validate workflow type."""
        valid_workflows = {"unsloth", "gpt-oss-lora", "gpt-oss-full"}
        if v not in valid_workflows:
            raise ValueError(
                f"Workflow must be one of {valid_workflows}, got '{v}'"
            )
        return v

    @field_validator("cloud")
    @classmethod
    def validate_cloud(cls, v: str) -> str:
        """Validate cloud provider."""
        valid_clouds = {"nebius", "aws", "gcp", "azure"}
        if v not in valid_clouds:
            raise ValueError(
                f"Cloud must be one of {valid_clouds}, got '{v}'"
            )
        return v


class JobHandle(BaseModel):
    """Handle to a running or completed training job."""

    job_id: str = Field(min_length=1, description="Unique job identifier")
    cluster_name: str = Field(min_length=1, description="SkyPilot cluster name")
    status: JobStatus = Field(description="Current job status")
    cost_estimate: Optional[CostEstimate] = Field(
        default=None, description="Cost estimation for the job"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Job creation timestamp"
    )
    model: str = Field(min_length=1, description="Model being trained")
    dataset: str = Field(min_length=1, description="Dataset being used")
    workflow: str = Field(min_length=1, description="Workflow type")
    gpu_type: str = Field(min_length=1, description="GPU type being used")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional job metadata"
    )

    model_config = ConfigDict(use_enum_values=True)
