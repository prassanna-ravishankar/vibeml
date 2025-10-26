"""Tests for Pydantic models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from vibeml.models import (
    JobStatus,
    CostEstimate,
    WorkflowMetadata,
    TrainingRequest,
    JobHandle,
)


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_job_status_values(self) -> None:
        """Test that all job status values are correct."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.TERMINATED.value == "terminated"


class TestCostEstimate:
    """Tests for CostEstimate model."""

    def test_valid_cost_estimate(self) -> None:
        """Test creating a valid cost estimate."""
        estimate = CostEstimate(
            hourly_rate=2.5,
            estimated_duration_hours=4.0,
            min_cost=8.0,
            max_cost=12.0,
            expected_cost=10.0,
        )
        assert estimate.hourly_rate == 2.5
        assert estimate.estimated_duration_hours == 4.0
        assert estimate.min_cost == 8.0
        assert estimate.max_cost == 12.0
        assert estimate.expected_cost == 10.0
        assert estimate.currency == "USD"

    def test_cost_estimate_immutable(self) -> None:
        """Test that cost estimate is immutable."""
        estimate = CostEstimate(
            hourly_rate=2.5,
            estimated_duration_hours=4.0,
            min_cost=8.0,
            max_cost=12.0,
            expected_cost=10.0,
        )
        with pytest.raises(ValidationError):
            estimate.hourly_rate = 3.0  # type: ignore[misc]

    def test_invalid_hourly_rate(self) -> None:
        """Test validation of hourly rate."""
        with pytest.raises(ValidationError):
            CostEstimate(
                hourly_rate=0,
                estimated_duration_hours=4.0,
                min_cost=8.0,
                max_cost=12.0,
                expected_cost=10.0,
            )

        with pytest.raises(ValidationError):
            CostEstimate(
                hourly_rate=-1.0,
                estimated_duration_hours=4.0,
                min_cost=8.0,
                max_cost=12.0,
                expected_cost=10.0,
            )

    def test_max_cost_validation(self) -> None:
        """Test that max_cost must be >= min_cost."""
        with pytest.raises(ValidationError, match="max_cost must be greater"):
            CostEstimate(
                hourly_rate=2.5,
                estimated_duration_hours=4.0,
                min_cost=12.0,
                max_cost=8.0,
                expected_cost=10.0,
            )

    def test_expected_cost_validation(self) -> None:
        """Test that expected_cost must be between min and max."""
        with pytest.raises(ValidationError, match="expected_cost must be between"):
            CostEstimate(
                hourly_rate=2.5,
                estimated_duration_hours=4.0,
                min_cost=8.0,
                max_cost=12.0,
                expected_cost=15.0,
            )


class TestWorkflowMetadata:
    """Tests for WorkflowMetadata model."""

    def test_valid_workflow_metadata(self) -> None:
        """Test creating valid workflow metadata."""
        metadata = WorkflowMetadata(
            name="unsloth",
            description="Efficient fine-tuning",
            gpu_requirements=["L40S", "RTX4090"],
            typical_duration_hours=(2.0, 8.0),
            cost_range_usd=(1.0, 5.0),
            supported_model_sizes=["7b", "13b"],
        )
        assert metadata.name == "unsloth"
        assert len(metadata.gpu_requirements) == 2
        assert metadata.typical_duration_hours == (2.0, 8.0)

    def test_invalid_duration_range(self) -> None:
        """Test validation of duration range."""
        with pytest.raises(ValidationError, match="Min value must be less than"):
            WorkflowMetadata(
                name="test",
                description="Test workflow",
                gpu_requirements=["H100"],
                typical_duration_hours=(10.0, 5.0),  # Invalid: min > max
                cost_range_usd=(1.0, 5.0),
            )

    def test_negative_range_values(self) -> None:
        """Test that range values cannot be negative."""
        with pytest.raises(ValidationError, match="Range values must be non-negative"):
            WorkflowMetadata(
                name="test",
                description="Test workflow",
                gpu_requirements=["H100"],
                typical_duration_hours=(-1.0, 5.0),
                cost_range_usd=(1.0, 5.0),
            )


class TestTrainingRequest:
    """Tests for TrainingRequest model."""

    def test_valid_training_request(self) -> None:
        """Test creating a valid training request."""
        request = TrainingRequest(
            model="mistralai/Mistral-7B-v0.1",
            dataset="tatsu-lab/alpaca",
            workflow="unsloth",
            gpu_type="L40S",
        )
        assert request.model == "mistralai/Mistral-7B-v0.1"
        assert request.dataset == "tatsu-lab/alpaca"
        assert request.workflow == "unsloth"
        assert request.gpu_type == "L40S"
        assert request.cloud == "nebius"

    def test_model_size_format(self) -> None:
        """Test that model can be a size like '20b'."""
        request = TrainingRequest(
            model="20b",
            dataset="tatsu-lab/alpaca",
            workflow="gpt-oss-lora",
        )
        assert request.model == "20b"

    def test_invalid_model_format(self) -> None:
        """Test validation of model format."""
        with pytest.raises(ValidationError, match="Model must be in format"):
            TrainingRequest(
                model="invalid-model-name",  # Missing org/
                dataset="tatsu-lab/alpaca",
            )

    def test_invalid_gpu_type(self) -> None:
        """Test validation of GPU type."""
        with pytest.raises(ValidationError, match="GPU type must be one of"):
            TrainingRequest(
                model="mistralai/Mistral-7B-v0.1",
                dataset="tatsu-lab/alpaca",
                gpu_type="InvalidGPU",
            )

    def test_invalid_workflow(self) -> None:
        """Test validation of workflow type."""
        with pytest.raises(ValidationError, match="Workflow must be one of"):
            TrainingRequest(
                model="mistralai/Mistral-7B-v0.1",
                dataset="tatsu-lab/alpaca",
                workflow="invalid-workflow",
            )

    def test_invalid_cloud(self) -> None:
        """Test validation of cloud provider."""
        with pytest.raises(ValidationError, match="Cloud must be one of"):
            TrainingRequest(
                model="mistralai/Mistral-7B-v0.1",
                dataset="tatsu-lab/alpaca",
                cloud="invalid-cloud",
            )

    def test_max_cost_non_negative(self) -> None:
        """Test that max_cost must be non-negative."""
        with pytest.raises(ValidationError):
            TrainingRequest(
                model="mistralai/Mistral-7B-v0.1",
                dataset="tatsu-lab/alpaca",
                max_cost=-10.0,
            )

    def test_hyperparameters(self) -> None:
        """Test that hyperparameters are stored correctly."""
        request = TrainingRequest(
            model="mistralai/Mistral-7B-v0.1",
            dataset="tatsu-lab/alpaca",
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
        )
        assert request.hyperparameters["learning_rate"] == 0.001
        assert request.hyperparameters["batch_size"] == 32


class TestJobHandle:
    """Tests for JobHandle model."""

    def test_valid_job_handle(self) -> None:
        """Test creating a valid job handle."""
        handle = JobHandle(
            job_id="vibeml-test-123",
            cluster_name="vibeml-test-123",
            status=JobStatus.RUNNING,
            model="mistralai/Mistral-7B-v0.1",
            dataset="tatsu-lab/alpaca",
            workflow="unsloth",
            gpu_type="L40S",
        )
        assert handle.job_id == "vibeml-test-123"
        assert handle.status == JobStatus.RUNNING
        assert isinstance(handle.created_at, datetime)

    def test_job_handle_with_cost_estimate(self) -> None:
        """Test job handle with cost estimate."""
        cost_estimate = CostEstimate(
            hourly_rate=2.5,
            estimated_duration_hours=4.0,
            min_cost=8.0,
            max_cost=12.0,
            expected_cost=10.0,
        )
        handle = JobHandle(
            job_id="vibeml-test-123",
            cluster_name="vibeml-test-123",
            status=JobStatus.RUNNING,
            cost_estimate=cost_estimate,
            model="mistralai/Mistral-7B-v0.1",
            dataset="tatsu-lab/alpaca",
            workflow="unsloth",
            gpu_type="L40S",
        )
        assert handle.cost_estimate == cost_estimate

    def test_job_handle_metadata(self) -> None:
        """Test that metadata can store additional info."""
        handle = JobHandle(
            job_id="vibeml-test-123",
            cluster_name="vibeml-test-123",
            status=JobStatus.RUNNING,
            model="mistralai/Mistral-7B-v0.1",
            dataset="tatsu-lab/alpaca",
            workflow="unsloth",
            gpu_type="L40S",
            metadata={"custom_field": "value", "retries": 3},
        )
        assert handle.metadata["custom_field"] == "value"
        assert handle.metadata["retries"] == 3

    def test_empty_strings_validation(self) -> None:
        """Test that empty strings are not allowed."""
        with pytest.raises(ValidationError):
            JobHandle(
                job_id="",  # Empty string not allowed
                cluster_name="test",
                status=JobStatus.RUNNING,
                model="mistralai/Mistral-7B-v0.1",
                dataset="tatsu-lab/alpaca",
                workflow="unsloth",
                gpu_type="L40S",
            )
