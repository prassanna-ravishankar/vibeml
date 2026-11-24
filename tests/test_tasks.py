"""Tests for task generation functions."""

import pytest
import sky

from vibeml.exceptions import ValidationError
from vibeml.tasks import (
    WORKFLOWS,
    build_task,
    create_gpt_oss_full_task,
    create_gpt_oss_lora_task,
    create_unsloth_task,
    get_workflow,
)


def _resources(task: sky.Task) -> sky.Resources:
    """Return the single Resources entry of a task."""
    (resources,) = task.resources
    return resources


class TestUnslothTask:
    """Test Unsloth task generation."""

    def test_create_unsloth_task_basic(self):
        """Test basic Unsloth task creation."""
        task = create_unsloth_task(
            model="mistralai/Mistral-7B-v0.1",
            dataset="tatsu-lab/alpaca",
            gpu_type="L40S",
        )

        assert isinstance(task, sky.Task)
        assert "vibeml-unsloth" in task.name
        assert str(_resources(task).cloud) == "nebius"
        assert _resources(task).accelerators == {"L40S": 1}
        assert _resources(task).use_spot is True

    def test_create_unsloth_task_with_custom_params(self):
        """Test Unsloth task with custom parameters."""
        task = create_unsloth_task(
            model="meta-llama/Llama-3.2-1B",
            dataset="yahma/alpaca-cleaned",
            gpu_type="H100",
            max_steps=100,
            learning_rate=1e-4,
            lora_r=32,
        )

        assert _resources(task).accelerators == {"H100": 1}
        assert "max_steps=100" in task.run
        assert "learning_rate=0.0001" in task.run
        assert "r=32" in task.run

    def test_create_unsloth_task_invalid_model(self):
        """Test Unsloth task with invalid model format."""
        with pytest.raises(ValidationError) as exc_info:
            create_unsloth_task(
                model="invalid_model",  # Missing org/model format
                dataset="tatsu-lab/alpaca",
            )
        assert "Invalid model format" in str(exc_info.value)

    def test_create_unsloth_task_invalid_gpu(self):
        """Test Unsloth task with unsupported GPU."""
        with pytest.raises(ValidationError) as exc_info:
            create_unsloth_task(
                model="mistralai/Mistral-7B-v0.1",
                dataset="tatsu-lab/alpaca",
                gpu_type="V100",  # Not in supported list
            )
        assert "Unsupported GPU type" in str(exc_info.value)

    def test_create_unsloth_task_empty_dataset(self):
        """Test Unsloth task with empty dataset."""
        with pytest.raises(ValidationError) as exc_info:
            create_unsloth_task(
                model="mistralai/Mistral-7B-v0.1",
                dataset="",
            )
        assert "Dataset cannot be empty" in str(exc_info.value)

    def test_create_unsloth_task_with_cloud_bucket(self):
        """Test Unsloth task with cloud bucket for checkpoints."""
        task = create_unsloth_task(
            model="mistralai/Mistral-7B-v0.1",
            dataset="tatsu-lab/alpaca",
            cloud_bucket="my-training-bucket",
        )

        assert task.file_mounts is not None
        assert "/outputs" in task.file_mounts
        assert task.file_mounts["/outputs"]["name"] == "my-training-bucket"


class TestGPTOSSLoRATask:
    """Test GPT-OSS LoRA task generation."""

    def test_create_gpt_oss_lora_20b(self):
        """Test GPT-OSS LoRA task for 20B model."""
        task = create_gpt_oss_lora_task(
            model_size="20b",
            dataset="tatsu-lab/alpaca",
        )

        assert isinstance(task, sky.Task)
        assert "vibeml-gpt-oss-lora-20b" in task.name
        assert str(_resources(task).cloud) == "nebius"
        assert _resources(task).accelerators == {"H100": 2}
        assert _resources(task).disk_size == 200

    def test_create_gpt_oss_lora_120b(self):
        """Test GPT-OSS LoRA task for 120B model."""
        task = create_gpt_oss_lora_task(
            model_size="120b",
            dataset="tatsu-lab/alpaca",
        )

        assert _resources(task).accelerators == {"H100": 8}
        assert _resources(task).disk_size == 500

    def test_create_gpt_oss_lora_invalid_size(self):
        """Test GPT-OSS LoRA task with invalid model size."""
        with pytest.raises(ValidationError) as exc_info:
            create_gpt_oss_lora_task(
                model_size="40b",  # Not supported
                dataset="tatsu-lab/alpaca",
            )
        assert "Invalid model size" in str(exc_info.value)


class TestGPTOSSFullTask:
    """Test GPT-OSS full fine-tuning task generation."""

    def test_create_gpt_oss_full_20b(self):
        """Test GPT-OSS full task for 20B model."""
        task = create_gpt_oss_full_task(
            model_size="20b",
            dataset="tatsu-lab/alpaca",
        )

        assert isinstance(task, sky.Task)
        assert "vibeml-gpt-oss-full-20b" in task.name
        assert str(_resources(task).cloud) == "nebius"
        assert _resources(task).accelerators == {"H100": 8}
        assert (
            _resources(task).use_spot is False
        )  # Full training needs stable instances
        assert task.num_nodes == 1

    def test_create_gpt_oss_full_120b(self):
        """Test GPT-OSS full task for 120B model."""
        task = create_gpt_oss_full_task(
            model_size="120b",
            dataset="tatsu-lab/alpaca",
        )

        assert _resources(task).accelerators == {"H100": 8}
        assert _resources(task).disk_size == 1000
        assert task.num_nodes == 4  # Multi-node for 120B


class TestWorkflowRegistry:
    """Test workflow registry functions."""

    def test_get_workflow_unsloth(self):
        """Test getting Unsloth workflow."""
        workflow_func = get_workflow("unsloth")
        assert workflow_func == create_unsloth_task

    def test_get_workflow_gpt_oss_lora(self):
        """Test getting GPT-OSS LoRA workflow."""
        workflow_func = get_workflow("gpt-oss-lora")
        assert workflow_func == create_gpt_oss_lora_task

    def test_get_workflow_invalid(self):
        """Test getting invalid workflow."""
        with pytest.raises(ValueError) as exc_info:
            get_workflow("invalid-workflow")
        assert "Unknown workflow" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_workflows_registry_contents(self):
        """Test that all expected workflows are in registry."""
        assert "unsloth" in WORKFLOWS
        assert "gpt-oss-lora" in WORKFLOWS
        assert "gpt-oss-full" in WORKFLOWS
        assert len(WORKFLOWS) == 3


class TestBuildTask:
    """build_task maps tool-level parameters onto a workflow's arguments."""

    def test_unsloth_defaults_to_l40s(self) -> None:
        """Without a GPU type, Unsloth uses the cost-effective L40S."""
        task = build_task(
            "unsloth", model="mistralai/Mistral-7B-v0.1", dataset="tatsu-lab/alpaca"
        )

        assert _resources(task).accelerators == {"L40S": 1}

    def test_unsloth_passes_hyperparameters(self) -> None:
        """Given hyperparameters reach the training script; others keep defaults."""
        task = build_task(
            "unsloth",
            model="meta-llama/Llama-3.2-1B",
            dataset="yahma/alpaca-cleaned",
            gpu_type="H100",
            max_steps=100,
            learning_rate=1e-4,
        )

        assert _resources(task).accelerators == {"H100": 1}
        assert "max_steps=100" in task.run
        assert "learning_rate=0.0001" in task.run

    def test_gpt_oss_uses_model_as_size(self) -> None:
        """For GPT-OSS workflows the model argument is the size."""
        task = build_task("gpt-oss-full", model="120b", dataset="tatsu-lab/alpaca")

        assert _resources(task).accelerators == {"H100": 8}
        assert task.num_nodes == 4

    def test_gpt_oss_rejects_unknown_size(self) -> None:
        """Unknown sizes are rejected rather than replaced with a default."""
        with pytest.raises(ValidationError, match="Invalid model size"):
            build_task("gpt-oss-lora", model="70b", dataset="tatsu-lab/alpaca")

    def test_unknown_workflow(self) -> None:
        """Unknown workflows raise ValidationError."""
        with pytest.raises(ValidationError, match="Unknown workflow"):
            build_task("nope", model="m", dataset="d")
