"""Tests for validation utilities."""

import pytest
from unittest.mock import Mock, patch

from vibeml.validation import ModelValidator, DatasetValidator, ResourceValidator
from vibeml.validation.resource_validator import ResourceRequirements
from vibeml.exceptions import ModelNotFoundError, DatasetError, ResourceError, ValidationError


class TestModelValidator:
    """Tests for ModelValidator."""

    def test_validate_gpt_oss_size(self) -> None:
        """Test validating GPT-OSS size format."""
        validator = ModelValidator()

        result = validator.validate_model("20b")
        assert result["model_id"] == "20b"
        assert result["model_type"] == "gpt-oss"
        assert result["accessible"] is True

    def test_estimate_model_size_from_id(self) -> None:
        """Test estimating model size from ID."""
        validator = ModelValidator()

        assert validator._estimate_model_size("mistralai/Mistral-7B-v0.1") == "7b"
        assert validator._estimate_model_size("meta-llama/Llama-2-13b-chat") == "13b"
        assert validator._estimate_model_size("gpt-70B-model") == "70b"

    def test_calculate_gpu_memory_base(self) -> None:
        """Test GPU memory calculation."""
        validator = ModelValidator()

        memory = validator.calculate_gpu_memory_required("7b", batch_size=1)
        assert memory == 16  # 7B model base memory

    def test_calculate_gpu_memory_with_quantization(self) -> None:
        """Test GPU memory with quantization."""
        validator = ModelValidator()

        memory = validator.calculate_gpu_memory_required("7b", use_quantization=True)
        assert memory == 4  # 16GB / 4

    def test_calculate_gpu_memory_with_batch_size(self) -> None:
        """Test GPU memory with larger batch size."""
        validator = ModelValidator()

        memory = validator.calculate_gpu_memory_required("7b", batch_size=4)
        assert memory > 16  # Should be more than base

    def test_recommend_gpu_small_model(self) -> None:
        """Test GPU recommendation for small model."""
        validator = ModelValidator()

        gpu = validator.recommend_gpu_type("7b", use_quantization=True)
        assert gpu in ["RTX4090", "L40S"]  # Small model can use cheaper GPU

    def test_recommend_gpu_large_model(self) -> None:
        """Test GPU recommendation for large model."""
        validator = ModelValidator()

        gpu = validator.recommend_gpu_type("70b")
        assert gpu in ["A100", "H100"]  # Large model needs big GPU


class TestDatasetValidator:
    """Tests for DatasetValidator."""

    def test_estimate_dataset_size(self) -> None:
        """Test dataset size estimation."""
        validator = DatasetValidator()

        size = validator.estimate_dataset_size("tatsu-lab/alpaca")
        assert isinstance(size, float)
        assert size > 0


class TestResourceValidator:
    """Tests for ResourceValidator."""

    def test_calculate_requirements_basic(self) -> None:
        """Test basic resource calculation."""
        validator = ResourceValidator()

        reqs = validator.calculate_requirements(
            model_size_gb=16,
            dataset_size_gb=10,
        )

        assert isinstance(reqs, ResourceRequirements)
        assert reqs.min_gpu_memory > 0
        assert reqs.min_disk_size > 0
        assert len(reqs.suitable_gpus) > 0

    def test_calculate_requirements_with_quantization(self) -> None:
        """Test requirements with quantization."""
        validator = ResourceValidator()

        reqs_no_quant = validator.calculate_requirements(
            model_size_gb=16,
            use_quantization=False,
        )
        reqs_quant = validator.calculate_requirements(
            model_size_gb=16,
            use_quantization=True,
        )

        assert reqs_quant.min_gpu_memory < reqs_no_quant.min_gpu_memory

    def test_calculate_requirements_large_model(self) -> None:
        """Test requirements for large model."""
        validator = ResourceValidator()

        reqs = validator.calculate_requirements(
            model_size_gb=60,  # Large model that fits in H100/A100
        )

        assert "H100" in reqs.suitable_gpus or "A100" in reqs.suitable_gpus
        assert "RTX4090" not in reqs.suitable_gpus  # Too small

    def test_calculate_requirements_too_large(self) -> None:
        """Test error for model too large for any GPU."""
        validator = ResourceValidator()

        with pytest.raises(ResourceError, match="No GPU has enough memory"):
            validator.calculate_requirements(
                model_size_gb=500,  # Too large
                use_quantization=False,
            )

    def test_select_optimal_gpu_cost_effective(self) -> None:
        """Test optimal GPU selection favors cost."""
        validator = ResourceValidator()

        suitable = ["RTX4090", "L40S", "A100"]
        optimal = validator._select_optimal_gpu(suitable, memory_required=20)

        # Should prefer cheaper option that meets requirements
        assert optimal in suitable

    def test_validate_gpu_valid(self) -> None:
        """Test validating a suitable GPU."""
        validator = ResourceValidator()

        assert validator.validate_gpu_for_model("L40S", memory_required=30)

    def test_validate_gpu_insufficient_memory(self) -> None:
        """Test error for insufficient GPU memory."""
        validator = ResourceValidator()

        with pytest.raises(ResourceError, match="memory"):
            validator.validate_gpu_for_model("RTX4090", memory_required=50)

    def test_validate_gpu_unknown_type(self) -> None:
        """Test error for unknown GPU type."""
        validator = ResourceValidator()

        with pytest.raises(ResourceError, match="Unknown GPU"):
            validator.validate_gpu_for_model("InvalidGPU", memory_required=10)

    def test_optimize_spot_short_job(self) -> None:
        """Test spot recommendation for short job."""
        validator = ResourceValidator()

        result = validator.optimize_spot_vs_ondemand(estimated_duration_hours=1.5)

        assert result["recommendation"] == "spot"
        assert "risk" in result["reason"].lower()

    def test_optimize_spot_long_job(self) -> None:
        """Test on-demand recommendation for long job."""
        validator = ResourceValidator()

        result = validator.optimize_spot_vs_ondemand(estimated_duration_hours=10)

        assert result["recommendation"] == "on-demand"

    def test_calculate_disk_size_basic(self) -> None:
        """Test basic disk size calculation."""
        validator = ResourceValidator()

        disk = validator.calculate_disk_size(
            model_size_gb=16,
            dataset_size_gb=10,
        )

        assert disk >= 26  # At least model + dataset
        assert disk % 10 == 0  # Rounded to 10GB

    def test_calculate_disk_size_no_checkpoints(self) -> None:
        """Test disk size without checkpoint storage."""
        validator = ResourceValidator()

        disk_with = validator.calculate_disk_size(
            model_size_gb=16,
            dataset_size_gb=10,
            checkpoint_storage=True,
        )
        disk_without = validator.calculate_disk_size(
            model_size_gb=16,
            dataset_size_gb=10,
            checkpoint_storage=False,
        )

        assert disk_without < disk_with
