"""Tests for exception hierarchy and error handling."""

import time
from typing import Any

import pytest

from vibeml.exceptions import (
    VibeMLError,
    ConfigurationError,
    BudgetExceededError,
    CloudProviderError,
    ModelNotFoundError,
    DatasetError,
    SkyPilotError,
    retry_with_backoff,
    ErrorTranslator,
    get_region_fallbacks,
)


class TestVibeMLError:
    """Tests for base VibeMLError."""

    def test_basic_error(self) -> None:
        """Test creating a basic error."""
        error = VibeMLError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.technical_details is None
        assert error.recovery_suggestion is None

    def test_error_with_all_fields(self) -> None:
        """Test error with all optional fields."""
        error = VibeMLError(
            message="Operation failed",
            technical_details="Stack trace here",
            recovery_suggestion="Try again later",
            context={"user_id": "123", "operation": "launch"},
        )
        assert error.message == "Operation failed"
        assert error.technical_details == "Stack trace here"
        assert error.recovery_suggestion == "Try again later"
        assert error.context["user_id"] == "123"

    def test_error_string_with_suggestion(self) -> None:
        """Test error string formatting with suggestion."""
        error = VibeMLError(
            message="Operation failed",
            recovery_suggestion="Try again later",
        )
        assert "Operation failed" in str(error)
        assert "Suggestion: Try again later" in str(error)

    def test_debug_info(self) -> None:
        """Test get_debug_info returns all details."""
        error = VibeMLError(
            message="Test error",
            technical_details="Technical info",
            recovery_suggestion="Do this",
            context={"key": "value"},
        )
        debug_info = error.get_debug_info()
        assert debug_info["error_type"] == "VibeMLError"
        assert debug_info["message"] == "Test error"
        assert debug_info["technical_details"] == "Technical info"
        assert debug_info["recovery_suggestion"] == "Do this"
        assert debug_info["context"]["key"] == "value"


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_basic_config_error(self) -> None:
        """Test basic configuration error."""
        error = ConfigurationError("Invalid configuration")
        assert error.message == "Invalid configuration"

    def test_config_error_with_key(self) -> None:
        """Test configuration error with config key."""
        error = ConfigurationError(
            message="Invalid value",
            config_key="api_key",
            expected_type="string",
        )
        assert error.context["config_key"] == "api_key"
        assert error.context["expected_type"] == "string"


class TestBudgetExceededError:
    """Tests for BudgetExceededError."""

    def test_budget_exceeded_basic(self) -> None:
        """Test basic budget exceeded error."""
        error = BudgetExceededError("Cost limit exceeded")
        assert "Cost limit exceeded" in error.message
        assert "max_cost" in error.recovery_suggestion

    def test_budget_exceeded_with_costs(self) -> None:
        """Test budget error with cost information."""
        error = BudgetExceededError(
            message="Budget exceeded",
            estimated_cost=150.0,
            max_cost=100.0,
        )
        assert error.context["estimated_cost"] == 150.0
        assert error.context["max_cost"] == 100.0

    def test_budget_exceeded_default_suggestion(self) -> None:
        """Test default recovery suggestion."""
        error = BudgetExceededError("Budget exceeded")
        assert "spot instances" in error.recovery_suggestion


class TestCloudProviderError:
    """Tests for CloudProviderError."""

    def test_cloud_error_basic(self) -> None:
        """Test basic cloud provider error."""
        error = CloudProviderError("Cloud operation failed")
        assert error.message == "Cloud operation failed"

    def test_cloud_error_with_provider(self) -> None:
        """Test cloud error with provider info."""
        error = CloudProviderError(
            message="Operation failed",
            cloud_provider="nebius",
            region="eu-north1",
        )
        assert error.context["cloud_provider"] == "nebius"
        assert error.context["region"] == "eu-north1"


class TestModelNotFoundError:
    """Tests for ModelNotFoundError."""

    def test_model_not_found_basic(self) -> None:
        """Test basic model not found error."""
        error = ModelNotFoundError("Model not found")
        assert "Model not found" in error.message
        assert "HuggingFace" in error.recovery_suggestion

    def test_model_not_found_with_id(self) -> None:
        """Test model error with model ID."""
        error = ModelNotFoundError(
            message="Model not found",
            model_id="invalid/model",
        )
        assert error.context["model_id"] == "invalid/model"


class TestDatasetError:
    """Tests for DatasetError."""

    def test_dataset_error_basic(self) -> None:
        """Test basic dataset error."""
        error = DatasetError("Dataset access failed")
        assert "Dataset access failed" in error.message
        assert "HuggingFace" in error.recovery_suggestion

    def test_dataset_error_with_id(self) -> None:
        """Test dataset error with dataset ID."""
        error = DatasetError(
            message="Dataset not accessible",
            dataset_id="invalid/dataset",
        )
        assert error.context["dataset_id"] == "invalid/dataset"


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_successful_function(self) -> None:
        """Test that successful functions execute normally."""
        call_count = {"count": 0}

        @retry_with_backoff(max_retries=3)
        def successful_func() -> str:
            call_count["count"] += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count["count"] == 1

    def test_transient_failure(self) -> None:
        """Test retry on transient failure."""
        call_count = {"count": 0}

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def fails_twice() -> str:
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise ValueError("Transient error")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert call_count["count"] == 3

    def test_permanent_failure(self) -> None:
        """Test that permanent failures raise SkyPilotError."""
        @retry_with_backoff(max_retries=2, initial_delay=0.01)
        def always_fails() -> None:
            raise ValueError("Permanent error")

        with pytest.raises(SkyPilotError) as exc_info:
            always_fails()

        error = exc_info.value
        assert "failed after 2 retries" in error.message
        assert "Permanent error" in error.technical_details

    def test_exponential_backoff_timing(self) -> None:
        """Test that backoff delays increase exponentially."""
        call_times = []

        @retry_with_backoff(max_retries=3, initial_delay=0.1, backoff_factor=2.0)
        def track_timing() -> None:
            call_times.append(time.time())
            if len(call_times) < 4:
                raise ValueError("Retry")

        track_timing()

        # Check that delays roughly double (with some tolerance for timing)
        delays = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
        assert len(delays) == 3
        # First delay ~0.1s, second ~0.2s, third ~0.4s
        assert delays[0] < delays[1] < delays[2]


class TestErrorTranslator:
    """Tests for ErrorTranslator."""

    def test_translate_no_resource_available(self) -> None:
        """Test translating no resource error."""
        error = Exception("No resource available in region")
        translated = ErrorTranslator.translate(error)

        assert "GPU resources currently available" in translated.message
        assert "different GPU type" in translated.recovery_suggestion
        assert translated.technical_details == "No resource available in region"

    def test_translate_quota_exceeded(self) -> None:
        """Test translating quota exceeded error."""
        error = Exception("Quota exceeded for GPU instances")
        translated = ErrorTranslator.translate(error)

        assert "quota limit reached" in translated.message.lower()
        assert "increase quota" in translated.recovery_suggestion

    def test_translate_permission_denied(self) -> None:
        """Test translating permission error."""
        error = Exception("Permission denied to access resource")
        translated = ErrorTranslator.translate(error)

        assert "permissions" in translated.message.lower()
        assert "credentials" in translated.recovery_suggestion

    def test_translate_unknown_error(self) -> None:
        """Test translating unknown error."""
        error = Exception("Some unknown error occurred")
        translated = ErrorTranslator.translate(error)

        assert "unexpected error" in translated.message.lower()
        assert translated.technical_details == "Some unknown error occurred"
        assert translated.context["original_error_type"] == "Exception"

    def test_translate_skypilot_cluster_not_found(self) -> None:
        """Test translating SkyPilot cluster not found error."""
        error = Exception("Cluster not found: vibeml-test")
        translated = ErrorTranslator.translate_skypilot_error(error)

        assert isinstance(translated, SkyPilotError)
        assert "cluster not found" in translated.message.lower()
        assert "terminated" in translated.recovery_suggestion

    def test_translate_skypilot_launch_failed(self) -> None:
        """Test translating SkyPilot launch failed error."""
        error = Exception("Launch failed due to capacity issues")
        translated = ErrorTranslator.translate_skypilot_error(error)

        assert isinstance(translated, SkyPilotError)
        assert "launch" in translated.message.lower()
        assert "cloud provider" in translated.recovery_suggestion

    def test_context_preservation(self) -> None:
        """Test that error context is preserved."""
        error = ValueError("Test error")
        translated = ErrorTranslator.translate(error)

        assert translated.context["original_error_type"] == "ValueError"


class TestGetRegionFallbacks:
    """Tests for get_region_fallbacks function."""

    def test_nebius_fallbacks(self) -> None:
        """Test Nebius region fallbacks."""
        fallbacks = get_region_fallbacks("nebius", "eu-north1")
        assert "eu-west1" in fallbacks
        assert "eu-north1" not in fallbacks  # Current region excluded

    def test_aws_fallbacks(self) -> None:
        """Test AWS region fallbacks."""
        fallbacks = get_region_fallbacks("aws", "us-east-1")
        assert "us-west-2" in fallbacks
        assert "eu-west-1" in fallbacks
        assert "us-east-1" not in fallbacks

    def test_gcp_fallbacks(self) -> None:
        """Test GCP region fallbacks."""
        fallbacks = get_region_fallbacks("gcp", "us-central1")
        assert "us-east1" in fallbacks
        assert "europe-west1" in fallbacks
        assert "us-central1" not in fallbacks

    def test_azure_fallbacks(self) -> None:
        """Test Azure region fallbacks."""
        fallbacks = get_region_fallbacks("azure", "eastus")
        assert "westus2" in fallbacks
        assert "westeurope" in fallbacks
        assert "eastus" not in fallbacks

    def test_unknown_cloud(self) -> None:
        """Test unknown cloud provider returns empty list."""
        fallbacks = get_region_fallbacks("unknown-cloud", "some-region")
        assert fallbacks == []
