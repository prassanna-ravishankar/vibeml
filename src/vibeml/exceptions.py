"""Custom exception hierarchy for VibeML."""

import time
from typing import Optional, Any, Dict, List, Callable
from functools import wraps


class VibeMLError(Exception):
    """Base exception for all VibeML operations.

    Attributes:
        message: Human-readable error message
        technical_details: Technical error information for debugging
        recovery_suggestion: Suggested action for recovery
        context: Additional context information
    """

    def __init__(
        self,
        message: str,
        technical_details: Optional[str] = None,
        recovery_suggestion: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize VibeMLError.

        Args:
            message: User-friendly error message
            technical_details: Technical details for debugging
            recovery_suggestion: Suggested recovery action
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.technical_details = technical_details
        self.recovery_suggestion = recovery_suggestion
        self.context = context or {}

    def __str__(self) -> str:
        """Return user-friendly error message."""
        parts = [self.message]
        if self.recovery_suggestion:
            parts.append(f"\nSuggestion: {self.recovery_suggestion}")
        return "\n".join(parts)

    def get_debug_info(self) -> Dict[str, Any]:
        """Get full debugging information.

        Returns:
            Dictionary with all error details
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "technical_details": self.technical_details,
            "recovery_suggestion": self.recovery_suggestion,
            "context": self.context,
        }


class TaskGenerationError(VibeMLError):
    """Raised when task generation fails."""

    pass


class SkyPilotError(VibeMLError):
    """Raised when SkyPilot operations fail."""

    pass


class NebiusError(VibeMLError):
    """Raised for Nebius-specific cloud provider errors."""

    pass


class ResourceError(VibeMLError):
    """Raised when resource allocation or validation fails."""

    pass


class ValidationError(VibeMLError):
    """Raised when input validation fails."""

    pass


class ConfigurationError(VibeMLError):
    """Raised when configuration settings are invalid."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        expected_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ConfigurationError.

        Args:
            message: Error message
            config_key: Configuration key that caused the error
            expected_type: Expected type for the configuration value
            **kwargs: Additional arguments passed to VibeMLError
        """
        context = kwargs.pop("context", {})
        if config_key:
            context["config_key"] = config_key
        if expected_type:
            context["expected_type"] = expected_type
        super().__init__(message, context=context, **kwargs)


class BudgetExceededError(VibeMLError):
    """Raised when cost limits are exceeded."""

    def __init__(
        self,
        message: str,
        estimated_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize BudgetExceededError.

        Args:
            message: Error message
            estimated_cost: Estimated cost that exceeded the limit
            max_cost: Maximum allowed cost
            **kwargs: Additional arguments passed to VibeMLError
        """
        context = kwargs.pop("context", {})
        if estimated_cost is not None:
            context["estimated_cost"] = estimated_cost
        if max_cost is not None:
            context["max_cost"] = max_cost

        recovery_suggestion = kwargs.pop(
            "recovery_suggestion",
            "Consider increasing max_cost, using spot instances, or choosing a smaller GPU type"
        )
        super().__init__(message, recovery_suggestion=recovery_suggestion, context=context, **kwargs)


class CloudProviderError(VibeMLError):
    """Raised for cloud-specific failures."""

    def __init__(
        self,
        message: str,
        cloud_provider: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CloudProviderError.

        Args:
            message: Error message
            cloud_provider: Cloud provider name (nebius, aws, gcp, azure)
            region: Cloud region that failed
            **kwargs: Additional arguments passed to VibeMLError
        """
        context = kwargs.pop("context", {})
        if cloud_provider:
            context["cloud_provider"] = cloud_provider
        if region:
            context["region"] = region
        super().__init__(message, context=context, **kwargs)


class ModelNotFoundError(VibeMLError):
    """Raised when model ID is invalid or not found."""

    def __init__(
        self,
        message: str,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ModelNotFoundError.

        Args:
            message: Error message
            model_id: Model ID that was not found
            **kwargs: Additional arguments passed to VibeMLError
        """
        context = kwargs.pop("context", {})
        if model_id:
            context["model_id"] = model_id

        recovery_suggestion = kwargs.pop(
            "recovery_suggestion",
            "Check the model ID on HuggingFace Hub or use a different model"
        )
        super().__init__(message, recovery_suggestion=recovery_suggestion, context=context, **kwargs)


class DatasetError(VibeMLError):
    """Raised when dataset access or validation fails."""

    def __init__(
        self,
        message: str,
        dataset_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize DatasetError.

        Args:
            message: Error message
            dataset_id: Dataset ID that caused the error
            **kwargs: Additional arguments passed to VibeMLError
        """
        context = kwargs.pop("context", {})
        if dataset_id:
            context["dataset_id"] = dataset_id

        recovery_suggestion = kwargs.pop(
            "recovery_suggestion",
            "Verify dataset exists on HuggingFace or check access permissions"
        )
        super().__init__(message, recovery_suggestion=recovery_suggestion, context=context, **kwargs)


# Error Recovery Strategies

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        # Final attempt failed
                        raise SkyPilotError(
                            f"Operation failed after {max_retries} retries",
                            technical_details=str(e),
                            recovery_suggestion="Check cloud provider status or try a different region",
                        ) from e

            return None  # Should never reach here
        return wrapper
    return decorator


class ErrorTranslator:
    """Translates technical errors into user-friendly messages."""

    # Common error patterns and their translations
    ERROR_PATTERNS: Dict[str, Dict[str, str]] = {
        "No resource available": {
            "message": "No GPU resources currently available in the selected region",
            "suggestion": "Try a different GPU type, region, or wait a few minutes and retry",
        },
        "Quota exceeded": {
            "message": "Cloud provider quota limit reached",
            "suggestion": "Contact cloud provider support to increase quota limits",
        },
        "Permission denied": {
            "message": "Insufficient permissions to access cloud resources",
            "suggestion": "Check your cloud credentials and IAM permissions",
        },
        "Authentication failed": {
            "message": "Cloud authentication failed",
            "suggestion": "Verify your API keys and cloud credentials are correctly configured",
        },
        "Network timeout": {
            "message": "Network connection timeout",
            "suggestion": "Check your internet connection and try again",
        },
        "Out of memory": {
            "message": "GPU ran out of memory during training",
            "suggestion": "Reduce batch size, use gradient checkpointing, or choose a larger GPU",
        },
        "CUDA error": {
            "message": "GPU computation error occurred",
            "suggestion": "This may be a transient error - try restarting the job",
        },
    }

    @classmethod
    def translate(cls, error: Exception) -> VibeMLError:
        """Translate a technical error into a user-friendly VibeMLError.

        Args:
            error: The original exception

        Returns:
            Translated VibeMLError with user-friendly message
        """
        error_str = str(error)

        # Check for known error patterns
        for pattern, translation in cls.ERROR_PATTERNS.items():
            if pattern.lower() in error_str.lower():
                return VibeMLError(
                    message=translation["message"],
                    technical_details=error_str,
                    recovery_suggestion=translation["suggestion"],
                    context={"original_error_type": type(error).__name__},
                )

        # Default translation for unknown errors
        return VibeMLError(
            message="An unexpected error occurred during training",
            technical_details=error_str,
            recovery_suggestion="Check logs for details or contact support",
            context={"original_error_type": type(error).__name__},
        )

    @classmethod
    def translate_skypilot_error(cls, error: Exception) -> SkyPilotError:
        """Translate SkyPilot-specific errors.

        Args:
            error: The original SkyPilot exception

        Returns:
            Translated SkyPilotError
        """
        error_str = str(error)

        # SkyPilot-specific patterns
        if "cluster not found" in error_str.lower():
            return SkyPilotError(
                message="Training cluster not found",
                technical_details=error_str,
                recovery_suggestion="The cluster may have been terminated. Check cluster status.",
            )
        elif "launch failed" in error_str.lower():
            return SkyPilotError(
                message="Failed to launch training cluster",
                technical_details=error_str,
                recovery_suggestion="Check cloud provider status and resource availability",
            )

        # Use general translation
        base_error = cls.translate(error)
        return SkyPilotError(
            message=base_error.message,
            technical_details=base_error.technical_details,
            recovery_suggestion=base_error.recovery_suggestion,
            context=base_error.context,
        )


def get_region_fallbacks(cloud: str, current_region: str) -> List[str]:
    """Get fallback regions for a cloud provider.

    Args:
        cloud: Cloud provider name
        current_region: Current region that failed

    Returns:
        List of alternative regions to try
    """
    REGION_FALLBACKS = {
        "nebius": ["eu-north1", "eu-west1"],
        "aws": ["us-east-1", "us-west-2", "eu-west-1"],
        "gcp": ["us-central1", "us-east1", "europe-west1"],
        "azure": ["eastus", "westus2", "westeurope"],
    }

    fallbacks = REGION_FALLBACKS.get(cloud, [])
    # Remove current region from fallbacks
    return [r for r in fallbacks if r != current_region]
