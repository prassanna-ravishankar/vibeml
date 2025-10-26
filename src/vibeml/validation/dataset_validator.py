"""Dataset validation using HuggingFace Hub."""

from typing import Dict, Any

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

from ..exceptions import DatasetError, ValidationError


class DatasetValidator:
    """Validates HuggingFace datasets."""

    def __init__(self) -> None:
        """Initialize DatasetValidator."""
        self.api = HfApi()

    def validate_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Validate that a dataset exists and is accessible.

        Args:
            dataset_id: HuggingFace dataset ID (e.g., "tatsu-lab/alpaca")

        Returns:
            Dictionary with dataset information

        Raises:
            DatasetError: If dataset doesn't exist or is not accessible
        """
        try:
            # Get dataset info from HuggingFace Hub
            dataset_info = self.api.dataset_info(dataset_id)

            return {
                "dataset_id": dataset_id,
                "accessible": True,
                "gated": getattr(dataset_info, "gated", False),
                "tags": getattr(dataset_info, "tags", []),
                "card_data": getattr(dataset_info, "card_data", {}),
            }

        except RepositoryNotFoundError:
            raise DatasetError(
                f"Dataset '{dataset_id}' not found on HuggingFace Hub",
                dataset_id=dataset_id,
            )
        except GatedRepoError:
            raise DatasetError(
                f"Dataset '{dataset_id}' is gated and requires access approval",
                dataset_id=dataset_id,
                recovery_suggestion="Request access on HuggingFace Hub and configure your HF token",
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to validate dataset '{dataset_id}'",
                technical_details=str(e),
            )

    def estimate_dataset_size(self, dataset_id: str) -> float:
        """Estimate dataset size in GB.

        Args:
            dataset_id: HuggingFace dataset ID

        Returns:
            Estimated size in GB (default 10GB if unknown)
        """
        try:
            dataset_info = self.api.dataset_info(dataset_id)
            # Try to get size from dataset info
            # This is approximate as HF doesn't always provide exact sizes
            return 10.0  # Default conservative estimate
        except Exception:
            return 10.0  # Default
