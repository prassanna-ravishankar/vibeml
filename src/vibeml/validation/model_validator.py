"""Model validation using HuggingFace Hub."""

from typing import Optional, Dict, Any
import re

from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

from ..exceptions import ModelNotFoundError, ValidationError


class ModelValidator:
    """Validates HuggingFace models and calculates requirements."""

    # Approximate memory requirements (GB) by model parameter size
    MODEL_SIZE_MEMORY_MAP = {
        "1b": 4,
        "3b": 8,
        "7b": 16,
        "13b": 32,
        "20b": 48,
        "30b": 64,
        "70b": 160,
        "120b": 256,
    }

    def __init__(self) -> None:
        """Initialize ModelValidator."""
        self.api = HfApi()

    def validate_model(self, model_id: str) -> Dict[str, Any]:
        """Validate that a model exists and is accessible.

        Args:
            model_id: HuggingFace model ID (e.g., "mistralai/Mistral-7B-v0.1")

        Returns:
            Dictionary with model information

        Raises:
            ModelNotFoundError: If model doesn't exist or is not accessible
        """
        # Handle GPT-OSS size format (20b, 120b)
        if model_id in ["20b", "120b"]:
            return {
                "model_id": model_id,
                "model_type": "gpt-oss",
                "size": model_id,
                "accessible": True,
                "gated": False,
            }

        try:
            # Get model info from HuggingFace Hub
            model_info = self.api.model_info(model_id)

            return {
                "model_id": model_id,
                "model_type": getattr(model_info, "pipeline_tag", "unknown"),
                "size": self._estimate_model_size(model_id),
                "accessible": True,
                "gated": getattr(model_info, "gated", False),
                "tags": getattr(model_info, "tags", []),
            }

        except RepositoryNotFoundError:
            raise ModelNotFoundError(
                f"Model '{model_id}' not found on HuggingFace Hub",
                model_id=model_id,
            )
        except GatedRepoError:
            raise ModelNotFoundError(
                f"Model '{model_id}' is gated and requires access approval",
                model_id=model_id,
                recovery_suggestion="Request access on HuggingFace Hub and configure your HF token",
            )
        except Exception as e:
            raise ValidationError(
                f"Failed to validate model '{model_id}'",
                technical_details=str(e),
            )

    def _estimate_model_size(self, model_id: str) -> Optional[str]:
        """Estimate model size from model ID.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Estimated size string (e.g., "7b") or None
        """
        # Common patterns: "7B", "7b", "7-B", "7_B"
        patterns = [
            r"(\d+\.?\d*)B",  # 7B, 13B
            r"(\d+\.?\d*)b",  # 7b, 13b
            r"(\d+\.?\d*)-B",  # 7-B
            r"(\d+\.?\d*)_B",  # 7_B
        ]

        for pattern in patterns:
            match = re.search(pattern, model_id)
            if match:
                size = match.group(1)
                return f"{int(float(size))}b"

        return None

    def calculate_gpu_memory_required(
        self,
        model_id: str,
        batch_size: int = 1,
        use_quantization: bool = False,
    ) -> float:
        """Calculate GPU memory requirements for a model.

        Args:
            model_id: HuggingFace model ID or size
            batch_size: Training batch size
            use_quantization: Whether 4-bit quantization is used

        Returns:
            Estimated GPU memory in GB
        """
        # Get model size
        if model_id in self.MODEL_SIZE_MEMORY_MAP:
            size = model_id
        else:
            size = self._estimate_model_size(model_id)

        # Get base memory requirement
        base_memory = self.MODEL_SIZE_MEMORY_MAP.get(size, 32)  # Default 32GB

        # Apply quantization reduction (4-bit = ~4x reduction)
        if use_quantization:
            base_memory = base_memory / 4

        # Add overhead for batch size (approximate)
        memory_per_batch = base_memory * 0.2  # 20% overhead per batch item
        total_memory = base_memory + (memory_per_batch * (batch_size - 1))

        return total_memory

    def recommend_gpu_type(
        self,
        model_id: str,
        batch_size: int = 1,
        use_quantization: bool = False,
    ) -> str:
        """Recommend GPU type based on model requirements.

        Args:
            model_id: HuggingFace model ID or size
            batch_size: Training batch size
            use_quantization: Whether 4-bit quantization is used

        Returns:
            Recommended GPU type
        """
        memory_required = self.calculate_gpu_memory_required(
            model_id, batch_size, use_quantization
        )

        # GPU memory capacities
        gpu_memory = {
            "RTX4090": 24,
            "L40S": 48,
            "A100": 80,
            "H100": 80,
        }

        # Find smallest GPU that fits
        for gpu, capacity in sorted(gpu_memory.items(), key=lambda x: x[1]):
            if memory_required <= capacity * 0.8:  # Leave 20% headroom
                return gpu

        # If nothing fits, recommend H100
        return "H100"
