"""Cost estimation for cloud training jobs."""

from typing import Dict, Optional
from dataclasses import dataclass

from ..models import CostEstimate


@dataclass
class GPUPricing:
    """GPU pricing information."""

    hourly_rate: float
    spot_discount: float = 0.7  # 30% of on-demand price


class CostEstimator:
    """Estimates costs for training jobs."""

    # Nebius GPU pricing ($/hour, on-demand)
    NEBIUS_PRICING = {
        "RTX4090": GPUPricing(hourly_rate=0.8),
        "L40S": GPUPricing(hourly_rate=1.2),
        "A100": GPUPricing(hourly_rate=2.4),
        "H100": GPUPricing(hourly_rate=3.5),
    }

    def estimate_cost(
        self,
        gpu_type: str,
        estimated_hours: float,
        gpu_count: int = 1,
        use_spot: bool = True,
        cloud: str = "nebius",
    ) -> CostEstimate:
        """Estimate cost for a training job.

        Args:
            gpu_type: GPU type
            estimated_hours: Estimated duration in hours
            gpu_count: Number of GPUs
            use_spot: Whether to use spot instances
            cloud: Cloud provider

        Returns:
            CostEstimate object
        """
        # Get pricing
        pricing = self._get_pricing(gpu_type, cloud)

        # Calculate base hourly rate
        hourly_rate = pricing.hourly_rate * gpu_count

        # Apply spot discount if applicable
        if use_spot:
            hourly_rate *= pricing.spot_discount

        # Calculate costs with variance
        min_hours = estimated_hours * 0.8
        max_hours = estimated_hours * 1.3
        expected_hours = estimated_hours

        min_cost = hourly_rate * min_hours
        max_cost = hourly_rate * max_hours
        expected_cost = hourly_rate * expected_hours

        return CostEstimate(
            hourly_rate=hourly_rate,
            estimated_duration_hours=estimated_hours,
            min_cost=min_cost,
            max_cost=max_cost,
            expected_cost=expected_cost,
        )

    def estimate_duration(
        self,
        model_size: str,
        dataset_rows: int = 10000,
        max_steps: int = 1000,
    ) -> float:
        """Estimate training duration in hours.

        Args:
            model_size: Model size (e.g., "7b")
            dataset_rows: Number of dataset rows
            max_steps: Maximum training steps

        Returns:
            Estimated duration in hours
        """
        # Base duration estimates (hours)
        base_durations = {
            "1b": 1.0,
            "3b": 2.0,
            "7b": 4.0,
            "13b": 6.0,
            "20b": 10.0,
            "70b": 24.0,
            "120b": 48.0,
        }

        base_duration = base_durations.get(model_size, 4.0)

        # Adjust for dataset size and steps
        steps_factor = max_steps / 1000  # Baseline 1000 steps
        data_factor = dataset_rows / 10000  # Baseline 10k rows

        estimated = base_duration * steps_factor * (0.5 + 0.5 * data_factor)

        return max(0.5, estimated)  # Minimum 30 minutes

    def _get_pricing(self, gpu_type: str, cloud: str) -> GPUPricing:
        """Get pricing for GPU type and cloud.

        Args:
            gpu_type: GPU type
            cloud: Cloud provider

        Returns:
            GPUPricing object
        """
        if cloud == "nebius":
            return self.NEBIUS_PRICING.get(
                gpu_type, GPUPricing(hourly_rate=2.0)
            )

        # Default pricing for other clouds
        return GPUPricing(hourly_rate=2.0)
