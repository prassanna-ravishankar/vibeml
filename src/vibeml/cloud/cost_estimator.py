"""Cost estimation for cloud training jobs."""

import math
from dataclasses import dataclass
from typing import ClassVar

import sky

from ..models import CostEstimate


@dataclass
class GPUPricing:
    """GPU pricing information."""

    hourly_rate: float
    spot_discount: float = 0.7  # 30% of on-demand price


@dataclass(frozen=True)
class TaskCost:
    """The priced resources of a task and its cost estimate."""

    gpu_type: str
    gpu_count: int
    use_spot: bool
    estimate: CostEstimate


class CostEstimator:
    """Estimates costs for training jobs."""

    # Nebius GPU pricing ($/hour, on-demand)
    NEBIUS_PRICING: ClassVar[dict[str, GPUPricing]] = {
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

    def estimate_task(self, task: sky.Task, hours: float) -> TaskCost:
        """Price exactly the resources a task requests.

        Args:
            task: Task with a single GPU resource request.
            hours: Estimated runtime in hours.

        Returns:
            The priced resources and the cost estimate.
        """
        (resources,) = task.resources
        ((gpu_type, per_node),) = resources.accelerators.items()
        gpu_count = int(per_node) * task.num_nodes
        return TaskCost(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            use_spot=resources.use_spot,
            estimate=self.estimate_cost(
                gpu_type=gpu_type,
                estimated_hours=hours,
                gpu_count=gpu_count,
                use_spot=resources.use_spot,
            ),
        )

    def affordable_minutes(self, task: sky.Task, max_cost: float) -> int:
        """Minutes of runtime a budget buys for a task, rounded down.

        Args:
            task: Task with a single GPU resource request.
            max_cost: Budget in USD.

        Returns:
            Whole minutes the budget covers at the task's hourly rate.
        """
        hourly_rate = self.estimate_task(task, hours=1).estimate.hourly_rate
        return math.floor(max_cost / hourly_rate * 60)

    def _get_pricing(self, gpu_type: str, cloud: str) -> GPUPricing:
        """Get pricing for GPU type and cloud.

        Args:
            gpu_type: GPU type
            cloud: Cloud provider

        Returns:
            GPUPricing object
        """
        if cloud == "nebius":
            return self.NEBIUS_PRICING.get(gpu_type, GPUPricing(hourly_rate=2.0))

        # Default pricing for other clouds
        return GPUPricing(hourly_rate=2.0)
