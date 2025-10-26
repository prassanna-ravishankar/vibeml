"""Resource validation and optimization."""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from ..exceptions import ResourceError


@dataclass
class ResourceRequirements:
    """Resource requirements for a training job."""

    min_gpu_memory: float  # GB
    recommended_gpu_memory: float  # GB
    min_disk_size: int  # GB
    recommended_disk_size: int  # GB
    suitable_gpus: List[str]
    preferred_gpu: str
    multi_gpu_required: bool = False
    min_gpu_count: int = 1


class ResourceValidator:
    """Validates and optimizes resource configurations."""

    # GPU specifications
    GPU_SPECS = {
        "RTX4090": {
            "memory_gb": 24,
            "cost_per_hour": 0.8,
            "performance_tier": 2,
        },
        "L40S": {
            "memory_gb": 48,
            "cost_per_hour": 1.2,
            "performance_tier": 3,
        },
        "A100": {
            "memory_gb": 80,
            "cost_per_hour": 2.4,
            "performance_tier": 4,
        },
        "H100": {
            "memory_gb": 80,
            "cost_per_hour": 3.5,
            "performance_tier": 5,
        },
    }

    def calculate_requirements(
        self,
        model_size_gb: float,
        dataset_size_gb: float = 10.0,
        batch_size: int = 1,
        use_quantization: bool = False,
    ) -> ResourceRequirements:
        """Calculate resource requirements for a training job.

        Args:
            model_size_gb: Model size in GB
            dataset_size_gb: Dataset size in GB
            batch_size: Training batch size
            use_quantization: Whether to use quantization

        Returns:
            ResourceRequirements object
        """
        # Calculate GPU memory requirements
        if use_quantization:
            model_size_gb = model_size_gb / 4

        min_memory = model_size_gb * 1.2  # 20% overhead
        recommended_memory = model_size_gb * 1.5 + (batch_size * 2)

        # Calculate disk requirements
        min_disk = int(model_size_gb + dataset_size_gb + 10)  # 10GB buffer
        recommended_disk = int((model_size_gb + dataset_size_gb) * 2 + 20)

        # Find suitable GPUs
        suitable_gpus = []
        for gpu_name, specs in self.GPU_SPECS.items():
            if specs["memory_gb"] >= min_memory:
                suitable_gpus.append(gpu_name)

        if not suitable_gpus:
            raise ResourceError(
                f"No GPU has enough memory ({min_memory:.1f}GB required)",
                recovery_suggestion="Use quantization or reduce batch size",
            )

        # Prefer cost-effective GPU that meets requirements
        preferred_gpu = self._select_optimal_gpu(suitable_gpus, recommended_memory)

        return ResourceRequirements(
            min_gpu_memory=min_memory,
            recommended_gpu_memory=recommended_memory,
            min_disk_size=min_disk,
            recommended_disk_size=recommended_disk,
            suitable_gpus=suitable_gpus,
            preferred_gpu=preferred_gpu,
        )

    def _select_optimal_gpu(
        self,
        suitable_gpus: List[str],
        memory_required: float,
    ) -> str:
        """Select the most cost-effective GPU that meets requirements.

        Args:
            suitable_gpus: List of GPUs that meet memory requirements
            memory_required: Required memory in GB

        Returns:
            Optimal GPU name
        """
        best_gpu = None
        best_cost_performance = float("inf")

        for gpu in suitable_gpus:
            specs = self.GPU_SPECS[gpu]
            memory_gb = specs["memory_gb"]

            # Skip if not enough headroom (need 20% margin)
            if memory_gb < memory_required * 1.2:
                continue

            # Calculate cost-performance ratio
            cost_perf = specs["cost_per_hour"] / specs["performance_tier"]

            if cost_perf < best_cost_performance:
                best_cost_performance = cost_perf
                best_gpu = gpu

        return best_gpu or suitable_gpus[0]

    def validate_gpu_for_model(
        self,
        gpu_type: str,
        memory_required: float,
    ) -> bool:
        """Validate that a GPU type is suitable for model requirements.

        Args:
            gpu_type: GPU type name
            memory_required: Required memory in GB

        Returns:
            True if GPU is suitable

        Raises:
            ResourceError: If GPU is not suitable
        """
        if gpu_type not in self.GPU_SPECS:
            raise ResourceError(
                f"Unknown GPU type: {gpu_type}",
                recovery_suggestion=f"Choose from: {', '.join(self.GPU_SPECS.keys())}",
            )

        gpu_memory = self.GPU_SPECS[gpu_type]["memory_gb"]

        if gpu_memory < memory_required:
            raise ResourceError(
                f"{gpu_type} has {gpu_memory}GB memory, but {memory_required:.1f}GB required",
                recovery_suggestion=f"Use a larger GPU or enable quantization",
            )

        # Check for adequate headroom
        if gpu_memory < memory_required * 1.2:
            raise ResourceError(
                f"{gpu_type} may not have enough memory headroom",
                recovery_suggestion="Consider using a GPU with more memory",
            )

        return True

    def optimize_spot_vs_ondemand(
        self,
        estimated_duration_hours: float,
        max_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Recommend spot vs on-demand based on cost and duration.

        Args:
            estimated_duration_hours: Estimated job duration
            max_cost: Maximum cost budget

        Returns:
            Dictionary with recommendation
        """
        # Spot instances are ~70% cheaper but can be preempted
        spot_discount = 0.7

        # For short jobs (<2h), spot risk is lower
        # For long jobs, on-demand is safer
        if estimated_duration_hours < 2:
            recommendation = "spot"
            reason = "Short job duration reduces preemption risk"
        elif estimated_duration_hours < 8:
            recommendation = "spot"
            reason = "Good cost savings with moderate preemption risk"
        else:
            recommendation = "on-demand"
            reason = "Long job - avoid potential restart overhead from preemption"

        return {
            "recommendation": recommendation,
            "reason": reason,
            "spot_discount": spot_discount,
            "estimated_savings_pct": (1 - spot_discount) * 100,
        }

    def calculate_disk_size(
        self,
        model_size_gb: float,
        dataset_size_gb: float,
        checkpoint_storage: bool = True,
    ) -> int:
        """Calculate optimal disk size.

        Args:
            model_size_gb: Model size in GB
            dataset_size_gb: Dataset size in GB
            checkpoint_storage: Whether to store checkpoints

        Returns:
            Recommended disk size in GB
        """
        base_size = model_size_gb + dataset_size_gb

        # Add space for checkpoints (2x model size)
        if checkpoint_storage:
            base_size += model_size_gb * 2

        # Add system overhead and buffer
        total_size = int(base_size * 1.5 + 20)

        # Round up to nearest 10GB
        return ((total_size + 9) // 10) * 10
