"""Tests for cloud integration."""

import pytest

from vibeml.cloud import CostEstimator
from vibeml.models import CostEstimate
from vibeml.tasks import create_gpt_oss_full_task, create_unsloth_task


class TestCostEstimator:
    """Tests for CostEstimator."""

    def test_estimate_cost_basic(self) -> None:
        """Test basic cost estimation."""
        estimator = CostEstimator()

        cost = estimator.estimate_cost(
            gpu_type="L40S",
            estimated_hours=4.0,
        )

        assert isinstance(cost, CostEstimate)
        assert cost.hourly_rate > 0
        assert cost.expected_cost > 0
        assert cost.min_cost < cost.expected_cost < cost.max_cost

    def test_estimate_cost_spot_cheaper(self) -> None:
        """Test that spot instances are cheaper."""
        estimator = CostEstimator()

        on_demand = estimator.estimate_cost(
            gpu_type="L40S",
            estimated_hours=4.0,
            use_spot=False,
        )
        spot = estimator.estimate_cost(
            gpu_type="L40S",
            estimated_hours=4.0,
            use_spot=True,
        )

        assert spot.expected_cost < on_demand.expected_cost

    def test_estimate_cost_multi_gpu(self) -> None:
        """Test cost estimation with multiple GPUs."""
        estimator = CostEstimator()

        single = estimator.estimate_cost(
            gpu_type="H100",
            estimated_hours=4.0,
            gpu_count=1,
        )
        multi = estimator.estimate_cost(
            gpu_type="H100",
            estimated_hours=4.0,
            gpu_count=8,
        )

        assert multi.expected_cost == single.expected_cost * 8

    def test_estimate_duration_small_model(self) -> None:
        """Test duration estimation for small model."""
        estimator = CostEstimator()

        duration = estimator.estimate_duration(
            model_size="7b",
            max_steps=1000,
        )

        assert duration > 0
        assert duration < 24  # Should be reasonable

    def test_estimate_duration_large_model(self) -> None:
        """Test duration estimation for large model."""
        estimator = CostEstimator()

        small = estimator.estimate_duration(model_size="7b")
        large = estimator.estimate_duration(model_size="70b")

        assert large > small

    def test_estimate_duration_more_steps(self) -> None:
        """Test that more steps increases duration."""
        estimator = CostEstimator()

        short = estimator.estimate_duration(model_size="7b", max_steps=500)
        long = estimator.estimate_duration(model_size="7b", max_steps=2000)

        assert long > short


class TestTaskPricing:
    """Pricing a task uses exactly the resources it requests."""

    def test_estimate_task_counts_every_gpu_on_every_node(self) -> None:
        """Full 120B training is 4 nodes of 8 on-demand H100s."""
        cost = CostEstimator().estimate_task(create_gpt_oss_full_task("120b"), hours=1)

        assert (cost.gpu_type, cost.gpu_count, cost.use_spot) == ("H100", 32, False)
        assert cost.estimate.expected_cost == pytest.approx(3.5 * 32)

    def test_affordable_minutes(self) -> None:
        """A budget buys budget / hourly rate of runtime, rounded down."""
        task = create_unsloth_task(model="org/model", dataset="d")  # 1 spot L40S

        assert CostEstimator().affordable_minutes(task, max_cost=4.2) == 300

    def test_zero_budget_buys_nothing(self) -> None:
        """A zero budget affords no runtime."""
        task = create_unsloth_task(model="org/model", dataset="d")

        assert CostEstimator().affordable_minutes(task, max_cost=0) == 0
