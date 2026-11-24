"""Tests for the MCP server, exercised through an in-memory MCP client."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from fastmcp.exceptions import ValidationError as FastMCPValidationError
from pydantic import ValidationError as PydanticValidationError

from vibeml import server
from vibeml.cloud.skypilot import ClusterInfo
from vibeml.exceptions import SkyPilotError, TaskGenerationError
from vibeml.models import JobStatus

LAUNCHED_AT = datetime(2026, 11, 24, 9, 0, tzinfo=timezone.utc)


def cluster(name: str, status: JobStatus, sky_status: str) -> ClusterInfo:
    """Build a ClusterInfo as the adapter returns it."""
    return ClusterInfo(
        name=name, sky_status=sky_status, status=status, launched_at=LAUNCHED_AT
    )


async def call(tool: str, **arguments: Any) -> dict[str, Any]:
    """Call an MCP tool and return its structured result."""
    async with Client(server.mcp) as client:
        result = await client.call_tool(tool, arguments)
    assert isinstance(result.data, dict)
    return result.data


@pytest.fixture
def adapter() -> Any:
    """Patch the SkyPilot adapter as seen by the server."""
    with patch.object(server, "skypilot") as mock_adapter:
        for name in ("launch", "get_cluster", "list_clusters", "down", "download_logs"):
            setattr(mock_adapter, name, AsyncMock())
        yield mock_adapter


async def test_exposes_training_tools() -> None:
    """The server's socket is its tool list."""
    async with Client(server.mcp) as client:
        names = {tool.name for tool in await client.list_tools()}

    assert names == {
        "launch_training",
        "get_training_status",
        "stop_training",
        "list_available_workflows",
        "estimate_training_cost",
        "list_active_training_jobs",
    }


class TestLaunchTraining:
    """launch_training builds the task and submits it without blocking."""

    async def test_submits_task_and_returns_request_id(self, adapter: Any) -> None:
        """The response carries the cluster name and the launch request ID."""
        adapter.launch.return_value = MagicMock(request_id="req-123")

        result = await call("launch_training", workflow="unsloth", gpu_type="H100")

        (task,), kwargs = adapter.launch.call_args
        (resources,) = task.resources
        assert resources.accelerators == {"H100": 1}
        assert kwargs["cluster_name"].startswith("vibeml-unsloth-")
        assert result["status"] == "submitted"
        assert result["cluster_id"] == kwargs["cluster_name"]
        assert result["launch_request_id"] == "req-123"
        assert result["logs_command"] == "sky api logs req-123"

    async def test_invalid_request_is_rejected_before_launch(
        self, adapter: Any
    ) -> None:
        """Validation errors never reach SkyPilot."""
        result = await call("launch_training", workflow="not-a-workflow")

        adapter.launch.assert_not_called()
        assert result["status"] == "error"
        assert result["error"] == "validation_error"

    async def test_unknown_model_size_is_rejected(self, adapter: Any) -> None:
        """GPT-OSS sizes are validated instead of silently replaced."""
        result = await call("launch_training", workflow="gpt-oss-lora", model="70b")

        adapter.launch.assert_not_called()
        assert result["status"] == "error"

    async def test_task_generation_failure_is_reported(self, adapter: Any) -> None:
        """Task construction errors use the same error envelope."""
        with patch.object(
            server, "build_task", side_effect=TaskGenerationError("bad template")
        ):
            result = await call("launch_training")

        adapter.launch.assert_not_called()
        assert result == {
            "status": "error",
            "error": "task_generation_error",
            "message": "bad template",
        }

    async def test_sdk_failure_is_reported(self, adapter: Any) -> None:
        """A failed submission is returned as an error, not raised."""
        adapter.launch.side_effect = SkyPilotError("no credentials")

        result = await call("launch_training")

        assert result == {
            "status": "error",
            "error": "skypilot_error",
            "message": "no credentials",
        }


class TestCostGuard:
    """max_cost is enforced by the launcher, before and during the run."""

    async def test_over_budget_is_refused_before_launch(self, adapter: Any) -> None:
        """An estimate above max_cost never reaches SkyPilot."""
        result = await call("launch_training", max_cost=1.0, estimated_hours=10)

        adapter.launch.assert_not_called()
        assert result["status"] == "error"
        assert result["error"] == "over_budget"
        assert result["estimated_cost"] == pytest.approx(1.2 * 0.7 * 10)

    async def test_within_budget_launches_with_a_runtime_cap(
        self, adapter: Any
    ) -> None:
        """The cap becomes a teardown after the minutes the budget buys."""
        adapter.launch.return_value = MagicMock(request_id="req-1")

        result = await call("launch_training", max_cost=4.2, estimated_hours=1)

        assert adapter.launch.call_args.kwargs["max_minutes"] == 300
        assert result["cost_guard"] == {"max_cost": 4.2, "teardown_after_minutes": 300}

    async def test_sub_minute_budget_is_refused(self, adapter: Any) -> None:
        """A budget that buys less than a minute cannot be enforced, so it is refused."""
        result = await call("launch_training", max_cost=0.001, estimated_hours=0.001)

        adapter.launch.assert_not_called()
        assert result["error"] == "over_budget"

    async def test_infinite_budget_is_rejected(self, adapter: Any) -> None:
        """max_cost must be finite.

        JSON clients can send it (Python parses 1e999 as inf); the MCP client
        itself serializes inf to null, so validate through the tool directly.
        """
        tool = await server.mcp.get_tool("launch_training")
        assert tool is not None

        # FastMCP may surface argument validation as its own ValidationError.
        with pytest.raises(
            (PydanticValidationError, FastMCPValidationError), match="finite"
        ):
            await tool.run({"max_cost": float("inf")})

        adapter.launch.assert_not_called()

    async def test_without_max_cost_there_is_no_cap(self, adapter: Any) -> None:
        """No budget, no runtime cap."""
        adapter.launch.return_value = MagicMock(request_id="req-1")

        result = await call("launch_training")

        assert adapter.launch.call_args.kwargs["max_minutes"] is None
        assert "cost_guard" not in result


class TestLaunchOptions:
    """Bucket and failure-cleanup options reach the launch."""

    async def test_cloud_bucket_reaches_the_task(self, adapter: Any) -> None:
        """The bucket is mounted where checkpoints are written."""
        adapter.launch.return_value = MagicMock(request_id="req-1")

        await call("launch_training", cloud_bucket="ckpts")

        (task,), _ = adapter.launch.call_args
        assert task.storage_mounts["/outputs"].name == "ckpts"

    async def test_failed_launches_are_cleaned_up_by_default(
        self, adapter: Any
    ) -> None:
        """The launcher tears down failed launches unless asked to keep them."""
        adapter.launch.return_value = MagicMock(request_id="req-1")

        await call("launch_training")
        assert adapter.launch.call_args.kwargs["keep_on_failure"] is False

        await call("launch_training", keep_on_failure=True)
        assert adapter.launch.call_args.kwargs["keep_on_failure"] is True


class TestGetTrainingStatus:
    """get_training_status reports the cluster's state."""

    async def test_reports_status(self, adapter: Any) -> None:
        """Status comes from the adapter's ClusterInfo."""
        adapter.get_cluster.return_value = cluster("vibeml-a", JobStatus.RUNNING, "UP")

        result = await call("get_training_status", cluster_id="vibeml-a")

        assert result == {
            "cluster_id": "vibeml-a",
            "status": "running",
            "sky_status": "UP",
            "launched_at": LAUNCHED_AT.isoformat(),
        }

    async def test_missing_cluster(self, adapter: Any) -> None:
        """A cluster that is not (yet) known points at the launch request logs."""
        adapter.get_cluster.return_value = None

        result = await call("get_training_status", cluster_id="vibeml-a")

        assert result["status"] == "error"
        assert "sky api logs" in result["message"]


class TestStopTraining:
    """stop_training optionally downloads logs, then tears the cluster down."""

    async def test_downloads_logs_then_tears_down(self, adapter: Any) -> None:
        """Logs are fetched before teardown; paths are those SkyPilot reports."""
        adapter.get_cluster.return_value = cluster("vibeml-a", JobStatus.RUNNING, "UP")
        adapter.download_logs.return_value = {"1": "/logs/vibeml-a/1"}

        result = await call("stop_training", cluster_id="vibeml-a")

        adapter.download_logs.assert_awaited_once_with("vibeml-a")
        adapter.down.assert_awaited_once_with("vibeml-a")
        assert result == {
            "status": "stopped",
            "cluster_id": "vibeml-a",
            "log_paths": {"1": "/logs/vibeml-a/1"},
        }

    async def test_tears_down_even_if_logs_fail(self, adapter: Any) -> None:
        """Log download is best effort; a running cluster costs money."""
        adapter.get_cluster.return_value = cluster(
            "vibeml-a", JobStatus.TERMINATED, "STOPPED"
        )
        adapter.download_logs.side_effect = SkyPilotError("cluster is not up")

        result = await call("stop_training", cluster_id="vibeml-a")

        adapter.down.assert_awaited_once_with("vibeml-a")
        assert result["status"] == "stopped"
        assert result["log_paths"] == {}
        assert result["log_error"] == "cluster is not up"

    async def test_teardown_failure_is_reported(self, adapter: Any) -> None:
        """If teardown fails, the caller is told the cluster may still be running."""
        adapter.get_cluster.return_value = cluster("vibeml-a", JobStatus.RUNNING, "UP")
        adapter.down.side_effect = SkyPilotError("cloud API timeout")

        result = await call("stop_training", cluster_id="vibeml-a")

        assert result["status"] == "error"
        assert result["error"] == "skypilot_error"

    async def test_skips_logs_when_asked(self, adapter: Any) -> None:
        """download_results=False tears down without fetching logs."""
        adapter.get_cluster.return_value = cluster("vibeml-a", JobStatus.RUNNING, "UP")

        result = await call(
            "stop_training", cluster_id="vibeml-a", download_results=False
        )

        adapter.download_logs.assert_not_called()
        adapter.down.assert_awaited_once_with("vibeml-a")
        assert result["log_paths"] == {}

    async def test_missing_cluster_is_not_torn_down(self, adapter: Any) -> None:
        """Nothing is torn down for an unknown cluster."""
        adapter.get_cluster.return_value = None

        result = await call("stop_training", cluster_id="vibeml-a")

        adapter.down.assert_not_called()
        assert result["status"] == "error"


async def test_list_active_jobs_excludes_stopped(adapter: Any) -> None:
    """Only pending and running vibeml clusters are active."""
    adapter.list_clusters.return_value = [
        cluster("vibeml-a", JobStatus.RUNNING, "UP"),
        cluster("vibeml-b", JobStatus.TERMINATED, "STOPPED"),
        cluster("vibeml-c", JobStatus.PENDING, "INIT"),
    ]

    result = await call("list_active_training_jobs")

    adapter.list_clusters.assert_awaited_once_with(prefix="vibeml-")
    assert [job["cluster_id"] for job in result["active_jobs"]] == [
        "vibeml-a",
        "vibeml-c",
    ]
    assert result["count"] == 2


class TestEstimateTrainingCost:
    """Cost estimates price the resources the workflow would actually request."""

    async def test_unsloth_defaults_to_one_spot_l40s(self) -> None:
        """Unsloth runs on a single spot L40S unless told otherwise."""
        result = await call(
            "estimate_training_cost", workflow="unsloth", estimated_hours=10
        )

        assert result["gpu_type"] == "L40S"
        assert result["gpu_count"] == 1
        assert result["use_spot"] is True
        assert result["expected_cost"] == pytest.approx(1.2 * 0.7 * 10)

    async def test_full_120b_counts_every_node(self) -> None:
        """Full 120B training is 4 nodes of 8 H100s, on demand."""
        result = await call(
            "estimate_training_cost",
            workflow="gpt-oss-full",
            model="120b",
            estimated_hours=1,
        )

        assert result["gpu_type"] == "H100"
        assert result["gpu_count"] == 32
        assert result["use_spot"] is False
        assert result["expected_cost"] == pytest.approx(3.5 * 32)

    async def test_rejects_non_positive_hours(self) -> None:
        """Duration must be positive."""
        async with Client(server.mcp) as client:
            with pytest.raises(ToolError):
                await client.call_tool("estimate_training_cost", {"estimated_hours": 0})

    async def test_unknown_workflow(self) -> None:
        """Unknown workflows are reported, not priced."""
        result = await call("estimate_training_cost", workflow="nope")

        assert result["status"] == "error"


async def test_list_available_workflows_matches_registry() -> None:
    """Every registered workflow is described."""
    result = await call("list_available_workflows")

    assert set(result["workflows"]) == {"unsloth", "gpt-oss-lora", "gpt-oss-full"}
