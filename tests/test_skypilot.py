"""Tests for the SkyPilot adapter, against mocked request-based SDK responses."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import sky
from sky.schemas.api.responses import StatusResponse

from vibeml.cloud import skypilot
from vibeml.cloud.skypilot import ClusterInfo, to_job_status
from vibeml.exceptions import SkyPilotError
from vibeml.models import JobStatus

LAUNCHED_AT = 1_700_000_000


def status_response(name: str, status: sky.ClusterStatus) -> StatusResponse:
    """Build a real SkyPilot status record with only the fields vibeml reads varying."""
    return StatusResponse(
        name=name,
        status=status,
        launched_at=LAUNCHED_AT,
        autostop=-1,
        to_down=False,
        cluster_hash="hash",
        cluster_ever_up=True,
        user_hash="user",
        user_name="user",
        workspace="default",
        is_managed=False,
        nodes=1,
    )


@pytest.fixture
def sdk() -> Any:
    """Patch the SkyPilot SDK as seen by the adapter.

    Every SDK call returns a request ID; ``sky.get`` resolves it.
    """
    with patch.object(skypilot, "sky") as mock_sky:
        mock_sky.ClusterStatus = sky.ClusterStatus
        mock_sky.StatusRefreshMode = sky.StatusRefreshMode
        mock_sky.status.return_value = "req-status"
        mock_sky.launch.return_value = "req-launch"
        mock_sky.down.return_value = "req-down"
        yield mock_sky


class TestToJobStatus:
    """SkyPilot cluster statuses map onto vibeml job statuses."""

    @pytest.mark.parametrize(
        ("cluster_status", "job_status"),
        [
            (sky.ClusterStatus.INIT, JobStatus.PENDING),
            (sky.ClusterStatus.PENDING, JobStatus.PENDING),
            (sky.ClusterStatus.UP, JobStatus.RUNNING),
            (sky.ClusterStatus.AUTOSTOPPING, JobStatus.RUNNING),
            (sky.ClusterStatus.STOPPED, JobStatus.TERMINATED),
        ],
    )
    def test_every_cluster_status_is_mapped(
        self, cluster_status: sky.ClusterStatus, job_status: JobStatus
    ) -> None:
        """Each ClusterStatus enum member has an explicit mapping."""
        assert to_job_status(cluster_status) == job_status

    def test_all_members_covered(self) -> None:
        """A new SkyPilot status must be mapped deliberately, not fall through."""
        for member in sky.ClusterStatus:
            to_job_status(member)


class TestGetCluster:
    """get_cluster resolves a status request for one cluster."""

    async def test_returns_cluster_info(self, sdk: MagicMock) -> None:
        """A found cluster is translated into vibeml's ClusterInfo."""
        sdk.get.return_value = [status_response("vibeml-a", sky.ClusterStatus.UP)]

        info = await skypilot.get_cluster("vibeml-a")

        sdk.status.assert_called_once_with(
            cluster_names=["vibeml-a"], refresh=sky.StatusRefreshMode.AUTO
        )
        sdk.get.assert_called_once_with("req-status")
        assert info == ClusterInfo(
            name="vibeml-a",
            sky_status="UP",
            status=JobStatus.RUNNING,
            launched_at=datetime.fromtimestamp(LAUNCHED_AT, tz=timezone.utc),
        )

    async def test_missing_cluster_returns_none(self, sdk: MagicMock) -> None:
        """An empty status result means the cluster does not exist."""
        sdk.get.return_value = []

        assert await skypilot.get_cluster("vibeml-missing") is None

    async def test_sdk_failure_raises(self, sdk: MagicMock) -> None:
        """SDK errors surface as SkyPilotError, not as 'not found'."""
        sdk.get.side_effect = RuntimeError("API server unreachable")

        with pytest.raises(SkyPilotError, match="API server unreachable"):
            await skypilot.get_cluster("vibeml-a")


class TestListClusters:
    """list_clusters returns clusters whose name has the given prefix."""

    async def test_filters_by_prefix(self, sdk: MagicMock) -> None:
        """Clusters not owned by vibeml are left out."""
        sdk.get.return_value = [
            status_response("vibeml-a", sky.ClusterStatus.UP),
            status_response("someone-else", sky.ClusterStatus.UP),
            status_response("vibeml-b", sky.ClusterStatus.STOPPED),
        ]

        clusters = await skypilot.list_clusters(prefix="vibeml-")

        sdk.status.assert_called_once_with(refresh=sky.StatusRefreshMode.AUTO)
        assert [c.name for c in clusters] == ["vibeml-a", "vibeml-b"]
        assert [c.status for c in clusters] == [JobStatus.RUNNING, JobStatus.TERMINATED]


class TestLaunch:
    """launch submits the task and returns without waiting for provisioning."""

    async def test_returns_request_id_without_blocking(self, sdk: MagicMock) -> None:
        """Provisioning takes minutes, so the request ID is returned immediately."""
        task = sky.Task(run="echo hi")

        request_id = await skypilot.launch(task, cluster_name="vibeml-a")

        sdk.launch.assert_called_once_with(task, cluster_name="vibeml-a")
        sdk.get.assert_not_called()
        assert request_id == "req-launch"

    async def test_submit_failure_raises(self, sdk: MagicMock) -> None:
        """A rejected submission surfaces as SkyPilotError."""
        sdk.launch.side_effect = RuntimeError("no credentials")

        with pytest.raises(SkyPilotError, match="no credentials"):
            await skypilot.launch(sky.Task(run="echo hi"), cluster_name="vibeml-a")


class TestDownAndLogs:
    """down waits for teardown; download_logs returns SkyPilot's local paths."""

    async def test_down_waits_for_request(self, sdk: MagicMock) -> None:
        """Teardown is awaited so the caller knows the cluster is gone."""
        await skypilot.down("vibeml-a")

        sdk.down.assert_called_once_with("vibeml-a")
        sdk.get.assert_called_once_with("req-down")

    async def test_download_logs_returns_paths(self, sdk: MagicMock) -> None:
        """Logs for all jobs are downloaded; SkyPilot chooses the local paths."""
        sdk.download_logs.return_value = {"1": "/home/u/sky_logs/vibeml-a/1"}

        paths = await skypilot.download_logs("vibeml-a")

        sdk.download_logs.assert_called_once_with("vibeml-a", job_ids=None)
        assert paths == {"1": "/home/u/sky_logs/vibeml-a/1"}
