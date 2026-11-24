"""Tests for the SkyPilot adapter, against mocked request-based SDK responses."""

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import sky
from sky.schemas.api.responses import StatusResponse
from sky.server import common as server_common
from sky.skylet.autostop_lib import AutostopWaitFor

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


@pytest.fixture(autouse=True)
def no_observe_delay() -> Any:
    """Re-observe launch requests immediately in tests."""
    with patch.object(skypilot, "_OBSERVE_RETRY_SECONDS", 0):
        yield


@pytest.fixture
def sdk() -> Any:
    """Patch the SkyPilot SDK as seen by the adapter.

    Every SDK call returns a request ID; ``sky.get`` resolves it.
    """
    with (
        patch.object(skypilot, "sky", autospec=True) as mock_sky,
        patch.object(
            server_common, "check_server_healthy_or_start_fn", autospec=True
        ) as start_server,
    ):
        mock_sky.start_server = start_server
        mock_sky.ClusterStatus = sky.ClusterStatus
        mock_sky.StatusRefreshMode = sky.StatusRefreshMode
        mock_sky.status.return_value = "req-status"
        mock_sky.launch.return_value = "req-launch"
        mock_sky.down.return_value = "req-down"
        mock_sky.api_info.return_value.api_version = "13"
        mock_sky.api_status.return_value = [
            MagicMock(request_id="req-launch", status="FAILED")
        ]
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

    async def test_missing_launch_time(self, sdk: MagicMock) -> None:
        """A cluster that never launched has no launch time."""
        record = status_response("vibeml-a", sky.ClusterStatus.INIT)
        record.launched_at = 0
        sdk.get.return_value = [record]

        info = await skypilot.get_cluster("vibeml-a")

        assert info is not None
        assert info.launched_at is None

    async def test_unexpected_record_shape_raises(self, sdk: MagicMock) -> None:
        """A record vibeml cannot translate is an SDK error, not a crash."""
        sdk.get.return_value = [object()]

        with pytest.raises(SkyPilotError):
            await skypilot.get_cluster("vibeml-a")

    async def test_blocking_call_does_not_block_event_loop(
        self, sdk: MagicMock
    ) -> None:
        """SDK calls block on the API server, so they run off the event loop."""
        release = threading.Event()

        def slow_get(request_id: str) -> list[StatusResponse]:
            release.wait(timeout=5)
            return []

        sdk.get.side_effect = slow_get
        lookup = asyncio.create_task(skypilot.get_cluster("vibeml-a"))
        await asyncio.sleep(0.05)  # the loop keeps running while the SDK blocks
        assert not lookup.done()
        release.set()

        assert await lookup is None

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

    async def test_returns_before_provisioning_finishes(self, sdk: MagicMock) -> None:
        """Provisioning takes minutes, so launch returns while it is in flight."""
        provisioned = threading.Event()
        sdk.get.side_effect = lambda request_id: provisioned.wait(timeout=5)
        task = sky.Task(run="echo hi")

        launch = await skypilot.launch(task, cluster_name="vibeml-a")

        sdk.launch.assert_called_once_with(task, cluster_name="vibeml-a")
        assert launch.request_id == "req-launch"
        assert not launch.settled.done()
        provisioned.set()
        await launch.settled
        sdk.down.assert_not_called()

    async def test_failed_launch_is_torn_down(self, sdk: MagicMock) -> None:
        """A launch that fails during provisioning is cleaned up by the launcher."""

        def get(request_id: str) -> None:
            if request_id == "req-launch":
                raise RuntimeError("storage mount failed")

        sdk.get.side_effect = get

        launch = await skypilot.launch(sky.Task(run="echo hi"), cluster_name="vibeml-a")
        await launch.settled

        sdk.down.assert_called_once_with("vibeml-a")
        sdk.get.assert_any_call("req-down")

    async def test_transient_observation_error_does_not_tear_down(
        self, sdk: MagicMock
    ) -> None:
        """A dropped connection while watching is not a launch failure."""
        observations = iter([ConnectionError("API server unreachable"), None])

        def get(request_id: str) -> None:
            outcome = next(observations)
            if outcome is not None:
                raise outcome

        sdk.get.side_effect = get
        sdk.api_status.return_value = [
            MagicMock(request_id="req-launch", status="RUNNING")
        ]

        launch = await skypilot.launch(sky.Task(run="echo hi"), cluster_name="vibeml-a")
        await launch.settled

        assert sdk.get.call_count == 2
        sdk.down.assert_not_called()

    async def test_unobservable_status_does_not_tear_down(self, sdk: MagicMock) -> None:
        """If the status cannot be read either, keep watching instead of tearing down."""
        observations = iter([ConnectionError("down"), None])

        def get(request_id: str) -> None:
            outcome = next(observations)
            if outcome is not None:
                raise outcome

        sdk.get.side_effect = get
        sdk.api_status.side_effect = ConnectionError("still down")

        launch = await skypilot.launch(sky.Task(run="echo hi"), cluster_name="vibeml-a")
        await launch.settled

        sdk.down.assert_not_called()

    async def test_confirmed_success_after_get_error_returns(
        self, sdk: MagicMock
    ) -> None:
        """If the status says the launch succeeded, nothing is torn down or retried."""
        sdk.get.side_effect = ConnectionError("dropped")
        sdk.api_status.return_value = [
            MagicMock(request_id="req-launch", status="SUCCEEDED")
        ]

        launch = await skypilot.launch(sky.Task(run="echo hi"), cluster_name="vibeml-a")
        await launch.settled

        assert sdk.get.call_count == 1
        sdk.down.assert_not_called()

    @pytest.mark.parametrize(
        "records", [[], [MagicMock(request_id="req-other", status="FAILED")]]
    )
    async def test_missing_status_record_keeps_observing(
        self, sdk: MagicMock, records: list[Any]
    ) -> None:
        """No matching status record is not evidence of failure."""
        observations = iter([ConnectionError("dropped"), None])

        def get(request_id: str) -> None:
            outcome = next(observations)
            if outcome is not None:
                raise outcome

        sdk.get.side_effect = get
        sdk.api_status.return_value = records

        launch = await skypilot.launch(sky.Task(run="echo hi"), cluster_name="vibeml-a")
        await launch.settled

        sdk.down.assert_not_called()

    async def test_cancelling_a_waiter_does_not_cancel_the_launch(
        self, sdk: MagicMock
    ) -> None:
        """Timing out on settled or settle_all leaves the launch owned and watched."""
        release = threading.Event()

        def get(request_id: str) -> None:
            if request_id == "req-launch":
                release.wait(timeout=5)
                raise RuntimeError("provisioning failed")

        sdk.get.side_effect = get

        launch = await skypilot.launch(sky.Task(run="echo hi"), cluster_name="vibeml-a")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(launch.settled, timeout=0.05)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(skypilot.settle_all(), timeout=0.05)
        release.set()
        await skypilot.settle_all()

        sdk.down.assert_called_once_with("vibeml-a")

    async def test_cancelled_request_is_torn_down(self, sdk: MagicMock) -> None:
        """A cancelled launch request may leave partial resources, so it is cleaned up."""
        sdk.get.side_effect = RuntimeError("request cancelled")
        sdk.api_status.return_value = [
            MagicMock(request_id="req-launch", status="CANCELLED")
        ]

        launch = await skypilot.launch(sky.Task(run="echo hi"), cluster_name="vibeml-a")
        await launch.settled

        sdk.down.assert_called_once_with("vibeml-a")

    async def test_caller_cancellation_keeps_ownership(self, sdk: MagicMock) -> None:
        """Cancelling the caller mid-submission still watches and cleans up the launch."""
        submitting, release = threading.Event(), threading.Event()

        def slow_launch(task: sky.Task, cluster_name: str) -> str:
            submitting.set()
            release.wait(timeout=5)
            return "req-launch"

        sdk.launch.side_effect = slow_launch
        sdk.get.side_effect = RuntimeError("provisioning failed")

        caller = asyncio.create_task(
            skypilot.launch(sky.Task(run="echo hi"), cluster_name="vibeml-a")
        )
        await asyncio.to_thread(submitting.wait, 5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        release.set()
        await skypilot.settle_all()

        sdk.down.assert_called_once_with("vibeml-a")

    async def test_failed_launch_kept_on_request(self, sdk: MagicMock) -> None:
        """keep_on_failure leaves a failed cluster up for debugging."""
        sdk.get.side_effect = RuntimeError("provisioning failed")

        launch = await skypilot.launch(
            sky.Task(run="echo hi"), cluster_name="vibeml-a", keep_on_failure=True
        )
        await launch.settled

        sdk.down.assert_not_called()

    async def test_failed_cleanup_does_not_raise(self, sdk: MagicMock) -> None:
        """If teardown fails too, the watcher logs it instead of crashing."""
        sdk.get.side_effect = RuntimeError("provisioning failed")
        sdk.down.side_effect = RuntimeError("cloud API down")

        launch = await skypilot.launch(sky.Task(run="echo hi"), cluster_name="vibeml-a")
        await launch.settled  # does not raise

        sdk.down.assert_called_once_with("vibeml-a")

    async def test_runtime_cap_becomes_unconditional_autostop(
        self, sdk: MagicMock
    ) -> None:
        """A runtime cap tears the cluster down after that many minutes, jobs or not."""
        task = sky.Task(run="echo hi")

        launch = await skypilot.launch(task, cluster_name="vibeml-a", max_minutes=120)
        await launch.settled

        sdk.launch.assert_called_once_with(
            task,
            cluster_name="vibeml-a",
            idle_minutes_to_autostop=120,
            down=True,
            wait_for=AutostopWaitFor.NONE,
        )

    async def test_runtime_cap_starts_api_server_before_checking_version(
        self, sdk: MagicMock
    ) -> None:
        """On a cold machine the local API server must be up before api_info."""
        order = MagicMock()
        order.attach_mock(sdk.start_server, "start")
        order.attach_mock(sdk.api_info, "api_info")
        order.attach_mock(sdk.launch, "launch")

        launch = await skypilot.launch(
            sky.Task(run="echo hi"), cluster_name="vibeml-a", max_minutes=60
        )
        await launch.settled

        assert [
            c[0] for c in order.mock_calls if c[0] in {"start", "api_info", "launch"}
        ] == [
            "start",
            "api_info",
            "launch",
        ]

    async def test_runtime_cap_fails_closed_on_old_api_server(
        self, sdk: MagicMock
    ) -> None:
        """Old API servers drop wait_for, so a capped launch must not proceed."""
        sdk.api_info.return_value.api_version = "12"

        with pytest.raises(SkyPilotError, match="API server"):
            await skypilot.launch(
                sky.Task(run="echo hi"), cluster_name="vibeml-a", max_minutes=60
            )

        sdk.launch.assert_not_called()

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
