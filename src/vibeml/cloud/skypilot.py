"""Adapter over SkyPilot's request-based client SDK.

Most SkyPilot SDK calls submit a request to the SkyPilot API server and return
a request ID; ``sky.get`` blocks until the request finishes. (``download_logs``
resolves its request internally and returns the result.) This module owns that
protocol and translates SkyPilot's records into vibeml's types, so callers never
handle request IDs, blocking calls, or SkyPilot enums directly.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

import sky
from sky.server import common as server_common
from sky.skylet.autostop_lib import AutostopWaitFor

from ..exceptions import SkyPilotError
from ..models import JobStatus

T = TypeVar("T")

logger = logging.getLogger(__name__)

# Each launch is owned by a background task (submission plus settlement).
# Keep strong references until they finish so they are not garbage-collected
# mid-flight, and so they outlive a cancelled caller.
_owners: set["asyncio.Task[None]"] = set()

# Launch requests that end in these states left nothing worth keeping.
_FAILED_REQUEST_STATES = {"FAILED", "CANCELLED"}

# How long to wait before re-observing a launch request whose outcome could
# not be confirmed (for example, the API server was briefly unreachable).
_OBSERVE_RETRY_SECONDS = 30.0

# First SkyPilot API version that honours autostop ``wait_for``; older servers
# silently drop it, which would turn the runtime cap into an idle-only cap.
_MIN_API_VERSION_FOR_WAIT_FOR = 13

_JOB_STATUS = {
    sky.ClusterStatus.INIT: JobStatus.PENDING,
    sky.ClusterStatus.PENDING: JobStatus.PENDING,
    sky.ClusterStatus.UP: JobStatus.RUNNING,
    sky.ClusterStatus.AUTOSTOPPING: JobStatus.RUNNING,
    sky.ClusterStatus.STOPPED: JobStatus.TERMINATED,
}


@dataclass(frozen=True)
class ClusterInfo:
    """A SkyPilot cluster as vibeml sees it."""

    name: str
    sky_status: str
    status: JobStatus
    launched_at: datetime | None


def to_job_status(cluster_status: sky.ClusterStatus) -> JobStatus:
    """Map a SkyPilot cluster status onto a vibeml job status.

    Args:
        cluster_status: Status reported by SkyPilot.

    Returns:
        The corresponding job status.
    """
    return _JOB_STATUS[cluster_status]


async def _call(fn: Callable[[], T]) -> T:
    """Run a blocking SDK call in a worker thread, wrapping its errors.

    Translation of SkyPilot records belongs inside ``fn`` too, so an
    unexpected record shape is reported as a SkyPilot error.

    Args:
        fn: Zero-argument callable that performs the SDK call.

    Returns:
        Whatever ``fn`` returns.

    Raises:
        SkyPilotError: If the SDK call fails.
    """
    try:
        return await asyncio.to_thread(fn)
    except Exception as e:
        raise SkyPilotError(str(e)) from e


def _to_cluster_info(record: Any) -> ClusterInfo:
    """Translate a SkyPilot status record into a ClusterInfo.

    Args:
        record: A ``StatusResponse`` from ``sky.status``.

    Returns:
        The cluster as vibeml sees it.
    """
    launched_at = (
        datetime.fromtimestamp(record.launched_at, tz=timezone.utc)
        if record.launched_at
        else None
    )
    return ClusterInfo(
        name=record.name,
        sky_status=record.status.value,
        status=to_job_status(record.status),
        launched_at=launched_at,
    )


async def get_cluster(name: str) -> ClusterInfo | None:
    """Look up one cluster.

    Args:
        name: Cluster name.

    Returns:
        The cluster, or None if SkyPilot does not know it.
    """
    clusters = await _call(
        lambda: [
            _to_cluster_info(r)
            for r in sky.get(
                sky.status(cluster_names=[name], refresh=sky.StatusRefreshMode.AUTO)
            )
        ]
    )
    return clusters[0] if clusters else None


async def list_clusters(prefix: str) -> list[ClusterInfo]:
    """List clusters whose name starts with ``prefix``.

    Args:
        prefix: Cluster name prefix.

    Returns:
        Matching clusters, in SkyPilot's order.
    """
    return await _call(
        lambda: [
            _to_cluster_info(r)
            for r in sky.get(sky.status(refresh=sky.StatusRefreshMode.AUTO))
            if r.name.startswith(prefix)
        ]
    )


@dataclass(frozen=True)
class Launch:
    """A submitted launch."""

    request_id: str
    _owner: "asyncio.Task[None]"

    @property
    def settled(self) -> "asyncio.Future[None]":
        """Finishes when SkyPilot's launch request does.

        If the request failed and the cluster was not kept, teardown has run
        by then. Cancelling this (for example with ``asyncio.wait_for``) only
        stops waiting; the launch stays owned and watched.
        """
        return asyncio.shield(self._owner)


async def launch(
    task: sky.Task,
    cluster_name: str,
    max_minutes: int | None = None,
    keep_on_failure: bool = False,
) -> Launch:
    """Submit a launch request without waiting for it to finish.

    Provisioning and setup take minutes; the request ID can be followed with
    ``sky api logs <request_id>``. In the background, the launcher waits for
    the request, and if it fails (during provisioning, file or storage sync,
    or setup) tears the cluster down, so a failed launch is not left running.
    That needs this process to stay alive until the request settles.

    Args:
        task: Task to run.
        cluster_name: Name for the new cluster.
        max_minutes: Runtime cap. The cluster is torn down about this many
            minutes after the job starts, whether or not it is still running
            (SkyPilot autostop with ``wait_for=NONE`` and ``down``). The timer
            is polled about once a minute, so the cap is approximate.
        keep_on_failure: Leave a cluster whose launch failed up for debugging.

    Returns:
        The submitted launch.
    """
    cap: dict[str, Any] = {}
    if max_minutes is not None:
        cap = {
            "idle_minutes_to_autostop": max_minutes,
            "down": True,
            "wait_for": AutostopWaitFor.NONE,
        }

    def submit() -> Any:
        if cap:
            # api_info does not start the local API server the way launch
            # does, so bring it up through the same lifecycle first.
            server_common.check_server_healthy_or_start_fn()
            api_version = int(sky.api_info().api_version)
            if api_version < _MIN_API_VERSION_FOR_WAIT_FOR:
                raise SkyPilotError(
                    f"SkyPilot API server version {api_version} cannot enforce a runtime "
                    f"cap (needs >= {_MIN_API_VERSION_FOR_WAIT_FOR}); not launched."
                )
        return sky.launch(task, cluster_name=cluster_name, **cap)

    loop = asyncio.get_running_loop()
    submitted: asyncio.Future[str] = loop.create_future()
    # Mark the result retrieved even if the caller was cancelled and never reads it.
    submitted.add_done_callback(lambda f: f.cancelled() or f.exception())

    async def own() -> None:
        try:
            request_id = await _call(submit)
        except BaseException as e:
            submitted.set_exception(e)
            return
        submitted.set_result(str(request_id))
        await _settle(request_id, cluster_name, keep_on_failure)

    owner = asyncio.create_task(own())
    _owners.add(owner)
    owner.add_done_callback(_owners.discard)
    # Cancelling the caller must not cancel the owner: a submission already
    # in flight still needs watching and, on failure, cleanup.
    request_id = await asyncio.shield(submitted)
    return Launch(request_id=request_id, _owner=owner)


async def settle_all() -> None:
    """Wait until every launch owned by this process has settled.

    Failed launches are cleaned up only while this process runs; an embedding
    application can await this before exiting. Cancelling the wait does not
    cancel the launches.
    """
    while _owners:
        # Shielded, so cancelling this waiter does not cancel the launches.
        await asyncio.gather(
            *(asyncio.shield(o) for o in _owners), return_exceptions=True
        )


async def _request_status(request_id: Any) -> str | None:
    """Read a request's status from the API server.

    Args:
        request_id: The request to look up.

    Returns:
        The status (for example ``"RUNNING"`` or ``"FAILED"``), or None if it
        could not be read.
    """
    try:
        payloads = await asyncio.to_thread(sky.api_status, [request_id])
    except Exception as e:
        logger.warning("Could not read status of request %s: %s", request_id, e)
        return None
    for payload in payloads:
        if payload.request_id == str(request_id):
            return str(payload.status)
    return None


async def _settle(request_id: Any, cluster_name: str, keep_on_failure: bool) -> None:
    """Wait for a launch request; tear the cluster down if it failed.

    Teardown happens only when the API server confirms the request failed or
    was cancelled. Errors observing the request (a dropped connection, an
    unreadable status) are retried, never treated as failure, because tearing
    down a healthy job would destroy its work.

    Args:
        request_id: The launch request.
        cluster_name: Cluster the request launches.
        keep_on_failure: Leave a failed cluster up for debugging.
    """
    while True:
        try:
            await asyncio.to_thread(sky.get, request_id)
            return
        except Exception as e:
            error = e
        status = await _request_status(request_id)
        if status in _FAILED_REQUEST_STATES:
            break
        if status == "SUCCEEDED":
            return
        logger.warning(
            "Launch request %s for %s not confirmed (status %s): %s; re-checking in %ss",
            request_id,
            cluster_name,
            status,
            error,
            _OBSERVE_RETRY_SECONDS,
        )
        await asyncio.sleep(_OBSERVE_RETRY_SECONDS)

    if keep_on_failure:
        logger.warning(
            "Launch of %s failed, kept for debugging: %s", cluster_name, error
        )
        return
    logger.warning("Launch of %s failed, tearing it down: %s", cluster_name, error)
    try:
        await asyncio.to_thread(lambda: sky.get(sky.down(cluster_name)))
    except Exception as down_error:
        logger.error(
            "Teardown of failed launch %s also failed; check `sky status`: %s",
            cluster_name,
            down_error,
        )


async def down(name: str) -> None:
    """Tear a cluster down and wait until it is gone.

    Args:
        name: Cluster name.
    """
    await _call(lambda: sky.get(sky.down(name)))


async def download_logs(name: str) -> dict[str, str]:
    """Download the logs of every job on a cluster.

    Args:
        name: Cluster name.

    Returns:
        Mapping of job ID to the local directory SkyPilot wrote its logs to.
    """
    return await _call(lambda: sky.download_logs(name, job_ids=None))
