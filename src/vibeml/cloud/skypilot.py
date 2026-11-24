"""Adapter over SkyPilot's request-based client SDK.

Most SkyPilot SDK calls submit a request to the SkyPilot API server and return
a request ID; ``sky.get`` blocks until the request finishes. (``download_logs``
resolves its request internally and returns the result.) This module owns that
protocol and translates SkyPilot's records into vibeml's types, so callers never
handle request IDs, blocking calls, or SkyPilot enums directly.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

import sky

from ..exceptions import SkyPilotError
from ..models import JobStatus

T = TypeVar("T")

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


async def launch(task: sky.Task, cluster_name: str) -> str:
    """Submit a launch request without waiting for it to finish.

    Provisioning and setup take minutes; the returned request ID can be
    followed with ``sky api logs <request_id>``.

    Args:
        task: Task to run.
        cluster_name: Name for the new cluster.

    Returns:
        The launch request ID.
    """
    request_id = await _call(lambda: sky.launch(task, cluster_name=cluster_name))
    return str(request_id)


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
