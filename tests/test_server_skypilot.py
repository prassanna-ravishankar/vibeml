"""Tests for SkyPilot-backed MCP server functionality."""

import pytest
from datetime import datetime, UTC
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Any, Dict, List

from vibeml.server import (
    _map_sky_status_to_job_status,
    _get_cluster_from_skypilot,
)
from vibeml.models import JobStatus

# We'll test by importing and calling the server functions directly
# The MCP decorator doesn't affect the underlying async function behavior


# Mock SkyPilot cluster object
class MockCluster:
    """Mock SkyPilot cluster info object."""

    def __init__(
        self,
        name: str = "vibeml-unsloth-abc123",
        status: str = "UP",
        launched_at: datetime = None,
    ):
        self.name = name
        self.status = status
        self.launched_at = launched_at or datetime.now(UTC)
        self.resources = MagicMock(accelerators="L40S:1")
        self.handle = {"cloud": "nebius", "region": "eu-north1"}


class TestStatusMapping:
    """Test SkyPilot status to JobStatus mapping."""

    def test_map_init_to_pending(self):
        """Test INIT maps to PENDING."""
        assert _map_sky_status_to_job_status("INIT") == JobStatus.PENDING

    def test_map_up_to_running(self):
        """Test UP maps to RUNNING."""
        assert _map_sky_status_to_job_status("UP") == JobStatus.RUNNING

    def test_map_stopped_to_terminated(self):
        """Test STOPPED maps to TERMINATED."""
        assert _map_sky_status_to_job_status("STOPPED") == JobStatus.TERMINATED

    def test_map_terminated_to_terminated(self):
        """Test TERMINATED maps to TERMINATED."""
        assert _map_sky_status_to_job_status("TERMINATED") == JobStatus.TERMINATED

    def test_map_unknown_status(self):
        """Test unknown status maps to FAILED."""
        assert _map_sky_status_to_job_status("WEIRD_STATE") == JobStatus.FAILED


class TestGetClusterFromSkyPilot:
    """Test _get_cluster_from_skypilot helper."""

    @pytest.mark.asyncio
    async def test_get_existing_cluster(self, monkeypatch):
        """Test retrieving an existing cluster."""
        mock_cluster = MockCluster(name="test-cluster", status="UP")

        def mock_status(*args, **kwargs):
            return [mock_cluster]

        with patch("vibeml.server.sky.status", side_effect=mock_status):
            result = await _get_cluster_from_skypilot("test-cluster")

        assert result is not None
        assert result["cluster_name"] == "test-cluster"
        assert result["status"] == "UP"
        assert result["launched_at"] == mock_cluster.launched_at

    @pytest.mark.asyncio
    async def test_get_nonexistent_cluster(self, monkeypatch):
        """Test retrieving a cluster that doesn't exist."""

        def mock_status(*args, **kwargs):
            return []

        with patch("vibeml.server.sky.status", side_effect=mock_status):
            result = await _get_cluster_from_skypilot("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_cluster_error_handling(self, monkeypatch):
        """Test error handling when SkyPilot fails."""

        def mock_status(*args, **kwargs):
            raise Exception("SkyPilot connection error")

        with patch("vibeml.server.sky.status", side_effect=mock_status):
            result = await _get_cluster_from_skypilot("test-cluster")

        assert result is None


# Note: MCP tool tests would require more complex setup with FastMCP
# The core logic is tested via the helper functions above
# Integration testing with actual MCP calls would be done at system level
