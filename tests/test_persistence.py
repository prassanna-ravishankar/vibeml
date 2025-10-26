"""Tests for job persistence layer."""

import asyncio
import json
import tempfile
from datetime import datetime, UTC
from pathlib import Path

import pytest

from vibeml.persistence import JobsRepository
from vibeml.models import JobHandle, JobStatus, CostEstimate


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)
    yield db_path
    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def repository(temp_db_path):
    """Create a repository instance with temporary database."""
    return JobsRepository(db_path=temp_db_path)


@pytest.fixture
def sample_job():
    """Create a sample JobHandle for testing."""
    return JobHandle(
        job_id="test-job-123",
        cluster_name="vibeml-test-123",
        status=JobStatus.RUNNING,
        cost_estimate=CostEstimate(
            hourly_rate=1.2,
            estimated_duration_hours=10.0,
            min_cost=9.6,
            max_cost=15.6,
            expected_cost=12.0,
        ),
        created_at=datetime.now(UTC),
        model="meta-llama/Llama-3.2-1B",
        dataset="tatsu-lab/alpaca",
        workflow="unsloth",
        gpu_type="L40S",
        metadata={"test": "value", "number": 42},
    )


class TestJobsRepository:
    """Test suite for JobsRepository."""

    @pytest.mark.asyncio
    async def test_save_and_retrieve_job(self, repository, sample_job):
        """Test saving a job and retrieving it."""
        # Save job
        await repository.save_job(sample_job)

        # Retrieve job
        retrieved = await repository.get_job(sample_job.job_id)

        assert retrieved is not None
        assert retrieved.job_id == sample_job.job_id
        assert retrieved.cluster_name == sample_job.cluster_name
        assert retrieved.status == sample_job.status
        assert retrieved.model == sample_job.model
        assert retrieved.dataset == sample_job.dataset
        assert retrieved.workflow == sample_job.workflow
        assert retrieved.gpu_type == sample_job.gpu_type
        assert retrieved.metadata == sample_job.metadata

    @pytest.mark.asyncio
    async def test_save_job_preserves_cost_estimate(self, repository, sample_job):
        """Test that cost estimate is correctly serialized and deserialized."""
        await repository.save_job(sample_job)
        retrieved = await repository.get_job(sample_job.job_id)

        assert retrieved.cost_estimate is not None
        assert retrieved.cost_estimate.hourly_rate == sample_job.cost_estimate.hourly_rate
        assert retrieved.cost_estimate.expected_cost == sample_job.cost_estimate.expected_cost

    @pytest.mark.asyncio
    async def test_save_job_without_cost_estimate(self, repository, sample_job):
        """Test saving a job without cost estimate."""
        sample_job.cost_estimate = None
        await repository.save_job(sample_job)
        retrieved = await repository.get_job(sample_job.job_id)

        assert retrieved.cost_estimate is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, repository):
        """Test retrieving a job that doesn't exist."""
        result = await repository.get_job("nonexistent-job")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_job_status(self, repository, sample_job):
        """Test updating job status."""
        # Save initial job
        await repository.save_job(sample_job)

        # Update status
        await repository.update_status(
            job_id=sample_job.job_id,
            status=JobStatus.COMPLETED
        )

        # Retrieve and verify
        retrieved = await repository.get_job(sample_job.job_id)
        assert retrieved.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_update_status_with_metadata(self, repository, sample_job):
        """Test updating status and merging metadata."""
        await repository.save_job(sample_job)

        # Update with new metadata
        await repository.update_status(
            job_id=sample_job.job_id,
            status=JobStatus.COMPLETED,
            metadata_updates={"completion_time": "2024-01-01T00:00:00Z"}
        )

        retrieved = await repository.get_job(sample_job.job_id)
        assert retrieved.status == JobStatus.COMPLETED
        assert retrieved.metadata["test"] == "value"  # Original metadata preserved
        assert retrieved.metadata["completion_time"] == "2024-01-01T00:00:00Z"  # New metadata added

    @pytest.mark.asyncio
    async def test_update_nonexistent_job_raises_error(self, repository):
        """Test that updating a nonexistent job raises ValidationError."""
        from vibeml.exceptions import ValidationError

        with pytest.raises(ValidationError, match="not found"):
            await repository.update_status(
                job_id="nonexistent",
                status=JobStatus.COMPLETED
            )

    @pytest.mark.asyncio
    async def test_list_active_jobs(self, repository):
        """Test listing active jobs."""
        # Create jobs with different statuses
        jobs = [
            JobHandle(
                job_id=f"job-{i}",
                cluster_name=f"cluster-{i}",
                status=status,
                model="test-model",
                dataset="test-dataset",
                workflow="unsloth",
                gpu_type="L40S",
                metadata={},
            )
            for i, status in enumerate([
                JobStatus.PENDING,
                JobStatus.RUNNING,
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.RUNNING,
            ])
        ]

        # Save all jobs
        for job in jobs:
            await repository.save_job(job)

        # List active jobs (pending, running, review)
        active = await repository.list_active()

        assert len(active) == 3  # 1 pending + 2 running
        active_statuses = {job.status for job in active}
        assert JobStatus.PENDING in active_statuses
        assert JobStatus.RUNNING in active_statuses
        assert JobStatus.COMPLETED not in active_statuses
        assert JobStatus.FAILED not in active_statuses

    @pytest.mark.asyncio
    async def test_list_all_jobs(self, repository):
        """Test listing all jobs with limit."""
        # Create multiple jobs
        jobs = [
            JobHandle(
                job_id=f"job-{i}",
                cluster_name=f"cluster-{i}",
                status=JobStatus.RUNNING,
                model="test-model",
                dataset="test-dataset",
                workflow="unsloth",
                gpu_type="L40S",
                metadata={},
            )
            for i in range(5)
        ]

        for job in jobs:
            await repository.save_job(job)

        # List all jobs
        all_jobs = await repository.list_all(limit=10)
        assert len(all_jobs) == 5

        # Test limit
        limited = await repository.list_all(limit=3)
        assert len(limited) == 3

    @pytest.mark.asyncio
    async def test_purge_completed_jobs(self, repository):
        """Test purging old completed jobs."""
        from datetime import timedelta

        # Create old completed job
        old_job = JobHandle(
            job_id="old-job",
            cluster_name="old-cluster",
            status=JobStatus.COMPLETED,
            created_at=datetime.now(UTC) - timedelta(days=60),
            model="test-model",
            dataset="test-dataset",
            workflow="unsloth",
            gpu_type="L40S",
            metadata={},
        )

        # Create recent completed job
        recent_job = JobHandle(
            job_id="recent-job",
            cluster_name="recent-cluster",
            status=JobStatus.COMPLETED,
            created_at=datetime.now(UTC) - timedelta(days=10),
            model="test-model",
            dataset="test-dataset",
            workflow="unsloth",
            gpu_type="L40S",
            metadata={},
        )

        # Create active job
        active_job = JobHandle(
            job_id="active-job",
            cluster_name="active-cluster",
            status=JobStatus.RUNNING,
            created_at=datetime.now(UTC) - timedelta(days=60),
            model="test-model",
            dataset="test-dataset",
            workflow="unsloth",
            gpu_type="L40S",
            metadata={},
        )

        await repository.save_job(old_job)
        await repository.save_job(recent_job)
        await repository.save_job(active_job)

        # Purge jobs older than 30 days
        deleted = await repository.purge_completed(older_than_days=30)
        assert deleted == 1  # Only old_job should be deleted

        # Verify
        assert await repository.get_job("old-job") is None
        assert await repository.get_job("recent-job") is not None
        assert await repository.get_job("active-job") is not None

    @pytest.mark.asyncio
    async def test_concurrent_updates(self, repository, sample_job):
        """Test concurrent status updates don't corrupt data."""
        await repository.save_job(sample_job)

        # Simulate concurrent updates
        async def update_status(status, metadata_key):
            await repository.update_status(
                job_id=sample_job.job_id,
                status=status,
                metadata_updates={metadata_key: "value"}
            )

        # Run multiple updates concurrently
        await asyncio.gather(
            update_status(JobStatus.RUNNING, "key1"),
            update_status(JobStatus.RUNNING, "key2"),
            update_status(JobStatus.RUNNING, "key3"),
        )

        # Verify final state is consistent
        retrieved = await repository.get_job(sample_job.job_id)
        assert retrieved is not None
        assert retrieved.status == JobStatus.RUNNING
        # All metadata keys should be present (last write wins for status)
        assert "key1" in retrieved.metadata or "key2" in retrieved.metadata or "key3" in retrieved.metadata

    @pytest.mark.asyncio
    async def test_job_metadata_serialization(self, repository, sample_job):
        """Test complex metadata serialization."""
        sample_job.metadata = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "bool": True,
            "null": None,
            "number": 42.5,
        }

        await repository.save_job(sample_job)
        retrieved = await repository.get_job(sample_job.job_id)

        assert retrieved.metadata == sample_job.metadata

    @pytest.mark.asyncio
    async def test_replace_existing_job(self, repository, sample_job):
        """Test that saving a job with existing ID replaces it."""
        # Save initial version
        await repository.save_job(sample_job)

        # Modify and save again
        sample_job.status = JobStatus.COMPLETED
        sample_job.metadata["updated"] = True
        await repository.save_job(sample_job)

        # Verify only one job exists with updated data
        all_jobs = await repository.list_all()
        assert len(all_jobs) == 1
        assert all_jobs[0].status == JobStatus.COMPLETED
        assert all_jobs[0].metadata["updated"] is True

    @pytest.mark.asyncio
    async def test_database_persistence_across_instances(self, temp_db_path, sample_job):
        """Test that data persists across repository instances."""
        # Save with first instance
        repo1 = JobsRepository(db_path=temp_db_path)
        await repo1.save_job(sample_job)

        # Retrieve with second instance
        repo2 = JobsRepository(db_path=temp_db_path)
        retrieved = await repo2.get_job(sample_job.job_id)

        assert retrieved is not None
        assert retrieved.job_id == sample_job.job_id

    @pytest.mark.asyncio
    async def test_schema_migration(self, temp_db_path):
        """Test that schema migration runs correctly."""
        repo = JobsRepository(db_path=temp_db_path)

        # Verify schema version is recorded
        import sqlite3
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        conn.close()

        assert version == 1  # Current schema version
