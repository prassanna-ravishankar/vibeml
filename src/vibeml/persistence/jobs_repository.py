"""SQLite-backed job persistence repository for VibeML."""

import asyncio
import json
import sqlite3
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from pydantic import BaseModel, ConfigDict, Field

from ..models import JobHandle, JobStatus, CostEstimate
from ..exceptions import SkyPilotError, ValidationError


class JobRecord(BaseModel):
    """Database representation of a JobHandle.

    This model handles serialization of complex types for SQLite storage.
    """

    job_id: str
    cluster_name: str
    status: str  # Stored as string, converted to/from JobStatus enum
    cost_estimate_json: Optional[str] = None  # Serialized CostEstimate
    created_at: str  # ISO 8601 datetime string
    updated_at: str  # ISO 8601 datetime string
    model: str
    dataset: str
    workflow: str
    gpu_type: str
    metadata_json: str  # Serialized metadata dict

    model_config = ConfigDict(frozen=False)

    @classmethod
    def from_job_handle(cls, job: JobHandle) -> "JobRecord":
        """Convert JobHandle to JobRecord for database storage."""
        now = datetime.now(UTC).isoformat()

        return cls(
            job_id=job.job_id,
            cluster_name=job.cluster_name,
            status=job.status.value if isinstance(job.status, JobStatus) else job.status,
            cost_estimate_json=job.cost_estimate.model_dump_json() if job.cost_estimate else None,
            created_at=job.created_at.isoformat(),
            updated_at=now,
            model=job.model,
            dataset=job.dataset,
            workflow=job.workflow,
            gpu_type=job.gpu_type,
            metadata_json=json.dumps(job.metadata),
        )

    def to_job_handle(self) -> JobHandle:
        """Convert JobRecord back to JobHandle."""
        return JobHandle(
            job_id=self.job_id,
            cluster_name=self.cluster_name,
            status=JobStatus(self.status),
            cost_estimate=CostEstimate.model_validate_json(self.cost_estimate_json) if self.cost_estimate_json else None,
            created_at=datetime.fromisoformat(self.created_at),
            model=self.model,
            dataset=self.dataset,
            workflow=self.workflow,
            gpu_type=self.gpu_type,
            metadata=json.loads(self.metadata_json),
        )


class JobsRepository:
    """SQLite-backed repository for persistent job storage.

    Features:
    - Single-writer pattern with WAL mode for concurrency
    - Async-safe operations using asyncio.to_thread
    - Automatic schema migration
    - Defensive error handling
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the jobs repository.

        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            # Use default path in user's config directory
            from ..config.preferences import PreferencesManager
            prefs = PreferencesManager()
            db_path = prefs.config_dir / "jobs.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with WAL mode enabled."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Access columns by name

        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema with migrations support."""
        with self._get_connection() as conn:
            # Create schema version table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)

            # Check current version
            cursor = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            row = cursor.fetchone()
            current_version = row[0] if row else 0

            # Apply migrations
            if current_version < 1:
                self._migrate_to_v1(conn)

            conn.commit()

    def _migrate_to_v1(self, conn: sqlite3.Connection) -> None:
        """Create initial schema (version 1)."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                cluster_name TEXT NOT NULL,
                status TEXT NOT NULL,
                cost_estimate_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                model TEXT NOT NULL,
                dataset TEXT NOT NULL,
                workflow TEXT NOT NULL,
                gpu_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
        """)

        # Create index on status for efficient filtering
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_status
            ON jobs(status)
        """)

        # Create index on created_at for chronological queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_created_at
            ON jobs(created_at DESC)
        """)

        # Record schema version
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (1, datetime.now(UTC).isoformat())
        )

    async def save_job(self, job: JobHandle) -> None:
        """Save a new job or update an existing one.

        Args:
            job: JobHandle to persist

        Raises:
            ValidationError: If job validation fails
            SkyPilotError: If database operation fails
        """
        try:
            record = JobRecord.from_job_handle(job)

            def _save():
                with self._get_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO jobs (
                            job_id, cluster_name, status, cost_estimate_json,
                            created_at, updated_at, model, dataset, workflow,
                            gpu_type, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.job_id,
                        record.cluster_name,
                        record.status,
                        record.cost_estimate_json,
                        record.created_at,
                        record.updated_at,
                        record.model,
                        record.dataset,
                        record.workflow,
                        record.gpu_type,
                        record.metadata_json,
                    ))
                    conn.commit()

            # Run blocking DB operation in thread pool
            await asyncio.to_thread(_save)

        except Exception as e:
            raise SkyPilotError(
                f"Failed to save job {job.job_id}",
                technical_details=str(e),
                recovery_suggestion="Check database permissions and disk space",
            ) from e

    async def get_job(self, job_id: str) -> Optional[JobHandle]:
        """Retrieve a job by ID.

        Args:
            job_id: Job identifier

        Returns:
            JobHandle if found, None otherwise

        Raises:
            SkyPilotError: If database operation fails
        """
        try:
            def _get():
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT * FROM jobs WHERE job_id = ?",
                        (job_id,)
                    )
                    row = cursor.fetchone()

                    if row is None:
                        return None

                    record = JobRecord(
                        job_id=row["job_id"],
                        cluster_name=row["cluster_name"],
                        status=row["status"],
                        cost_estimate_json=row["cost_estimate_json"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        model=row["model"],
                        dataset=row["dataset"],
                        workflow=row["workflow"],
                        gpu_type=row["gpu_type"],
                        metadata_json=row["metadata_json"],
                    )

                    return record.to_job_handle()

            return await asyncio.to_thread(_get)

        except Exception as e:
            raise SkyPilotError(
                f"Failed to retrieve job {job_id}",
                technical_details=str(e),
                recovery_suggestion="Check database integrity",
            ) from e

    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update job status and optionally merge metadata.

        Args:
            job_id: Job identifier
            status: New status
            metadata_updates: Additional metadata to merge

        Raises:
            ValidationError: If job not found
            SkyPilotError: If database operation fails
        """
        try:
            def _update():
                with self._get_connection() as conn:
                    # Get current job
                    cursor = conn.execute(
                        "SELECT metadata_json FROM jobs WHERE job_id = ?",
                        (job_id,)
                    )
                    row = cursor.fetchone()

                    if row is None:
                        raise ValidationError(
                            f"Job {job_id} not found",
                            recovery_suggestion="Verify job ID is correct",
                        )

                    # Merge metadata if provided
                    current_metadata = json.loads(row["metadata_json"])
                    if metadata_updates:
                        current_metadata.update(metadata_updates)

                    # Update status and metadata
                    conn.execute("""
                        UPDATE jobs
                        SET status = ?,
                            updated_at = ?,
                            metadata_json = ?
                        WHERE job_id = ?
                    """, (
                        status.value,
                        datetime.now(UTC).isoformat(),
                        json.dumps(current_metadata),
                        job_id,
                    ))
                    conn.commit()

            await asyncio.to_thread(_update)

        except ValidationError:
            raise
        except Exception as e:
            raise SkyPilotError(
                f"Failed to update job {job_id} status",
                technical_details=str(e),
                recovery_suggestion="Check database connectivity",
            ) from e

    async def list_active(self) -> List[JobHandle]:
        """List all non-terminal jobs (pending, running, review).

        Returns:
            List of active JobHandles

        Raises:
            SkyPilotError: If database operation fails
        """
        try:
            def _list():
                jobs = []
                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        SELECT * FROM jobs
                        WHERE status IN ('pending', 'running', 'review')
                        ORDER BY created_at DESC
                    """)

                    for row in cursor.fetchall():
                        record = JobRecord(
                            job_id=row["job_id"],
                            cluster_name=row["cluster_name"],
                            status=row["status"],
                            cost_estimate_json=row["cost_estimate_json"],
                            created_at=row["created_at"],
                            updated_at=row["updated_at"],
                            model=row["model"],
                            dataset=row["dataset"],
                            workflow=row["workflow"],
                            gpu_type=row["gpu_type"],
                            metadata_json=row["metadata_json"],
                        )
                        jobs.append(record.to_job_handle())

                return jobs

            return await asyncio.to_thread(_list)

        except Exception as e:
            raise SkyPilotError(
                "Failed to list active jobs",
                technical_details=str(e),
                recovery_suggestion="Check database integrity",
            ) from e

    async def list_all(self, limit: int = 100) -> List[JobHandle]:
        """List all jobs, most recent first.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of JobHandles

        Raises:
            SkyPilotError: If database operation fails
        """
        try:
            def _list():
                jobs = []
                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        SELECT * FROM jobs
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (limit,))

                    for row in cursor.fetchall():
                        record = JobRecord(
                            job_id=row["job_id"],
                            cluster_name=row["cluster_name"],
                            status=row["status"],
                            cost_estimate_json=row["cost_estimate_json"],
                            created_at=row["created_at"],
                            updated_at=row["updated_at"],
                            model=row["model"],
                            dataset=row["dataset"],
                            workflow=row["workflow"],
                            gpu_type=row["gpu_type"],
                            metadata_json=row["metadata_json"],
                        )
                        jobs.append(record.to_job_handle())

                return jobs

            return await asyncio.to_thread(_list)

        except Exception as e:
            raise SkyPilotError(
                "Failed to list jobs",
                technical_details=str(e),
                recovery_suggestion="Check database integrity",
            ) from e

    async def purge_completed(self, older_than_days: int = 30) -> int:
        """Remove completed/failed/terminated jobs older than specified days.

        Args:
            older_than_days: Remove jobs older than this many days

        Returns:
            Number of jobs purged

        Raises:
            SkyPilotError: If database operation fails
        """
        try:
            def _purge():
                from datetime import timedelta
                cutoff = (datetime.now(UTC) - timedelta(days=older_than_days)).isoformat()

                with self._get_connection() as conn:
                    cursor = conn.execute("""
                        DELETE FROM jobs
                        WHERE status IN ('completed', 'failed', 'terminated')
                        AND created_at < ?
                    """, (cutoff,))
                    deleted = cursor.rowcount
                    conn.commit()
                    return deleted

            return await asyncio.to_thread(_purge)

        except Exception as e:
            raise SkyPilotError(
                "Failed to purge completed jobs",
                technical_details=str(e),
                recovery_suggestion="Check database permissions",
            ) from e
