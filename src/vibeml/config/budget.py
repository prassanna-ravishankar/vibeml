"""Budget management and cost tracking."""

import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, UTC
from enum import Enum

from pydantic import BaseModel, Field

from ..exceptions import BudgetExceededError, ConfigurationError


class SpendingRecord(BaseModel):
    """Record of a spending event."""

    job_id: str = Field(description="Job identifier")
    timestamp: datetime = Field(description="When spending occurred")
    amount: float = Field(ge=0, description="Amount spent in USD")
    description: str = Field(description="Description of spending")
    workflow: str = Field(description="Workflow type")
    gpu_type: str = Field(description="GPU type used")


class BudgetStatus(str, Enum):
    """Status of budget."""

    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"


class BudgetManager:
    """Manages budget limits, tracking, and alerts."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize BudgetManager.

        Args:
            config_dir: Directory for storing budget data (default: ~/.vibeml)
        """
        self.config_dir = config_dir or Path.home() / ".vibeml"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.spending_file = self.config_dir / "spending.json"
        self._spending_records: List[SpendingRecord] = []
        self._load_spending()

    def _load_spending(self) -> None:
        """Load spending records from file."""
        if not self.spending_file.exists():
            self._spending_records = []
            return

        try:
            data = json.loads(self.spending_file.read_text())
            self._spending_records = [
                SpendingRecord(**record) for record in data.get("records", [])
            ]
        except Exception:
            # If loading fails, start fresh
            self._spending_records = []

    def _save_spending(self) -> None:
        """Save spending records to file."""
        try:
            data = {
                "records": [record.model_dump(mode="json") for record in self._spending_records],
                "last_updated": datetime.now(UTC).isoformat(),
            }
            self.spending_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            raise ConfigurationError(
                "Failed to save spending records",
                technical_details=str(e),
            )

    def record_spending(
        self,
        job_id: str,
        amount: float,
        description: str,
        workflow: str,
        gpu_type: str,
    ) -> None:
        """Record a spending event.

        Args:
            job_id: Job identifier
            amount: Amount spent in USD
            description: Description of spending
            workflow: Workflow type
            gpu_type: GPU type used
        """
        record = SpendingRecord(
            job_id=job_id,
            timestamp=datetime.now(UTC),
            amount=amount,
            description=description,
            workflow=workflow,
            gpu_type=gpu_type,
        )
        self._spending_records.append(record)
        self._save_spending()

    def get_total_spending(
        self,
        since: Optional[datetime] = None,
    ) -> float:
        """Get total spending amount.

        Args:
            since: Only count spending after this time (None = all time)

        Returns:
            Total spending in USD
        """
        total = 0.0
        for record in self._spending_records:
            if since is None or record.timestamp >= since:
                total += record.amount
        return total

    def get_spending_by_workflow(self) -> Dict[str, float]:
        """Get spending broken down by workflow type.

        Returns:
            Dictionary mapping workflow to total spending
        """
        spending = {}
        for record in self._spending_records:
            workflow = record.workflow
            spending[workflow] = spending.get(workflow, 0.0) + record.amount
        return spending

    def get_spending_by_gpu(self) -> Dict[str, float]:
        """Get spending broken down by GPU type.

        Returns:
            Dictionary mapping GPU type to total spending
        """
        spending = {}
        for record in self._spending_records:
            gpu = record.gpu_type
            spending[gpu] = spending.get(gpu, 0.0) + record.amount
        return spending

    def check_budget(
        self,
        estimated_cost: float,
        max_cost: Optional[float] = None,
        warning_threshold: float = 0.8,
    ) -> BudgetStatus:
        """Check if a job is within budget.

        Args:
            estimated_cost: Estimated cost of the job
            max_cost: Maximum allowed cost (None = no limit)
            warning_threshold: Fraction of max_cost to trigger warning

        Returns:
            BudgetStatus indicating whether budget allows the job

        Raises:
            BudgetExceededError: If estimated cost exceeds max_cost
        """
        if max_cost is None:
            return BudgetStatus.OK

        if estimated_cost > max_cost:
            raise BudgetExceededError(
                f"Estimated cost ${estimated_cost:.2f} exceeds budget limit ${max_cost:.2f}",
                estimated_cost=estimated_cost,
                max_cost=max_cost,
            )

        if estimated_cost > max_cost * warning_threshold:
            return BudgetStatus.WARNING

        return BudgetStatus.OK

    def require_approval(
        self,
        estimated_cost: float,
        max_cost: Optional[float] = None,
        auto_approve_threshold: float = 10.0,
    ) -> bool:
        """Check if a job requires manual approval.

        Args:
            estimated_cost: Estimated cost of the job
            max_cost: Maximum allowed cost
            auto_approve_threshold: Auto-approve if cost below this (USD)

        Returns:
            True if manual approval required, False otherwise
        """
        # Auto-approve small jobs
        if estimated_cost <= auto_approve_threshold:
            return False

        # Check if within budget
        try:
            status = self.check_budget(estimated_cost, max_cost)
            # Require approval for warnings
            return status == BudgetStatus.WARNING
        except BudgetExceededError:
            # Over budget always requires approval (or rejection)
            return True

    def get_spending_summary(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, any]:
        """Get spending summary with breakdown.

        Args:
            since: Only include spending after this time

        Returns:
            Dictionary with spending summary
        """
        total = self.get_total_spending(since)
        by_workflow = self.get_spending_by_workflow()
        by_gpu = self.get_spending_by_gpu()

        return {
            "total": total,
            "by_workflow": by_workflow,
            "by_gpu": by_gpu,
            "num_jobs": len(self._spending_records),
        }

    def clear_history(self) -> None:
        """Clear all spending history."""
        self._spending_records = []
        self._save_spending()

    def export_records(self, output_file: Path) -> None:
        """Export spending records to a file.

        Args:
            output_file: Path to output file
        """
        data = {
            "records": [record.model_dump(mode="json") for record in self._spending_records],
            "exported_at": datetime.now(UTC).isoformat(),
            "summary": self.get_spending_summary(),
        }
        output_file.write_text(json.dumps(data, indent=2))
