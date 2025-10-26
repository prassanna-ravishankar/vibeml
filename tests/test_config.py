"""Tests for configuration management."""

import json
from pathlib import Path
from datetime import datetime, UTC, timedelta
import tempfile
import pytest

from vibeml.config import CredentialManager, PreferencesManager, BudgetManager
from vibeml.config.budget import BudgetStatus, SpendingRecord
from vibeml.exceptions import ConfigurationError, BudgetExceededError


class TestCredentialManager:
    """Tests for CredentialManager."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test credential manager initialization."""
        cm = CredentialManager(config_dir=tmp_path)
        assert cm.config_dir == tmp_path
        assert cm.config_dir.exists()
        assert cm.key_file.exists()

    def test_store_and_retrieve_credential(self, tmp_path: Path) -> None:
        """Test storing and retrieving credentials."""
        cm = CredentialManager(config_dir=tmp_path)

        cm.store_credential("nebius", "api_key", "test-key-123")
        value = cm.get_credential("nebius", "api_key")

        assert value == "test-key-123"

    def test_multiple_providers(self, tmp_path: Path) -> None:
        """Test storing credentials for multiple providers."""
        cm = CredentialManager(config_dir=tmp_path)

        cm.store_credential("nebius", "api_key", "nebius-key")
        cm.store_credential("aws", "access_key", "aws-access")
        cm.store_credential("aws", "secret_key", "aws-secret")

        assert cm.get_credential("nebius", "api_key") == "nebius-key"
        assert cm.get_credential("aws", "access_key") == "aws-access"
        assert cm.get_credential("aws", "secret_key") == "aws-secret"

    def test_fallback_to_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test fallback to environment variables."""
        monkeypatch.setenv("TEST_API_KEY", "env-value")

        cm = CredentialManager(config_dir=tmp_path)
        value = cm.get_credential("test", "api_key", fallback_env="TEST_API_KEY")

        assert value == "env-value"

    def test_delete_specific_credential(self, tmp_path: Path) -> None:
        """Test deleting a specific credential."""
        cm = CredentialManager(config_dir=tmp_path)

        cm.store_credential("nebius", "api_key", "key1")
        cm.store_credential("nebius", "secret", "secret1")

        cm.delete_credential("nebius", "api_key")

        assert cm.get_credential("nebius", "api_key") is None
        assert cm.get_credential("nebius", "secret") == "secret1"

    def test_delete_all_provider_credentials(self, tmp_path: Path) -> None:
        """Test deleting all credentials for a provider."""
        cm = CredentialManager(config_dir=tmp_path)

        cm.store_credential("nebius", "api_key", "key1")
        cm.store_credential("nebius", "secret", "secret1")

        cm.delete_credential("nebius")

        assert cm.get_credential("nebius", "api_key") is None
        assert cm.get_credential("nebius", "secret") is None

    def test_list_providers(self, tmp_path: Path) -> None:
        """Test listing providers with credentials."""
        cm = CredentialManager(config_dir=tmp_path)

        cm.store_credential("nebius", "api_key", "key1")
        cm.store_credential("aws", "access_key", "key2")

        providers = cm.list_providers()
        assert set(providers) == {"nebius", "aws"}

    def test_validate_credentials(self, tmp_path: Path) -> None:
        """Test credential validation."""
        cm = CredentialManager(config_dir=tmp_path)

        cm.store_credential("nebius", "api_key", "valid-key")
        cm.store_credential("nebius", "empty", "")

        validation = cm.validate_credentials("nebius")
        assert validation["api_key"] is True
        assert validation["empty"] is False

    def test_encryption_persistence(self, tmp_path: Path) -> None:
        """Test that credentials persist and remain encrypted."""
        # Store credentials
        cm1 = CredentialManager(config_dir=tmp_path)
        cm1.store_credential("nebius", "api_key", "secret-value")

        # Create new instance (reload from disk)
        cm2 = CredentialManager(config_dir=tmp_path)
        value = cm2.get_credential("nebius", "api_key")

        assert value == "secret-value"

        # Check that file is encrypted (not plaintext)
        encrypted_content = cm1.credentials_file.read_text()
        assert "secret-value" not in encrypted_content

    def test_reset_credentials(self, tmp_path: Path) -> None:
        """Test resetting all credentials."""
        cm = CredentialManager(config_dir=tmp_path)

        cm.store_credential("nebius", "api_key", "key1")
        cm.reset()

        assert cm.get_credential("nebius", "api_key") is None
        assert cm.list_providers() == []


class TestPreferencesManager:
    """Tests for PreferencesManager."""

    def test_initialization_creates_defaults(self, tmp_path: Path) -> None:
        """Test that initialization creates default preferences."""
        pm = PreferencesManager(config_dir=tmp_path)
        prefs = pm.load()

        assert prefs.default_gpu_type == "L40S"
        assert prefs.default_cloud == "nebius"
        assert prefs.default_workflow == "unsloth"
        assert prefs.use_spot_instances is True

    def test_get_preference(self, tmp_path: Path) -> None:
        """Test getting a preference value."""
        pm = PreferencesManager(config_dir=tmp_path)
        value = pm.get("default_gpu_type")

        assert value == "L40S"

    def test_set_preference(self, tmp_path: Path) -> None:
        """Test setting a preference value."""
        pm = PreferencesManager(config_dir=tmp_path)

        pm.set("default_gpu_type", "H100")
        value = pm.get("default_gpu_type")

        assert value == "H100"

    def test_invalid_preference_key(self, tmp_path: Path) -> None:
        """Test setting an invalid preference key."""
        pm = PreferencesManager(config_dir=tmp_path)

        with pytest.raises(ConfigurationError, match="Invalid preference key"):
            pm.set("nonexistent_key", "value")

    def test_invalid_gpu_type(self, tmp_path: Path) -> None:
        """Test setting an invalid GPU type."""
        pm = PreferencesManager(config_dir=tmp_path)

        with pytest.raises(ConfigurationError):
            pm.set("default_gpu_type", "InvalidGPU")

    def test_preferences_persistence(self, tmp_path: Path) -> None:
        """Test that preferences persist across instances."""
        pm1 = PreferencesManager(config_dir=tmp_path)
        pm1.set("default_gpu_type", "H100")
        pm1.set("max_budget_per_job", 100.0)

        pm2 = PreferencesManager(config_dir=tmp_path)
        assert pm2.get("default_gpu_type") == "H100"
        assert pm2.get("max_budget_per_job") == 100.0

    def test_get_all_preferences(self, tmp_path: Path) -> None:
        """Test getting all preferences."""
        pm = PreferencesManager(config_dir=tmp_path)
        all_prefs = pm.get_all()

        assert "default_gpu_type" in all_prefs
        assert "default_cloud" in all_prefs
        assert "use_spot_instances" in all_prefs

    def test_reset_preferences(self, tmp_path: Path) -> None:
        """Test resetting preferences to defaults."""
        pm = PreferencesManager(config_dir=tmp_path)

        pm.set("default_gpu_type", "H100")
        pm.reset()

        assert pm.get("default_gpu_type") == "L40S"

    def test_config_versioning(self, tmp_path: Path) -> None:
        """Test that config version is saved."""
        pm = PreferencesManager(config_dir=tmp_path)
        pm.load()

        config_data = json.loads(pm.config_file.read_text())
        assert "version" in config_data
        assert config_data["version"] == "1.0"


class TestBudgetManager:
    """Tests for BudgetManager."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Test budget manager initialization."""
        bm = BudgetManager(config_dir=tmp_path)
        assert bm.config_dir == tmp_path
        assert bm.get_total_spending() == 0.0

    def test_record_spending(self, tmp_path: Path) -> None:
        """Test recording spending."""
        bm = BudgetManager(config_dir=tmp_path)

        bm.record_spending(
            job_id="job-1",
            amount=25.50,
            description="Training job",
            workflow="unsloth",
            gpu_type="L40S",
        )

        assert bm.get_total_spending() == 25.50

    def test_multiple_spending_records(self, tmp_path: Path) -> None:
        """Test multiple spending records."""
        bm = BudgetManager(config_dir=tmp_path)

        bm.record_spending("job-1", 10.0, "Job 1", "unsloth", "L40S")
        bm.record_spending("job-2", 20.0, "Job 2", "gpt-oss-lora", "H100")
        bm.record_spending("job-3", 15.0, "Job 3", "unsloth", "A100")

        assert bm.get_total_spending() == 45.0

    def test_spending_by_workflow(self, tmp_path: Path) -> None:
        """Test getting spending by workflow."""
        bm = BudgetManager(config_dir=tmp_path)

        bm.record_spending("job-1", 10.0, "Job 1", "unsloth", "L40S")
        bm.record_spending("job-2", 20.0, "Job 2", "gpt-oss-lora", "H100")
        bm.record_spending("job-3", 15.0, "Job 3", "unsloth", "A100")

        by_workflow = bm.get_spending_by_workflow()
        assert by_workflow["unsloth"] == 25.0
        assert by_workflow["gpt-oss-lora"] == 20.0

    def test_spending_by_gpu(self, tmp_path: Path) -> None:
        """Test getting spending by GPU type."""
        bm = BudgetManager(config_dir=tmp_path)

        bm.record_spending("job-1", 10.0, "Job 1", "unsloth", "L40S")
        bm.record_spending("job-2", 20.0, "Job 2", "gpt-oss-lora", "H100")
        bm.record_spending("job-3", 15.0, "Job 3", "unsloth", "L40S")

        by_gpu = bm.get_spending_by_gpu()
        assert by_gpu["L40S"] == 25.0
        assert by_gpu["H100"] == 20.0

    def test_check_budget_ok(self, tmp_path: Path) -> None:
        """Test budget check with acceptable cost."""
        bm = BudgetManager(config_dir=tmp_path)

        status = bm.check_budget(estimated_cost=50.0, max_cost=100.0)
        assert status == BudgetStatus.OK

    def test_check_budget_warning(self, tmp_path: Path) -> None:
        """Test budget check warning threshold."""
        bm = BudgetManager(config_dir=tmp_path)

        status = bm.check_budget(
            estimated_cost=85.0,
            max_cost=100.0,
            warning_threshold=0.8,
        )
        assert status == BudgetStatus.WARNING

    def test_check_budget_exceeded(self, tmp_path: Path) -> None:
        """Test budget exceeded error."""
        bm = BudgetManager(config_dir=tmp_path)

        with pytest.raises(BudgetExceededError):
            bm.check_budget(estimated_cost=150.0, max_cost=100.0)

    def test_check_budget_no_limit(self, tmp_path: Path) -> None:
        """Test budget check with no limit."""
        bm = BudgetManager(config_dir=tmp_path)

        status = bm.check_budget(estimated_cost=1000.0, max_cost=None)
        assert status == BudgetStatus.OK

    def test_require_approval_small_job(self, tmp_path: Path) -> None:
        """Test that small jobs don't require approval."""
        bm = BudgetManager(config_dir=tmp_path)

        requires = bm.require_approval(
            estimated_cost=5.0,
            max_cost=100.0,
            auto_approve_threshold=10.0,
        )
        assert requires is False

    def test_require_approval_large_job(self, tmp_path: Path) -> None:
        """Test that large jobs require approval."""
        bm = BudgetManager(config_dir=tmp_path)

        requires = bm.require_approval(
            estimated_cost=50.0,
            max_cost=100.0,
            auto_approve_threshold=10.0,
        )
        assert requires is False  # Within budget, no approval needed

    def test_require_approval_over_budget(self, tmp_path: Path) -> None:
        """Test that over-budget jobs require approval."""
        bm = BudgetManager(config_dir=tmp_path)

        requires = bm.require_approval(
            estimated_cost=150.0,
            max_cost=100.0,
            auto_approve_threshold=10.0,
        )
        assert requires is True

    def test_spending_summary(self, tmp_path: Path) -> None:
        """Test getting spending summary."""
        bm = BudgetManager(config_dir=tmp_path)

        bm.record_spending("job-1", 10.0, "Job 1", "unsloth", "L40S")
        bm.record_spending("job-2", 20.0, "Job 2", "gpt-oss-lora", "H100")

        summary = bm.get_spending_summary()
        assert summary["total"] == 30.0
        assert summary["num_jobs"] == 2
        assert "unsloth" in summary["by_workflow"]
        assert "L40S" in summary["by_gpu"]

    def test_clear_history(self, tmp_path: Path) -> None:
        """Test clearing spending history."""
        bm = BudgetManager(config_dir=tmp_path)

        bm.record_spending("job-1", 10.0, "Job 1", "unsloth", "L40S")
        bm.clear_history()

        assert bm.get_total_spending() == 0.0

    def test_export_records(self, tmp_path: Path) -> None:
        """Test exporting spending records."""
        bm = BudgetManager(config_dir=tmp_path)

        bm.record_spending("job-1", 10.0, "Job 1", "unsloth", "L40S")

        export_file = tmp_path / "export.json"
        bm.export_records(export_file)

        assert export_file.exists()
        data = json.loads(export_file.read_text())
        assert "records" in data
        assert "summary" in data
        assert len(data["records"]) == 1
