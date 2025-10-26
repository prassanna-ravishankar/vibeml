"""User preferences management."""

import json
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from ..exceptions import ConfigurationError


class UserPreferences(BaseModel):
    """User preferences for VibeML."""

    default_gpu_type: str = Field(default="L40S", description="Default GPU type")
    default_cloud: str = Field(default="nebius", description="Default cloud provider")
    default_region: Optional[str] = Field(
        default=None, description="Default region (auto-selected if None)"
    )
    default_workflow: str = Field(default="unsloth", description="Default workflow type")
    max_budget_per_job: Optional[float] = Field(
        default=None, ge=0, description="Maximum budget per job in USD"
    )
    use_spot_instances: bool = Field(
        default=True, description="Prefer spot instances for cost savings"
    )
    auto_terminate_idle: bool = Field(
        default=True, description="Automatically terminate idle clusters"
    )
    idle_timeout_minutes: int = Field(
        default=60, ge=5, description="Minutes before terminating idle clusters"
    )
    enable_cost_alerts: bool = Field(
        default=True, description="Enable cost alert notifications"
    )
    cost_alert_threshold: float = Field(
        default=50.0, ge=0, description="Cost threshold for alerts in USD"
    )

    @field_validator("default_gpu_type")
    @classmethod
    def validate_gpu_type(cls, v: str) -> str:
        """Validate GPU type."""
        valid_gpus = {"L40S", "RTX4090", "H100", "A100"}
        if v not in valid_gpus:
            raise ValueError(f"GPU type must be one of {valid_gpus}")
        return v

    @field_validator("default_cloud")
    @classmethod
    def validate_cloud(cls, v: str) -> str:
        """Validate cloud provider."""
        valid_clouds = {"nebius", "aws", "gcp", "azure"}
        if v not in valid_clouds:
            raise ValueError(f"Cloud must be one of {valid_clouds}")
        return v

    @field_validator("default_workflow")
    @classmethod
    def validate_workflow(cls, v: str) -> str:
        """Validate workflow type."""
        valid_workflows = {"unsloth", "gpt-oss-lora", "gpt-oss-full"}
        if v not in valid_workflows:
            raise ValueError(f"Workflow must be one of {valid_workflows}")
        return v


class PreferencesManager:
    """Manages user preferences with versioning support."""

    CONFIG_VERSION = "1.0"

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize PreferencesManager.

        Args:
            config_dir: Directory for storing preferences (default: ~/.vibeml)
        """
        self.config_dir = config_dir or Path.home() / ".vibeml"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.config_dir / "config.json"
        self._preferences: Optional[UserPreferences] = None

    def load(self) -> UserPreferences:
        """Load user preferences from file.

        Returns:
            UserPreferences instance

        Raises:
            ConfigurationError: If loading fails
        """
        if self._preferences is not None:
            return self._preferences

        if not self.config_file.exists():
            # Create default preferences
            self._preferences = UserPreferences()
            self.save()
            return self._preferences

        try:
            data = json.loads(self.config_file.read_text())

            # Check version and migrate if needed
            file_version = data.get("version", "1.0")
            if file_version != self.CONFIG_VERSION:
                data = self._migrate_config(data, file_version)

            # Extract preferences (skip metadata)
            prefs_data = {k: v for k, v in data.items() if k not in ["version", "last_updated"]}
            self._preferences = UserPreferences(**prefs_data)
            return self._preferences

        except Exception as e:
            raise ConfigurationError(
                "Failed to load preferences",
                technical_details=str(e),
                recovery_suggestion="Check config file at ~/.vibeml/config.json or reset preferences",
            )

    def save(self) -> None:
        """Save user preferences to file.

        Raises:
            ConfigurationError: If saving fails
        """
        if self._preferences is None:
            return

        try:
            # Add metadata
            data = self._preferences.model_dump()
            data["version"] = self.CONFIG_VERSION
            data["last_updated"] = datetime.now().isoformat()

            self.config_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            raise ConfigurationError(
                "Failed to save preferences",
                technical_details=str(e),
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific preference value.

        Args:
            key: Preference key
            default: Default value if key not found

        Returns:
            Preference value or default
        """
        prefs = self.load()
        return getattr(prefs, key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a specific preference value.

        Args:
            key: Preference key
            value: New value

        Raises:
            ConfigurationError: If key is invalid or value is invalid
        """
        prefs = self.load()

        if not hasattr(prefs, key):
            raise ConfigurationError(
                f"Invalid preference key: {key}",
                config_key=key,
            )

        try:
            setattr(prefs, key, value)
            # Re-validate
            UserPreferences(**prefs.model_dump())
            self._preferences = prefs
            self.save()
        except Exception as e:
            raise ConfigurationError(
                f"Invalid value for preference {key}",
                technical_details=str(e),
                config_key=key,
            )

    def reset(self) -> None:
        """Reset preferences to defaults."""
        self._preferences = UserPreferences()
        self.save()

    def get_all(self) -> Dict[str, Any]:
        """Get all preferences as a dictionary.

        Returns:
            Dictionary of all preferences
        """
        prefs = self.load()
        return prefs.model_dump()

    def _migrate_config(self, data: Dict[str, Any], from_version: str) -> Dict[str, Any]:
        """Migrate configuration from old version to current.

        Args:
            data: Configuration data
            from_version: Source version

        Returns:
            Migrated configuration data
        """
        # Currently at version 1.0, no migrations needed
        # Future versions would add migration logic here
        return data
