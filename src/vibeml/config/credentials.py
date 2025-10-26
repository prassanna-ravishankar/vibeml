"""Secure credential storage and management."""

import os
from pathlib import Path
from typing import Optional, Dict
import json

from cryptography.fernet import Fernet

from ..exceptions import ConfigurationError


class CredentialManager:
    """Manages encrypted storage and retrieval of cloud credentials."""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize CredentialManager.

        Args:
            config_dir: Directory for storing credentials (default: ~/.vibeml)
        """
        self.config_dir = config_dir or Path.home() / ".vibeml"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.credentials_file = self.config_dir / "credentials.enc"
        self.key_file = self.config_dir / ".key"

        # Initialize encryption key
        self._ensure_key()

    def _ensure_key(self) -> None:
        """Ensure encryption key exists, create if needed."""
        if not self.key_file.exists():
            # Generate a new encryption key
            key = Fernet.generate_key()
            self.key_file.write_bytes(key)
            self.key_file.chmod(0o600)  # Restrict permissions

    def _get_cipher(self) -> Fernet:
        """Get Fernet cipher for encryption/decryption.

        Returns:
            Fernet cipher instance
        """
        key = self.key_file.read_bytes()
        return Fernet(key)

    def store_credential(
        self,
        provider: str,
        credential_type: str,
        value: str,
    ) -> None:
        """Store an encrypted credential.

        Args:
            provider: Cloud provider name (nebius, aws, gcp, azure)
            credential_type: Type of credential (api_key, access_key, etc.)
            value: Credential value

        Raises:
            ConfigurationError: If credential storage fails
        """
        try:
            # Load existing credentials
            credentials = self._load_credentials()

            # Update credentials
            if provider not in credentials:
                credentials[provider] = {}
            credentials[provider][credential_type] = value

            # Save encrypted credentials
            self._save_credentials(credentials)

        except Exception as e:
            raise ConfigurationError(
                f"Failed to store credential for {provider}",
                technical_details=str(e),
            )

    def get_credential(
        self,
        provider: str,
        credential_type: str,
        fallback_env: Optional[str] = None,
    ) -> Optional[str]:
        """Retrieve a credential.

        Args:
            provider: Cloud provider name
            credential_type: Type of credential
            fallback_env: Environment variable name to check as fallback

        Returns:
            Credential value or None if not found
        """
        # First try environment variable if specified
        if fallback_env:
            env_value = os.getenv(fallback_env)
            if env_value:
                return env_value

        # Try encrypted storage
        credentials = self._load_credentials()
        return credentials.get(provider, {}).get(credential_type)

    def delete_credential(
        self,
        provider: str,
        credential_type: Optional[str] = None,
    ) -> None:
        """Delete a credential or all credentials for a provider.

        Args:
            provider: Cloud provider name
            credential_type: Specific credential type to delete (None = delete all)
        """
        credentials = self._load_credentials()

        if provider in credentials:
            if credential_type:
                credentials[provider].pop(credential_type, None)
                if not credentials[provider]:
                    del credentials[provider]
            else:
                del credentials[provider]

            self._save_credentials(credentials)

    def list_providers(self) -> list[str]:
        """List all providers with stored credentials.

        Returns:
            List of provider names
        """
        credentials = self._load_credentials()
        return list(credentials.keys())

    def validate_credentials(self, provider: str) -> Dict[str, bool]:
        """Validate credentials for a provider.

        Args:
            provider: Cloud provider name

        Returns:
            Dictionary mapping credential types to validation status
        """
        credentials = self._load_credentials()
        provider_creds = credentials.get(provider, {})

        validation = {}
        for cred_type, value in provider_creds.items():
            # Basic validation - check non-empty
            validation[cred_type] = bool(value and len(value) > 0)

        return validation

    def _load_credentials(self) -> Dict[str, Dict[str, str]]:
        """Load and decrypt credentials from file.

        Returns:
            Dictionary of credentials by provider
        """
        if not self.credentials_file.exists():
            return {}

        try:
            cipher = self._get_cipher()
            encrypted_data = self.credentials_file.read_bytes()
            decrypted_data = cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        except Exception as e:
            raise ConfigurationError(
                "Failed to load credentials",
                technical_details=str(e),
                recovery_suggestion="Credentials file may be corrupted. Consider resetting credentials.",
            )

    def _save_credentials(self, credentials: Dict[str, Dict[str, str]]) -> None:
        """Encrypt and save credentials to file.

        Args:
            credentials: Credentials dictionary to save
        """
        try:
            cipher = self._get_cipher()
            json_data = json.dumps(credentials).encode()
            encrypted_data = cipher.encrypt(json_data)
            self.credentials_file.write_bytes(encrypted_data)
            self.credentials_file.chmod(0o600)  # Restrict permissions
        except Exception as e:
            raise ConfigurationError(
                "Failed to save credentials",
                technical_details=str(e),
            )

    def reset(self) -> None:
        """Reset all credentials and encryption keys."""
        if self.credentials_file.exists():
            self.credentials_file.unlink()
        if self.key_file.exists():
            self.key_file.unlink()
        self._ensure_key()
