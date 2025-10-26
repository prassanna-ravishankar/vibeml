# Configuration

Configure VibeML to work with your cloud providers and set your preferences.

## Cloud Provider Credentials

VibeML uses encrypted credential storage for security. All credentials are stored at `~/.vibeml/credentials.enc`.

### Nebius Cloud (Recommended)

```bash
# Store your Nebius API key
export NEBIUS_API_KEY="your-api-key-here"

# Or use the credential manager
vibeml config set-credential nebius api_key "your-api-key-here"
```

Get your API key from [Nebius Console](https://console.nebius.com/).

### AWS

```bash
# Option 1: Use AWS CLI
aws configure

# Option 2: Environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-west-2"

# Option 3: VibeML credential manager
vibeml config set-credential aws access_key_id "your-access-key"
vibeml config set-credential aws secret_access_key "your-secret-key"
```

### Google Cloud Platform

```bash
# Authenticate with gcloud
gcloud auth application-default login

# Or use service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### Azure

```bash
# Login via Azure CLI
az login

# Or use service principal
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"
export AZURE_TENANT_ID="your-tenant-id"
```

## User Preferences

Set default values for common parameters:

```bash
# Set default cloud provider
vibeml config set default_cloud nebius

# Set default GPU type
vibeml config set default_gpu_type L40S

# Enable spot instances by default
vibeml config set use_spot true

# Set default budget limit
vibeml config set max_cost 50.0

# Set default workflow
vibeml config set default_workflow unsloth
```

View your configuration:

```bash
vibeml config show
```

## Budget Management

Set up budget tracking and alerts:

```bash
# Set monthly budget limit
vibeml config set-budget monthly 500.0

# Set per-job budget limit
vibeml config set-budget per_job 50.0

# Enable budget alerts at 80% threshold
vibeml config set budget_alert_threshold 0.8
```

View spending:

```bash
# Show current month spending
vibeml config show-spending

# Show spending history
vibeml config show-spending --history
```

## Configuration File

VibeML stores preferences in `~/.vibeml/config.json`:

```json
{
  "version": "1.0",
  "preferences": {
    "default_cloud": "nebius",
    "default_gpu_type": "L40S",
    "use_spot": true,
    "max_cost": 50.0,
    "default_workflow": "unsloth"
  },
  "budget": {
    "monthly_limit": 500.0,
    "per_job_limit": 50.0,
    "alert_threshold": 0.8,
    "current_month_spending": 0.0
  }
}
```

## Security Best Practices

### Credential Storage

VibeML encrypts credentials using Fernet symmetric encryption:

- Encryption key stored at `~/.vibeml/.key` with 0600 permissions
- Credentials file at `~/.vibeml/credentials.enc` is also restricted
- Environment variables take precedence over stored credentials

### Recommended Setup

1. **Use environment variables** for CI/CD and production
2. **Use credential manager** for local development
3. **Never commit credentials** to version control
4. **Rotate keys regularly** following your organization's policy

### Accessing Credentials

```python
from vibeml.config import CredentialManager

manager = CredentialManager()

# Retrieve with environment fallback
api_key = manager.get_credential(
    provider="nebius",
    credential_type="api_key",
    fallback_env="NEBIUS_API_KEY"
)
```

## MCP Server Configuration

To use VibeML with Claude Code or other MCP clients, create `.mcp.json`:

```json
{
  "mcpServers": {
    "vibeml": {
      "command": "vibeml",
      "args": ["--mcp"],
      "env": {
        "NEBIUS_API_KEY": "${NEBIUS_API_KEY}",
        "VIBEML_DEFAULT_CLOUD": "nebius",
        "VIBEML_DEFAULT_GPU": "L40S",
        "VIBEML_MAX_COST": "50.0"
      }
    }
  }
}
```

## Advanced Configuration

### Custom Pricing

Override default GPU pricing:

```python
from vibeml.cloud import CostEstimator

estimator = CostEstimator()
estimator.NEBIUS_PRICING["L40S"].hourly_rate = 1.5  # Custom rate
```

### Custom Workflows

Register a custom training workflow:

```python
from vibeml import tasks

def my_workflow(model: str, dataset: str, **kwargs):
    # Custom task creation logic
    pass

tasks.WORKFLOWS["my-workflow"] = my_workflow
```

### Logging Configuration

Set log level via environment variable:

```bash
export VIBEML_LOG_LEVEL=DEBUG
```

Or in code:

```python
import logging
logging.getLogger("vibeml").setLevel(logging.DEBUG)
```

## Verification

Test your configuration:

```bash
# Dry run a training job
vibeml launch-training \
  --model meta-llama/Llama-3.2-1B \
  --dataset tatsu-lab/alpaca \
  --dry-run

# Verify cloud credentials
vibeml config verify-credentials
```

## Troubleshooting

### Credential Errors

!!! error "ConfigurationError: Credential not found"
    The requested credential is not configured.

**Solution**: Set the credential or environment variable:

```bash
export NEBIUS_API_KEY="your-key"
```

### Permission Denied

!!! error "PermissionError: Cannot access ~/.vibeml/credentials.enc"
    File permissions are incorrect.

**Solution**: Fix permissions:

```bash
chmod 600 ~/.vibeml/credentials.enc
chmod 600 ~/.vibeml/.key
```

### Budget Exceeded

!!! error "BudgetExceededError: Monthly budget limit reached"
    You've exceeded your budget limit.

**Solution**: Increase budget or wait until next month:

```bash
vibeml config set-budget monthly 1000.0
```

## Next Steps

- **[Launch your first job](quickstart.md)** with your configured credentials
- **[Learn about cost management](../concepts/cost-management.md)**
- **[Manage credentials securely](../how-to/manage-credentials.md)**
