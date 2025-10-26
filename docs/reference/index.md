# API Reference

Complete reference documentation for VibeML's CLI commands, Python API, and MCP server.

## Quick Links

- **[CLI Commands](cli.md)** - Command-line interface reference
- **[Data Models](api/models.md)** - Pydantic models for requests and responses
- **[Cloud Integration](api/cloud.md)** - Cost estimation and cloud provider interfaces
- **[Configuration](api/config.md)** - Credential and preference management
- **[Validation](api/validation.md)** - Model, dataset, and resource validators
- **[Scripts](api/scripts.md)** - Training script generation
- **[Exceptions](api/exceptions.md)** - Error handling and custom exceptions
- **[MCP Server](mcp.md)** - Model Context Protocol server endpoints

## API Design Principles

VibeML's API follows these principles:

### Type Safety

All public APIs use Pydantic models for validation:

```python
from vibeml.models import TrainingRequest

request = TrainingRequest(
    model="meta-llama/Llama-3.2-1B",
    dataset="tatsu-lab/alpaca",
    workflow="unsloth",
    max_cost=15.0
)  # Validates at construction
```

### Async-First

Core functions are async for non-blocking operations:

```python
from vibeml import launch_training

result = await launch_training(
    model="meta-llama/Llama-3.2-1B",
    dataset="tatsu-lab/alpaca"
)
```

### Error Handling

Comprehensive exception hierarchy with context:

```python
from vibeml.exceptions import BudgetExceededError, retry_with_backoff

try:
    result = await launch_training(...)
except BudgetExceededError as e:
    print(f"Cost: {e.context['estimated_cost']}")
    print(f"Limit: {e.context['max_cost']}")
    print(f"Suggestion: {e.recovery_suggestion}")
```

### Immutable Data

Cost estimates and other critical data use frozen models:

```python
from vibeml.models import CostEstimate

estimate = CostEstimate(
    hourly_rate=1.2,
    estimated_duration_hours=10,
    min_cost=9.6,
    max_cost=15.6,
    expected_cost=12.0
)

# estimate.hourly_rate = 2.0  # Raises FrozenInstanceError
```

## Common Patterns

### Validating Models

```python
from vibeml.validation import ModelValidator

validator = ModelValidator()

# Check if model exists
is_valid = validator.validate_model("meta-llama/Llama-3.2-1B")

# Get GPU requirements
memory_gb = validator.calculate_gpu_memory_required(
    model_id="meta-llama/Llama-3.2-7B",
    batch_size=2,
    use_quantization=True
)

# Get GPU recommendation
gpu_type = validator.recommend_gpu_type(
    model_id="meta-llama/Llama-3.2-7B",
    use_quantization=True
)
```

### Managing Credentials

```python
from vibeml.config import CredentialManager

manager = CredentialManager()

# Store encrypted credential
manager.store_credential(
    provider="nebius",
    credential_type="api_key",
    value="your-key"
)

# Retrieve with fallback
api_key = manager.get_credential(
    provider="nebius",
    credential_type="api_key",
    fallback_env="NEBIUS_API_KEY"
)
```

### Estimating Costs

```python
from vibeml.cloud import CostEstimator

estimator = CostEstimator()

# Estimate training cost
cost = estimator.estimate_cost(
    gpu_type="L40S",
    estimated_hours=12.0,
    gpu_count=1,
    use_spot=True,
    cloud="nebius"
)

print(f"Expected: ${cost.expected_cost:.2f}")
print(f"Range: ${cost.min_cost:.2f} - ${cost.max_cost:.2f}")
```

### Generating Scripts

```python
from vibeml.scripts import ScriptGenerator

generator = ScriptGenerator()

script = generator.generate_script(
    workflow="unsloth",
    model="meta-llama/Llama-3.2-1B",
    dataset="tatsu-lab/alpaca",
    max_steps=1000,
    learning_rate=2e-4,
    lora_r=16
)

with open("train.py", "w") as f:
    f.write(script)
```

## Version Compatibility

| VibeML Version | Python | FastMCP | SkyPilot | Pydantic |
|----------------|--------|---------|----------|----------|
| 0.1.x          | ≥3.10  | ≥0.1.0  | ≥0.8.1   | ≥2.0.0   |

## Migration Guides

- **Upgrading to 0.2.x** - Coming soon

## API Stability

- **Stable**: `vibeml.models`, `vibeml.cloud`, `vibeml.config`
- **Beta**: `vibeml.scripts`, `vibeml.validation`
- **Experimental**: MCP server endpoints

Breaking changes will be documented in the [changelog](https://github.com/prassanna-ravishankar/vibeml/releases).
