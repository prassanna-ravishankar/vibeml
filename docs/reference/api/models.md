# Data Models

VibeML uses Pydantic v2 for all data models, providing runtime validation and type safety.

## Core Models

### TrainingRequest

::: vibeml.models.TrainingRequest
    options:
      show_root_heading: true
      show_source: true
      members_order: source

### JobHandle

::: vibeml.models.JobHandle
    options:
      show_root_heading: true
      show_source: true
      members_order: source

### CostEstimate

::: vibeml.models.CostEstimate
    options:
      show_root_heading: true
      show_source: true
      members_order: source

### WorkflowMetadata

::: vibeml.models.WorkflowMetadata
    options:
      show_root_heading: true
      show_source: true
      members_order: source

## Enumerations

### JobStatus

::: vibeml.models.JobStatus
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Creating a Training Request

```python
from vibeml.models import TrainingRequest

# Basic request
request = TrainingRequest(
    model="meta-llama/Llama-3.2-1B",
    dataset="tatsu-lab/alpaca"
)

# With options
request = TrainingRequest(
    model="meta-llama/Llama-3.2-7B",
    dataset="tatsu-lab/alpaca",
    workflow="unsloth",
    gpu_type="L40S",
    cloud="nebius",
    max_cost=20.0,
    hyperparameters={
        "max_steps": 1000,
        "learning_rate": 2e-4,
        "lora_r": 16
    }
)

# Validation happens automatically
try:
    invalid = TrainingRequest(
        model="invalid",  # No "/" in model name
        dataset="tatsu-lab/alpaca"
    )
except ValidationError as e:
    print(e.errors())
```

### Working with Job Handles

```python
from vibeml.models import JobHandle, JobStatus, CostEstimate
from datetime import datetime, UTC

# Create a job handle
job = JobHandle(
    job_id="vibeml-llama-20251026",
    cluster_name="vibeml-llama-20251026",
    status=JobStatus.RUNNING,
    model="meta-llama/Llama-3.2-1B",
    dataset="tatsu-lab/alpaca",
    workflow="unsloth",
    gpu_type="L40S",
    cost_estimate=CostEstimate(
        hourly_rate=1.008,
        estimated_duration_hours=12.0,
        min_cost=9.67,
        max_cost=15.70,
        expected_cost=12.10
    ),
    created_at=datetime.now(UTC)
)

# Serialize to JSON
job_json = job.model_dump_json(indent=2)

# Deserialize from JSON
job_restored = JobHandle.model_validate_json(job_json)
```

### Cost Estimates

```python
from vibeml.models import CostEstimate

# Create cost estimate (immutable)
estimate = CostEstimate(
    hourly_rate=1.2,
    estimated_duration_hours=10.0,
    min_cost=9.6,   # 80% of expected
    max_cost=15.6,  # 130% of expected
    expected_cost=12.0
)

# Cannot modify frozen model
try:
    estimate.hourly_rate = 2.0
except ValidationError:
    print("Cost estimates are immutable")

# Create new estimate with changes
new_estimate = estimate.model_copy(
    update={"hourly_rate": 2.0}
)
```

## Validation Rules

### Model Validation

- Model must be in format `org/name` (e.g., `meta-llama/Llama-3.2-1B`)
- Or one of the size identifiers: `20b`, `120b`

### Cost Validation

- All cost values must be >= 0
- `hourly_rate` must be > 0
- `min_cost` < `expected_cost` < `max_cost`

### Job Handle Validation

- `job_id` and `cluster_name` must be non-empty strings
- `created_at` must be timezone-aware datetime
- `status` must be a valid `JobStatus` enum value

## Model Configuration

All models use Pydantic v2 `ConfigDict`:

```python
from pydantic import BaseModel, ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,           # Immutable
        validate_assignment=True,  # Validate on field assignment
        str_strip_whitespace=True,  # Strip whitespace from strings
    )
```

## JSON Schema

Generate JSON schema for API documentation:

```python
from vibeml.models import TrainingRequest

schema = TrainingRequest.model_json_schema()
print(json.dumps(schema, indent=2))
```

Output:

```json
{
  "title": "TrainingRequest",
  "type": "object",
  "properties": {
    "model": {
      "title": "Model",
      "type": "string",
      "minLength": 1
    },
    "dataset": {
      "title": "Dataset",
      "type": "string",
      "minLength": 1
    },
    ...
  },
  "required": ["model", "dataset"]
}
```

## See Also

- [Cloud Integration API](cloud.md) - Cost estimation using these models
- [Validation API](validation.md) - Model and dataset validation
- [MCP Server](../mcp.md) - MCP endpoints using these models
