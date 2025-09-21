# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VibeML is a conversational interface that democratizes AI model training by providing natural language access to multi-cloud GPU resources through MCP (Model Context Protocol) and SkyPilot.

## Development Principles

### KISS (Keep It Simple, Stupid)
- Keep files small and focused on a single responsibility
- Break large modules into smaller, manageable components
- Prefer simple solutions over complex abstractions

### SOLID
- Build good, simple abstractions that follow SOLID principles
- Single Responsibility: Each class/function should have one reason to change
- Open/Closed: Open for extension, closed for modification
- Liskov Substitution: Derived classes must be substitutable for base classes
- Interface Segregation: Many specific interfaces are better than one general interface
- Dependency Inversion: Depend on abstractions, not concretions

### DBL (Don't Be Lazy)
- Write complete implementations, not placeholder code
- Add proper type hints and docstrings
- Handle edge cases properly
- Don't skip validation or error handling

### AOEH (Avoid Over-Exception Handling)
- Don't catch exceptions you can't handle meaningfully
- Let exceptions bubble up when appropriate
- Use specific exception types, not bare `except:`
- Log errors at the appropriate level

## Development Commands

```bash
# Project management with UV (package manager)
uv sync                          # Install dependencies
uv tool install vibeml          # Install as UV tool
uv run vibeml                   # Run the CLI

# Testing
nox -s tests                    # Run test suite
nox -s tests -- tests/test_main.py  # Run specific test
nox -s coverage                 # Generate coverage report
nox -s typeguard                # Run tests with runtime type checking

# Code quality
nox -s pre-commit              # Run all linting checks
nox -s mypy                    # Type checking
ruff check src/                # Linting
ruff format src/               # Auto-format code

# Documentation
nox -s docs                    # Build and serve docs with live reload
nox -s docs-build             # Build docs once
```

## Architecture

### Tech Stack
- **Python 3.9+** with modern async/await patterns
- **FastMCP**: MCP server implementation for AI assistant integration
- **SkyPilot**: Multi-cloud orchestration (AWS, GCP, Azure, Nebius)
- **UV**: Modern Python package management
- **Nox**: Task automation and testing across Python versions

### Core Architecture Flow
```
User Request → Claude/ChatGPT → MCP Protocol → VibeML FastMCP Server → SkyPilot → Nebius Cloud
```

### Project Structure (Cookiecutter Template)
```
src/vibeml/           # Main package code
├── __init__.py       # Package initialization
├── __main__.py       # CLI entry point (Click-based)
├── server.py         # FastMCP server implementation (to be created)
├── tasks.py          # SkyPilot task generation functions (to be created)
└── exceptions.py     # Custom exception hierarchy (to be created)

tests/                # Test suite
docs/                 # Sphinx documentation
noxfile.py           # Test automation configuration
```

## Implementation Strategy

### Phase 1: MVP - Nebius Cloud Integration
Focus on core training job launcher with Nebius as primary cloud provider.

#### Key Components

**FastMCP Server** (`src/vibeml/server.py`):
```python
from fastmcp.server import FastMCP
from fastmcp.server.middleware import make_async_background
import sky

mcp = FastMCP("vibeml")

@mcp.tool()
@make_async_background
async def launch_training(
    model: str,
    dataset: str,
    gpu_type: str = "H100"
) -> dict:
    """Launch training job on Nebius Cloud."""
    task = create_unsloth_task(model, dataset, gpu_type)
    cluster = await sky.launch_async(task)
    return {"cluster": cluster.name, "status": "launched"}
```

**Task Generation** (`src/vibeml/tasks.py`):
```python
def create_unsloth_task(
    model: str,
    dataset: str,
    gpu_type: str = "H100"
) -> sky.Task:
    """Generate SkyPilot Task for Nebius Cloud."""
    return sky.Task(
        name=f"vibeml-{model.replace('/', '-')}",
        setup=setup_script,
        run=training_script,
        resources=sky.Resources(
            cloud="nebius",  # Changed from runpod
            accelerators=f"{gpu_type}:1",
            use_spot=True,
            disk_size=100
        )
    )
```

**Exception Hierarchy** (`src/vibeml/exceptions.py`):
```python
class VibeMLError(Exception):
    """Base exception for VibeML."""
    pass

class SkyPilotError(VibeMLError):
    """SkyPilot operation failures."""
    pass

class NebiusError(VibeMLError):
    """Nebius-specific errors."""
    pass
```

### Nebius Cloud Specifics

**GPU Types Available**:
- H100 (80GB) - Premium performance
- A100 (40GB/80GB) - Production workloads
- L40S (48GB) - Cost-effective training
- RTX 4090 (24GB) - Development/testing

**Resource Configuration**:
```python
sky.Resources(
    cloud="nebius",
    region="eu-north1",  # Primary region
    accelerators="H100:1",
    instance_type="gpu-h100-sxm",  # Nebius-specific instance
    disk_size=100,
    use_spot=True
)
```

### Critical Implementation Notes

1. **Async Handling**: Use FastMCP's `make_async_background` decorator for SkyPilot operations
2. **Direct Task Objects**: Create SkyPilot Task objects directly, no YAML templates
3. **Cost Guards**: Validate GPU selection and estimate costs before launch
4. **Error Messages**: Provide clear, actionable error messages for common failures

## Testing Strategy

```python
# tests/test_tasks.py
def test_create_unsloth_task():
    """Test Nebius task generation."""
    task = create_unsloth_task(
        model="meta-llama/Llama-3.2-1B",
        dataset="tatsu-lab/alpaca",
        gpu_type="L40S"
    )
    assert task.resources.cloud == "nebius"
    assert "L40S:1" in task.resources.accelerators

# tests/test_server.py
@pytest.mark.asyncio
async def test_launch_training():
    """Test training job launch."""
    with mock.patch("sky.launch_async") as mock_launch:
        result = await launch_training(
            model="meta-llama/Llama-3.2-1B",
            dataset="tatsu-lab/alpaca"
        )
        assert result["status"] == "launched"
```

## Workflow Registry Pattern

```python
# src/vibeml/tasks.py
WORKFLOWS = {
    "unsloth": create_unsloth_task,
    "lora": create_lora_task,  # Future
    "full": create_full_training_task,  # Future
}

def get_workflow(workflow_type: str):
    """Get workflow function by type."""
    if workflow_type not in WORKFLOWS:
        raise ValueError(f"Unknown workflow: {workflow_type}")
    return WORKFLOWS[workflow_type]
```

## MVP Success Criteria

- Successfully launch Unsloth training jobs on Nebius Cloud
- Handle common failures gracefully (no GPU availability, quota limits)
- Response time <30 seconds for job launch
- Clear cost estimation before launch
- Proper async operation without blocking

## Common Pitfalls to Avoid

1. **Don't use subprocess** - Use SkyPilot Python API directly
2. **Don't generate YAML** - Create Task objects programmatically
3. **Don't catch all exceptions** - Let SkyPilot errors bubble up with context
4. **Don't hardcode regions** - Make Nebius regions configurable
5. **Don't skip validation** - Always validate model/dataset compatibility