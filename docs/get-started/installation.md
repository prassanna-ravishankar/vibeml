# Installation

This guide will help you install VibeML and its dependencies.

## Method 1: UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package manager that provides the best installation experience.

### Install UV

=== "macOS/Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### Install VibeML

```bash
# Install as a UV tool (recommended)
uv tool install vibeml

# Verify installation
vibeml --version
```

## Method 2: pip

```bash
# Install from PyPI
pip install vibeml

# Verify installation
vibeml --version
```

## Method 3: From Source

For development or the latest features:

```bash
# Clone the repository
git clone https://github.com/prassanna-ravishankar/vibeml.git
cd vibeml

# Install with UV
uv sync

# Or with pip
pip install -e .

# Verify installation
vibeml --version
```

## Verify Installation

Check that VibeML is properly installed:

```bash
vibeml --help
```

You should see output similar to:

```
Usage: vibeml [OPTIONS] COMMAND [ARGS]...

  VibeML: Conversational AI model training on multi-cloud GPUs

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  launch-training  Launch a model training job
  list-jobs        List active training jobs
  get-job-status   Get status of a training job
  terminate-job    Terminate a running job
```

## Install Cloud Provider CLIs

Depending on your cloud provider, you may need additional tools:

=== "Nebius"

    ```bash
    # Nebius is supported directly through SkyPilot
    # No additional CLI needed
    ```

=== "AWS"

    ```bash
    # Install AWS CLI
    pip install awscli

    # Configure credentials
    aws configure
    ```

=== "GCP"

    ```bash
    # Install Google Cloud SDK
    # See: https://cloud.google.com/sdk/docs/install
    ```

=== "Azure"

    ```bash
    # Install Azure CLI
    pip install azure-cli

    # Login
    az login
    ```

## Optional: MCP Server Setup

To use VibeML with Claude Code or other MCP-enabled assistants, add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "vibeml": {
      "command": "vibeml",
      "args": ["--mcp"],
      "env": {
        "NEBIUS_API_KEY": "your-key-here"
      }
    }
  }
}
```

## Troubleshooting

### Python Version Error

!!! error "Python 3.10+ Required"
    VibeML requires Python 3.10 or higher due to FastMCP dependencies.

**Solution**: Upgrade Python or use a virtual environment:

```bash
# Create Python 3.10+ environment
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install VibeML
uv pip install vibeml
```

### UV Not Found

!!! error "UV command not found"
    The `uv` command is not in your PATH.

**Solution**: Add UV to your PATH or use the full path:

```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.cargo/bin:$PATH"

# Reload shell configuration
source ~/.bashrc  # or source ~/.zshrc
```

### Import Errors

!!! error "ModuleNotFoundError: No module named 'fastmcp'"
    Dependencies are not properly installed.

**Solution**: Reinstall with all dependencies:

```bash
pip install --force-reinstall vibeml
```

## Next Steps

Now that VibeML is installed:

1. **[Configure](configuration.md)** your cloud credentials
2. **[Launch your first job](quickstart.md)**
3. Explore the **[CLI reference](../reference/cli.md)**
