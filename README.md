# VibeML

**Conversational AI model training on multi-cloud GPU resources**

[![PyPI](https://img.shields.io/pypi/v/vibeml.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/vibeml.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/vibeml)][pypi status]
[![License](https://img.shields.io/pypi/l/vibeml)][license]

[![Read the documentation at https://vibeml.readthedocs.io/](https://img.shields.io/readthedocs/vibeml/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/prassanna-ravishankar/vibeml/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/prassanna-ravishankar/vibeml/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Ruff codestyle][ruff badge]][ruff project]

[pypi status]: https://pypi.org/project/vibeml/
[read the docs]: https://vibeml.readthedocs.io/
[tests]: https://github.com/prassanna-ravishankar/vibeml/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/prassanna-ravishankar/vibeml
[pre-commit]: https://github.com/pre-commit/pre-commit
[ruff badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff project]: https://github.com/charliermarsh/ruff

VibeML democratizes AI model training by providing natural language access to multi-cloud GPU resources through MCP (Model Context Protocol) and SkyPilot.

## ✨ Features

- **Conversational Interface**: Launch training jobs using natural language through Claude or ChatGPT
- **Multi-Cloud Support**: Seamlessly deploy across AWS, GCP, Azure, and Nebius Cloud
- **Cost Optimization**: Automatic GPU selection, spot instance management, and budget tracking
- **Pre-configured Workflows**: Unsloth, LoRA, and full fine-tuning templates ready to use
- **MCP Integration**: First-class support for Model Context Protocol
- **Production Ready**: Comprehensive error handling, validation, and monitoring

## 🚀 Quick Start

```bash
# Install VibeML
uv tool install vibeml

# Launch your first training job
vibeml launch-training \
  --model meta-llama/Llama-3.2-1B \
  --dataset tatsu-lab/alpaca \
  --workflow unsloth \
  --gpu-type L40S \
  --max-cost 15.0
```

Or use with Claude Code via MCP:
```
"Train Llama-3.2-1B on the Alpaca dataset using Nebius L40S GPUs with a $15 budget"
```

## 📋 Requirements

- Python 3.10 or higher
- Cloud provider account (Nebius, AWS, GCP, or Azure)
- HuggingFace account (for accessing models and datasets)

## Installation

You can install _vibeml_ via [pip] from [PyPI]. The package is distributed as a pure Python package, but also with pre-compiled wheels for major platforms, which include performance optimizations.

```console
$ pip install vibeml
```

The pre-compiled wheels are built using `mypyc` and will be used automatically if your platform is supported. You can check the files on PyPI to see the list of available wheels.

## Usage

Please see the [Command-line Reference] for details.

## 📚 Documentation

Comprehensive documentation is available at [vibeml.readthedocs.io](https://vibeml.readthedocs.io/):

- **[Get Started](https://vibeml.readthedocs.io/en/latest/get-started/)** - Installation and quick start guide
- **[Concepts](https://vibeml.readthedocs.io/en/latest/concepts/)** - Architecture and core concepts
- **[How-to Guides](https://vibeml.readthedocs.io/en/latest/how-to/)** - Practical task-oriented guides
- **[API Reference](https://vibeml.readthedocs.io/en/latest/reference/)** - Complete CLI and Python API documentation

### Building Documentation Locally

```bash
# Install documentation dependencies
uv pip install -r docs/requirements.txt

# Serve documentation with live reload
mkdocs serve

# Build static site
mkdocs build
```

The documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material theme](https://squidfunk.github.io/mkdocs-material/).

## 🛠️ Development

To contribute to this project, please see the [Contributor Guide].

### Development Setup

```bash
# Clone the repository
git clone https://github.com/prassanna-ravishankar/vibeml.git
cd vibeml

# Install development dependencies
uv sync --group dev

# Run tests
nox -s tests

# Run linting
nox -s pre-commit

# Run type checking
nox -s mypy
```

### Documentation Workflow

1. Make changes to documentation in `docs/`
2. Preview locally with `mkdocs serve`
3. Pre-commit hooks automatically validate documentation
4. GitHub Actions builds and deploys on push to main
5. ReadTheDocs builds on all commits for versioned docs

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_vibeml_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [uv hypermodern python cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[uv hypermodern python cookiecutter]: https://github.com/bosd/cookiecutter-uv-hypermodern-python
[file an issue]: https://github.com/prassanna-ravishankar/vibeml/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/prassanna-ravishankar/vibeml/blob/main/LICENSE
[contributor guide]: https://github.com/prassanna-ravishankar/vibeml/blob/main/CONTRIBUTING.md
[command-line reference]: https://vibeml.readthedocs.io/en/latest/usage.html
