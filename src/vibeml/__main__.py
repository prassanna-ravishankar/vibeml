"""Command-line interface."""

import click


@click.group()
@click.version_option()
def cli() -> None:
    """VibeML - Natural language interface for AI model training."""
    pass


@cli.command()
def server() -> None:
    """Start the VibeML MCP server."""
    from .server import run_server

    click.echo("Starting VibeML MCP Server...")
    click.echo("Use this server with Claude or other MCP-compatible AI assistants")
    click.echo("Press Ctrl+C to stop the server")

    try:
        run_server()
    except KeyboardInterrupt:
        click.echo("\nServer stopped.")


@cli.command()
def info() -> None:
    """Show information about available workflows."""
    from .tasks import WORKFLOWS

    click.echo("VibeML - AI Training Workflows")
    click.echo("=" * 40)
    click.echo("\nAvailable workflows:")

    workflow_info = {
        "unsloth": "Efficient 4-bit fine-tuning (7B-13B models)",
        "gpt-oss-lora": "LoRA fine-tuning for large models (20B-120B)",
        "gpt-oss-full": "Full parameter training (maximum quality)",
    }

    for name, description in workflow_info.items():
        if name in WORKFLOWS:
            click.echo(f"  • {name}: {description}")

    click.echo("\nSupported Nebius GPUs:")
    click.echo("  • L40S (48GB) - Cost-effective")
    click.echo("  • RTX4090 (24GB) - Development")
    click.echo("  • H100 (80GB) - Premium performance")
    click.echo("  • A100 (40GB/80GB) - Production")

    click.echo("\nUsage:")
    click.echo("  vibeml server  - Start MCP server")
    click.echo("  vibeml info    - Show this information")


if __name__ == "__main__":
    cli(prog_name="vibeml")  # pragma: no cover
