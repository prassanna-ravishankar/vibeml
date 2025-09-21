"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """vibeml."""


if __name__ == "__main__":
    main(prog_name="vibeml")  # pragma: no cover
