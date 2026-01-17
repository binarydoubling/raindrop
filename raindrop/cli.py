"""Main CLI entry point for raindrop."""

import click

from raindrop import __version__
from raindrop.commands import (
    current,
    hourly,
    daily,
    aqi,
    alerts,
    discussion,
    precip,
    compare,
    history,
    config,
    fav,
    astro,
    clothing,
    route,
    completions,
    dashboard,
    marine,
)


@click.group()
@click.version_option(version=__version__)
def cli():
    """A simple, absolutely stunning weather CLI tool."""
    pass


# Register all commands
cli.add_command(current)
cli.add_command(hourly)
cli.add_command(daily)
cli.add_command(aqi)
cli.add_command(alerts)
cli.add_command(discussion)
cli.add_command(precip)
cli.add_command(compare)
cli.add_command(history)
cli.add_command(config)
cli.add_command(fav)
cli.add_command(astro)
cli.add_command(clothing)
cli.add_command(route)
cli.add_command(completions)
cli.add_command(dashboard)
cli.add_command(marine)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
