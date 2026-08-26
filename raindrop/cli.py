"""Main CLI entry point for raindrop."""

import os

import click

from raindrop import __version__
from raindrop.cache import set_cache_enabled
from raindrop.commands import (
    alerts,
    aqi,
    astro,
    clothing,
    compare,
    completions,
    config,
    current,
    daily,
    dashboard,
    discussion,
    fav,
    history,
    hourly,
    marine,
    precip,
    route,
    stations,
    window,
)
from raindrop.open_meteo import OpenMeteoError
from raindrop.providers.xweather import XweatherError


@click.group()
@click.version_option(version=__version__)
@click.option("--no-cache", is_flag=True, help="Bypass the API response cache")
def cli(no_cache: bool) -> None:
    """A simple, absolutely stunning weather CLI tool."""
    if no_cache:
        os.environ["RAINDROP_NO_CACHE"] = "1"
        set_cache_enabled(False)


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
cli.add_command(fav, "favorites")
cli.add_command(astro)
cli.add_command(clothing)
cli.add_command(route)
cli.add_command(stations)
cli.add_command(completions)
cli.add_command(dashboard)
cli.add_command(marine)
cli.add_command(window)
cli.add_command(window, "outside")


def main() -> None:
    """Entry point for the CLI."""
    try:
        cli(standalone_mode=False)
    except click.ClickException as e:
        e.show()
        raise SystemExit(e.exit_code) from e
    except click.exceptions.Exit as e:
        raise SystemExit(e.exit_code) from e
    except click.Abort as e:
        click.echo("Aborted!", err=True)
        raise SystemExit(1) from e
    except (OpenMeteoError, XweatherError) as e:
        click.ClickException(str(e)).show()
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
