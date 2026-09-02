"""Main CLI entry point for raindrop."""

import os

import click

from raindrop.cache import set_cache_enabled
from raindrop.commands.alerts import alerts
from raindrop.commands.aqi import aqi
from raindrop.commands.astro import astro
from raindrop.commands.clothing import clothing
from raindrop.commands.compare import compare
from raindrop.commands.config import config
from raindrop.commands.current import current
from raindrop.commands.daily import daily
from raindrop.commands.dashboard import dashboard
from raindrop.commands.discussion import discussion
from raindrop.commands.ensemble import ensemble
from raindrop.commands.favorites import fav
from raindrop.commands.history import history
from raindrop.commands.hourly import hourly
from raindrop.commands.marine import marine
from raindrop.commands.precip import precip
from raindrop.commands.route import route
from raindrop.commands.stations import stations
from raindrop.commands.window import window
from raindrop.open_meteo import OpenMeteoError
from raindrop.providers.xweather import XweatherError


@click.group()
@click.version_option(package_name="rdrop")
@click.option("--no-cache", is_flag=True, help="Bypass the API response cache")
def cli(no_cache: bool) -> None:
    """A simple, absolutely stunning weather CLI tool."""
    if no_cache:
        os.environ["RAINDROP_NO_CACHE"] = "1"
        set_cache_enabled(False)


cli.add_command(current)
cli.add_command(hourly)
cli.add_command(daily)
cli.add_command(aqi)
cli.add_command(alerts)
cli.add_command(discussion)
cli.add_command(ensemble)
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
