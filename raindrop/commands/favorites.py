"""Favorites management commands."""

import click
from rich.console import Console
from rich.table import Table
from rich import box

from open_meteo import OpenMeteo
from settings import get_settings, Favorite

om = OpenMeteo()
console = Console()


def geocode(location: str, country: str | None = None):
    results = om.geocode(location, country_code=country)
    return results[0]


@click.group()
def fav():
    """Manage favorite locations."""
    pass


@fav.command("list")
def fav_list():
    """List all saved favorites."""
    settings = get_settings()

    if not settings.favorites:
        console.print("[dim]No favorites saved yet.[/dim]")
        console.print(
            "[dim]Use 'raindrop fav add <alias> <location>' to add one.[/dim]"
        )
        return

    table = Table(show_header=True, box=box.ROUNDED, header_style="bold")
    table.add_column("Alias", style="cyan")
    table.add_column("Location")
    table.add_column("Country", style="dim")

    for alias, favorite in sorted(settings.favorites.items()):
        table.add_row(alias, favorite.name, favorite.country_code or "\u2014")

    console.print(table)


@fav.command("add")
@click.argument("alias")
@click.argument("location")
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
def fav_add(alias: str, location: str, country: str | None):
    """Add a favorite location.

    \b
    Examples:
      raindrop fav add home "San Francisco"
      raindrop fav add work "New York" -c US
      raindrop fav add parents "Paris" -c FR
    """
    settings = get_settings()

    # Validate by attempting to geocode
    try:
        result = geocode(location, country)
    except Exception as e:
        raise click.ClickException(f"Could not find location: {e}")

    settings.favorites[alias] = Favorite(
        name=result.name,
        country_code=country.upper() if country else None,
    )
    settings.save()

    console.print(
        f"[green]Added favorite '{alias}' -> {result.name}, {result.admin1}, {result.country}[/green]"
    )


@fav.command("remove")
@click.argument("alias")
def fav_remove(alias: str):
    """Remove a favorite location."""
    settings = get_settings()

    if alias not in settings.favorites:
        raise click.ClickException(f"Favorite '{alias}' not found")

    del settings.favorites[alias]
    settings.save()

    console.print(f"[green]Removed favorite '{alias}'[/green]")
