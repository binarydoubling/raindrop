"""Favorites management commands."""

import click
from rich import box
from rich.table import Table

from raindrop.commands.common import console, format_location, geocode
from raindrop.settings import Favorite, get_settings


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
        console.print("[dim]Use 'raindrop fav add <alias> <location>' to add one.[/dim]")
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
@click.option("-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)")
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

    favorite_name = f"{result.name}, {result.admin1}" if result.admin1 else result.name
    settings.favorites[alias] = Favorite(
        name=favorite_name,
        country_code=result.country_code or (country.upper() if country else None),
    )
    settings.save()

    console.print(f"[green]Added favorite '{alias}' -> {format_location(result)}[/green]")


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
