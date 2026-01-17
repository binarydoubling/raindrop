"""Configuration commands."""

import click
from rich.console import Console
from rich.table import Table
from rich import box

from settings import get_settings, AVAILABLE_MODELS
from raindrop.cache import get_cache

console = Console()


@click.group()
def config():
    """View and manage settings."""
    pass


@config.command("show")
def config_show():
    """Show current settings."""
    settings = get_settings()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("setting", style="dim")
    table.add_column("value", style="bold")

    table.add_row("location", settings.location or "(not set)")
    table.add_row("country_code", settings.country_code or "(not set)")
    table.add_row("temperature_unit", settings.temperature_unit)
    table.add_row("wind_speed_unit", settings.wind_speed_unit)
    table.add_row("precipitation_unit", settings.precipitation_unit)
    table.add_row("model", settings.model or "(auto)")

    console.print(table)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a configuration value.

    \b
    Available settings:
      location           Default location name
      country_code       Default country code (e.g., US, ES, DE)
      temperature_unit   celsius or fahrenheit
      wind_speed_unit    kmh, ms, mph, or kn
      precipitation_unit mm or inch
      model              Weather model (see 'weather config models')
    """
    settings = get_settings()

    if key == "location":
        settings.location = value
    elif key == "country_code":
        settings.country_code = value.upper()
    elif key == "temperature_unit":
        if value not in ("celsius", "fahrenheit"):
            raise click.ClickException(
                "temperature_unit must be 'celsius' or 'fahrenheit'"
            )
        settings.temperature_unit = value  # type: ignore
    elif key == "wind_speed_unit":
        if value not in ("kmh", "ms", "mph", "kn"):
            raise click.ClickException(
                "wind_speed_unit must be 'kmh', 'ms', 'mph', or 'kn'"
            )
        settings.wind_speed_unit = value  # type: ignore
    elif key == "precipitation_unit":
        if value not in ("mm", "inch"):
            raise click.ClickException("precipitation_unit must be 'mm' or 'inch'")
        settings.precipitation_unit = value  # type: ignore
    elif key == "model":
        if value == "auto":
            settings.model = None
        elif value not in AVAILABLE_MODELS:
            raise click.ClickException(
                f"Unknown model: {value}. Run 'weather config models' to see available models."
            )
        else:
            settings.model = value
    else:
        raise click.ClickException(f"Unknown setting: {key}")

    settings.save()
    console.print(f"[green]Set {key} = {value}[/green]")


@config.command("unset")
@click.argument("key")
def config_unset(key: str):
    """Unset a configuration value (reset to default)."""
    settings = get_settings()

    if key == "location":
        settings.location = None
    elif key == "country_code":
        settings.country_code = None
    elif key == "model":
        settings.model = None
    elif key in ("temperature_unit", "wind_speed_unit", "precipitation_unit"):
        raise click.ClickException(f"Cannot unset {key}, use 'config set' to change it")
    else:
        raise click.ClickException(f"Unknown setting: {key}")

    settings.save()
    console.print(f"[green]Unset {key}[/green]")


@config.command("models")
def config_models():
    """List available weather models."""
    console.print("\n[bold]Available weather models:[/bold]\n")
    console.print("[dim]Use 'weather config set model <name>' to set a default.[/dim]")
    console.print("[dim]Or use '--model <name>' flag on any command.[/dim]\n")

    console.print("[cyan]Auto (default)[/cyan]")
    console.print("  [dim]Omit --model to let Open-Meteo choose the best model[/dim]\n")

    models_by_category = {
        "ECMWF (European)": ["ecmwf", "ecmwf_aifs"],
        "US (NOAA)": ["gfs", "hrrr"],
        "German (DWD)": ["icon", "icon_eu", "icon_d2"],
        "French (Meteo-France)": ["arpege", "arome"],
        "UK (Met Office)": ["ukmo"],
        "Canadian (GEM)": ["gem", "gem_hrdps"],
        "Japanese (JMA)": ["jma"],
        "Norwegian (MET)": ["metno"],
    }

    for category, models in models_by_category.items():
        console.print(f"[cyan]{category}[/cyan]")
        for model in models:
            console.print(f"  {model}")
    console.print()


@config.command("cache")
@click.option("--clear", is_flag=True, help="Clear all cached data")
def config_cache(clear: bool):
    """View or manage the API response cache."""
    cache = get_cache()

    if clear:
        count = cache.clear()
        console.print(f"[green]Cleared {count} cached entries[/green]")
        return

    stats = cache.stats()

    console.print("\n[bold]Cache Status[/bold]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="dim")
    table.add_column("value")

    table.add_row(
        "Enabled", "[green]Yes[/green]" if stats["enabled"] else "[red]No[/red]"
    )
    table.add_row("Location", stats.get("cache_dir", "N/A"))
    table.add_row("Total entries", str(stats.get("entries", 0)))
    table.add_row("Valid entries", str(stats.get("valid", 0)))
    table.add_row("Expired entries", str(stats.get("expired", 0)))

    size_bytes = stats.get("size_bytes", 0)
    if size_bytes > 1024 * 1024:
        size_str = f"{size_bytes / 1024 / 1024:.1f} MB"
    elif size_bytes > 1024:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes} bytes"
    table.add_row("Size", size_str)

    console.print(table)
    console.print("\n[dim]Use --clear to remove all cached data[/dim]")
