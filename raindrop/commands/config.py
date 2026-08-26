"""Configuration commands."""

from typing import cast

import click
from rich.console import Console
from rich.table import Table

from raindrop.cache import get_cache
from raindrop.open_meteo import PrecipitationUnit, TemperatureUnit, WindSpeedUnit
from raindrop.providers.xweather import get_xweather_credential_status
from raindrop.settings import (
    AVAILABLE_MODELS,
    PRECIPITATION_UNITS,
    TEMPERATURE_UNITS,
    WEATHER_PROVIDERS,
    WIND_SPEED_UNITS,
    WeatherProviderName,
    get_settings,
    normalize_country_code,
)

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
    table.add_row("weather_provider", settings.weather_provider)
    table.add_row("model", settings.model or "(auto)")
    xweather = get_xweather_credential_status()
    if xweather.configured:
        table.add_row("xweather", f"configured ({xweather.source})")
    else:
        table.add_row("xweather", "not configured")

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
      units              imperial or metric shortcut
      temperature_unit   celsius or fahrenheit
      wind_speed_unit    kmh, ms, mph, or kn
      precipitation_unit mm or inch
      weather_provider  auto, open-meteo, or xweather
      model              Open-Meteo weather model (see 'raindrop config models')
    """
    settings = get_settings()

    normalized_value = value.lower()

    if key == "location":
        settings.location = value
    elif key == "country_code":
        try:
            settings.country_code = normalize_country_code(value)
        except ValueError as e:
            raise click.ClickException(str(e)) from e
    elif key == "units":
        if normalized_value == "metric":
            settings.temperature_unit = "celsius"
            settings.wind_speed_unit = "kmh"
            settings.precipitation_unit = "mm"
        elif normalized_value == "imperial":
            settings.temperature_unit = "fahrenheit"
            settings.wind_speed_unit = "mph"
            settings.precipitation_unit = "inch"
        else:
            raise click.ClickException("units must be 'metric' or 'imperial'")
    elif key == "temperature_unit":
        if normalized_value not in TEMPERATURE_UNITS:
            raise click.ClickException("temperature_unit must be 'celsius' or 'fahrenheit'")
        settings.temperature_unit = cast(TemperatureUnit, normalized_value)
    elif key == "wind_speed_unit":
        if normalized_value not in WIND_SPEED_UNITS:
            raise click.ClickException("wind_speed_unit must be 'kmh', 'ms', 'mph', or 'kn'")
        settings.wind_speed_unit = cast(WindSpeedUnit, normalized_value)
    elif key == "precipitation_unit":
        if normalized_value not in PRECIPITATION_UNITS:
            raise click.ClickException("precipitation_unit must be 'mm' or 'inch'")
        settings.precipitation_unit = cast(PrecipitationUnit, normalized_value)
    elif key == "weather_provider":
        if normalized_value not in WEATHER_PROVIDERS:
            raise click.ClickException(
                "weather_provider must be 'auto', 'open-meteo', or 'xweather'"
            )
        settings.weather_provider = cast(WeatherProviderName, normalized_value)
    elif key == "model":
        if normalized_value == "auto":
            settings.model = None
        elif normalized_value not in AVAILABLE_MODELS:
            raise click.ClickException(
                f"Unknown model: {value}. Run 'raindrop config models' to see available models."
            )
        else:
            settings.model = normalized_value
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
    elif key == "weather_provider":
        settings.weather_provider = "auto"
    elif key == "model":
        settings.model = None
    elif key in ("units", "temperature_unit", "wind_speed_unit", "precipitation_unit"):
        raise click.ClickException(f"Cannot unset {key}, use 'config set' to change it")
    else:
        raise click.ClickException(f"Unknown setting: {key}")

    settings.save()
    console.print(f"[green]Unset {key}[/green]")


@config.command("models")
def config_models():
    """List available weather models."""
    console.print("\n[bold]Available weather models:[/bold]\n")
    console.print("[dim]Use 'raindrop config set model <name>' to set a default.[/dim]")
    console.print("[dim]Or use '--model <name>' flag on any command.[/dim]\n")

    console.print("[cyan]Auto (default)[/cyan]")
    console.print("  [dim]Omit --model to let Open-Meteo choose the best model[/dim]\n")

    models_by_category = {
        "ECMWF (European)": ["ecmwf"],
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

    table.add_row("Enabled", "[green]Yes[/green]" if stats["enabled"] else "[red]No[/red]")
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
