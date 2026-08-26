"""Weather window command: describe what it feels like to stand somewhere."""

import json as json_lib
from dataclasses import asdict

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from raindrop.commands.common import (
    format_location,
    format_weather_source,
    geocode,
    location_payload,
    resolve_location_or_fail,
    resolve_weather_provider_or_fail,
)
from raindrop.scene import SceneReading, build_scene_report
from raindrop.settings import get_settings
from raindrop.utils import WEATHER_CODES, find_time_index, now_in_timezone

console = Console()


def _at_index[T](values: list[T] | None, index: int) -> T | None:
    """Safely read a value from an optional API array."""
    if values is None or index < 0 or index >= len(values):
        return None
    return values[index]


@click.command()
@click.argument("location", required=False)
@click.option("-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)")
@click.option(
    "-m",
    "--model",
    "model_name",
    help="Open-Meteo model to use (forces Open-Meteo in auto mode)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--compact", is_flag=True, help="Only show the scene summary")
def window(
    location: str | None,
    country: str | None,
    model_name: str | None,
    as_json: bool,
    compact: bool,
) -> None:
    """Peer through a weather window into a location.

    Translates temperature, humidity, cloud, wind, visibility, and weather
    codes into a first-person description of what it would feel like to
    stand outside there right now.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()
    location, country = resolve_location_or_fail(settings, location, country, "window")
    weather_provider = resolve_weather_provider_or_fail(model_name, settings)

    result = geocode(location, country)
    weather = weather_provider.forecast(
        result.latitude,
        result.longitude,
        current=[
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "rain",
            "showers",
            "snowfall",
            "weather_code",
            "cloud_cover",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "visibility",
            "uv_index",
            "is_day",
        ],
        hourly=[
            "precipitation_probability",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
        ],
        temperature_unit=settings.temperature_unit,
        wind_speed_unit=settings.wind_speed_unit,
        precipitation_unit=settings.precipitation_unit,
        forecast_days=1,
    )

    c = weather.current
    if c is None:
        raise click.ClickException("No current weather data returned")

    h = weather.hourly
    hourly_index = find_time_index(h.time, now_in_timezone(weather.timezone)) if h else 0

    reading = SceneReading(
        temperature=c.temperature_2m,
        apparent_temperature=c.apparent_temperature,
        humidity=c.relative_humidity_2m,
        dew_point=c.dew_point_2m,
        weather_code=c.weather_code,
        cloud_cover=c.cloud_cover,
        wind_speed=c.wind_speed_10m,
        wind_gusts=c.wind_gusts_10m,
        wind_direction=c.wind_direction_10m,
        pressure_msl=c.pressure_msl,
        visibility=c.visibility,
        uv_index=c.uv_index,
        precipitation=c.precipitation,
        rain=c.rain,
        showers=c.showers,
        snowfall=c.snowfall,
        is_day=c.is_day,
        temperature_unit=settings.temperature_unit,
        wind_speed_unit=settings.wind_speed_unit,
        precipitation_unit=settings.precipitation_unit,
        precipitation_probability=_at_index(
            h.precipitation_probability if h else None,
            hourly_index,
        ),
        low_cloud_cover=_at_index(h.cloud_cover_low if h else None, hourly_index),
        mid_cloud_cover=_at_index(h.cloud_cover_mid if h else None, hourly_index),
        high_cloud_cover=_at_index(h.cloud_cover_high if h else None, hourly_index),
    )
    report = build_scene_report(reading)

    if as_json:
        data = {
            "location": location_payload(result),
            "timezone": weather.timezone,
            "model": weather_provider.model_label,
            "source": {
                "provider": weather_provider.name,
                "label": weather_provider.label,
                "attribution": weather_provider.attribution,
            },
            "weather_description": WEATHER_CODES.get(c.weather_code or 0, "Unknown"),
            "reading": asdict(reading),
            "scene": report.to_dict(),
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    if compact:
        click.echo(report.summary)
        return

    console.print(f"\n[bold cyan]{format_location(result)}[/bold cyan]")
    console.print(
        f"[dim]Weather window · {weather.timezone} · {format_weather_source(weather_provider)}[/dim]\n"
    )

    console.print(
        Panel(
            report.summary,
            title="[bold]Standing there now[/bold]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
    table.add_column("sense", style="bold")
    table.add_column("description")
    for title, description in report.sections.items():
        table.add_row(title, description)
    console.print(table)

    if report.cues:
        console.print(f"\n[dim]Inferred from: {', '.join(report.cues)}[/dim]")
