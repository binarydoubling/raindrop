"""Current weather command."""

from datetime import datetime
import json as json_lib

import click
from rich.console import Console
from rich.table import Table
from rich import box

from open_meteo import OpenMeteo
from settings import get_settings, AVAILABLE_MODELS
from raindrop.utils import (
    WEATHER_CODES,
    TEMP_SYMBOLS,
    WIND_SYMBOLS,
    format_duration,
    format_visibility,
    format_uv,
    deg_to_compass,
)

om = OpenMeteo()
console = Console()


def geocode(location: str, country: str | None = None):
    results = om.geocode(location, country_code=country)
    return results[0]


@click.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option(
    "-m",
    "--model",
    "model_name",
    help="Weather model to use (see 'raindrop config models')",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--compact", is_flag=True, help="One-line output for shell prompts")
def current(
    location: str | None,
    country: str | None,
    model_name: str | None,
    as_json: bool,
    compact: bool,
):
    """Get current weather for a location.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop current <location>' or set a default with 'raindrop config set location <name>'"
        )

    # CLI country flag overrides resolved country
    if country is not None:
        resolved_country = country

    location = resolved_location
    country = resolved_country

    # Resolve model (CLI flag > settings > auto)
    model_key = model_name or settings.model
    api_model = AVAILABLE_MODELS.get(model_key) if model_key else None
    models = [api_model] if api_model else None

    result = geocode(location, country)

    weather = om.forecast(
        result.latitude,
        result.longitude,
        current=[
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "dew_point_2m",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "pressure_msl",
            "surface_pressure",
            "visibility",
            "uv_index",
            "weather_code",
            "is_day",
        ],
        daily=[
            "sunrise",
            "sunset",
            "uv_index_max",
        ],
        temperature_unit=settings.temperature_unit,
        wind_speed_unit=settings.wind_speed_unit,
        models=models,
        forecast_days=1,
    )
    c = weather.current
    d = weather.daily
    if c is None:
        raise click.ClickException("No current weather data returned")

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    wind_symbol = WIND_SYMBOLS[settings.wind_speed_unit]

    # JSON output
    if as_json:
        data = {
            "location": {
                "name": result.name,
                "admin1": result.admin1,
                "country": result.country,
                "latitude": result.latitude,
                "longitude": result.longitude,
            },
            "elevation": weather.elevation,
            "timezone": weather.timezone,
            "model": model_key or "auto",
            "current": {
                "time": c.time,
                "temperature": c.temperature_2m,
                "apparent_temperature": c.apparent_temperature,
                "humidity": c.relative_humidity_2m,
                "dew_point": c.dew_point_2m,
                "cloud_cover": c.cloud_cover,
                "wind_speed": c.wind_speed_10m,
                "wind_direction": c.wind_direction_10m,
                "wind_gusts": c.wind_gusts_10m,
                "pressure_msl": c.pressure_msl,
                "surface_pressure": c.surface_pressure,
                "visibility": c.visibility,
                "uv_index": c.uv_index,
                "weather_code": c.weather_code,
                "weather_description": WEATHER_CODES.get(
                    c.weather_code or 0, "Unknown"
                ),
                "is_day": c.is_day,
            },
            "daily": {
                "sunrise": d.sunrise[0] if d and d.sunrise else None,
                "sunset": d.sunset[0] if d and d.sunset else None,
                "uv_index_max": d.uv_index_max[0] if d and d.uv_index_max else None,
            },
            "units": {
                "temperature": settings.temperature_unit,
                "wind_speed": settings.wind_speed_unit,
                "precipitation": settings.precipitation_unit,
            },
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # Compact one-liner output
    if compact:
        code = c.weather_code or 0
        condition = WEATHER_CODES.get(code, "Unknown")
        click.echo(
            f"{result.name}: {c.temperature_2m:.0f}\u00b0{temp_symbol} {condition} | "
            f"Wind {c.wind_speed_10m:.0f} {wind_symbol} | "
            f"Humidity {c.relative_humidity_2m}%"
        )
        return

    # Parse times
    now = datetime.fromisoformat(c.time)
    sunrise = datetime.fromisoformat(d.sunrise[0]) if d and d.sunrise else None
    sunset = datetime.fromisoformat(d.sunset[0]) if d and d.sunset else None

    # Location header
    console.print(
        f"\n[bold cyan]{result.name}, {result.admin1}, {result.country}[/bold cyan]"
    )

    # Coordinates and metadata
    lat_dir = "N" if result.latitude >= 0 else "S"
    lon_dir = "E" if result.longitude >= 0 else "W"
    console.print(
        f"[dim]{abs(result.latitude):.4f}\u00b0{lat_dir} {abs(result.longitude):.4f}\u00b0{lon_dir} \u00b7 "
        f"Elev {weather.elevation:.0f}m \u00b7 {weather.timezone}[/dim]"
    )

    # Weather condition with WMO code
    code = c.weather_code or 0
    condition = WEATHER_CODES.get(code, "Unknown")
    is_day_str = "Day" if c.is_day else "Night"
    console.print(f"[dim]WMO {code}: {condition} ({is_day_str})[/dim]")

    # Model info
    model_display = model_key or "auto"
    console.print(f"[dim]Model: {model_display}[/dim]\n")

    # Main conditions table
    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
    table.add_column("label", style="dim")
    table.add_column("value", style="bold")
    table.add_column("label2", style="dim")
    table.add_column("value2", style="bold")

    # Row 1: Temperature / Feels like
    table.add_row(
        "Temperature",
        f"{c.temperature_2m}\u00b0{temp_symbol}",
        "Feels like",
        f"{c.apparent_temperature}\u00b0{temp_symbol}",
    )

    # Row 2: Humidity / Dew point
    table.add_row(
        "Humidity",
        f"{c.relative_humidity_2m}%",
        "Dew point",
        f"{c.dew_point_2m}\u00b0{temp_symbol}" if c.dew_point_2m else "\u2014",
    )

    # Row 3: Wind speed + direction / Gusts
    wind_dir = c.wind_direction_10m
    if wind_dir is not None:
        compass = deg_to_compass(wind_dir)
        wind_str = f"{c.wind_speed_10m} {wind_symbol} {compass} ({wind_dir}\u00b0)"
    else:
        wind_str = f"{c.wind_speed_10m} {wind_symbol}"
    table.add_row(
        "Wind",
        wind_str,
        "Gusts",
        f"{c.wind_gusts_10m} {wind_symbol}",
    )

    # Row 4: Pressure (MSL) / Surface pressure
    table.add_row(
        "Pressure (MSL)",
        f"{c.pressure_msl} hPa" if c.pressure_msl else "\u2014",
        "Surface",
        f"{c.surface_pressure} hPa" if c.surface_pressure else "\u2014",
    )

    # Row 5: Visibility / Cloud cover
    visibility_str = format_visibility(c.visibility) if c.visibility else "\u2014"
    table.add_row(
        "Visibility",
        visibility_str,
        "Cloud cover",
        f"{c.cloud_cover}%",
    )

    # Row 6: UV Index / UV Max today
    uv_str = format_uv(c.uv_index) if c.uv_index is not None else "\u2014"
    uv_max = d.uv_index_max[0] if d and d.uv_index_max else None
    uv_max_str = format_uv(uv_max) if uv_max is not None else "\u2014"
    table.add_row(
        "UV Index",
        uv_str,
        "UV Max today",
        uv_max_str,
    )

    console.print(table)

    # Sun info
    if sunrise and sunset:
        console.print()
        sun_table = Table(show_header=False, box=None, padding=(0, 2))
        sun_table.add_column("label", style="dim")
        sun_table.add_column("value", style="bold")
        sun_table.add_column("label2", style="dim")
        sun_table.add_column("value2", style="bold")

        sunrise_str = sunrise.strftime("%-I:%M %p").lower()
        sunset_str = sunset.strftime("%-I:%M %p").lower()

        # Calculate time until/since sunrise/sunset
        if now < sunrise:
            sun_status = f"[cyan]Sunrise in {format_duration(sunrise - now)}[/cyan]"
        elif now < sunset:
            sun_status = f"[yellow]Sunset in {format_duration(sunset - now)}[/yellow]"
        else:
            sun_status = f"[dim]Sun set {format_duration(now - sunset)} ago[/dim]"

        sun_table.add_row(
            "Sunrise",
            sunrise_str,
            "Sunset",
            sunset_str,
        )
        console.print(sun_table)
        console.print(sun_status)
