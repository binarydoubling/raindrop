"""Hourly forecast command."""

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
    WEATHER_LABELS,
    TEMP_SYMBOLS,
    WIND_SYMBOLS,
    sparkline,
    format_delta,
    format_precip_chance,
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
@click.option("-n", "--hours", default=12, help="Number of hours to show (default: 12)")
@click.option(
    "-m",
    "--model",
    "model_name",
    help="Weather model to use (see 'raindrop config models')",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--spark", is_flag=True, help="Show sparkline summary")
def hourly(
    location: str | None,
    country: str | None,
    hours: int,
    model_name: str | None,
    as_json: bool,
    spark: bool,
):
    """Show hourly forecast with deltas.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop hourly <location>' or set a default with 'raindrop config set location <name>'"
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
        hourly=[
            "temperature_2m",
            "apparent_temperature",
            "precipitation_probability",
            "weather_code",
            "wind_speed_10m",
            "relative_humidity_2m",
        ],
        temperature_unit=settings.temperature_unit,
        wind_speed_unit=settings.wind_speed_unit,
        models=models,
        forecast_days=2,  # Need 2 days to get enough hours
    )
    h = weather.hourly
    if h is None:
        raise click.ClickException("No hourly weather data returned")

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    wind_symbol = WIND_SYMBOLS[settings.wind_speed_unit]

    # Find current hour index
    now = datetime.now()
    current_hour_str = now.strftime("%Y-%m-%dT%H:00")

    try:
        start_idx = h.time.index(current_hour_str)
    except ValueError:
        # If exact match not found, find closest hour
        start_idx = 0
        for i, t in enumerate(h.time):
            if t >= current_hour_str:
                start_idx = i
                break

    # Get data arrays (with None safety)
    temps = h.temperature_2m or []
    feels = h.apparent_temperature or []
    precip_probs = h.precipitation_probability or []
    codes = h.weather_code or []
    winds = h.wind_speed_10m or []
    humidities = h.relative_humidity_2m or []

    # JSON output
    if as_json:
        hourly_data = []
        for i in range(start_idx, min(start_idx + hours, len(h.time))):
            code = codes[i] if i < len(codes) else 0
            hourly_data.append(
                {
                    "time": h.time[i],
                    "temperature": temps[i] if i < len(temps) else None,
                    "apparent_temperature": feels[i] if i < len(feels) else None,
                    "precipitation_probability": precip_probs[i]
                    if i < len(precip_probs)
                    else None,
                    "weather_code": code,
                    "weather_description": WEATHER_CODES.get(code, "Unknown"),
                    "wind_speed": winds[i] if i < len(winds) else None,
                    "humidity": humidities[i] if i < len(humidities) else None,
                }
            )

        data = {
            "location": {
                "name": result.name,
                "admin1": result.admin1,
                "country": result.country,
                "latitude": result.latitude,
                "longitude": result.longitude,
            },
            "model": model_key or "auto",
            "hours": hourly_data,
            "units": {
                "temperature": settings.temperature_unit,
                "wind_speed": settings.wind_speed_unit,
                "precipitation": settings.precipitation_unit,
            },
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # Sparkline output
    if spark:
        temp_vals = [
            temps[i] if i < len(temps) else None
            for i in range(start_idx, min(start_idx + hours, len(h.time)))
        ]
        precip_vals = [
            precip_probs[i] if i < len(precip_probs) else None
            for i in range(start_idx, min(start_idx + hours, len(h.time)))
        ]
        wind_vals = [
            winds[i] if i < len(winds) else None
            for i in range(start_idx, min(start_idx + hours, len(h.time)))
        ]

        temp_clean = [t for t in temp_vals if t is not None]
        wind_clean = [w for w in wind_vals if w is not None]
        precip_clean = [p for p in precip_vals if p is not None]

        temp_range = (
            f"{min(temp_clean):.0f}-{max(temp_clean):.0f}\u00b0{temp_symbol}"
            if temp_clean
            else "\u2014"
        )
        wind_range = (
            f"{min(wind_clean):.0f}-{max(wind_clean):.0f} {wind_symbol}"
            if wind_clean
            else "\u2014"
        )
        precip_max = (
            f"{max(precip_clean):.0f}%"
            if precip_clean and max(precip_clean) > 0
            else "\u2014"
        )

        console.print(
            f"\n[bold cyan]{result.name}[/bold cyan] [dim]Next {hours}h[/dim]\n"
        )
        console.print(f"[dim]Temp[/dim]   {sparkline(temp_vals)}  {temp_range}")
        console.print(f"[dim]Precip[/dim] {sparkline(precip_vals)}  {precip_max}")
        console.print(f"[dim]Wind[/dim]   {sparkline(wind_vals)}  {wind_range}")
        return

    # Location header
    console.print(f"\n[bold cyan]{result.name}, {result.admin1}[/bold cyan]")
    console.print(f"[dim]Next {hours} hours[/dim]\n")

    # Build the table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Time", style="cyan", justify="right")
    table.add_column("Weather", justify="left")
    table.add_column(f"Temp (\u00b0{temp_symbol})", justify="right")
    table.add_column("Feels", justify="right")
    table.add_column("Precip", justify="right")
    table.add_column(f"Wind ({wind_symbol})", justify="right")
    table.add_column("Humidity", justify="right")

    for i in range(start_idx, min(start_idx + hours, len(h.time))):
        time_str = h.time[i]
        hour_dt = datetime.fromisoformat(time_str)

        # Format time nicely
        if hour_dt.date() == now.date():
            if hour_dt.hour == now.hour:
                time_display = "[bold yellow]Now[/bold yellow]"
            else:
                time_display = hour_dt.strftime("%-I%p").lower()
        else:
            time_display = hour_dt.strftime("%a %-I%p").lower()

        # Get values for this hour
        temp = temps[i] if i < len(temps) else 0
        feel = feels[i] if i < len(feels) else 0
        precip_prob = precip_probs[i] if i < len(precip_probs) else 0
        code = codes[i] if i < len(codes) else 0
        wind = winds[i] if i < len(winds) else 0
        humidity = humidities[i] if i < len(humidities) else 0

        # Get previous values for deltas
        prev_idx = i - 1 if i > start_idx else i
        prev_temp = temps[prev_idx] if prev_idx < len(temps) else temp
        prev_feel = feels[prev_idx] if prev_idx < len(feels) else feel
        prev_precip_prob = (
            precip_probs[prev_idx] if prev_idx < len(precip_probs) else precip_prob
        )
        prev_wind = winds[prev_idx] if prev_idx < len(winds) else wind
        prev_humidity = humidities[prev_idx] if prev_idx < len(humidities) else humidity

        # Weather label
        label, color = WEATHER_LABELS.get(code, ("?", "white"))
        weather_str = f"[{color}]{label}[/{color}]"

        # Format each column with deltas
        temp_str = format_delta(temp, prev_temp, "\u00b0", 0)
        feel_str = format_delta(feel, prev_feel, "\u00b0", 0)
        precip_str = format_precip_chance(precip_prob, prev_precip_prob)
        wind_str = format_delta(wind, prev_wind, "", 0)
        humidity_str = format_delta(humidity, prev_humidity, "%", 0)

        table.add_row(
            time_display,
            weather_str,
            temp_str,
            feel_str,
            precip_str,
            wind_str,
            humidity_str,
        )

    console.print(table)

    # Legend
    console.print("\n[dim]\u2191 rising  \u2193 falling  \u00b7 steady[/dim]")
