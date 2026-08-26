"""Hourly forecast command."""

import json as json_lib
from datetime import datetime

import click
from rich import box
from rich.console import Console
from rich.table import Table

from raindrop.commands.common import (
    format_location,
    format_weather_source,
    geocode,
    resolve_location_or_fail,
    resolve_weather_provider_or_fail,
)
from raindrop.settings import get_settings
from raindrop.utils import (
    TEMP_SYMBOLS,
    WEATHER_CODES,
    WEATHER_LABELS,
    WIND_SYMBOLS,
    find_time_index,
    format_delta,
    format_precip_chance,
    format_time,
    now_in_timezone,
    sparkline,
)

console = Console()


@click.command()
@click.argument("location", required=False)
@click.option("-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)")
@click.option(
    "-n",
    "--hours",
    type=click.IntRange(1, 48),
    default=12,
    show_default=True,
    help="Number of hours to show",
)
@click.option(
    "-m",
    "--model",
    "model_name",
    help="Open-Meteo model to use (forces Open-Meteo in auto mode)",
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

    location, country = resolve_location_or_fail(settings, location, country, "hourly")

    weather_provider = resolve_weather_provider_or_fail(model_name, settings)

    result = geocode(location, country)

    weather = weather_provider.forecast(
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
        forecast_days=2,  # Need 2 days to get enough hours
    )
    h = weather.hourly
    if h is None:
        raise click.ClickException("No hourly weather data returned")

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    wind_symbol = WIND_SYMBOLS[settings.wind_speed_unit]

    # Find current hour index in the target location timezone.
    now = now_in_timezone(weather.timezone)
    start_idx = find_time_index(h.time, now)

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
                    "precipitation_probability": precip_probs[i] if i < len(precip_probs) else None,
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
            "model": weather_provider.model_label,
            "source": {
                "provider": weather_provider.name,
                "label": weather_provider.label,
                "attribution": weather_provider.attribution,
            },
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
            f"{min(wind_clean):.0f}-{max(wind_clean):.0f} {wind_symbol}" if wind_clean else "\u2014"
        )
        precip_max = (
            f"{max(precip_clean):.0f}%" if precip_clean and max(precip_clean) > 0 else "\u2014"
        )

        console.print(f"\n[bold cyan]{result.name}[/bold cyan] [dim]Next {hours}h[/dim]")
        console.print(f"[dim]{format_weather_source(weather_provider)}[/dim]\n")
        console.print(f"[dim]Temp[/dim]   {sparkline(temp_vals)}  {temp_range}")
        console.print(f"[dim]Precip[/dim] {sparkline(precip_vals)}  {precip_max}")
        console.print(f"[dim]Wind[/dim]   {sparkline(wind_vals)}  {wind_range}")
        return

    # Location header
    console.print(f"\n[bold cyan]{format_location(result, include_country=False)}[/bold cyan]")
    console.print(f"[dim]Next {hours} hours · {format_weather_source(weather_provider)}[/dim]\n")

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
                time_display = format_time(hour_dt, "{hour}%p")
        else:
            time_display = f"{hour_dt.strftime('%a')} {format_time(hour_dt, '{hour}%p')}"

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
        prev_precip_prob = precip_probs[prev_idx] if prev_idx < len(precip_probs) else precip_prob
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
