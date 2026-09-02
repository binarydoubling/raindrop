"""Historical weather comparison command."""

import click
from rich import box
from rich.table import Table

from raindrop.commands.common import (
    console,
    echo_json,
    format_location,
    geocode,
    location_payload,
    om,
    resolve_location_or_fail,
)
from raindrop.settings import get_settings
from raindrop.utils import (
    TEMP_SYMBOLS,
    WEATHER_LABELS,
    now_in_timezone,
)


@click.command()
@click.argument("location", required=False)
@click.option("-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)")
@click.option(
    "-y",
    "--years",
    type=click.IntRange(1, 30),
    default=1,
    show_default=True,
    help="How many years back to compare",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def history(location: str | None, country: str | None, years: int, as_json: bool):
    """Compare today's weather with the same day in past years.

    Shows how today compares to this day in previous years.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    location, country = resolve_location_or_fail(settings, location, country, "history")

    result = geocode(location, country)

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    precip_symbol = settings.precipitation_unit

    # Get today's date
    today = now_in_timezone(result.timezone).date()
    today_str = today.strftime("%Y-%m-%d")

    # Get current weather
    current_weather = om.forecast(
        result.latitude,
        result.longitude,
        current=["temperature_2m"],
        daily=[
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "weather_code",
        ],
        temperature_unit=settings.temperature_unit,
        precipitation_unit=settings.precipitation_unit,
        forecast_days=1,
    )

    # Get historical data for same day in past years
    historical_data = []
    for y in range(1, years + 1):
        try:
            past_date = today.replace(year=today.year - y)
            past_str = past_date.strftime("%Y-%m-%d")

            hist = om.historical(
                result.latitude,
                result.longitude,
                start_date=past_str,
                end_date=past_str,
                daily=[
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "weather_code",
                ],
                temperature_unit=settings.temperature_unit,
                precipitation_unit=settings.precipitation_unit,
            )

            if "daily" in hist and hist["daily"].get("time"):
                d = hist["daily"]
                historical_data.append(
                    {
                        "year": today.year - y,
                        "date": past_str,
                        "temp_max": d.get("temperature_2m_max", [None])[0],
                        "temp_min": d.get("temperature_2m_min", [None])[0],
                        "precip": d.get("precipitation_sum", [None])[0],
                        "weather_code": d.get("weather_code", [None])[0],
                    }
                )
        except Exception:
            # Skip years with missing data
            pass

    # Current day data
    c = current_weather.current
    d = current_weather.daily

    current_data = {
        "year": today.year,
        "date": today_str,
        "temp_max": d.temperature_2m_max[0] if d and d.temperature_2m_max else None,
        "temp_min": d.temperature_2m_min[0] if d and d.temperature_2m_min else None,
        "precip": d.precipitation_sum[0] if d and d.precipitation_sum else None,
        "weather_code": d.weather_code[0] if d and d.weather_code else None,
        "temp_current": c.temperature_2m if c else None,
    }

    # JSON output
    if as_json:
        data = {
            "location": location_payload(result),
            "today": current_data,
            "historical": historical_data,
            "units": {
                "temperature": settings.temperature_unit,
                "precipitation": settings.precipitation_unit,
            },
        }
        echo_json(data)
        return

    # Display
    console.print(f"\n[bold cyan]{format_location(result)}[/bold cyan]")
    console.print(f"[dim]Historical comparison for {today.strftime('%B %d')}[/dim]\n")

    # Table
    table = Table(show_header=True, box=box.ROUNDED, header_style="bold")
    table.add_column("Year", style="cyan", justify="right")
    table.add_column("High", justify="right")
    table.add_column("Low", justify="right")
    table.add_column("Precip", justify="right")
    table.add_column("Weather", justify="left")
    table.add_column("vs Today", justify="right")

    # Add current year first
    code = current_data.get("weather_code", 0) or 0
    label, color = WEATHER_LABELS.get(code, ("?", "white"))
    table.add_row(
        f"[bold yellow]{current_data['year']}[/bold yellow]",
        f"{current_data['temp_max']:.0f}\u00b0{temp_symbol}"
        if current_data["temp_max"] is not None
        else "\u2014",
        f"{current_data['temp_min']:.0f}\u00b0{temp_symbol}"
        if current_data["temp_min"] is not None
        else "\u2014",
        f"{current_data['precip']:.1f} {precip_symbol}"
        if current_data["precip"] is not None
        else "\u2014",
        f"[{color}]{label}[/{color}]",
        "[bold]Today[/bold]",
    )

    # Add historical years
    today_max = current_data.get("temp_max")
    for h in historical_data:
        code = h.get("weather_code", 0) or 0
        label, color = WEATHER_LABELS.get(code, ("?", "white"))

        # Calculate difference from today
        if today_max is not None and h.get("temp_max") is not None:
            diff = today_max - h["temp_max"]
            if diff > 0:
                diff_str = f"[red]+{diff:.0f}\u00b0[/red]"
            elif diff < 0:
                diff_str = f"[cyan]{diff:.0f}\u00b0[/cyan]"
            else:
                diff_str = "[dim]same[/dim]"
        else:
            diff_str = "\u2014"

        table.add_row(
            str(h["year"]),
            f"{h['temp_max']:.0f}\u00b0{temp_symbol}"
            if h.get("temp_max") is not None
            else "\u2014",
            f"{h['temp_min']:.0f}\u00b0{temp_symbol}"
            if h.get("temp_min") is not None
            else "\u2014",
            f"{h['precip']:.1f} {precip_symbol}" if h.get("precip") is not None else "\u2014",
            f"[{color}]{label}[/{color}]",
            diff_str,
        )

    console.print(table)

    # Summary stats
    if historical_data and today_max is not None:
        hist_maxes = [h["temp_max"] for h in historical_data if h.get("temp_max") is not None]
        if hist_maxes:
            avg_max = sum(hist_maxes) / len(hist_maxes)
            diff_from_avg = today_max - avg_max
            if diff_from_avg > 0:
                console.print(
                    f"\n[dim]Today is [red]{diff_from_avg:.1f}\u00b0{temp_symbol} warmer[/red] than average for this date[/dim]"
                )
            elif diff_from_avg < 0:
                console.print(
                    f"\n[dim]Today is [cyan]{abs(diff_from_avg):.1f}\u00b0{temp_symbol} cooler[/cyan] than average for this date[/dim]"
                )
            else:
                console.print("\n[dim]Today matches the historical average for this date[/dim]")
