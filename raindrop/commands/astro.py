"""Astronomical data command (sun, moon, golden hour)."""

from datetime import datetime, timedelta
import json as json_lib

import click
from rich.console import Console
from rich.table import Table
from rich import box

from open_meteo import OpenMeteo
from settings import get_settings
from raindrop.utils.astro import (
    moon_phase,
    moon_illumination,
    next_moon_phase,
    golden_hour,
    blue_hour,
    daylight_duration,
    format_daylight_duration,
    solar_noon,
)

om = OpenMeteo()
console = Console()


def geocode(location: str, country: str | None = None):
    results = om.geocode(location, country_code=country)
    return results[0]


def format_time(dt: datetime) -> str:
    """Format datetime as time string."""
    return dt.strftime("%-I:%M %p").lower()


def format_time_until(now: datetime, target: datetime) -> str:
    """Format time until a future event."""
    if target < now:
        return "passed"
    delta = target - now
    hours = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    if hours > 0:
        return f"in {hours}h {minutes}m"
    return f"in {minutes}m"


@click.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def astro(location: str | None, country: str | None, as_json: bool):
    """Show astronomical data: sun times, moon phase, golden hour.

    Displays sunrise, sunset, daylight duration, moon phase,
    and photography golden/blue hours.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop astro <location>' or set a default with 'raindrop config set location <name>'"
        )

    # CLI country flag overrides resolved country
    if country is not None:
        resolved_country = country

    location = resolved_location
    country = resolved_country

    result = geocode(location, country)

    # Get sunrise/sunset data for today and tomorrow
    weather = om.forecast(
        result.latitude,
        result.longitude,
        daily=[
            "sunrise",
            "sunset",
            "daylight_duration",
            "sunshine_duration",
            "uv_index_max",
        ],
        forecast_days=7,
    )

    d = weather.daily
    if d is None or not d.sunrise or not d.sunset:
        raise click.ClickException("No astronomical data returned")

    now = datetime.now()
    today = now.date()

    # Parse today's sun times
    sunrise = datetime.fromisoformat(d.sunrise[0])
    sunset = datetime.fromisoformat(d.sunset[0])

    # Calculate derived data
    daylight = daylight_duration(sunrise, sunset)
    noon = solar_noon(sunrise, sunset)
    morning_golden, evening_golden = golden_hour(sunrise, sunset)
    morning_blue, evening_blue = blue_hour(sunrise, sunset)

    # Moon data
    phase_val, phase_name, phase_symbol = moon_phase(now)
    illumination = moon_illumination(phase_val)

    # Next major phases
    next_new = next_moon_phase(now, 0)
    next_first_quarter = next_moon_phase(now, 0.25)
    next_full = next_moon_phase(now, 0.5)
    next_last_quarter = next_moon_phase(now, 0.75)

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
            "date": today.isoformat(),
            "sun": {
                "sunrise": d.sunrise[0],
                "sunset": d.sunset[0],
                "solar_noon": noon.isoformat(),
                "daylight_duration_seconds": daylight.total_seconds(),
                "daylight_duration_formatted": format_daylight_duration(daylight),
            },
            "golden_hour": {
                "morning": {
                    "start": morning_golden[0].isoformat(),
                    "end": morning_golden[1].isoformat(),
                },
                "evening": {
                    "start": evening_golden[0].isoformat(),
                    "end": evening_golden[1].isoformat(),
                },
            },
            "blue_hour": {
                "morning": {
                    "start": morning_blue[0].isoformat(),
                    "end": morning_blue[1].isoformat(),
                },
                "evening": {
                    "start": evening_blue[0].isoformat(),
                    "end": evening_blue[1].isoformat(),
                },
            },
            "moon": {
                "phase": phase_val,
                "phase_name": phase_name,
                "illumination": illumination,
                "next_new_moon": next_new.isoformat(),
                "next_first_quarter": next_first_quarter.isoformat(),
                "next_full_moon": next_full.isoformat(),
                "next_last_quarter": next_last_quarter.isoformat(),
            },
            "week_daylight": [
                {
                    "date": d.time[i],
                    "sunrise": d.sunrise[i],
                    "sunset": d.sunset[i],
                    "daylight_seconds": d.daylight_duration[i]
                    if d.daylight_duration
                    else None,
                }
                for i in range(min(7, len(d.time)))
            ],
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # Display
    console.print(
        f"\n[bold cyan]{result.name}, {result.admin1}, {result.country}[/bold cyan]"
    )
    console.print(f"[dim]Astronomical Data for {today.strftime('%A, %B %d')}[/dim]\n")

    # Sun section
    console.print("[bold yellow]Sun[/bold yellow]")

    sun_table = Table(show_header=False, box=None, padding=(0, 2))
    sun_table.add_column("label", style="dim")
    sun_table.add_column("value", style="bold")
    sun_table.add_column("label2", style="dim")
    sun_table.add_column("value2", style="bold")

    sun_table.add_row(
        "Sunrise",
        format_time(sunrise),
        "Sunset",
        format_time(sunset),
    )
    sun_table.add_row(
        "Solar Noon",
        format_time(noon),
        "Daylight",
        format_daylight_duration(daylight),
    )
    console.print(sun_table)

    # Current sun status
    if now < sunrise:
        console.print(f"  [cyan]Sunrise {format_time_until(now, sunrise)}[/cyan]")
    elif now < sunset:
        console.print(f"  [yellow]Sunset {format_time_until(now, sunset)}[/yellow]")
    else:
        console.print(f"  [dim]Sun has set[/dim]")

    console.print()

    # Golden & Blue Hour section
    console.print("[bold yellow]Photography Hours[/bold yellow]")

    photo_table = Table(show_header=True, box=box.SIMPLE, header_style="dim")
    photo_table.add_column("", style="dim")
    photo_table.add_column("Morning", justify="center")
    photo_table.add_column("Evening", justify="center")

    photo_table.add_row(
        "[blue]Blue Hour[/blue]",
        f"{format_time(morning_blue[0])} - {format_time(morning_blue[1])}",
        f"{format_time(evening_blue[0])} - {format_time(evening_blue[1])}",
    )
    photo_table.add_row(
        "[yellow]Golden Hour[/yellow]",
        f"{format_time(morning_golden[0])} - {format_time(morning_golden[1])}",
        f"{format_time(evening_golden[0])} - {format_time(evening_golden[1])}",
    )
    console.print(photo_table)
    console.print()

    # Moon section
    console.print("[bold white]Moon[/bold white]")
    console.print(
        f"  {phase_symbol} [bold]{phase_name}[/bold] ({illumination:.0f}% illuminated)"
    )
    console.print()

    # Upcoming phases
    console.print("[dim]Upcoming Phases:[/dim]")
    phases_table = Table(show_header=False, box=None, padding=(0, 1))
    phases_table.add_column("symbol")
    phases_table.add_column("name")
    phases_table.add_column("date", style="dim")

    upcoming = [
        ("\U0001f311", "New Moon", next_new),
        ("\U0001f313", "First Quarter", next_first_quarter),
        ("\U0001f315", "Full Moon", next_full),
        ("\U0001f317", "Last Quarter", next_last_quarter),
    ]
    # Sort by date
    upcoming.sort(key=lambda x: x[2])

    for symbol, name, date in upcoming[:4]:
        days_until = (date.date() - today).days
        if days_until == 0:
            date_str = "Today"
        elif days_until == 1:
            date_str = "Tomorrow"
        else:
            date_str = f"{date.strftime('%b %d')} ({days_until} days)"
        phases_table.add_row(f"  {symbol}", name, date_str)

    console.print(phases_table)
    console.print()

    # Weekly daylight trend
    console.print("[bold]Daylight This Week[/bold]")

    week_table = Table(show_header=True, box=box.SIMPLE, header_style="dim")
    week_table.add_column("Day", style="cyan")
    week_table.add_column("Sunrise")
    week_table.add_column("Sunset")
    week_table.add_column("Daylight")
    week_table.add_column("\u0394", justify="right")  # Delta

    prev_daylight = None
    for i in range(min(7, len(d.time))):
        date = datetime.fromisoformat(d.time[i]).date()
        sr = datetime.fromisoformat(d.sunrise[i])
        ss = datetime.fromisoformat(d.sunset[i])
        dl = daylight_duration(sr, ss)

        if date == today:
            day_str = "[bold yellow]Today[/bold yellow]"
        else:
            day_str = date.strftime("%a")

        # Calculate delta from previous day
        if prev_daylight is not None:
            delta_seconds = dl.total_seconds() - prev_daylight.total_seconds()
            delta_minutes = int(delta_seconds / 60)
            if delta_minutes > 0:
                delta_str = f"[green]+{delta_minutes}m[/green]"
            elif delta_minutes < 0:
                delta_str = f"[red]{delta_minutes}m[/red]"
            else:
                delta_str = "[dim]\u00b7[/dim]"
        else:
            delta_str = ""

        week_table.add_row(
            day_str,
            format_time(sr),
            format_time(ss),
            format_daylight_duration(dl),
            delta_str,
        )
        prev_daylight = dl

    console.print(week_table)
