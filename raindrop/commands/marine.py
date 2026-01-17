"""Marine weather forecast command."""

from datetime import datetime
import json as json_lib

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from open_meteo import OpenMeteo
from settings import get_settings
from raindrop.utils import (
    WEATHER_CODES,
    WEATHER_LABELS,
    sparkline,
    deg_to_compass,
)

om = OpenMeteo()
console = Console()


# Marine API endpoint
MARINE_BASE_URL = "https://marine-api.open-meteo.com/v1"


def geocode(location: str, country: str | None = None):
    results = om.geocode(location, country_code=country)
    return results[0]


def get_marine_forecast(lat: float, lon: float, settings) -> dict:
    """Fetch marine weather forecast from Open-Meteo Marine API."""
    import urllib.request
    import urllib.parse
    import json

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(
            [
                "wave_height",
                "wave_direction",
                "wave_period",
                "wind_wave_height",
                "wind_wave_direction",
                "swell_wave_height",
                "swell_wave_direction",
                "swell_wave_period",
            ]
        ),
        "daily": ",".join(
            [
                "wave_height_max",
                "wave_direction_dominant",
                "wave_period_max",
                "wind_wave_height_max",
                "swell_wave_height_max",
            ]
        ),
        "timezone": "auto",
        "forecast_days": 7,
    }

    query = urllib.parse.urlencode(params)
    url = f"{MARINE_BASE_URL}/marine?{query}"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        raise click.ClickException(f"Could not fetch marine forecast: {e}")


def wave_height_color(height: float) -> str:
    """Get color for wave height."""
    if height < 0.5:
        return "green"
    elif height < 1.0:
        return "cyan"
    elif height < 2.0:
        return "yellow"
    elif height < 3.0:
        return "orange1"
    else:
        return "red"


def wave_conditions(height: float) -> str:
    """Describe wave conditions based on height."""
    if height < 0.3:
        return "Calm"
    elif height < 0.6:
        return "Smooth"
    elif height < 1.25:
        return "Slight"
    elif height < 2.5:
        return "Moderate"
    elif height < 4.0:
        return "Rough"
    elif height < 6.0:
        return "Very Rough"
    elif height < 9.0:
        return "High"
    else:
        return "Phenomenal"


@click.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option("-n", "--days", default=5, help="Number of days to show (default: 5)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def marine(location: str | None, country: str | None, days: int, as_json: bool):
    """Show marine/ocean weather forecast.

    Displays wave heights, periods, directions, and swell information
    for coastal and offshore locations.

    Best used for coastal cities or coordinates near water.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop marine <location>' or set a default with 'raindrop config set location <name>'"
        )

    # CLI country flag overrides resolved country
    if country is not None:
        resolved_country = country

    location = resolved_location
    country = resolved_country

    result = geocode(location, country)

    # Fetch marine data
    data = get_marine_forecast(result.latitude, result.longitude, settings)

    hourly = data.get("hourly", {})
    daily = data.get("daily", {})

    if not hourly.get("time") and not daily.get("time"):
        raise click.ClickException(
            "No marine data available for this location. "
            "Marine forecasts work best for coastal areas."
        )

    # JSON output
    if as_json:
        output = {
            "location": {
                "name": result.name,
                "admin1": result.admin1,
                "country": result.country,
                "latitude": result.latitude,
                "longitude": result.longitude,
            },
            "hourly": hourly,
            "daily": daily,
        }
        click.echo(json_lib.dumps(output, indent=2))
        return

    # Display header
    console.print(
        f"\n[bold cyan]{result.name}, {result.admin1}, {result.country}[/bold cyan]"
    )
    console.print("[dim]Marine Weather Forecast[/dim]\n")

    # Current conditions (first available hour)
    if hourly.get("wave_height"):
        now = datetime.now()
        current_hour = now.strftime("%Y-%m-%dT%H:00")

        try:
            idx = hourly["time"].index(current_hour)
        except ValueError:
            idx = 0

        wave_h = (
            hourly["wave_height"][idx]
            if hourly.get("wave_height") and idx < len(hourly["wave_height"])
            else None
        )
        wave_d = (
            hourly["wave_direction"][idx]
            if hourly.get("wave_direction") and idx < len(hourly["wave_direction"])
            else None
        )
        wave_p = (
            hourly["wave_period"][idx]
            if hourly.get("wave_period") and idx < len(hourly["wave_period"])
            else None
        )
        swell_h = (
            hourly["swell_wave_height"][idx]
            if hourly.get("swell_wave_height")
            and idx < len(hourly["swell_wave_height"])
            else None
        )

        if wave_h is not None:
            color = wave_height_color(wave_h)
            condition = wave_conditions(wave_h)

            console.print("[bold]Current Conditions[/bold]")
            console.print(f"  Waves: [{color}]{wave_h:.1f}m ({condition})[/{color}]")

            if wave_d is not None:
                compass = deg_to_compass(int(wave_d))
                console.print(f"  Direction: {compass} ({wave_d:.0f}\u00b0)")

            if wave_p is not None:
                console.print(f"  Period: {wave_p:.0f}s")

            if swell_h is not None and swell_h > 0:
                console.print(f"  Swell: {swell_h:.1f}m")

            console.print()

    # Next 24 hours sparklines
    if hourly.get("wave_height"):
        wave_heights = hourly["wave_height"][:24]
        swell_heights = (hourly.get("swell_wave_height") or [])[:24]

        console.print("[bold]Next 24 Hours[/bold]")

        wh_clean = [w for w in wave_heights if w is not None]
        if wh_clean:
            console.print(
                f"  [dim]Waves:[/dim]  {sparkline(wave_heights)}  {min(wh_clean):.1f}-{max(wh_clean):.1f}m"
            )

        sh_clean = [s for s in swell_heights if s is not None]
        if sh_clean and max(sh_clean) > 0:
            console.print(
                f"  [dim]Swell:[/dim]  {sparkline(swell_heights)}  {min(sh_clean):.1f}-{max(sh_clean):.1f}m"
            )

        console.print()

    # Daily forecast table
    if daily.get("time"):
        console.print("[bold]Daily Forecast[/bold]\n")

        table = Table(show_header=True, box=box.ROUNDED, header_style="bold")
        table.add_column("Day", style="cyan")
        table.add_column("Max Waves", justify="right")
        table.add_column("Conditions")
        table.add_column("Direction")
        table.add_column("Swell", justify="right")

        today = datetime.now().date()

        for i in range(min(days, len(daily["time"]))):
            date = datetime.fromisoformat(daily["time"][i]).date()

            if date == today:
                day_str = "[bold yellow]Today[/bold yellow]"
            else:
                day_str = date.strftime("%a")

            wave_max = daily.get("wave_height_max", [None])[i]
            wave_dir = daily.get("wave_direction_dominant", [None])[i]
            swell_max = daily.get("swell_wave_height_max", [None])[i]

            if wave_max is not None:
                color = wave_height_color(wave_max)
                wave_str = f"[{color}]{wave_max:.1f}m[/{color}]"
                condition = wave_conditions(wave_max)
                cond_str = f"[{color}]{condition}[/{color}]"
            else:
                wave_str = "\u2014"
                cond_str = "\u2014"

            if wave_dir is not None:
                compass = deg_to_compass(int(wave_dir))
                dir_str = f"{compass} ({wave_dir:.0f}\u00b0)"
            else:
                dir_str = "\u2014"

            if swell_max is not None and swell_max > 0:
                swell_str = f"{swell_max:.1f}m"
            else:
                swell_str = "[dim]\u2014[/dim]"

            table.add_row(day_str, wave_str, cond_str, dir_str, swell_str)

        console.print(table)

    # Legend
    console.print(
        "\n[dim]Wave Scale: Calm <0.3m \u00b7 Smooth 0.3-0.6m \u00b7 Slight 0.6-1.25m \u00b7 Moderate 1.25-2.5m \u00b7 Rough 2.5-4m[/dim]"
    )
