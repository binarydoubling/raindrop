"""Air quality command."""

from datetime import datetime
import json as json_lib

import click
from rich.console import Console
from rich.table import Table
from rich import box

from open_meteo import OpenMeteo
from settings import get_settings
from raindrop.utils import (
    sparkline,
    format_uv,
    format_pollutant,
    format_us_aqi,
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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def aqi(location: str | None, country: str | None, as_json: bool):
    """Show air quality index and pollutants.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop aqi <location>' or set a default with 'raindrop config set location <name>'"
        )

    # CLI country flag overrides resolved country
    if country is not None:
        resolved_country = country

    location = resolved_location
    country = resolved_country

    result = geocode(location, country)

    aq = om.air_quality(
        result.latitude,
        result.longitude,
        current=[
            "us_aqi",
            "european_aqi",
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "dust",
            "uv_index",
        ],
        hourly=[
            "us_aqi",
            "pm2_5",
            "pm10",
        ],
        forecast_days=2,
    )

    c = aq.current
    h = aq.hourly
    if c is None:
        raise click.ClickException("No air quality data returned")

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
            "current": {
                "time": c.time,
                "us_aqi": c.us_aqi,
                "european_aqi": c.european_aqi,
                "pm10": c.pm10,
                "pm2_5": c.pm2_5,
                "carbon_monoxide": c.carbon_monoxide,
                "nitrogen_dioxide": c.nitrogen_dioxide,
                "sulphur_dioxide": c.sulphur_dioxide,
                "ozone": c.ozone,
                "dust": c.dust,
                "uv_index": c.uv_index,
            },
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # Location header
    console.print(
        f"\n[bold cyan]{result.name}, {result.admin1}, {result.country}[/bold cyan]"
    )
    console.print("[dim]Air Quality Index[/dim]\n")

    # Main AQI display
    console.print(f"[bold]US AQI:[/bold] {format_us_aqi(c.us_aqi)}")
    if c.european_aqi is not None:
        console.print(f"[dim]European AQI: {c.european_aqi}[/dim]")
    console.print()

    # Pollutants table
    table = Table(show_header=True, box=box.ROUNDED, header_style="bold")
    table.add_column("Pollutant", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("PM2.5", format_pollutant(c.pm2_5))
    table.add_row("PM10", format_pollutant(c.pm10))
    table.add_row("Ozone (O\u2083)", format_pollutant(c.ozone))
    table.add_row("Nitrogen Dioxide (NO\u2082)", format_pollutant(c.nitrogen_dioxide))
    table.add_row("Sulphur Dioxide (SO\u2082)", format_pollutant(c.sulphur_dioxide))
    table.add_row("Carbon Monoxide (CO)", format_pollutant(c.carbon_monoxide))
    if c.dust is not None and c.dust > 0:
        table.add_row("Dust", format_pollutant(c.dust))
    if c.uv_index is not None:
        table.add_row("UV Index", format_uv(c.uv_index))

    console.print(table)

    # Hourly sparkline if available
    if h and h.us_aqi:
        # Find current hour index
        now = datetime.now()
        current_hour_str = now.strftime("%Y-%m-%dT%H:00")
        try:
            start_idx = h.time.index(current_hour_str)
        except ValueError:
            start_idx = 0

        # Get next 24 hours of AQI
        aqi_vals = h.us_aqi[start_idx : start_idx + 24]
        if aqi_vals:
            console.print(f"\n[dim]Next 24h AQI:[/dim] {sparkline(aqi_vals)}")
            aqi_clean = [a for a in aqi_vals if a is not None]
            if aqi_clean:
                console.print(f"[dim]Range: {min(aqi_clean)}-{max(aqi_clean)}[/dim]")

    # Legend
    console.print(
        "\n[dim]US AQI: 0-50 Good \u00b7 51-100 Moderate \u00b7 101-150 Sensitive \u00b7 151-200 Unhealthy \u00b7 201+ Very Unhealthy[/dim]"
    )
