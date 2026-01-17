"""Compare weather across locations command."""

import json as json_lib

import click
from rich.console import Console
from rich.table import Table
from rich import box

from open_meteo import OpenMeteo
from settings import get_settings
from raindrop.utils import (
    WEATHER_LABELS,
    TEMP_SYMBOLS,
    WIND_SYMBOLS,
)

om = OpenMeteo()
console = Console()


def geocode(location: str, country: str | None = None):
    results = om.geocode(location, country_code=country)
    return results[0]


@click.command()
@click.argument("locations", nargs=-1, required=True)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def compare(locations: tuple[str, ...], as_json: bool):
    """Compare current weather across multiple locations.

    Provide 2 or more locations (or favorite aliases) to compare.

    \b
    Examples:
      raindrop compare "San Francisco" "New York" "Miami"
      raindrop compare home work  # using favorites
    """
    if len(locations) < 2:
        raise click.ClickException("Please provide at least 2 locations to compare")

    settings = get_settings()
    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    wind_symbol = WIND_SYMBOLS[settings.wind_speed_unit]

    # Fetch weather for all locations
    results = []
    for loc in locations:
        try:
            # Resolve favorites
            resolved_loc, resolved_country = settings.resolve_location(loc)
            geo = geocode(resolved_loc, resolved_country)

            weather = om.forecast(
                geo.latitude,
                geo.longitude,
                current=[
                    "temperature_2m",
                    "apparent_temperature",
                    "weather_code",
                    "wind_speed_10m",
                    "relative_humidity_2m",
                    "precipitation",
                ],
                temperature_unit=settings.temperature_unit,
                wind_speed_unit=settings.wind_speed_unit,
                forecast_days=1,
            )

            c = weather.current
            if c:
                results.append(
                    {
                        "input": loc,
                        "name": geo.name,
                        "admin1": geo.admin1,
                        "country": geo.country,
                        "latitude": geo.latitude,
                        "longitude": geo.longitude,
                        "temperature": c.temperature_2m,
                        "feels_like": c.apparent_temperature,
                        "weather_code": c.weather_code,
                        "wind_speed": c.wind_speed_10m,
                        "humidity": c.relative_humidity_2m,
                        "precipitation": c.precipitation,
                        "time": c.time,
                    }
                )
        except Exception as e:
            results.append(
                {
                    "input": loc,
                    "error": str(e),
                }
            )

    # JSON output
    if as_json:
        data = {
            "locations": results,
            "units": {
                "temperature": settings.temperature_unit,
                "wind_speed": settings.wind_speed_unit,
            },
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # Table output
    console.print("\n[bold]Weather Comparison[/bold]\n")

    table = Table(show_header=True, box=box.ROUNDED, header_style="bold")
    table.add_column("Location", style="cyan")
    table.add_column("Temp", justify="right")
    table.add_column("Feels", justify="right")
    table.add_column("Weather", justify="left")
    table.add_column("Wind", justify="right")
    table.add_column("Humidity", justify="right")

    for r in results:
        if "error" in r:
            table.add_row(
                r["input"],
                "[red]Error[/red]",
                "",
                f"[dim]{r['error'][:30]}...[/dim]"
                if len(r.get("error", "")) > 30
                else f"[dim]{r.get('error', '')}[/dim]",
                "",
                "",
            )
        else:
            # Format location
            loc_str = r["name"]
            if r.get("admin1"):
                loc_str += f", {r['admin1'][:2]}"  # Abbreviate state

            # Weather label
            code = r.get("weather_code", 0) or 0
            label, color = WEATHER_LABELS.get(code, ("?", "white"))
            weather_str = f"[{color}]{label}[/{color}]"

            # Temperature with color
            temp = r.get("temperature", 0)
            if settings.temperature_unit == "fahrenheit":
                if temp >= 90:
                    temp_str = f"[red]{temp:.0f}\u00b0{temp_symbol}[/red]"
                elif temp >= 75:
                    temp_str = f"[yellow]{temp:.0f}\u00b0{temp_symbol}[/yellow]"
                elif temp <= 32:
                    temp_str = f"[cyan]{temp:.0f}\u00b0{temp_symbol}[/cyan]"
                else:
                    temp_str = f"{temp:.0f}\u00b0{temp_symbol}"
            else:
                if temp >= 32:
                    temp_str = f"[red]{temp:.0f}\u00b0{temp_symbol}[/red]"
                elif temp >= 24:
                    temp_str = f"[yellow]{temp:.0f}\u00b0{temp_symbol}[/yellow]"
                elif temp <= 0:
                    temp_str = f"[cyan]{temp:.0f}\u00b0{temp_symbol}[/cyan]"
                else:
                    temp_str = f"{temp:.0f}\u00b0{temp_symbol}"

            feels = r.get("feels_like", 0)
            feels_str = f"{feels:.0f}\u00b0{temp_symbol}"

            wind = r.get("wind_speed", 0)
            wind_str = f"{wind:.0f} {wind_symbol}"

            humidity = r.get("humidity", 0)
            humidity_str = f"{humidity}%"

            table.add_row(
                loc_str, temp_str, feels_str, weather_str, wind_str, humidity_str
            )

    console.print(table)

    # Find extremes
    valid = [r for r in results if "error" not in r]
    if len(valid) >= 2:
        temps = [(r["name"], r.get("temperature", 0)) for r in valid]
        hottest = max(temps, key=lambda x: x[1])
        coldest = min(temps, key=lambda x: x[1])
        diff = hottest[1] - coldest[1]

        console.print(
            f"\n[dim]Warmest: {hottest[0]} ({hottest[1]:.0f}\u00b0{temp_symbol})[/dim]"
        )
        console.print(
            f"[dim]Coolest: {coldest[0]} ({coldest[1]:.0f}\u00b0{temp_symbol})[/dim]"
        )
        console.print(f"[dim]Difference: {diff:.0f}\u00b0{temp_symbol}[/dim]")
