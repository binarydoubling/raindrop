import click

from rich.console import Console
from rich.table import Table

from open_meteo import OpenMeteo, GeocodingResult

om = OpenMeteo()


def geocode(location: str, country: str | None = None) -> GeocodingResult:
    results = om.geocode(location, country_code=country)
    return results[0]


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """A simple, absolutely stunning weather CLI tool."""
    pass


WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@cli.command()
@click.argument("location")
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
def current(location: str, country: str | None):
    """Get current weather for a location."""
    result = geocode(location, country)

    weather = om.forecast(
        result.latitude,
        result.longitude,
        current=[
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "cloud_cover",
            "wind_speed_10m",
            "wind_gusts_10m",
            "weather_code",
            "is_day",
        ],
        temperature_unit="fahrenheit",
        wind_speed_unit="mph",
    )
    c = weather.current
    if c is None:
        raise click.ClickException("No current weather data returned")

    console = Console()

    # Location header
    console.print(f"\n[bold cyan]{result.name}, {result.admin1}[/bold cyan]")

    # Weather condition
    condition = WEATHER_CODES.get(c.weather_code or 0, "Unknown")
    console.print(f"[dim]{condition}[/dim]\n")

    # Main stats table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("label", style="dim")
    table.add_column("value", style="bold")

    table.add_row("Temperature", f"{c.temperature_2m}°F")
    table.add_row("Feels like", f"{c.apparent_temperature}°F")
    table.add_row("Humidity", f"{c.relative_humidity_2m}%")
    table.add_row("Cloud cover", f"{c.cloud_cover}%")
    table.add_row("Wind", f"{c.wind_speed_10m} mph")
    table.add_row("Gusts", f"{c.wind_gusts_10m} mph")

    console.print(table)


if __name__ == "__main__":
    cli()
