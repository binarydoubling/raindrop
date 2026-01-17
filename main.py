from datetime import datetime

import click

from rich.console import Console
from rich.table import Table
from rich import box

from open_meteo import OpenMeteo, GeocodingResult
from settings import get_settings, AVAILABLE_MODELS

om = OpenMeteo()
console = Console()


def geocode(location: str, country: str | None = None) -> GeocodingResult:
    results = om.geocode(location, country_code=country)
    return results[0]


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """A simple, absolutely stunning weather CLI tool."""
    pass


# =============================================================================
# Config commands
# =============================================================================


@cli.group()
def config():
    """View and manage settings."""
    pass


@config.command("show")
def config_show():
    """Show current settings."""
    settings = get_settings()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("setting", style="dim")
    table.add_column("value", style="bold")

    table.add_row("location", settings.location or "(not set)")
    table.add_row("country_code", settings.country_code or "(not set)")
    table.add_row("temperature_unit", settings.temperature_unit)
    table.add_row("wind_speed_unit", settings.wind_speed_unit)
    table.add_row("precipitation_unit", settings.precipitation_unit)
    table.add_row("model", settings.model or "(auto)")

    console.print(table)


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a configuration value.

    \b
    Available settings:
      location           Default location name
      country_code       Default country code (e.g., US, ES, DE)
      temperature_unit   celsius or fahrenheit
      wind_speed_unit    kmh, ms, mph, or kn
      precipitation_unit mm or inch
      model              Weather model (see 'weather config models')
    """
    settings = get_settings()

    if key == "location":
        settings.location = value
    elif key == "country_code":
        settings.country_code = value.upper()
    elif key == "temperature_unit":
        if value not in ("celsius", "fahrenheit"):
            raise click.ClickException(
                "temperature_unit must be 'celsius' or 'fahrenheit'"
            )
        settings.temperature_unit = value  # type: ignore
    elif key == "wind_speed_unit":
        if value not in ("kmh", "ms", "mph", "kn"):
            raise click.ClickException(
                "wind_speed_unit must be 'kmh', 'ms', 'mph', or 'kn'"
            )
        settings.wind_speed_unit = value  # type: ignore
    elif key == "precipitation_unit":
        if value not in ("mm", "inch"):
            raise click.ClickException("precipitation_unit must be 'mm' or 'inch'")
        settings.precipitation_unit = value  # type: ignore
    elif key == "model":
        if value == "auto":
            settings.model = None
        elif value not in AVAILABLE_MODELS:
            raise click.ClickException(
                f"Unknown model: {value}. Run 'weather config models' to see available models."
            )
        else:
            settings.model = value
    else:
        raise click.ClickException(f"Unknown setting: {key}")

    settings.save()
    console.print(f"[green]Set {key} = {value}[/green]")


@config.command("unset")
@click.argument("key")
def config_unset(key: str):
    """Unset a configuration value (reset to default)."""
    settings = get_settings()

    if key == "location":
        settings.location = None
    elif key == "country_code":
        settings.country_code = None
    elif key == "model":
        settings.model = None
    elif key in ("temperature_unit", "wind_speed_unit", "precipitation_unit"):
        raise click.ClickException(f"Cannot unset {key}, use 'config set' to change it")
    else:
        raise click.ClickException(f"Unknown setting: {key}")

    settings.save()
    console.print(f"[green]Unset {key}[/green]")


@config.command("models")
def config_models():
    """List available weather models."""
    console.print("\n[bold]Available weather models:[/bold]\n")
    console.print("[dim]Use 'weather config set model <name>' to set a default.[/dim]")
    console.print("[dim]Or use '--model <name>' flag on any command.[/dim]\n")

    console.print("[cyan]Auto (default)[/cyan]")
    console.print("  [dim]Omit --model to let Open-Meteo choose the best model[/dim]\n")

    models_by_category = {
        "ECMWF (European)": ["ecmwf", "ecmwf_aifs"],
        "US (NOAA)": ["gfs", "hrrr"],
        "German (DWD)": ["icon", "icon_eu", "icon_d2"],
        "French (Météo-France)": ["arpege", "arome"],
        "UK (Met Office)": ["ukmo"],
        "Canadian (GEM)": ["gem", "gem_hrdps"],
        "Japanese (JMA)": ["jma"],
        "Norwegian (MET)": ["metno"],
    }

    for category, models in models_by_category.items():
        console.print(f"[cyan]{category}[/cyan]")
        for model in models:
            console.print(f"  {model}")
    console.print()


# =============================================================================
# Weather commands
# =============================================================================


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


TEMP_SYMBOLS = {"celsius": "C", "fahrenheit": "F"}
WIND_SYMBOLS = {"kmh": "km/h", "ms": "m/s", "mph": "mph", "kn": "kn"}

# Weather code to short labels with colors
WEATHER_LABELS: dict[int, tuple[str, str]] = {
    0: ("Clear", "yellow"),
    1: ("Clear", "yellow"),
    2: ("Cloudy", "white"),
    3: ("Cloudy", "dim white"),
    45: ("Fog", "dim white"),
    48: ("Fog", "dim white"),
    51: ("Drizzle", "cyan"),
    53: ("Drizzle", "cyan"),
    55: ("Drizzle", "blue"),
    61: ("Rain", "cyan"),
    63: ("Rain", "blue"),
    65: ("Rain", "bold blue"),
    71: ("Snow", "white"),
    73: ("Snow", "bold white"),
    75: ("Snow", "bold white"),
    80: ("Showers", "cyan"),
    81: ("Showers", "blue"),
    82: ("Storms", "bold blue"),
    95: ("Storms", "bold magenta"),
    96: ("Storms", "bold magenta"),
    99: ("Storms", "bold magenta"),
}


def format_delta(
    current: float, previous: float, unit: str = "", precision: int = 0
) -> str:
    """Format a value with its delta from the previous value."""
    delta = current - previous
    if abs(delta) < 1:
        delta_str = "[dim]·[/dim]"
    elif delta > 0:
        delta_str = f"[red]↑{abs(delta):.0f}[/red]"
    else:
        delta_str = f"[cyan]↓{abs(delta):.0f}[/cyan]"
    return f"{current:.{precision}f}{unit} {delta_str}"


def format_precip_chance(chance: int, prev_chance: int) -> str:
    """Format precipitation chance with emoji and delta."""
    if chance == 0:
        return "[dim]—[/dim]"

    delta = chance - prev_chance
    if abs(delta) < 1:
        delta_str = " [dim]·[/dim]"
    elif delta > 0:
        delta_str = f" [red]↑{abs(delta)}[/red]"
    else:
        delta_str = f" [cyan]↓{abs(delta)}[/cyan]"

    if chance >= 70:
        return f"[bold blue]{chance}%[/bold blue]{delta_str}"
    elif chance >= 40:
        return f"[blue]{chance}%[/blue]{delta_str}"
    else:
        return f"[dim]{chance}%[/dim]{delta_str}"


@cli.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option(
    "-m",
    "--model",
    "model_name",
    help="Weather model to use (see 'weather config models')",
)
def current(location: str | None, country: str | None, model_name: str | None):
    """Get current weather for a location."""
    settings = get_settings()

    # Use defaults from settings if not provided
    if location is None:
        location = settings.location
    if location is None:
        raise click.ClickException(
            "No location provided. Use 'weather current <location>' or set a default with 'weather config set location <name>'"
        )

    if country is None:
        country = settings.country_code

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
            "cloud_cover",
            "wind_speed_10m",
            "wind_gusts_10m",
            "weather_code",
            "is_day",
        ],
        temperature_unit=settings.temperature_unit,
        wind_speed_unit=settings.wind_speed_unit,
        models=models,
    )
    c = weather.current
    if c is None:
        raise click.ClickException("No current weather data returned")

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    wind_symbol = WIND_SYMBOLS[settings.wind_speed_unit]

    # Location header
    console.print(f"\n[bold cyan]{result.name}, {result.admin1}[/bold cyan]")

    # Weather condition
    condition = WEATHER_CODES.get(c.weather_code or 0, "Unknown")
    console.print(f"[dim]{condition}[/dim]\n")

    # Main stats table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("label", style="dim")
    table.add_column("value", style="bold")

    table.add_row("Temperature", f"{c.temperature_2m}°{temp_symbol}")
    table.add_row("Feels like", f"{c.apparent_temperature}°{temp_symbol}")
    table.add_row("Humidity", f"{c.relative_humidity_2m}%")
    table.add_row("Cloud cover", f"{c.cloud_cover}%")
    table.add_row("Wind", f"{c.wind_speed_10m} {wind_symbol}")
    table.add_row("Gusts", f"{c.wind_gusts_10m} {wind_symbol}")

    console.print(table)


@cli.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option("-n", "--hours", default=12, help="Number of hours to show (default: 12)")
@click.option(
    "-m",
    "--model",
    "model_name",
    help="Weather model to use (see 'weather config models')",
)
def hourly(
    location: str | None, country: str | None, hours: int, model_name: str | None
):
    """Show hourly forecast with deltas."""
    settings = get_settings()

    # Use defaults from settings if not provided
    if location is None:
        location = settings.location
    if location is None:
        raise click.ClickException(
            "No location provided. Use 'weather hourly <location>' or set a default with 'weather config set location <name>'"
        )

    if country is None:
        country = settings.country_code

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
    precip_symbol = settings.precipitation_unit

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

    # Location header
    console.print(f"\n[bold cyan]{result.name}, {result.admin1}[/bold cyan]")
    console.print(f"[dim]Next {hours} hours[/dim]\n")

    # Build the table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Time", style="cyan", justify="right")
    table.add_column("Weather", justify="left")
    table.add_column(f"Temp (°{temp_symbol})", justify="right")
    table.add_column("Feels", justify="right")
    table.add_column("Precip", justify="right")
    table.add_column(f"Wind ({wind_symbol})", justify="right")
    table.add_column("Humidity", justify="right")

    # Get data arrays (with None safety)
    temps = h.temperature_2m or []
    feels = h.apparent_temperature or []
    precip_probs = h.precipitation_probability or []
    codes = h.weather_code or []
    winds = h.wind_speed_10m or []
    humidities = h.relative_humidity_2m or []

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
        temp_str = format_delta(temp, prev_temp, "°", 0)
        feel_str = format_delta(feel, prev_feel, "°", 0)
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
    console.print("\n[dim]↑ rising  ↓ falling  · steady[/dim]")


if __name__ == "__main__":
    cli()
