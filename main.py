from datetime import datetime, timedelta
import json as json_lib

import click

from rich.console import Console
from rich.table import Table
from rich import box

from open_meteo import OpenMeteo, GeocodingResult, NWSClient
from settings import get_settings, AVAILABLE_MODELS

om = OpenMeteo()
nws = NWSClient()
console = Console()


def geocode(location: str, country: str | None = None) -> GeocodingResult:
    results = om.geocode(location, country_code=country)
    return results[0]


def deg_to_compass(deg: int) -> str:
    """Convert degrees to compass direction."""
    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    idx = round(deg / 22.5) % 16
    return directions[idx]


def format_visibility(meters: float) -> str:
    """Format visibility in human-readable form."""
    if meters >= 10000:
        return f"{meters / 1000:.0f} km"
    elif meters >= 1000:
        return f"{meters / 1000:.1f} km"
    else:
        return f"{meters:.0f} m"


def format_uv(uv: float) -> str:
    """Format UV index with risk level."""
    if uv < 3:
        return f"{uv:.1f} [green](Low)[/green]"
    elif uv < 6:
        return f"{uv:.1f} [yellow](Moderate)[/yellow]"
    elif uv < 8:
        return f"{uv:.1f} [orange1](High)[/orange1]"
    elif uv < 11:
        return f"{uv:.1f} [red](Very High)[/red]"
    else:
        return f"{uv:.1f} [magenta](Extreme)[/magenta]"


def format_duration(td: timedelta) -> str:
    """Format a timedelta as human readable."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


# =============================================================================
# Sparkline Functions
# =============================================================================

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float | int | None]) -> str:
    """Generate a sparkline string from a list of values."""
    clean_values = [v for v in values if v is not None]
    if not clean_values:
        return ""

    min_val = min(clean_values)
    max_val = max(clean_values)
    val_range = max_val - min_val

    result = []
    for v in values:
        if v is None:
            result.append(" ")
        elif val_range == 0:
            result.append(SPARK_CHARS[3])
        else:
            idx = int((v - min_val) / val_range * 7)
            idx = min(7, max(0, idx))
            result.append(SPARK_CHARS[idx])

    return "".join(result)


# =============================================================================
# Technical Analysis Functions
# =============================================================================


def ema(values: list[float], period: int) -> list[float | None]:
    """Calculate Exponential Moving Average."""
    if len(values) < period:
        return [None] * len(values)

    result: list[float | None] = [None] * (period - 1)
    multiplier = 2 / (period + 1)

    # First EMA is SMA
    sma = sum(values[:period]) / period
    result.append(sma)

    # Calculate EMA for remaining values
    for i in range(period, len(values)):
        ema_val = (values[i] - result[-1]) * multiplier + result[-1]  # type: ignore
        result.append(ema_val)

    return result


def calc_roc(values: list[float], period: int = 3) -> list[float | None]:
    """
    Calculate Rate of Change over a period.
    Returns the temperature change over the last N days.
    """
    result: list[float | None] = [None] * period
    for i in range(period, len(values)):
        roc = values[i] - values[i - period]
        result.append(roc)
    return result


def calc_volatility(highs: list[float], lows: list[float]) -> list[float]:
    """Calculate daily temperature range (volatility)."""
    return [h - l for h, l in zip(highs, lows)]


def trend_signal(
    value: float, ema_short: float | None, ema_long: float | None
) -> tuple[str, str]:
    """
    Determine trend signal based on EMA crossover.
    Returns (signal, color).
    """
    if ema_short is None or ema_long is None:
        return ("—", "dim")

    diff = ema_short - ema_long
    pct_diff = (diff / ema_long) * 100 if ema_long != 0 else 0

    if pct_diff > 2:
        return ("▲ Hot", "red")
    elif pct_diff > 0.5:
        return ("↗ Warming", "yellow")
    elif pct_diff < -2:
        return ("▼ Cold", "cyan")
    elif pct_diff < -0.5:
        return ("↘ Cooling", "blue")
    else:
        return ("→ Stable", "dim")


def roc_signal(roc: float | None) -> tuple[str, str]:
    """
    Interpret rate of change as a momentum signal.
    Returns (description, color).
    """
    if roc is None:
        return ("—", "dim")

    if roc >= 10:
        return ("Surging", "bold red")
    elif roc >= 5:
        return ("Rising", "red")
    elif roc >= 2:
        return ("Warming", "yellow")
    elif roc <= -10:
        return ("Plunging", "bold cyan")
    elif roc <= -5:
        return ("Falling", "cyan")
    elif roc <= -2:
        return ("Cooling", "blue")
    else:
        return ("Steady", "dim")


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
# Favorites commands
# =============================================================================


@cli.group()
def fav():
    """Manage favorite locations."""
    pass


@fav.command("list")
def fav_list():
    """List all saved favorites."""
    settings = get_settings()

    if not settings.favorites:
        console.print("[dim]No favorites saved yet.[/dim]")
        console.print(
            "[dim]Use 'raindrop fav add <alias> <location>' to add one.[/dim]"
        )
        return

    table = Table(show_header=True, box=box.ROUNDED, header_style="bold")
    table.add_column("Alias", style="cyan")
    table.add_column("Location")
    table.add_column("Country", style="dim")

    for alias, fav in sorted(settings.favorites.items()):
        table.add_row(alias, fav.name, fav.country_code or "—")

    console.print(table)


@fav.command("add")
@click.argument("alias")
@click.argument("location")
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
def fav_add(alias: str, location: str, country: str | None):
    """Add a favorite location.

    \b
    Examples:
      raindrop fav add home "San Francisco"
      raindrop fav add work "New York" -c US
      raindrop fav add parents "Paris" -c FR
    """
    from settings import Favorite

    settings = get_settings()

    # Validate by attempting to geocode
    try:
        result = geocode(location, country)
    except Exception as e:
        raise click.ClickException(f"Could not find location: {e}")

    settings.favorites[alias] = Favorite(
        name=result.name,
        country_code=country.upper() if country else None,
    )
    settings.save()

    console.print(
        f"[green]Added favorite '{alias}' -> {result.name}, {result.admin1}, {result.country}[/green]"
    )


@fav.command("remove")
@click.argument("alias")
def fav_remove(alias: str):
    """Remove a favorite location."""
    settings = get_settings()

    if alias not in settings.favorites:
        raise click.ClickException(f"Favorite '{alias}' not found")

    del settings.favorites[alias]
    settings.save()

    console.print(f"[green]Removed favorite '{alias}'[/green]")


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
            f"{result.name}: {c.temperature_2m:.0f}°{temp_symbol} {condition} | "
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
        f"[dim]{abs(result.latitude):.4f}°{lat_dir} {abs(result.longitude):.4f}°{lon_dir} · "
        f"Elev {weather.elevation:.0f}m · {weather.timezone}[/dim]"
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
        f"{c.temperature_2m}°{temp_symbol}",
        "Feels like",
        f"{c.apparent_temperature}°{temp_symbol}",
    )

    # Row 2: Humidity / Dew point
    table.add_row(
        "Humidity",
        f"{c.relative_humidity_2m}%",
        "Dew point",
        f"{c.dew_point_2m}°{temp_symbol}" if c.dew_point_2m else "—",
    )

    # Row 3: Wind speed + direction / Gusts
    wind_dir = c.wind_direction_10m
    if wind_dir is not None:
        compass = deg_to_compass(wind_dir)
        wind_str = f"{c.wind_speed_10m} {wind_symbol} {compass} ({wind_dir}°)"
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
        f"{c.pressure_msl} hPa" if c.pressure_msl else "—",
        "Surface",
        f"{c.surface_pressure} hPa" if c.surface_pressure else "—",
    )

    # Row 5: Visibility / Cloud cover
    visibility_str = format_visibility(c.visibility) if c.visibility else "—"
    table.add_row(
        "Visibility",
        visibility_str,
        "Cloud cover",
        f"{c.cloud_cover}%",
    )

    # Row 6: UV Index / UV Max today
    uv_str = format_uv(c.uv_index) if c.uv_index is not None else "—"
    uv_max = d.uv_index_max[0] if d and d.uv_index_max else None
    uv_max_str = format_uv(uv_max) if uv_max is not None else "—"
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
            f"{min(temp_clean):.0f}-{max(temp_clean):.0f}°{temp_symbol}"
            if temp_clean
            else "—"
        )
        wind_range = (
            f"{min(wind_clean):.0f}-{max(wind_clean):.0f} {wind_symbol}"
            if wind_clean
            else "—"
        )
        precip_max = (
            f"{max(precip_clean):.0f}%"
            if precip_clean and max(precip_clean) > 0
            else "—"
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
    table.add_column(f"Temp (°{temp_symbol})", justify="right")
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


@cli.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option("-n", "--days", default=10, help="Number of days to show (default: 10)")
@click.option(
    "-m",
    "--model",
    "model_name",
    help="Weather model to use (see 'raindrop config models')",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def daily(
    location: str | None,
    country: str | None,
    days: int,
    model_name: str | None,
    as_json: bool,
):
    """Show daily forecast with technical analysis indicators.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop daily <location>' or set a default with 'raindrop config set location <name>'"
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
        daily=[
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "uv_index_max",
        ],
        temperature_unit=settings.temperature_unit,
        wind_speed_unit=settings.wind_speed_unit,
        precipitation_unit=settings.precipitation_unit,
        models=models,
        forecast_days=min(days, 16),
    )
    d = weather.daily
    if d is None:
        raise click.ClickException("No daily weather data returned")

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    wind_symbol = WIND_SYMBOLS[settings.wind_speed_unit]
    precip_symbol = settings.precipitation_unit

    # Get data arrays
    times = d.time
    codes = d.weather_code or []
    highs = d.temperature_2m_max or []
    lows = d.temperature_2m_min or []
    precip_probs = d.precipitation_probability_max or []
    precip_sums = d.precipitation_sum or []
    wind_maxs = d.wind_speed_10m_max or []
    uv_maxs = d.uv_index_max or []

    # Calculate technical indicators using average temperature
    avg_temps = [(h + l) / 2 for h, l in zip(highs, lows)]
    ema_3 = ema(avg_temps, 3)  # Short-term EMA
    ema_7 = ema(avg_temps, 7)  # Long-term EMA
    roc_vals = calc_roc(avg_temps, 3)  # 3-day rate of change
    volatility = calc_volatility(highs, lows)
    wind_gusts = d.wind_gusts_10m_max or []

    # JSON output
    if as_json:
        daily_data = []
        for i in range(min(len(times), days)):
            code = codes[i] if i < len(codes) else 0
            ema_s = ema_3[i] if i < len(ema_3) else None
            ema_l = ema_7[i] if i < len(ema_7) else None
            trend_txt, _ = trend_signal(avg_temps[i], ema_s, ema_l)
            roc = roc_vals[i] if i < len(roc_vals) else None

            daily_data.append(
                {
                    "date": times[i],
                    "temperature_max": highs[i] if i < len(highs) else None,
                    "temperature_min": lows[i] if i < len(lows) else None,
                    "temperature_avg": avg_temps[i] if i < len(avg_temps) else None,
                    "weather_code": code,
                    "weather_description": WEATHER_CODES.get(code, "Unknown"),
                    "precipitation_probability": precip_probs[i]
                    if i < len(precip_probs)
                    else None,
                    "precipitation_sum": precip_sums[i]
                    if i < len(precip_sums)
                    else None,
                    "wind_speed_max": wind_maxs[i] if i < len(wind_maxs) else None,
                    "wind_gusts_max": wind_gusts[i] if i < len(wind_gusts) else None,
                    "uv_index_max": uv_maxs[i] if i < len(uv_maxs) else None,
                    "analysis": {
                        "ema_3": ema_s,
                        "ema_7": ema_l,
                        "trend": trend_txt,
                        "rate_of_change_3d": roc,
                        "daily_range": volatility[i] if i < len(volatility) else None,
                    },
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
            "days": daily_data,
            "units": {
                "temperature": settings.temperature_unit,
                "wind_speed": settings.wind_speed_unit,
                "precipitation": settings.precipitation_unit,
            },
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # Location header
    console.print(
        f"\n[bold cyan]{result.name}, {result.admin1}, {result.country}[/bold cyan]"
    )
    console.print(f"[dim]{days}-day forecast · Model: {model_key or 'auto'}[/dim]\n")

    # Main forecast table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Date", style="cyan", justify="right")
    table.add_column("Weather", justify="left")
    table.add_column("High", justify="right")
    table.add_column("Low", justify="right")
    table.add_column("Range", justify="right")
    table.add_column("Precip", justify="right")
    table.add_column("Wind", justify="right")
    table.add_column("Trend", justify="center")
    table.add_column("Δ3d", justify="right")  # 3-day rate of change

    today = datetime.now().date()

    for i in range(min(len(times), days)):
        date = datetime.fromisoformat(times[i]).date()

        # Date display
        if date == today:
            date_str = "[bold yellow]Today[/bold yellow]"
        elif date == today + timedelta(days=1):
            date_str = "Tomorrow"
        else:
            date_str = date.strftime("%a %d")

        # Weather
        code = codes[i] if i < len(codes) else 0
        label, color = WEATHER_LABELS.get(code, ("?", "white"))
        weather_str = f"[{color}]{label}[/{color}]"

        # Temps
        high = highs[i] if i < len(highs) else 0
        low = lows[i] if i < len(lows) else 0
        vol = volatility[i] if i < len(volatility) else 0

        # High with delta
        if i > 0 and i < len(highs):
            high_delta = high - highs[i - 1]
            if abs(high_delta) < 1:
                high_str = f"{high:.0f}°{temp_symbol} [dim]·[/dim]"
            elif high_delta > 0:
                high_str = f"{high:.0f}°{temp_symbol} [red]↑{abs(high_delta):.0f}[/red]"
            else:
                high_str = (
                    f"{high:.0f}°{temp_symbol} [cyan]↓{abs(high_delta):.0f}[/cyan]"
                )
        else:
            high_str = f"{high:.0f}°{temp_symbol}"

        # Low with delta
        if i > 0 and i < len(lows):
            low_delta = low - lows[i - 1]
            if abs(low_delta) < 1:
                low_str = f"{low:.0f}°{temp_symbol} [dim]·[/dim]"
            elif low_delta > 0:
                low_str = f"{low:.0f}°{temp_symbol} [red]↑{abs(low_delta):.0f}[/red]"
            else:
                low_str = f"{low:.0f}°{temp_symbol} [cyan]↓{abs(low_delta):.0f}[/cyan]"
        else:
            low_str = f"{low:.0f}°{temp_symbol}"

        # Range (volatility)
        range_str = f"{vol:.0f}°"

        # Precipitation
        prob = precip_probs[i] if i < len(precip_probs) else 0
        amount = precip_sums[i] if i < len(precip_sums) else 0
        if prob == 0:
            precip_str = "[dim]—[/dim]"
        elif prob >= 70:
            precip_str = f"[bold blue]{prob}%[/bold blue] {amount:.1f}{precip_symbol}"
        elif prob >= 40:
            precip_str = f"[blue]{prob}%[/blue] {amount:.1f}{precip_symbol}"
        else:
            precip_str = f"[dim]{prob}%[/dim]"

        # Wind
        wind = wind_maxs[i] if i < len(wind_maxs) else 0
        wind_str = f"{wind:.0f} {wind_symbol}"

        # Trend signal (EMA crossover)
        ema_s = ema_3[i] if i < len(ema_3) else None
        ema_l = ema_7[i] if i < len(ema_7) else None
        trend_txt, trend_color = trend_signal(avg_temps[i], ema_s, ema_l)
        trend_str = f"[{trend_color}]{trend_txt}[/{trend_color}]"

        # Rate of change (3-day)
        roc = roc_vals[i] if i < len(roc_vals) else None
        if roc is not None:
            roc_txt, roc_color = roc_signal(roc)
            roc_str = f"[{roc_color}]{roc:+.0f}°[/{roc_color}]"
        else:
            roc_str = "[dim]—[/dim]"

        table.add_row(
            date_str,
            weather_str,
            high_str,
            low_str,
            range_str,
            precip_str,
            wind_str,
            trend_str,
            roc_str,
        )

    console.print(table)

    # Technical analysis summary
    console.print("\n[bold]Technical Analysis[/bold]")

    # Current trend
    if len(ema_3) > 0 and len(ema_7) > 0:
        latest_ema3 = ema_3[-1]
        latest_ema7 = ema_7[-1]
        if latest_ema3 is not None and latest_ema7 is not None:
            trend_txt, trend_color = trend_signal(
                avg_temps[-1], latest_ema3, latest_ema7
            )
            ema_diff = latest_ema3 - latest_ema7
            console.print(
                f"[dim]EMA(3):[/dim] {latest_ema3:.1f}°  "
                f"[dim]EMA(7):[/dim] {latest_ema7:.1f}°  "
                f"[dim]Spread:[/dim] [{trend_color}]{ema_diff:+.1f}°[/{trend_color}]  "
                f"[dim]Signal:[/dim] [{trend_color}]{trend_txt}[/{trend_color}]"
            )

    # Rate of change
    if len(roc_vals) > 0 and roc_vals[-1] is not None:
        roc_val = roc_vals[-1]
        roc_txt, roc_color = roc_signal(roc_val)
        console.print(
            f"[dim]3-day Δ:[/dim] [{roc_color}]{roc_val:+.1f}°[/{roc_color}]  "
            f"[dim]Rate:[/dim] [{roc_color}]{roc_txt}[/{roc_color}]"
        )

    # Volatility trend
    if len(volatility) >= 3:
        recent_vol = sum(volatility[-3:]) / 3
        earlier_vol = sum(volatility[:3]) / 3 if len(volatility) >= 6 else recent_vol
        vol_change = recent_vol - earlier_vol
        if vol_change > 2:
            vol_trend = "[red]Increasing[/red]"
        elif vol_change < -2:
            vol_trend = "[green]Decreasing[/green]"
        else:
            vol_trend = "[dim]Stable[/dim]"
        console.print(
            f"[dim]Avg Range:[/dim] {recent_vol:.1f}°  "
            f"[dim]Volatility:[/dim] {vol_trend}"
        )

    # Legend
    console.print(
        "\n[dim]Trend: EMA(3)/EMA(7) crossover · Δ3d: 3-day temperature change[/dim]"
    )
    console.print("[dim]▲ Hot · ↗ Warming · → Stable · ↘ Cooling · ▼ Cold[/dim]")


# =============================================================================
# Air Quality command
# =============================================================================

# US AQI levels and colors
US_AQI_LEVELS = [
    (50, "Good", "green"),
    (100, "Moderate", "yellow"),
    (150, "Unhealthy (Sensitive)", "orange1"),
    (200, "Unhealthy", "red"),
    (300, "Very Unhealthy", "magenta"),
    (500, "Hazardous", "bold red"),
]


def format_us_aqi(aqi: int | None) -> str:
    """Format US AQI with level and color."""
    if aqi is None:
        return "—"
    for threshold, level, color in US_AQI_LEVELS:
        if aqi <= threshold:
            return f"[{color}]{aqi} ({level})[/{color}]"
    return f"[bold red]{aqi} (Hazardous)[/bold red]"


def format_pollutant(value: float | None, unit: str = "μg/m³") -> str:
    """Format a pollutant value."""
    if value is None:
        return "—"
    return f"{value:.1f} {unit}"


@cli.command()
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
    console.print(f"[dim]Air Quality Index[/dim]\n")

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
    table.add_row("Ozone (O₃)", format_pollutant(c.ozone))
    table.add_row("Nitrogen Dioxide (NO₂)", format_pollutant(c.nitrogen_dioxide))
    table.add_row("Sulphur Dioxide (SO₂)", format_pollutant(c.sulphur_dioxide))
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
        "\n[dim]US AQI: 0-50 Good · 51-100 Moderate · 101-150 Sensitive · 151-200 Unhealthy · 201+ Very Unhealthy[/dim]"
    )


# =============================================================================
# NWS Forecast Discussion command
# =============================================================================


def format_discussion(text: str) -> str:
    """Format NWS forecast discussion text for pretty printing."""
    import re

    lines = text.strip().split("\n")
    formatted_lines = []

    for line in lines:
        # Skip header lines (first few lines with codes)
        if line.startswith("000") or line.startswith("FX") or line.startswith("AFD"):
            continue

        # Section headers start with . and end with ...
        if line.startswith(".") and "..." in line:
            section_match = re.match(r"\.([A-Z][A-Z\s/]+)\.\.\.", line)
            if section_match:
                current_section = section_match.group(1).strip()
                formatted_lines.append(f"\n[bold cyan]{current_section}[/bold cyan]")
                # Get any text after the ...
                after = line.split("...")[-1].strip()
                if after:
                    formatted_lines.append(after)
                continue

        # Key messages header
        if "KEY MESSAGES" in line:
            formatted_lines.append(f"\n[bold yellow]KEY MESSAGES[/bold yellow]")
            continue

        # Issued/Updated timestamps
        if line.strip().startswith("Issued at") or line.strip().startswith(
            "Updated at"
        ):
            formatted_lines.append(f"[dim]{line.strip()}[/dim]")
            continue

        # Skip && separators
        if line.strip() == "&&":
            continue

        # Skip $$ end markers
        if line.strip() == "$$":
            break

        # Bullet points (lines starting with -)
        if line.strip().startswith("-"):
            formatted_lines.append(f"  [yellow]*[/yellow]{line.strip()[1:]}")
            continue

        # Regular content
        if line.strip():
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


@cli.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option("--raw", is_flag=True, help="Show raw unformatted text")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def discussion(location: str | None, country: str | None, raw: bool, as_json: bool):
    """Show NWS Area Forecast Discussion.

    Displays the meteorologist's technical forecast discussion from the
    National Weather Service. Only available for US locations.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop discussion <location>' or set a default with 'raindrop config set location <name>'"
        )

    # CLI country flag overrides resolved country
    if country is not None:
        resolved_country = country

    location = resolved_location
    country = resolved_country

    result = geocode(location, country)

    # Check if in US
    if result.country_code != "US":
        raise click.ClickException(
            f"NWS forecast discussions are only available for US locations. "
            f"{result.name} is in {result.country}."
        )

    # Get NWS office and discussion
    try:
        office = nws.get_office_for_point(result.latitude, result.longitude)
        disc = nws.get_latest_discussion(office.id)
    except Exception as e:
        raise click.ClickException(f"Could not fetch NWS discussion: {e}")

    # Parse issuance time
    from datetime import datetime as dt

    try:
        issued = dt.fromisoformat(disc.issuance_time.replace("Z", "+00:00"))
        issued_str = issued.strftime("%B %d, %Y at %-I:%M %p %Z")
    except Exception:
        issued_str = disc.issuance_time

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
            "office": {
                "id": office.id,
                "name": office.name,
            },
            "discussion": {
                "id": disc.id,
                "issuance_time": disc.issuance_time,
                "text": disc.product_text,
            },
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # Header
    console.print(f"\n[bold cyan]{result.name}, {result.admin1}[/bold cyan]")
    console.print(f"[dim]NWS {office.id} Area Forecast Discussion[/dim]")
    console.print(f"[dim]Issued: {issued_str}[/dim]\n")

    # Discussion text
    if raw:
        console.print(disc.product_text)
    else:
        formatted = format_discussion(disc.product_text)
        console.print(formatted)

    console.print(f"\n[dim]Source: NWS {office.id} | forecast.weather.gov[/dim]")


# =============================================================================
# Precipitation command
# =============================================================================


@cli.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option("-n", "--days", default=7, help="Number of days to show (default: 7)")
@click.option(
    "-m",
    "--model",
    "model_name",
    help="Weather model to use (see 'raindrop config models')",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def precip(
    location: str | None,
    country: str | None,
    days: int,
    model_name: str | None,
    as_json: bool,
):
    """Show precipitation forecast and accumulation.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop precip <location>' or set a default with 'raindrop config set location <name>'"
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
        daily=[
            "precipitation_sum",
            "precipitation_probability_max",
            "precipitation_hours",
            "rain_sum",
            "showers_sum",
            "snowfall_sum",
            "weather_code",
        ],
        hourly=[
            "precipitation",
            "precipitation_probability",
        ],
        precipitation_unit=settings.precipitation_unit,
        models=models,
        forecast_days=min(days, 16),
    )

    d = weather.daily
    h = weather.hourly
    if d is None:
        raise click.ClickException("No precipitation data returned")

    precip_symbol = settings.precipitation_unit
    times = d.time
    precip_sums = d.precipitation_sum or []
    precip_probs = d.precipitation_probability_max or []
    precip_hours = d.precipitation_hours or []
    rain_sums = d.rain_sum or []
    snow_sums = d.snowfall_sum or []
    codes = d.weather_code or []

    # Calculate totals
    total_precip = sum(p for p in precip_sums[:days] if p is not None)
    total_rain = sum(r for r in rain_sums[:days] if r is not None)
    total_snow = sum(s for s in snow_sums[:days] if s is not None)
    total_hours = sum(h for h in precip_hours[:days] if h is not None)

    # JSON output
    if as_json:
        daily_data = []
        for i in range(min(len(times), days)):
            daily_data.append(
                {
                    "date": times[i],
                    "precipitation_sum": precip_sums[i]
                    if i < len(precip_sums)
                    else None,
                    "precipitation_probability": precip_probs[i]
                    if i < len(precip_probs)
                    else None,
                    "precipitation_hours": precip_hours[i]
                    if i < len(precip_hours)
                    else None,
                    "rain_sum": rain_sums[i] if i < len(rain_sums) else None,
                    "snowfall_sum": snow_sums[i] if i < len(snow_sums) else None,
                    "weather_code": codes[i] if i < len(codes) else None,
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
            "totals": {
                "precipitation": total_precip,
                "rain": total_rain,
                "snow": total_snow,
                "hours": total_hours,
            },
            "days": daily_data,
            "units": {
                "precipitation": settings.precipitation_unit,
            },
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # Header
    console.print(
        f"\n[bold cyan]{result.name}, {result.admin1}, {result.country}[/bold cyan]"
    )
    console.print(f"[dim]{days}-day precipitation forecast[/dim]\n")

    # Summary
    if total_precip > 0:
        console.print(
            f"[bold]Total Expected:[/bold] {total_precip:.1f} {precip_symbol}"
        )
        if total_rain > 0:
            console.print(f"  [blue]Rain:[/blue] {total_rain:.1f} {precip_symbol}")
        if total_snow > 0:
            console.print(f"  [white]Snow:[/white] {total_snow:.1f} {precip_symbol}")
        if total_hours > 0:
            console.print(f"  [dim]~{total_hours:.0f} hours of precipitation[/dim]")
        console.print()
    else:
        console.print("[green]No precipitation expected[/green]\n")

    # Daily breakdown table
    table = Table(show_header=True, box=box.ROUNDED, header_style="bold")
    table.add_column("Date", style="cyan", justify="right")
    table.add_column("Chance", justify="right")
    table.add_column("Amount", justify="right")
    table.add_column("Type", justify="left")
    table.add_column("Hours", justify="right")
    table.add_column("Accumulation", justify="right")

    today = datetime.now().date()
    running_total = 0.0

    for i in range(min(len(times), days)):
        date = datetime.fromisoformat(times[i]).date()

        # Date display
        if date == today:
            date_str = "[bold yellow]Today[/bold yellow]"
        elif date == today + timedelta(days=1):
            date_str = "Tomorrow"
        else:
            date_str = date.strftime("%a %d")

        # Values
        prob = precip_probs[i] if i < len(precip_probs) else 0
        amount = precip_sums[i] if i < len(precip_sums) else 0
        rain = rain_sums[i] if i < len(rain_sums) else 0
        snow = snow_sums[i] if i < len(snow_sums) else 0
        hours = precip_hours[i] if i < len(precip_hours) else 0

        # Chance formatting
        if prob == 0:
            chance_str = "[dim]—[/dim]"
        elif prob >= 70:
            chance_str = f"[bold blue]{prob}%[/bold blue]"
        elif prob >= 40:
            chance_str = f"[blue]{prob}%[/blue]"
        else:
            chance_str = f"[dim]{prob}%[/dim]"

        # Amount formatting
        if amount == 0:
            amount_str = "[dim]—[/dim]"
        elif amount >= 10:
            amount_str = f"[bold blue]{amount:.1f} {precip_symbol}[/bold blue]"
        elif amount >= 2:
            amount_str = f"[blue]{amount:.1f} {precip_symbol}[/blue]"
        else:
            amount_str = f"{amount:.1f} {precip_symbol}"

        # Type
        if snow > rain and snow > 0:
            type_str = "[white]Snow[/white]"
        elif rain > 0:
            type_str = "[blue]Rain[/blue]"
        elif amount > 0:
            type_str = "[cyan]Mixed[/cyan]"
        else:
            type_str = "[dim]—[/dim]"

        # Hours
        hours_str = f"{hours:.0f}h" if hours > 0 else "[dim]—[/dim]"

        # Running total
        running_total += amount if amount else 0
        if running_total > 0:
            accum_str = f"{running_total:.1f} {precip_symbol}"
        else:
            accum_str = "[dim]—[/dim]"

        table.add_row(date_str, chance_str, amount_str, type_str, hours_str, accum_str)

    console.print(table)

    # Hourly sparkline for next 24h
    if h and h.precipitation_probability:
        now = datetime.now()
        current_hour_str = now.strftime("%Y-%m-%dT%H:00")
        try:
            start_idx = h.time.index(current_hour_str)
        except ValueError:
            start_idx = 0

        probs = h.precipitation_probability[start_idx : start_idx + 24]
        amounts = (h.precipitation or [])[start_idx : start_idx + 24]

        if probs:
            console.print(f"\n[dim]Next 24h chance:[/dim] {sparkline(probs)}")
        if amounts and any(a > 0 for a in amounts if a is not None):
            console.print(f"[dim]Next 24h amount:[/dim] {sparkline(amounts)}")


# =============================================================================
# Compare command
# =============================================================================


@cli.command()
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
                    temp_str = f"[red]{temp:.0f}°{temp_symbol}[/red]"
                elif temp >= 75:
                    temp_str = f"[yellow]{temp:.0f}°{temp_symbol}[/yellow]"
                elif temp <= 32:
                    temp_str = f"[cyan]{temp:.0f}°{temp_symbol}[/cyan]"
                else:
                    temp_str = f"{temp:.0f}°{temp_symbol}"
            else:
                if temp >= 32:
                    temp_str = f"[red]{temp:.0f}°{temp_symbol}[/red]"
                elif temp >= 24:
                    temp_str = f"[yellow]{temp:.0f}°{temp_symbol}[/yellow]"
                elif temp <= 0:
                    temp_str = f"[cyan]{temp:.0f}°{temp_symbol}[/cyan]"
                else:
                    temp_str = f"{temp:.0f}°{temp_symbol}"

            feels = r.get("feels_like", 0)
            feels_str = f"{feels:.0f}°{temp_symbol}"

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
            f"\n[dim]Warmest: {hottest[0]} ({hottest[1]:.0f}°{temp_symbol})[/dim]"
        )
        console.print(
            f"[dim]Coolest: {coldest[0]} ({coldest[1]:.0f}°{temp_symbol})[/dim]"
        )
        console.print(f"[dim]Difference: {diff:.0f}°{temp_symbol}[/dim]")


# =============================================================================
# History command
# =============================================================================


@cli.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option(
    "-y", "--years", default=1, help="How many years back to compare (default: 1)"
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def history(location: str | None, country: str | None, years: int, as_json: bool):
    """Compare today's weather with the same day in past years.

    Shows how today compares to this day in previous years.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop history <location>' or set a default with 'raindrop config set location <name>'"
        )

    # CLI country flag overrides resolved country
    if country is not None:
        resolved_country = country

    location = resolved_location
    country = resolved_country

    result = geocode(location, country)

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    precip_symbol = settings.precipitation_unit

    # Get today's date
    today = datetime.now().date()
    today_str = today.strftime("%Y-%m-%d")

    # Get current weather
    current_weather = om.forecast(
        result.latitude,
        result.longitude,
        current=[
            "temperature_2m",
            "weather_code",
        ],
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
            "location": {
                "name": result.name,
                "admin1": result.admin1,
                "country": result.country,
                "latitude": result.latitude,
                "longitude": result.longitude,
            },
            "today": current_data,
            "historical": historical_data,
            "units": {
                "temperature": settings.temperature_unit,
                "precipitation": settings.precipitation_unit,
            },
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # Display
    console.print(
        f"\n[bold cyan]{result.name}, {result.admin1}, {result.country}[/bold cyan]"
    )
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
        f"{current_data['temp_max']:.0f}°{temp_symbol}"
        if current_data["temp_max"]
        else "—",
        f"{current_data['temp_min']:.0f}°{temp_symbol}"
        if current_data["temp_min"]
        else "—",
        f"{current_data['precip']:.1f} {precip_symbol}"
        if current_data["precip"]
        else "—",
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
                diff_str = f"[red]+{diff:.0f}°[/red]"
            elif diff < 0:
                diff_str = f"[cyan]{diff:.0f}°[/cyan]"
            else:
                diff_str = "[dim]same[/dim]"
        else:
            diff_str = "—"

        table.add_row(
            str(h["year"]),
            f"{h['temp_max']:.0f}°{temp_symbol}" if h.get("temp_max") else "—",
            f"{h['temp_min']:.0f}°{temp_symbol}" if h.get("temp_min") else "—",
            f"{h['precip']:.1f} {precip_symbol}" if h.get("precip") else "—",
            f"[{color}]{label}[/{color}]",
            diff_str,
        )

    console.print(table)

    # Summary stats
    if historical_data and today_max is not None:
        hist_maxes = [
            h["temp_max"] for h in historical_data if h.get("temp_max") is not None
        ]
        if hist_maxes:
            avg_max = sum(hist_maxes) / len(hist_maxes)
            diff_from_avg = today_max - avg_max
            if diff_from_avg > 0:
                console.print(
                    f"\n[dim]Today is [red]{diff_from_avg:.1f}°{temp_symbol} warmer[/red] than average for this date[/dim]"
                )
            elif diff_from_avg < 0:
                console.print(
                    f"\n[dim]Today is [cyan]{abs(diff_from_avg):.1f}°{temp_symbol} cooler[/cyan] than average for this date[/dim]"
                )
            else:
                console.print(
                    f"\n[dim]Today matches the historical average for this date[/dim]"
                )


if __name__ == "__main__":
    cli()
