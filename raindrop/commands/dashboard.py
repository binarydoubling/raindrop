"""Full-screen TUI dashboard for weather monitoring."""

from datetime import datetime
import time

import click
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box

from open_meteo import OpenMeteo, NWSClient
from settings import get_settings
from raindrop.utils import (
    WEATHER_CODES,
    WEATHER_LABELS,
    TEMP_SYMBOLS,
    WIND_SYMBOLS,
    sparkline,
    format_duration,
    deg_to_compass,
)
from raindrop.utils.astro import (
    moon_phase,
    moon_illumination,
    daylight_duration,
    format_daylight_duration,
)

om = OpenMeteo()
nws = NWSClient()
console = Console()


def geocode(location: str, country: str | None = None):
    results = om.geocode(location, country_code=country)
    return results[0]


def make_layout() -> Layout:
    """Create the dashboard layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )

    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )

    layout["left"].split(
        Layout(name="current", ratio=1),
        Layout(name="hourly", ratio=1),
    )

    layout["right"].split(
        Layout(name="daily", ratio=2),
        Layout(name="astro", ratio=1),
    )

    return layout


def render_header(location_name: str, timezone: str) -> Panel:
    """Render the header panel."""
    now = datetime.now()
    time_str = now.strftime("%A, %B %d, %Y  %I:%M:%S %p")

    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right")

    grid.add_row(
        f"[bold cyan]{location_name}[/bold cyan]  [dim]{timezone}[/dim]",
        f"[bold]{time_str}[/bold]",
    )

    return Panel(grid, box=box.ROUNDED, style="white on blue")


def render_current(weather, settings) -> Panel:
    """Render current conditions panel."""
    c = weather.current
    if c is None:
        return Panel("No data", title="Current Conditions")

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    wind_symbol = WIND_SYMBOLS[settings.wind_speed_unit]

    code = c.weather_code or 0
    condition = WEATHER_CODES.get(code, "Unknown")
    label, color = WEATHER_LABELS.get(code, ("?", "white"))

    # Main display
    table = Table.grid(padding=(0, 2))
    table.add_column()
    table.add_column()

    # Big temperature
    temp_text = Text(f"{c.temperature_2m:.0f}\u00b0{temp_symbol}", style="bold white")
    temp_text.stylize("bold", 0, len(temp_text))

    table.add_row(
        Text.from_markup(
            f"[bold white on {color}] {label.upper()} [/bold white on {color}]"
        ),
        "",
    )
    table.add_row(
        Text(f"{c.temperature_2m:.0f}\u00b0", style="bold"),
        Text(
            f"Feels like {c.apparent_temperature:.0f}\u00b0{temp_symbol}", style="dim"
        ),
    )
    table.add_row("", "")

    # Details grid
    details = Table.grid(padding=(0, 1))
    details.add_column(style="dim")
    details.add_column()
    details.add_column(style="dim")
    details.add_column()

    wind_dir = c.wind_direction_10m
    compass = deg_to_compass(wind_dir) if wind_dir is not None else ""

    details.add_row(
        "Humidity",
        f"{c.relative_humidity_2m}%",
        "Wind",
        f"{c.wind_speed_10m:.0f} {wind_symbol} {compass}",
    )
    details.add_row(
        "Cloud",
        f"{c.cloud_cover}%",
        "Pressure",
        f"{c.pressure_msl:.0f} hPa" if c.pressure_msl else "\u2014",
    )

    content = Group(table, Text(""), details)

    return Panel(content, title="[bold]Current Conditions[/bold]", border_style="green")


def render_hourly(weather, settings) -> Panel:
    """Render hourly forecast panel with sparklines."""
    h = weather.hourly
    if h is None:
        return Panel("No data", title="Hourly Forecast")

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]

    now = datetime.now()
    current_hour = now.strftime("%Y-%m-%dT%H:00")

    try:
        start_idx = h.time.index(current_hour)
    except ValueError:
        start_idx = 0

    # Get next 24 hours
    temps = (h.temperature_2m or [])[start_idx : start_idx + 24]
    precips = (h.precipitation_probability or [])[start_idx : start_idx + 24]
    winds = (h.wind_speed_10m or [])[start_idx : start_idx + 24]

    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", width=8)
    table.add_column(width=26)
    table.add_column(width=12)

    # Temperature sparkline
    temp_clean = [t for t in temps if t is not None]
    temp_range = (
        f"{min(temp_clean):.0f}-{max(temp_clean):.0f}\u00b0" if temp_clean else ""
    )
    table.add_row("Temp", sparkline(temps), temp_range)

    # Precipitation sparkline
    precip_clean = [p for p in precips if p is not None]
    precip_max = (
        f"max {max(precip_clean):.0f}%"
        if precip_clean and max(precip_clean) > 0
        else ""
    )
    table.add_row("Precip", sparkline(precips), precip_max)

    # Wind sparkline
    wind_clean = [w for w in winds if w is not None]
    wind_range = f"{min(wind_clean):.0f}-{max(wind_clean):.0f}" if wind_clean else ""
    table.add_row("Wind", sparkline(winds), wind_range)

    # Hour labels
    hours_row = ""
    for i in [0, 6, 12, 18, 23]:
        if start_idx + i < len(h.time):
            hr = datetime.fromisoformat(h.time[start_idx + i]).strftime("%-I%p").lower()
            hours_row += f"{hr:>4}"

    content = Group(table, Text(""), Text(f"[dim]{hours_row}[/dim]"))

    return Panel(content, title="[bold]Next 24 Hours[/bold]", border_style="cyan")


def render_daily(weather, settings) -> Panel:
    """Render daily forecast panel."""
    d = weather.daily
    if d is None:
        return Panel("No data", title="Daily Forecast")

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]

    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column(style="cyan", width=5)  # Day
    table.add_column(width=8)  # Weather
    table.add_column(justify="right", width=4)  # High
    table.add_column(justify="right", width=4)  # Low

    today = datetime.now().date()

    for i in range(min(7, len(d.time))):
        date = datetime.fromisoformat(d.time[i]).date()

        if date == today:
            day_str = "[bold]Today[/bold]"
        else:
            day_str = date.strftime("%a")

        code = (
            (d.weather_code or [])[i]
            if d.weather_code and i < len(d.weather_code)
            else 0
        )
        label, color = WEATHER_LABELS.get(code, ("?", "white"))

        high = (
            (d.temperature_2m_max or [])[i]
            if d.temperature_2m_max and i < len(d.temperature_2m_max)
            else 0
        )
        low = (
            (d.temperature_2m_min or [])[i]
            if d.temperature_2m_min and i < len(d.temperature_2m_min)
            else 0
        )

        table.add_row(
            day_str,
            f"[{color}]{label}[/{color}]",
            f"{high:.0f}\u00b0",
            f"[dim]{low:.0f}\u00b0[/dim]",
        )

    return Panel(table, title="[bold]7-Day Forecast[/bold]", border_style="yellow")


def render_astro(weather) -> Panel:
    """Render astronomical data panel."""
    d = weather.daily
    if d is None or not d.sunrise or not d.sunset:
        return Panel("No data", title="Sun & Moon")

    now = datetime.now()
    sunrise = datetime.fromisoformat(d.sunrise[0])
    sunset = datetime.fromisoformat(d.sunset[0])

    daylight = daylight_duration(sunrise, sunset)

    # Moon phase
    phase_val, phase_name, phase_symbol = moon_phase(now)
    illumination = moon_illumination(phase_val)

    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()

    table.add_row("Sunrise", sunrise.strftime("%-I:%M %p").lower())
    table.add_row("Sunset", sunset.strftime("%-I:%M %p").lower())
    table.add_row("Daylight", format_daylight_duration(daylight))
    table.add_row("", "")
    table.add_row("Moon", f"{phase_symbol} {phase_name}")
    table.add_row("", f"{illumination:.0f}% illuminated")

    return Panel(table, title="[bold]Sun & Moon[/bold]", border_style="magenta")


def render_footer() -> Panel:
    """Render the footer panel."""
    return Panel(
        "[dim]Press [bold]Ctrl+C[/bold] to exit  |  Refreshes every 60 seconds  |  Raindrop Weather Dashboard[/dim]",
        box=box.ROUNDED,
        style="dim",
    )


def fetch_weather_data(geo, settings):
    """Fetch all weather data needed for dashboard."""
    return om.forecast(
        geo.latitude,
        geo.longitude,
        current=[
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "pressure_msl",
            "weather_code",
        ],
        hourly=[
            "temperature_2m",
            "precipitation_probability",
            "wind_speed_10m",
            "weather_code",
        ],
        daily=[
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "sunrise",
            "sunset",
        ],
        temperature_unit=settings.temperature_unit,
        wind_speed_unit=settings.wind_speed_unit,
        forecast_days=7,
    )


@click.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option(
    "-r", "--refresh", default=60, help="Refresh interval in seconds (default: 60)"
)
def dashboard(location: str | None, country: str | None, refresh: int):
    """Launch full-screen weather dashboard.

    Shows a live-updating dashboard with current conditions, hourly forecast,
    daily forecast, and astronomical data.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').

    Press Ctrl+C to exit.
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop dashboard <location>' or set a default with 'raindrop config set location <name>'"
        )

    # CLI country flag overrides resolved country
    if country is not None:
        resolved_country = country

    location = resolved_location
    country = resolved_country

    geo = geocode(location, country)
    location_name = f"{geo.name}, {geo.admin1}, {geo.country}"

    # Create layout
    layout = make_layout()

    def update_dashboard():
        """Update all dashboard panels."""
        weather = fetch_weather_data(geo, settings)

        layout["header"].update(render_header(location_name, weather.timezone))
        layout["current"].update(render_current(weather, settings))
        layout["hourly"].update(render_hourly(weather, settings))
        layout["daily"].update(render_daily(weather, settings))
        layout["astro"].update(render_astro(weather))
        layout["footer"].update(render_footer())

    # Initial update
    update_dashboard()

    # Live display with refresh
    try:
        with Live(layout, console=console, screen=True, refresh_per_second=1) as live:
            last_update = time.time()
            while True:
                # Check if we need to refresh weather data
                if time.time() - last_update >= refresh:
                    update_dashboard()
                    last_update = time.time()
                else:
                    # Just update the header for time
                    weather = fetch_weather_data(geo, settings)  # For timezone
                    layout["header"].update(
                        render_header(location_name, weather.timezone)
                    )

                time.sleep(1)
    except KeyboardInterrupt:
        pass  # Clean exit on Ctrl+C
