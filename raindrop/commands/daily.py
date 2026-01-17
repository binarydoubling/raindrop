"""Daily forecast command."""

from datetime import datetime, timedelta
import json as json_lib

import click
from rich.console import Console
from rich.table import Table
from rich import box

from open_meteo import OpenMeteo
from settings import get_settings, AVAILABLE_MODELS
from raindrop.utils import (
    WEATHER_CODES,
    WEATHER_LABELS,
    TEMP_SYMBOLS,
    WIND_SYMBOLS,
    ema,
    calc_roc,
    calc_volatility,
    trend_signal,
    roc_signal,
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
    console.print(
        f"[dim]{days}-day forecast \u00b7 Model: {model_key or 'auto'}[/dim]\n"
    )

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
    table.add_column("\u0394 3d", justify="right")  # 3-day rate of change

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
                high_str = f"{high:.0f}\u00b0{temp_symbol} [dim]\u00b7[/dim]"
            elif high_delta > 0:
                high_str = f"{high:.0f}\u00b0{temp_symbol} [red]\u2191{abs(high_delta):.0f}[/red]"
            else:
                high_str = f"{high:.0f}\u00b0{temp_symbol} [cyan]\u2193{abs(high_delta):.0f}[/cyan]"
        else:
            high_str = f"{high:.0f}\u00b0{temp_symbol}"

        # Low with delta
        if i > 0 and i < len(lows):
            low_delta = low - lows[i - 1]
            if abs(low_delta) < 1:
                low_str = f"{low:.0f}\u00b0{temp_symbol} [dim]\u00b7[/dim]"
            elif low_delta > 0:
                low_str = f"{low:.0f}\u00b0{temp_symbol} [red]\u2191{abs(low_delta):.0f}[/red]"
            else:
                low_str = f"{low:.0f}\u00b0{temp_symbol} [cyan]\u2193{abs(low_delta):.0f}[/cyan]"
        else:
            low_str = f"{low:.0f}\u00b0{temp_symbol}"

        # Range (volatility)
        range_str = f"{vol:.0f}\u00b0"

        # Precipitation
        prob = precip_probs[i] if i < len(precip_probs) else 0
        amount = precip_sums[i] if i < len(precip_sums) else 0
        if prob == 0:
            precip_str = "[dim]\u2014[/dim]"
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
            roc_str = f"[{roc_color}]{roc:+.0f}\u00b0[/{roc_color}]"
        else:
            roc_str = "[dim]\u2014[/dim]"

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
                f"[dim]EMA(3):[/dim] {latest_ema3:.1f}\u00b0  "
                f"[dim]EMA(7):[/dim] {latest_ema7:.1f}\u00b0  "
                f"[dim]Spread:[/dim] [{trend_color}]{ema_diff:+.1f}\u00b0[/{trend_color}]  "
                f"[dim]Signal:[/dim] [{trend_color}]{trend_txt}[/{trend_color}]"
            )

    # Rate of change
    if len(roc_vals) > 0 and roc_vals[-1] is not None:
        roc_val = roc_vals[-1]
        roc_txt, roc_color = roc_signal(roc_val)
        console.print(
            f"[dim]3-day \u0394:[/dim] [{roc_color}]{roc_val:+.1f}\u00b0[/{roc_color}]  "
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
            f"[dim]Avg Range:[/dim] {recent_vol:.1f}\u00b0  "
            f"[dim]Volatility:[/dim] {vol_trend}"
        )

    # Legend
    console.print(
        "\n[dim]Trend: EMA(3)/EMA(7) crossover \u00b7 \u0394 3d: 3-day temperature change[/dim]"
    )
    console.print(
        "[dim]\u25b2 Hot \u00b7 \u2197 Warming \u00b7 \u2192 Stable \u00b7 \u2198 Cooling \u00b7 \u25bc Cold[/dim]"
    )
