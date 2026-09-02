"""Daily forecast command."""

from datetime import datetime, timedelta

import click
from rich import box
from rich.table import Table

from raindrop.commands.common import (
    console,
    echo_json,
    format_location,
    format_weather_source,
    geocode,
    location_payload,
    resolve_location_or_fail,
    resolve_weather_provider_or_fail,
)
from raindrop.settings import get_settings
from raindrop.utils import (
    TEMP_SYMBOLS,
    WEATHER_CODES,
    WEATHER_LABELS,
    WIND_SYMBOLS,
    calc_roc,
    calc_volatility,
    ema,
    format_delta,
    now_in_timezone,
    roc_signal,
    trend_signal,
)
from raindrop.weather_provider import provider_source_payload


@click.command()
@click.argument("location", required=False)
@click.option("-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)")
@click.option(
    "-n",
    "--days",
    type=click.IntRange(1, 16),
    default=10,
    show_default=True,
    help="Number of days to show",
)
@click.option(
    "-m",
    "--model",
    "model_name",
    help="Open-Meteo model to use (forces Open-Meteo in auto mode)",
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

    location, country = resolve_location_or_fail(settings, location, country, "daily")

    weather_provider = resolve_weather_provider_or_fail(model_name, settings)

    result = geocode(location, country)

    weather = weather_provider.forecast(
        result.latitude,
        result.longitude,
        daily=[
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "uv_index_max",
        ],
        temperature_unit=settings.temperature_unit,
        wind_speed_unit=settings.wind_speed_unit,
        precipitation_unit=settings.precipitation_unit,
        forecast_days=min(days, 16),
    )
    d = weather.daily
    if d is None:
        raise click.ClickException("No daily weather data returned")

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    wind_symbol = WIND_SYMBOLS[settings.wind_speed_unit]
    precip_symbol = settings.precipitation_unit

    # Get data arrays, stripping trailing None days (short-range models)
    raw_highs = d.temperature_2m_max or []
    raw_lows = d.temperature_2m_min or []
    valid_days = len(d.time)
    for i in range(len(d.time) - 1, -1, -1):
        h_val = raw_highs[i] if i < len(raw_highs) else None
        l_val = raw_lows[i] if i < len(raw_lows) else None
        if h_val is not None and l_val is not None:
            valid_days = i + 1
            break
    else:
        valid_days = 0

    if valid_days == 0:
        raise click.ClickException("No temperature data returned by this model")

    times = d.time[:valid_days]
    codes = (d.weather_code or [])[:valid_days]
    highs = raw_highs[:valid_days]
    lows = raw_lows[:valid_days]
    precip_probs = (d.precipitation_probability_max or [])[:valid_days]
    precip_sums = (d.precipitation_sum or [])[:valid_days]
    wind_maxs = (d.wind_speed_10m_max or [])[:valid_days]
    uv_maxs = (d.uv_index_max or [])[:valid_days]

    # Calculate technical indicators using average temperature
    avg_temps = [((high or 0) + (low or 0)) / 2 for high, low in zip(highs, lows, strict=False)]
    ema_3 = ema(avg_temps, 3)  # Short-term EMA
    ema_7 = ema(avg_temps, 7)  # Long-term EMA
    roc_vals = calc_roc(avg_temps, 3)  # 3-day rate of change
    volatility = calc_volatility(highs, lows)
    wind_gusts = (d.wind_gusts_10m_max or [])[:valid_days]

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
                    "precipitation_probability": precip_probs[i] if i < len(precip_probs) else None,
                    "precipitation_sum": precip_sums[i] if i < len(precip_sums) else None,
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
            "location": location_payload(result),
            "model": weather_provider.model_label,
            "source": provider_source_payload(weather_provider),
            "days": daily_data,
            "units": {
                "temperature": settings.temperature_unit,
                "wind_speed": settings.wind_speed_unit,
                "precipitation": settings.precipitation_unit,
            },
        }
        echo_json(data)
        return

    # Location header
    console.print(f"\n[bold cyan]{format_location(result)}[/bold cyan]")
    console.print(f"[dim]{days}-day forecast · {format_weather_source(weather_provider)}[/dim]\n")

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

    today = now_in_timezone(weather.timezone).date()

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

        high_str = (
            format_delta(high, highs[i - 1], f"\u00b0{temp_symbol}")
            if 0 < i < len(highs)
            else f"{high:.0f}\u00b0{temp_symbol}"
        )
        low_str = (
            format_delta(low, lows[i - 1], f"\u00b0{temp_symbol}")
            if 0 < i < len(lows)
            else f"{low:.0f}\u00b0{temp_symbol}"
        )

        # Range (volatility)
        range_str = f"{vol:.0f}\u00b0"

        # Precipitation
        prob = precip_probs[i] if i < len(precip_probs) else 0
        amount = precip_sums[i] if i < len(precip_sums) else 0
        if prob == 0:
            precip_str = "[dim]\u2014[/dim]"
        elif prob >= 70:
            precip_str = f"[bold blue]{prob}%[/bold blue] {amount:.2f}{precip_symbol}"
        elif prob >= 40:
            precip_str = f"[blue]{prob}%[/blue] {amount:.2f}{precip_symbol}"
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
            trend_txt, trend_color = trend_signal(avg_temps[-1], latest_ema3, latest_ema7)
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
            f"[dim]Avg Range:[/dim] {recent_vol:.1f}\u00b0  [dim]Volatility:[/dim] {vol_trend}"
        )

    # Legend
    console.print("\n[dim]Trend: EMA(3)/EMA(7) crossover · Δ 3d: 3-day temperature change[/dim]")
    console.print("[dim]▲ Hot · ↗ Warming · → Stable · ↘ Cooling · ▼ Cold[/dim]")
