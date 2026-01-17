"""Precipitation forecast command."""

from datetime import datetime, timedelta
import json as json_lib

import click
from rich.console import Console
from rich.table import Table
from rich import box

from open_meteo import OpenMeteo
from settings import get_settings, AVAILABLE_MODELS
from raindrop.utils import sparkline

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
    total_hours = sum(hr for hr in precip_hours[:days] if hr is not None)

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
            chance_str = "[dim]\u2014[/dim]"
        elif prob >= 70:
            chance_str = f"[bold blue]{prob}%[/bold blue]"
        elif prob >= 40:
            chance_str = f"[blue]{prob}%[/blue]"
        else:
            chance_str = f"[dim]{prob}%[/dim]"

        # Amount formatting
        if amount == 0:
            amount_str = "[dim]\u2014[/dim]"
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
            type_str = "[dim]\u2014[/dim]"

        # Hours
        hours_str = f"{hours:.0f}h" if hours > 0 else "[dim]\u2014[/dim]"

        # Running total
        running_total += amount if amount else 0
        if running_total > 0:
            accum_str = f"{running_total:.1f} {precip_symbol}"
        else:
            accum_str = "[dim]\u2014[/dim]"

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
