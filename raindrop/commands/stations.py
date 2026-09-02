"""Measured station observation commands."""

from datetime import UTC, datetime, timedelta
from typing import cast

import click
from rich import box
from rich.table import Table

from raindrop.commands.common import (
    console,
    echo_json,
    format_location,
    geocode,
    location_payload,
    resolve_location_or_fail,
)
from raindrop.observations import (
    CORE_MEASUREMENTS,
    Measurement,
    StationObservation,
    StationSort,
    filter_observations,
    observation_to_dict,
    sort_observations,
)
from raindrop.providers.xweather import StationKindFilter, XweatherClient, XweatherError
from raindrop.settings import Settings, get_settings
from raindrop.utils import TEMP_SYMBOLS, WIND_SYMBOLS, deg_to_compass


@click.group()
def stations() -> None:
    """Show measured observations from nearby weather stations."""


@stations.command("nearby")
@click.argument("location", required=False)
@click.option("-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)")
@click.option(
    "--kind",
    type=click.Choice(["pws", "official", "all"]),
    default="pws",
    show_default=True,
    help="Station source class to request.",
)
@click.option("--radius", type=click.FloatRange(min=0.1), default=25.0, show_default=True)
@click.option("--radius-unit", type=click.Choice(["mi", "km"]), help="Distance unit for radius")
@click.option("-n", "--limit", type=click.IntRange(1, 50), default=10, show_default=True)
@click.option(
    "--max-age",
    type=click.IntRange(1, 1440),
    default=60,
    show_default=True,
    help="Freshness window in minutes",
)
@click.option("--include-stale", is_flag=True, help="Include stale observations")
@click.option(
    "--sort",
    "sort_name",
    type=click.Choice(["best", "distance", "freshness"]),
    default="best",
    show_default=True,
)
@click.option("--json", "as_json", is_flag=True, help="Output normalized JSON")
@click.option("--raw", is_flag=True, help="Output sanitized provider JSON")
def nearby(
    location: str | None,
    country: str | None,
    kind: str,
    radius: float,
    radius_unit: str | None,
    limit: int,
    max_age: int,
    include_stale: bool,
    sort_name: str,
    as_json: bool,
    raw: bool,
) -> None:
    """List live nearby measured station observations.

    LOCATION can be a city name or a favorite alias. The default source class
    is PWS, meaning Xweather/PWSweather personal stations rather than official
    airport observations.
    """
    if as_json and raw:
        raise click.ClickException("--json and --raw are mutually exclusive")

    settings = get_settings()
    location, country = resolve_location_or_fail(settings, location, country, "stations nearby")
    result = geocode(location, country)
    radius_unit = radius_unit or _default_distance_unit(settings)
    radius_km = _distance_to_km(radius, radius_unit)
    client = XweatherClient()
    kind_filter = cast(StationKindFilter, kind)

    try:
        if raw:
            payload = client.nearby_raw(
                result.latitude,
                result.longitude,
                radius_km=radius_km,
                limit=limit,
                kind=kind_filter,
            )
            echo_json(payload)
            return

        observations = client.nearby_observations(
            result.latitude,
            result.longitude,
            radius_km=radius_km,
            limit=limit,
            kind=kind_filter,
        )
    except XweatherError as e:
        raise click.ClickException(str(e)) from e

    now = datetime.now(UTC)
    observations = filter_observations(
        observations,
        radius_km=radius_km,
        max_age_minutes=max_age,
        include_stale=include_stale,
        now=now,
    )
    observations = sort_observations(
        observations,
        sort=cast(StationSort, sort_name),
        max_age_minutes=max_age,
        now=now,
    )

    if as_json:
        echo_json(
            {
                "provider": "xweather",
                "attribution": "Powered by Vaisala Xweather",
                "location": location_payload(result),
                "query": {
                    "kind": kind,
                    "radius_km": radius_km,
                    "limit": limit,
                    "max_age_minutes": max_age,
                    "include_stale": include_stale,
                    "sort": sort_name,
                },
                "units": "SI",
                "observations": [observation_to_dict(observation) for observation in observations],
            }
        )
        return

    _render_nearby(
        observations,
        location_name=format_location(result),
        kind=kind_filter,
        radius=radius,
        radius_unit=radius_unit,
        max_age=max_age,
        settings=settings,
        include_stale=include_stale,
    )


@stations.command("current")
@click.argument("station_id")
@click.option(
    "--max-age",
    type=click.IntRange(1, 1440),
    default=60,
    show_default=True,
    help="Freshness window in minutes",
)
@click.option("--include-stale", is_flag=True, help="Show stale observations")
@click.option("--json", "as_json", is_flag=True, help="Output normalized JSON")
@click.option("--raw", is_flag=True, help="Output sanitized provider JSON")
def current(
    station_id: str,
    max_age: int,
    include_stale: bool,
    as_json: bool,
    raw: bool,
) -> None:
    """Show the latest measured observation for STATION_ID."""
    if as_json and raw:
        raise click.ClickException("--json and --raw are mutually exclusive")

    settings = get_settings()
    client = XweatherClient()
    try:
        if raw:
            payload = client.current_raw(station_id)
            echo_json(payload)
            return
        observation = client.current_observation(station_id)
    except XweatherError as e:
        raise click.ClickException(str(e)) from e

    now = datetime.now(UTC)
    filtered = filter_observations(
        [observation],
        max_age_minutes=max_age,
        include_stale=include_stale,
        now=now,
    )
    if not filtered:
        raise click.ClickException(
            f"Station '{station_id}' has no core measurement within {max_age} minutes. "
            "Use --include-stale to inspect it anyway."
        )

    if as_json:
        echo_json(
            {
                "provider": "xweather",
                "attribution": "Powered by Vaisala Xweather",
                "units": "SI",
                "observation": observation_to_dict(observation),
            }
        )
        return

    _render_current(observation, settings=settings)


def _render_nearby(
    observations: list[StationObservation],
    *,
    location_name: str,
    kind: StationKindFilter,
    radius: float,
    radius_unit: str,
    max_age: int,
    settings: Settings,
    include_stale: bool,
) -> None:
    title = {
        "pws": "Personal stations",
        "official": "Official stations",
        "all": "Measured stations",
    }[kind]
    console.print(f"\n[bold cyan]{title} near {location_name}[/bold cyan]")
    stale_note = " including stale" if include_stale else ""
    console.print(
        f"[dim]Xweather observations · radius {radius:g} {radius_unit} · max age {max_age}m{stale_note} · "
        "Powered by Vaisala Xweather[/dim]\n"
    )

    if not observations:
        console.print("[yellow]No matching fresh station observations found.[/yellow]")
        console.print(
            "[dim]Try --include-stale, --kind all, a larger --radius, or an official station query.[/dim]"
        )
        return

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    wind_symbol = WIND_SYMBOLS[settings.wind_speed_unit]
    table = Table(show_header=True, box=box.ROUNDED, header_style="bold")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Source", style="dim")
    table.add_column("Dist", justify="right")
    table.add_column("Age", justify="right")
    table.add_column(f"Temp °{temp_symbol}", justify="right")
    table.add_column("RH", justify="right")
    table.add_column(f"Wind {wind_symbol}", justify="right")
    table.add_column("QC", justify="center")

    now = datetime.now(UTC)
    for observation in observations:
        table.add_row(
            observation.station.station_id,
            _human_name(observation.station.name),
            _source_label(observation),
            _format_distance(observation.station.distance_km, radius_unit),
            _format_age(observation.age(now=now, names=CORE_MEASUREMENTS)),
            _format_temperature(_measurement(observation, "air_temperature"), settings),
            _format_percent(_measurement(observation, "relative_humidity")),
            _format_wind(observation, settings),
            _format_qc(observation),
        )

    console.print(table)
    console.print(
        "\n[dim]PWS readings are measured nearby, not official truth; siting, shielding, snow, "
        "and provider QC can affect values.[/dim]"
    )


def _render_current(observation: StationObservation, *, settings: Settings) -> None:
    station = observation.station
    console.print(
        f"\n[bold cyan]{station.station_id}[/bold cyan] [dim]{_human_name(station.name)}[/dim]"
    )
    console.print(
        f"[dim]{station.kind} · {station.network_name or station.network_id or 'unknown source'} · "
        "Powered by Vaisala Xweather[/dim]\n"
    )

    meta = Table(show_header=False, box=None, padding=(0, 2))
    meta.add_column("label", style="dim")
    meta.add_column("value")
    meta.add_row("Coordinates", f"{station.latitude:.5f}, {station.longitude:.5f}")
    meta.add_row(
        "Elevation", f"{station.elevation_m:.0f} m" if station.elevation_m is not None else "—"
    )
    meta.add_row("Observed", _format_datetime(observation.freshest_observed_at(CORE_MEASUREMENTS)))
    meta.add_row("Received", _format_datetime(observation.receipt_at))
    meta.add_row("Age", _format_age(observation.age(names=CORE_MEASUREMENTS)))
    meta.add_row("QC", _format_qc(observation))
    console.print(meta)

    table = Table(show_header=True, box=box.ROUNDED, header_style="bold")
    table.add_column("Measurement")
    table.add_column("Value", justify="right")
    table.add_column("QC", justify="center")

    rows = [
        (
            "Air temperature",
            _format_temperature(_measurement(observation, "air_temperature"), settings),
            "air_temperature",
        ),
        (
            "Dew point",
            _format_temperature(_measurement(observation, "dew_point"), settings),
            "dew_point",
        ),
        (
            "Relative humidity",
            _format_percent(_measurement(observation, "relative_humidity")),
            "relative_humidity",
        ),
        ("Wind", _format_wind(observation, settings), "wind_speed"),
        (
            "Pressure MSL",
            _format_pressure(_measurement(observation, "pressure_msl")),
            "pressure_msl",
        ),
        (
            "Altimeter",
            _format_pressure(_measurement(observation, "altimeter_setting")),
            "altimeter_setting",
        ),
        (
            "Precipitation",
            _format_precip(_measurement(observation, "precipitation"), settings),
            "precipitation",
        ),
        (
            "Precip rate",
            _format_precip_rate(_measurement(observation, "precipitation_rate"), settings),
            "precipitation_rate",
        ),
        (
            "Solar radiation",
            _format_plain(_measurement(observation, "solar_radiation"), "W/m²"),
            "solar_radiation",
        ),
        ("UV index", _format_plain(_measurement(observation, "uv_index"), ""), "uv_index"),
    ]
    for label, value, key in rows:
        if value == "—":
            continue
        table.add_row(label, value, _format_measurement_qc(_measurement(observation, key)))
    console.print(table)
    console.print(
        "\n[dim]Measured station observation. Do not blend with raindrop current's Open-Meteo model conditions.[/dim]"
    )


def _measurement(observation: StationObservation, name: str) -> Measurement | None:
    return observation.measurements.get(name)


def _format_temperature(measurement: Measurement | None, settings: Settings) -> str:
    if measurement is None or measurement.value_si is None:
        return "—"
    value = measurement.value_si
    if settings.temperature_unit == "fahrenheit":
        value = value * 9 / 5 + 32
    return f"{value:.1f}"


def _format_percent(measurement: Measurement | None) -> str:
    if measurement is None or measurement.value_si is None:
        return "—"
    return f"{measurement.value_si:.0f}%"


def _format_wind(observation: StationObservation, settings: Settings) -> str:
    speed = _measurement(observation, "wind_speed")
    gust = _measurement(observation, "wind_gust")
    direction = _measurement(observation, "wind_direction")
    if speed is None or speed.value_si is None:
        return "—"

    speed_value = _wind_from_ms(speed.value_si, settings.wind_speed_unit)
    text = f"{speed_value:.0f}"
    if direction is not None and direction.value_si is not None:
        text += f" {deg_to_compass(int(direction.value_si))}"
    if gust is not None and gust.value_si is not None and gust.value_si > speed.value_si:
        gust_value = _wind_from_ms(gust.value_si, settings.wind_speed_unit)
        text += f" g{gust_value:.0f}"
    return text


def _format_pressure(measurement: Measurement | None) -> str:
    if measurement is None or measurement.value_si is None:
        return "—"
    return f"{measurement.value_si / 100:.1f} hPa"


def _format_precip(measurement: Measurement | None, settings: Settings) -> str:
    if measurement is None or measurement.value_si is None:
        return "—"
    value = measurement.value_si
    unit = "mm"
    if settings.precipitation_unit == "inch":
        value /= 25.4
        unit = "in"
    return f"{value:.2f} {unit}"


def _format_precip_rate(measurement: Measurement | None, settings: Settings) -> str:
    value = _format_precip(measurement, settings)
    if value == "—":
        return value
    return f"{value}/h"


def _format_plain(measurement: Measurement | None, unit: str) -> str:
    if measurement is None or measurement.value_si is None:
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{measurement.value_si:.1f}{suffix}"


def _format_distance(distance_km: float | None, unit: str) -> str:
    if distance_km is None:
        return "—"
    if unit == "mi":
        return f"{distance_km * 0.621371:.1f} mi"
    return f"{distance_km:.1f} km"


def _format_age(age: timedelta | None) -> str:
    if age is None:
        return "—"
    minutes = max(int(age.total_seconds() // 60), 0)
    if minutes < 60:
        return f"{minutes}m"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m"


def _format_datetime(value: datetime | None) -> str:
    if value is None:
        return "—"
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _format_qc(observation: StationObservation) -> str:
    return _format_status(observation.worst_qc())


def _format_measurement_qc(measurement: Measurement | None) -> str:
    if measurement is None:
        return "—"
    return _format_status(measurement.qc)


def _format_status(status: str) -> str:
    if status == "pass":
        return "[green]pass[/green]"
    if status == "suspect":
        return "[yellow]suspect[/yellow]"
    if status == "fail":
        return "[red]fail[/red]"
    return "[dim]unknown[/dim]"


def _source_label(observation: StationObservation) -> str:
    station = observation.station
    if station.kind == "personal":
        return "PWS"
    if station.kind == "official":
        return "Official"
    if station.network_name:
        return station.network_name
    return station.network_id or station.kind


def _human_name(name: str) -> str:
    if name.islower():
        return name.title()
    return name


def _wind_from_ms(value: float, unit: str) -> float:
    if unit == "kmh":
        return value * 3.6
    if unit == "mph":
        return value * 2.23694
    if unit == "kn":
        return value * 1.94384
    return value


def _distance_to_km(value: float, unit: str) -> float:
    if unit == "mi":
        return value * 1.609344
    return value


def _default_distance_unit(settings: Settings) -> str:
    if settings.wind_speed_unit in {"kmh", "ms"}:
        return "km"
    return "mi"
