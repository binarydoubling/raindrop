"""Ensemble forecast analysis commands."""

import math
from datetime import datetime
from typing import Any, Literal, cast

import click
from rich import box
from rich.table import Table

from raindrop.commands.common import (
    console,
    echo_json,
    format_location,
    geocode,
    location_payload,
    om,
    resolve_location_or_fail,
)
from raindrop.ensemble import (
    ENSEMBLE_MODELS,
    distribution_summary,
    evaluate_member,
    member_series,
    rank_members,
    values_at,
)
from raindrop.open_meteo import GeocodingResult
from raindrop.settings import get_settings

DEFAULT_MODEL = "ncep_gefs_seamless"
DEFAULT_PERCENTILES = "10,25,50,75,90"
TemporalResolution = Literal["native", "hourly", "hourly_3", "hourly_6"]


def _load(
    location: str | None,
    country: str | None,
    model: str,
    variables: tuple[str, ...],
    daily: bool,
    periods: int | None,
    temporal_resolution: str,
    command_name: str,
) -> tuple[GeocodingResult, dict[str, Any], dict[str, str], list[str], list[str], int]:
    """Resolve a location and fetch one ensemble dataset."""
    settings = get_settings()
    if not model.strip() or "," in model:
        raise click.ClickException("Select exactly one ensemble model")
    location, country = resolve_location_or_fail(settings, location, country, command_name)
    result = geocode(location, country)
    interval: Literal["hourly", "daily"] = "daily" if daily else "hourly"
    steps = periods or (7 if daily else 24)
    if daily and steps > 36:
        raise click.ClickException("Daily ensemble forecasts support at most 36 periods")

    selected = list(dict.fromkeys(variable.strip() for variable in variables if variable.strip()))
    selected = selected or ["temperature_2m_mean" if daily else "temperature_2m"]
    payload = om.ensemble(
        result.latitude,
        result.longitude,
        model=model,
        variables=selected,
        interval=interval,
        steps=steps,
        temporal_resolution=cast(TemporalResolution, temporal_resolution),
        temperature_unit=settings.temperature_unit,
        wind_speed_unit=settings.wind_speed_unit,
        precipitation_unit=settings.precipitation_unit,
    )
    dataset = payload.get(interval)
    units = payload.get(f"{interval}_units")
    if not isinstance(dataset, dict) or not isinstance(units, dict):
        raise click.ClickException("No ensemble data returned")
    times = dataset.get("time")
    if not isinstance(times, list):
        raise click.ClickException("No ensemble forecast times returned")
    return result, dataset, units, selected, times[:steps], steps


def _members(dataset: dict[str, Any], variable: str) -> dict[int, list[float | None]]:
    """Return member data as a user-facing command error."""
    try:
        return member_series(dataset, variable)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def _parse_percentiles(value: str) -> tuple[float, ...]:
    """Parse a comma-separated percentile list."""
    try:
        percentiles = tuple(sorted({float(item) for item in value.split(",")}))
    except ValueError as e:
        raise click.ClickException("Percentiles must be comma-separated numbers") from e
    if not percentiles or any(
        not math.isfinite(percent) or percent < 0 or percent > 100 for percent in percentiles
    ):
        raise click.ClickException("Percentiles must be between 0 and 100")
    return percentiles


def _time_label(value: str, daily: bool) -> str:
    """Format one API timestamp for a compact table row."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    return parsed.strftime("%a %b %d" if daily else "%a %H:%M")


def _number(value: Any) -> str:
    """Format an optional numeric statistic."""
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.1f}"


def _center(summary: dict[str, Any]) -> str:
    """Format a distribution center for any supported variable kind."""
    if not summary["count"]:
        return "—"
    if summary["kind"] == "categorical":
        return _number(summary.get("mode"))
    return _number(summary.get("mean"))


def _interval(summary: dict[str, Any], percentiles: tuple[float, ...]) -> str:
    """Format uncertainty for any supported variable kind."""
    if not summary["count"]:
        return "—"
    if summary["kind"] == "categorical":
        return f"{summary['agreement']:.0%} agree"
    if summary["kind"] == "circular":
        return f"{summary['concentration']:.0%} agree"
    values = summary["percentiles"]
    low = values[str(percentiles[0]).removesuffix(".0")]
    high = values[str(percentiles[-1]).removesuffix(".0")]
    return f"{low:.1f}–{high:.1f}"


@click.group()
def ensemble() -> None:
    """Analyze individual ensemble forecast members and uncertainty."""


@ensemble.command("forecast")
@click.argument("location", required=False)
@click.option("-c", "--country", help="ISO country code")
@click.option("-m", "--model", default=DEFAULT_MODEL, show_default=True)
@click.option("-v", "--variable", "variables", multiple=True, help="API variable; repeatable")
@click.option("--daily", is_flag=True, help="Analyze daily instead of hourly variables")
@click.option("-n", "--periods", type=click.IntRange(1, 864), help="Hours, or days with --daily")
@click.option(
    "--temporal-resolution",
    type=click.Choice(["native", "hourly", "hourly_3", "hourly_6"]),
    default="hourly",
    show_default=True,
)
@click.option("--percentiles", default=DEFAULT_PERCENTILES, show_default=True)
@click.option("--threshold", type=float, help="Calculate probability above/below this value")
@click.option("--operator", type=click.Choice(["above", "below"]), default="above")
@click.option("--json", "as_json", is_flag=True, help="Output complete member analysis as JSON")
def forecast(
    location: str | None,
    country: str | None,
    model: str,
    variables: tuple[str, ...],
    daily: bool,
    periods: int | None,
    temporal_resolution: str,
    percentiles: str,
    threshold: float | None,
    operator: str,
    as_json: bool,
) -> None:
    """Show ensemble distributions, spread, outliers, and threshold probabilities."""
    selected_percentiles = _parse_percentiles(percentiles)
    if threshold is not None and not math.isfinite(threshold):
        raise click.ClickException("--threshold must be finite")
    if threshold is not None and len({value.strip() for value in variables if value.strip()}) > 1:
        raise click.ClickException("--threshold requires one --variable")
    result, dataset, units, selected, times, _ = _load(
        location,
        country,
        model,
        variables,
        daily,
        periods,
        temporal_resolution,
        "ensemble forecast",
    )

    analyses: dict[str, list[dict[str, Any]]] = {}
    for variable in selected:
        members = _members(dataset, variable)
        unit = str(units.get(variable, ""))
        analyses[variable] = []
        for index, time in enumerate(times):
            values = values_at(members, index)
            analyses[variable].append(
                {
                    "time": time,
                    "members": values,
                    "summary": distribution_summary(
                        values,
                        variable=variable,
                        unit=unit,
                        percentiles=selected_percentiles,
                        threshold=threshold,
                        operator=operator,
                    ),
                }
            )

    if as_json:
        echo_json(
            {
                "location": location_payload(result),
                "source": {"provider": "open-meteo", "model": model},
                "interval": "daily" if daily else "hourly",
                "units": {variable: units.get(variable) for variable in selected},
                "analysis": analyses,
            }
        )
        return

    console.print(f"\n[bold cyan]{format_location(result)}[/bold cyan]")
    console.print(f"[dim]Open-Meteo ensemble · {model} · consensus is not forecast skill[/dim]\n")
    for variable, rows in analyses.items():
        unit = units.get(variable, "")
        table = Table(
            title=f"{variable} ({unit})" if unit else variable,
            box=box.ROUNDED,
            header_style="bold",
        )
        table.add_column("Time", style="cyan")
        table.add_column("Control", justify="right")
        table.add_column("Center", justify="right")
        table.add_column("Uncertainty", justify="right")
        table.add_column("Members", justify="right")
        if threshold is not None:
            symbol = ">" if operator == "above" else "<"
            table.add_column(f"P({symbol}{threshold:g})", justify="right")
        for row in rows:
            summary = row["summary"]
            cells = [
                _time_label(row["time"], daily),
                _number(summary.get("control")),
                _center(summary),
                _interval(summary, selected_percentiles),
                str(summary["count"]),
            ]
            if threshold is not None:
                cells.append(f"{summary.get('threshold_probability', 0):.0%}")
            table.add_row(*cells)
        console.print(table)
        console.print()


@ensemble.command("member")
@click.argument("location", required=False)
@click.option("--member", type=click.IntRange(0, 99), required=True)
@click.option("-c", "--country", help="ISO country code")
@click.option("-m", "--model", default=DEFAULT_MODEL, show_default=True)
@click.option("-v", "--variable", "variables", multiple=True, help="API variable; repeatable")
@click.option("--daily", is_flag=True, help="Analyze daily instead of hourly variables")
@click.option("-n", "--periods", type=click.IntRange(1, 864), help="Hours, or days with --daily")
@click.option(
    "--temporal-resolution",
    type=click.Choice(["native", "hourly", "hourly_3", "hourly_6"]),
    default="hourly",
    show_default=True,
)
@click.option("--json", "as_json", is_flag=True)
def member(
    location: str | None,
    member: int,
    country: str | None,
    model: str,
    variables: tuple[str, ...],
    daily: bool,
    periods: int | None,
    temporal_resolution: str,
    as_json: bool,
) -> None:
    """Evaluate MEMBER against ensemble consensus at every forecast step."""
    result, dataset, units, selected, times, _ = _load(
        location,
        country,
        model,
        variables,
        daily,
        periods,
        temporal_resolution,
        "ensemble member",
    )
    evaluations: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any] | None] = {}
    for variable in selected:
        members = _members(dataset, variable)
        if member not in members:
            raise click.ClickException(f"Member {member} is unavailable for {variable}")
        unit = str(units.get(variable, ""))
        evaluations[variable] = [
            {
                "time": time,
                **(
                    evaluate_member(values_at(members, index), member, variable=variable, unit=unit)
                    or {}
                ),
            }
            for index, time in enumerate(times)
        ]
        summaries[variable] = next(
            (
                row
                for row in rank_members(members, variable=variable, unit=unit, steps=len(times))
                if row["member"] == member
            ),
            None,
        )

    if as_json:
        echo_json(
            {
                "location": location_payload(result),
                "source": {"provider": "open-meteo", "model": model},
                "member": member,
                "interval": "daily" if daily else "hourly",
                "units": {variable: units.get(variable) for variable in selected},
                "evaluation": evaluations,
                "summary": summaries,
            }
        )
        return

    console.print(f"\n[bold cyan]{format_location(result)} · member {member:02d}[/bold cyan]")
    console.print(f"[dim]{model} · evaluated against same-run consensus, not observations[/dim]\n")
    for variable, rows in evaluations.items():
        table = Table(title=f"{variable} ({units.get(variable, '')})", box=box.ROUNDED)
        table.add_column("Time", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Consensus", justify="right")
        table.add_column("Departure", justify="right")
        table.add_column("Position", justify="right")
        for row in rows:
            if "value" not in row:
                table.add_row(_time_label(row["time"], daily), "—", "—", "—", "missing")
                continue
            departure = row.get("difference", row.get("angular_difference"))
            if row.get("kind") == "categorical":
                departure = 0 if row.get("matches_consensus") else 1
                position = "match" if not departure else "diverge"
            else:
                position = (
                    f"p{row['percentile_rank'] * 100:.0f}"
                    if "percentile_rank" in row
                    else f"{row.get('directional_agreement', 0):.0%} agree"
                )
            table.add_row(
                _time_label(row["time"], daily),
                _number(row.get("value")),
                _number(row.get("consensus")),
                _number(departure),
                position,
            )
        console.print(table)
        summary = summaries[variable]
        if summary:
            console.print(
                f"[dim]Horizon: bias {_number(summary['bias_from_consensus'])} · "
                f"mean |Δ| {_number(summary['mean_absolute_departure'])} · "
                f"RMSE {_number(summary['rmse_from_consensus'])} · "
                f"outlier {summary['outlier_rate']:.0%}[/dim]"
            )
        console.print()


@ensemble.command("rank")
@click.argument("location", required=False)
@click.option("-c", "--country", help="ISO country code")
@click.option("-m", "--model", default=DEFAULT_MODEL, show_default=True)
@click.option("-v", "--variable", "variables", multiple=True, help="Exactly one API variable")
@click.option("--daily", is_flag=True, help="Analyze daily instead of hourly values")
@click.option("-n", "--periods", type=click.IntRange(1, 864), help="Hours, or days with --daily")
@click.option(
    "--temporal-resolution",
    type=click.Choice(["native", "hourly", "hourly_3", "hourly_6"]),
    default="hourly",
    show_default=True,
)
@click.option("--top", type=click.IntRange(1, 100), default=10, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def rank(
    location: str | None,
    country: str | None,
    model: str,
    variables: tuple[str, ...],
    daily: bool,
    periods: int | None,
    temporal_resolution: str,
    top: int,
    as_json: bool,
) -> None:
    """Rank members by closeness to consensus across the requested horizon."""
    if len({value.strip() for value in variables if value.strip()}) > 1:
        raise click.ClickException("rank accepts exactly one --variable")
    result, dataset, units, selected, times, _ = _load(
        location,
        country,
        model,
        variables,
        daily,
        periods,
        temporal_resolution,
        "ensemble rank",
    )
    variable = selected[0]
    unit = str(units.get(variable, ""))
    ranking = rank_members(
        _members(dataset, variable),
        variable=variable,
        unit=unit,
        steps=len(times),
    )

    if as_json:
        echo_json(
            {
                "location": location_payload(result),
                "source": {"provider": "open-meteo", "model": model},
                "variable": variable,
                "unit": unit,
                "metric": "departure from ensemble consensus, not forecast skill",
                "ranking": ranking,
            }
        )
        return

    console.print(f"\n[bold cyan]{format_location(result)}[/bold cyan]")
    console.print(
        f"[dim]{model} · {variable} ({unit}) · closest to consensus, not most accurate[/dim]\n"
    )
    table = Table(box=box.ROUNDED, header_style="bold")
    table.add_column("Rank", justify="right")
    table.add_column("Member", justify="right")
    table.add_column("Samples", justify="right")
    table.add_column("Bias", justify="right")
    table.add_column("Mean |Δ|", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("Outlier", justify="right")
    for place, row in enumerate(ranking[:top], 1):
        table.add_row(
            str(place),
            f"{row['member']:02d}",
            str(row["samples"]),
            _number(row["bias_from_consensus"]),
            _number(row["mean_absolute_departure"]),
            _number(row["rmse_from_consensus"]),
            f"{row['outlier_rate']:.0%}",
        )
    console.print(table)


@ensemble.command("models")
@click.option("--json", "as_json", is_flag=True)
def models(as_json: bool) -> None:
    """List known Open-Meteo ensemble model identifiers."""
    if as_json:
        echo_json(
            {
                key: {
                    "name": value[0],
                    "region": value[1],
                    "members": value[2],
                    "horizon": value[3],
                }
                for key, value in ENSEMBLE_MODELS.items()
            }
        )
        return

    table = Table(title="Open-Meteo Ensemble Models", box=box.ROUNDED, header_style="bold")
    table.add_column("Identifier", style="cyan")
    table.add_column("Model")
    table.add_column("Region")
    table.add_column("Members", justify="right")
    table.add_column("Horizon", justify="right")
    for key, (name, region, members, horizon) in ENSEMBLE_MODELS.items():
        table.add_row(key, name, region, str(members), horizon)
    console.print(table)
    console.print(
        "[dim]Model availability and supported variables differ by region and horizon.[/dim]"
    )
