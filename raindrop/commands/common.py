"""Shared helpers for CLI command modules."""

from typing import Any

import click

from raindrop.open_meteo import GeocodingResult, NWSClient, OpenMeteo, OpenMeteoError
from raindrop.settings import Settings, normalize_country_code, resolve_model

om = OpenMeteo()
nws = NWSClient()


def geocode(location: str, country: str | None = None) -> GeocodingResult:
    """Geocode a location and convert API failures to Click errors."""
    try:
        normalized_country = normalize_country_code(country)
        results = om.geocode(location, country_code=normalized_country)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    except OpenMeteoError as e:
        raise click.ClickException(str(e)) from e

    if not results:
        raise click.ClickException(f"No locations found for '{location}'")
    return results[0]


def resolve_location_or_fail(
    settings: Settings,
    location: str | None,
    country: str | None,
    command_name: str,
) -> tuple[str, str | None]:
    """Resolve favorites/defaults and validate CLI country overrides."""
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError as e:
        raise click.ClickException(
            f"No location provided. Use 'raindrop {command_name} <location>' "
            "or set a default with 'raindrop config set location <name>'"
        ) from e

    if country is not None:
        try:
            resolved_country = normalize_country_code(country)
        except ValueError as e:
            raise click.ClickException(str(e)) from e

    return resolved_location, resolved_country


def resolve_model_or_fail(
    model_name: str | None,
    settings: Settings,
) -> tuple[str | None, list[str] | None]:
    """Resolve a weather model and convert validation failures to Click errors."""
    try:
        return resolve_model(model_name, settings)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


def location_payload(result: GeocodingResult) -> dict[str, Any]:
    """Return a consistent JSON location payload."""
    return {
        "name": result.name,
        "admin1": result.admin1,
        "country": result.country,
        "country_code": result.country_code,
        "latitude": result.latitude,
        "longitude": result.longitude,
    }


def format_location(result: GeocodingResult, *, include_country: bool = True) -> str:
    """Format a location without rendering literal ``None`` values."""
    parts = [result.name]
    if result.admin1:
        parts.append(result.admin1)
    if include_country and result.country:
        parts.append(result.country)
    return ", ".join(parts)
