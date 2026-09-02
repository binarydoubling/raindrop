"""NWS forecast discussion command."""

import re
from datetime import datetime as dt

import click

from raindrop.commands.common import (
    console,
    echo_json,
    format_location,
    geocode,
    location_payload,
    nws,
    resolve_location_or_fail,
)
from raindrop.settings import get_settings
from raindrop.utils import format_time


def format_discussion(text: str) -> str:
    """Format NWS forecast discussion text for pretty printing."""
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
            formatted_lines.append("\n[bold yellow]KEY MESSAGES[/bold yellow]")
            continue

        # Issued/Updated timestamps
        if line.strip().startswith("Issued at") or line.strip().startswith("Updated at"):
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


@click.command()
@click.argument("location", required=False)
@click.option("-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)")
@click.option("--raw", is_flag=True, help="Show raw unformatted text")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def discussion(location: str | None, country: str | None, raw: bool, as_json: bool):
    """Show NWS Area Forecast Discussion.

    Displays the meteorologist's technical forecast discussion from the
    National Weather Service. Only available for US locations.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    location, country = resolve_location_or_fail(settings, location, country, "discussion")

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
    try:
        issued = dt.fromisoformat(disc.issuance_time.replace("Z", "+00:00"))
        issued_str = f"{issued.strftime('%B %d, %Y at')} {format_time(issued)} {issued.strftime('%Z')}".strip()
    except Exception:
        issued_str = disc.issuance_time

    # JSON output
    if as_json:
        data = {
            "location": location_payload(result),
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
        echo_json(data)
        return

    # Header
    console.print(f"\n[bold cyan]{format_location(result, include_country=False)}[/bold cyan]")
    console.print(f"[dim]NWS {office.id} Area Forecast Discussion[/dim]")
    console.print(f"[dim]Issued: {issued_str}[/dim]\n")

    # Discussion text
    if raw:
        console.print(disc.product_text)
    else:
        formatted = format_discussion(disc.product_text)
        console.print(formatted)

    console.print(f"\n[dim]Source: NWS {office.id} | forecast.weather.gov[/dim]")
