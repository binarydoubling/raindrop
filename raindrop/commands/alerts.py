"""Weather alerts command."""

import json as json_lib

import click
from rich.console import Console

from open_meteo import OpenMeteo, NWSClient
from settings import get_settings
from raindrop.utils import (
    SEVERITY_COLORS,
    URGENCY_COLORS,
    format_alert_time,
)

om = OpenMeteo()
nws = NWSClient()
console = Console()


def geocode(location: str, country: str | None = None):
    results = om.geocode(location, country_code=country)
    return results[0]


@click.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("-v", "--verbose", is_flag=True, help="Show full alert details")
def alerts(location: str | None, country: str | None, as_json: bool, verbose: bool):
    """Show active weather alerts and warnings.

    Displays NWS weather alerts including watches, warnings, and advisories.
    Only available for US locations.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop alerts <location>' or set a default with 'raindrop config set location <name>'"
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
            f"NWS weather alerts are only available for US locations. "
            f"{result.name} is in {result.country}."
        )

    # Get alerts
    try:
        alert_list = nws.get_alerts(result.latitude, result.longitude)
    except Exception as e:
        raise click.ClickException(f"Could not fetch alerts: {e}")

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
            "alerts": [
                {
                    "id": a.id,
                    "event": a.event,
                    "severity": a.severity,
                    "certainty": a.certainty,
                    "urgency": a.urgency,
                    "headline": a.headline,
                    "description": a.description,
                    "instruction": a.instruction,
                    "onset": a.onset,
                    "expires": a.expires,
                    "sender": a.sender_name,
                    "areas": a.areas,
                }
                for a in alert_list
            ],
            "count": len(alert_list),
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # Header
    console.print(
        f"\n[bold cyan]{result.name}, {result.admin1}, {result.country}[/bold cyan]"
    )

    if not alert_list:
        console.print("[green]No active weather alerts[/green]\n")
        return

    # Sort by severity
    severity_order = {
        "Extreme": 0,
        "Severe": 1,
        "Moderate": 2,
        "Minor": 3,
        "Unknown": 4,
    }
    alert_list.sort(key=lambda a: severity_order.get(a.severity, 5))

    console.print(
        f"[bold yellow]{len(alert_list)} Active Alert{'s' if len(alert_list) != 1 else ''}[/bold yellow]\n"
    )

    for alert in alert_list:
        sev_color = SEVERITY_COLORS.get(alert.severity, "white")
        urg_color = URGENCY_COLORS.get(alert.urgency, "white")

        # Alert header with event name
        console.print(f"[{sev_color}]{alert.event}[/{sev_color}]")
        console.print(
            f"  [dim]Severity:[/dim] [{sev_color}]{alert.severity}[/{sev_color}]  "
            f"[dim]Urgency:[/dim] [{urg_color}]{alert.urgency}[/{urg_color}]  "
            f"[dim]Certainty:[/dim] {alert.certainty}"
        )

        # Timing
        onset_str = format_alert_time(alert.onset)
        expires_str = format_alert_time(alert.expires)
        console.print(
            f"  [dim]From:[/dim] {onset_str}  [dim]Until:[/dim] {expires_str}"
        )

        # Headline
        if alert.headline:
            console.print(f"  [bold]{alert.headline}[/bold]")

        if verbose:
            # Full description
            if alert.description:
                console.print("\n  [dim]Description:[/dim]")
                # Word wrap the description
                desc_lines = alert.description.strip().split("\n")
                for line in desc_lines:
                    if line.strip():
                        console.print(f"  {line.strip()}")

            # Instructions
            if alert.instruction:
                console.print("\n  [yellow]Instructions:[/yellow]")
                inst_lines = alert.instruction.strip().split("\n")
                for line in inst_lines:
                    if line.strip():
                        console.print(f"  {line.strip()}")

            # Source
            console.print(f"\n  [dim]Source: {alert.sender_name}[/dim]")

        console.print()

    if not verbose:
        console.print("[dim]Use --verbose for full alert details[/dim]")
