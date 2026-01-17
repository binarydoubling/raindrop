"""Route weather command - weather along a driving route with turn-by-turn directions."""

from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import json as json_lib
import urllib.request
import urllib.parse

import click
from rich.console import Console
from rich.table import Table
from rich import box

from open_meteo import OpenMeteo
from settings import get_settings
from raindrop.utils import (
    WEATHER_CODES,
    WEATHER_LABELS,
    TEMP_SYMBOLS,
    WIND_SYMBOLS,
    sparkline,
)

om = OpenMeteo()
console = Console()

# OSRM public demo server
OSRM_BASE_URL = "https://router.project-osrm.org"

# Weather check interval in miles
WEATHER_INTERVAL_MILES = 50


def geocode(location: str, country: str | None = None):
    results = om.geocode(location, country_code=country)
    return results[0]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance in kilometers between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # Earth radius in km


def get_osrm_route(
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> dict:
    """
    Get driving route from OSRM with full turn-by-turn directions.

    Returns dict with:
        - distance: total distance in meters
        - duration: total duration in seconds
        - geometry: list of [lon, lat] coordinates along route
        - legs: list of route legs with steps
    """
    # OSRM expects lon,lat order
    coords = f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"

    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "true",  # Get turn-by-turn
        "annotations": "true",  # Get speed/duration per segment
    }

    query = urllib.parse.urlencode(params)
    url = f"{OSRM_BASE_URL}/route/v1/driving/{coords}?{query}"

    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "raindrop-weather-cli")

        with urllib.request.urlopen(req, timeout=15) as response:
            data = json_lib.loads(response.read().decode())

        if data.get("code") != "Ok":
            raise Exception(data.get("message", "Unknown routing error"))

        route = data["routes"][0]

        return {
            "distance": route["distance"],  # meters
            "duration": route["duration"],  # seconds
            "geometry": route["geometry"]["coordinates"],  # List of [lon, lat]
            "legs": route["legs"],  # Contains steps with directions
        }
    except urllib.error.URLError as e:
        raise Exception(f"Could not connect to routing service: {e}")
    except Exception as e:
        raise Exception(f"Routing error: {e}")


def parse_road_name(step: dict) -> str:
    """Extract road name from OSRM step."""
    name = step.get("name", "")
    ref = step.get("ref", "")  # Highway number like "I-5" or "US-101"

    if ref and name:
        return f"{ref} ({name})"
    elif ref:
        return ref
    elif name:
        return name
    else:
        return "Local road"


def parse_maneuver(step: dict) -> str:
    """Parse maneuver instruction from OSRM step."""
    maneuver = step.get("maneuver", {})
    maneuver_type = maneuver.get("type", "")
    modifier = maneuver.get("modifier", "")

    # Build instruction based on maneuver type
    if maneuver_type == "depart":
        return "Depart"
    elif maneuver_type == "arrive":
        return "Arrive"
    elif maneuver_type == "merge":
        return f"Merge {modifier}" if modifier else "Merge"
    elif maneuver_type == "on ramp":
        return "Take on-ramp"
    elif maneuver_type == "off ramp":
        return "Take exit"
    elif maneuver_type == "fork":
        return f"Keep {modifier}" if modifier else "Fork"
    elif maneuver_type == "turn":
        return f"Turn {modifier}" if modifier else "Turn"
    elif maneuver_type == "new name":
        return "Continue"
    elif maneuver_type == "continue":
        return f"Continue {modifier}" if modifier else "Continue"
    elif maneuver_type == "roundabout":
        return "Enter roundabout"
    elif maneuver_type == "rotary":
        return "Enter rotary"
    elif maneuver_type == "roundabout turn":
        return "Exit roundabout"
    elif maneuver_type == "end of road":
        return f"Turn {modifier}" if modifier else "End of road"
    else:
        return maneuver_type.replace("_", " ").title() if maneuver_type else "Continue"


def build_route_segments(route_data: dict) -> list[dict]:
    """
    Build route segments from OSRM response.
    Each segment represents a stretch on one road.
    """
    segments = []
    cumulative_distance = 0  # meters
    cumulative_duration = 0  # seconds

    for leg in route_data["legs"]:
        for step in leg.get("steps", []):
            distance = step.get("distance", 0)
            duration = step.get("duration", 0)

            # Get road info
            road_name = parse_road_name(step)
            maneuver = parse_maneuver(step)

            # Get the geometry for this step (first point is start of step)
            geometry = step.get("geometry", {}).get("coordinates", [])
            if geometry:
                start_coord = geometry[0]  # [lon, lat]
                end_coord = geometry[-1]
            else:
                start_coord = None
                end_coord = None

            segment = {
                "road": road_name,
                "maneuver": maneuver,
                "distance_m": distance,
                "distance_mi": distance * 0.000621371,
                "duration_s": duration,
                "cumulative_distance_m": cumulative_distance,
                "cumulative_distance_mi": cumulative_distance * 0.000621371,
                "cumulative_duration_s": cumulative_duration,
                "start_coord": start_coord,
                "end_coord": end_coord,
                "geometry": geometry,
            }

            segments.append(segment)
            cumulative_distance += distance
            cumulative_duration += duration

    return segments


def consolidate_road_segments(segments: list[dict]) -> list[dict]:
    """
    Consolidate consecutive segments on the same road.
    This simplifies the output - we don't need every tiny turn.
    """
    if not segments:
        return []

    consolidated = []
    current = None

    for seg in segments:
        road = seg["road"]

        # Skip very short segments (< 0.1 miles) unless it's a significant maneuver
        if seg["distance_mi"] < 0.1 and seg["maneuver"] in ["Continue", "new name"]:
            if current:
                # Add distance to current segment
                current["distance_m"] += seg["distance_m"]
                current["distance_mi"] += seg["distance_mi"]
                current["duration_s"] += seg["duration_s"]
                current["end_coord"] = seg["end_coord"]
                current["geometry"].extend(seg["geometry"][1:])  # Avoid duplicate point
            continue

        if current is None:
            current = seg.copy()
            current["geometry"] = list(seg["geometry"])
        elif road == current["road"] and seg["maneuver"] in ["Continue", "new name"]:
            # Same road, consolidate
            current["distance_m"] += seg["distance_m"]
            current["distance_mi"] += seg["distance_mi"]
            current["duration_s"] += seg["duration_s"]
            current["end_coord"] = seg["end_coord"]
            current["geometry"].extend(seg["geometry"][1:])
        else:
            # Different road or significant maneuver
            consolidated.append(current)
            current = seg.copy()
            current["geometry"] = list(seg["geometry"])

    if current:
        consolidated.append(current)

    # Recalculate cumulative distances
    cumulative_dist = 0
    cumulative_dur = 0
    for seg in consolidated:
        seg["cumulative_distance_m"] = cumulative_dist
        seg["cumulative_distance_mi"] = cumulative_dist * 0.000621371
        seg["cumulative_duration_s"] = cumulative_dur
        cumulative_dist += seg["distance_m"]
        cumulative_dur += seg["duration_s"]

    return consolidated


def get_weather_checkpoints(
    segments: list[dict],
    total_distance_mi: float,
    interval_mi: float = WEATHER_INTERVAL_MILES,
) -> list[dict]:
    """
    Generate weather checkpoints at regular intervals along the route.
    Returns list of checkpoints with coordinates and cumulative distance/time.
    """
    checkpoints = []
    next_checkpoint_mi = 0

    for seg in segments:
        seg_start_mi = seg["cumulative_distance_mi"]
        seg_end_mi = seg_start_mi + seg["distance_mi"]

        # Check if any checkpoints fall within this segment
        while (
            next_checkpoint_mi <= seg_end_mi and next_checkpoint_mi <= total_distance_mi
        ):
            # Calculate position within segment
            if seg["distance_mi"] > 0:
                progress_in_seg = (next_checkpoint_mi - seg_start_mi) / seg[
                    "distance_mi"
                ]
            else:
                progress_in_seg = 0

            progress_in_seg = max(0, min(1, progress_in_seg))

            # Interpolate coordinates
            if seg["geometry"] and len(seg["geometry"]) >= 2:
                # Find the right point in geometry
                geo_idx = int(progress_in_seg * (len(seg["geometry"]) - 1))
                geo_idx = min(geo_idx, len(seg["geometry"]) - 1)
                coord = seg["geometry"][geo_idx]
                lat, lon = coord[1], coord[0]
            elif seg["start_coord"]:
                lat, lon = seg["start_coord"][1], seg["start_coord"][0]
            else:
                lat, lon = None, None

            # Calculate cumulative duration to this point
            if seg["distance_mi"] > 0:
                duration_to_point = seg["cumulative_duration_s"] + (
                    seg["duration_s"] * progress_in_seg
                )
            else:
                duration_to_point = seg["cumulative_duration_s"]

            checkpoints.append(
                {
                    "mile": next_checkpoint_mi,
                    "lat": lat,
                    "lon": lon,
                    "cumulative_duration_s": duration_to_point,
                    "road": seg["road"],
                }
            )

            next_checkpoint_mi += interval_mi

    # Always include the final destination
    if segments:
        final_seg = segments[-1]
        final_mi = final_seg["cumulative_distance_mi"] + final_seg["distance_mi"]
        if not checkpoints or checkpoints[-1]["mile"] < final_mi - 1:
            if final_seg["end_coord"]:
                lat, lon = final_seg["end_coord"][1], final_seg["end_coord"][0]
            else:
                lat, lon = None, None

            checkpoints.append(
                {
                    "mile": final_mi,
                    "lat": lat,
                    "lon": lon,
                    "cumulative_duration_s": final_seg["cumulative_duration_s"]
                    + final_seg["duration_s"],
                    "road": final_seg["road"],
                }
            )

    return checkpoints


def get_weather_for_point(
    lat: float,
    lon: float,
    arrival_time: datetime,
    settings,
) -> dict:
    """Get weather forecast for a specific point at estimated arrival time."""
    if lat is None or lon is None:
        return {}

    weather = om.forecast(
        lat,
        lon,
        hourly=[
            "temperature_2m",
            "apparent_temperature",
            "precipitation_probability",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
            "visibility",
        ],
        temperature_unit=settings.temperature_unit,
        wind_speed_unit=settings.wind_speed_unit,
        precipitation_unit=settings.precipitation_unit,
        forecast_days=2,
    )

    h = weather.hourly
    if not h or not h.time:
        return {}

    target_hour = arrival_time.strftime("%Y-%m-%dT%H:00")

    try:
        idx = h.time.index(target_hour)
    except ValueError:
        idx = 0
        for i, t in enumerate(h.time):
            if t >= target_hour:
                idx = i
                break

    return {
        "temperature": h.temperature_2m[idx]
        if h.temperature_2m and idx < len(h.temperature_2m)
        else None,
        "feels_like": h.apparent_temperature[idx]
        if h.apparent_temperature and idx < len(h.apparent_temperature)
        else None,
        "precip_prob": h.precipitation_probability[idx]
        if h.precipitation_probability and idx < len(h.precipitation_probability)
        else None,
        "weather_code": h.weather_code[idx]
        if h.weather_code and idx < len(h.weather_code)
        else None,
        "wind_speed": h.wind_speed_10m[idx]
        if h.wind_speed_10m and idx < len(h.wind_speed_10m)
        else None,
        "visibility": h.visibility[idx]
        if h.visibility and idx < len(h.visibility)
        else None,
    }


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


@click.command()
@click.argument("origin")
@click.argument("destination")
@click.option("-d", "--depart", help="Departure time (HH:MM, default: now)")
@click.option(
    "-i", "--interval", default=50, help="Weather check interval in miles (default: 50)"
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--brief", is_flag=True, help="Show condensed output without full directions"
)
def route(
    origin: str,
    destination: str,
    depart: str | None,
    interval: int,
    as_json: bool,
    brief: bool,
):
    """Show weather conditions along a driving route with turn-by-turn directions.

    Uses real driving routes from OpenStreetMap (via OSRM) to show
    forecasted weather along your journey.

    \b
    Examples:
      raindrop route "Seattle" "Portland"
      raindrop route "Los Angeles" "San Francisco" -i 75
      raindrop route "New York" "Boston" -d 08:00
      raindrop route "Denver" "Salt Lake City" --brief
    """
    settings = get_settings()
    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    wind_symbol = WIND_SYMBOLS[settings.wind_speed_unit]

    # Parse departure time
    now = datetime.now()
    if depart:
        try:
            hour, minute = map(int, depart.split(":"))
            departure_time = now.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            if departure_time < now:
                departure_time += timedelta(days=1)
        except ValueError:
            raise click.ClickException("Invalid time format. Use HH:MM (e.g., 08:00)")
    else:
        departure_time = now

    # Geocode origin and destination
    console.print("[dim]Finding locations...[/dim]")
    try:
        origin_geo = geocode(origin)
        dest_geo = geocode(destination)
    except Exception as e:
        raise click.ClickException(f"Could not find location: {e}")

    # Get real route from OSRM
    console.print("[dim]Calculating route...[/dim]")
    try:
        route_data = get_osrm_route(
            (origin_geo.latitude, origin_geo.longitude),
            (dest_geo.latitude, dest_geo.longitude),
        )
    except Exception as e:
        raise click.ClickException(str(e))

    total_distance_m = route_data["distance"]
    total_distance_mi = total_distance_m * 0.000621371
    total_duration_s = route_data["duration"]
    total_duration = timedelta(seconds=total_duration_s)

    # Build and consolidate route segments
    segments = build_route_segments(route_data)
    consolidated = consolidate_road_segments(segments)

    # Get weather checkpoints
    checkpoints = get_weather_checkpoints(consolidated, total_distance_mi, interval)

    # Fetch weather for each checkpoint
    console.print("[dim]Fetching weather data...[/dim]")
    for cp in checkpoints:
        arrival_time = departure_time + timedelta(seconds=cp["cumulative_duration_s"])
        cp["arrival_time"] = arrival_time
        cp["weather"] = get_weather_for_point(
            cp["lat"], cp["lon"], arrival_time, settings
        )

    # JSON output
    if as_json:
        data = {
            "origin": {
                "name": origin_geo.name,
                "admin1": origin_geo.admin1,
                "latitude": origin_geo.latitude,
                "longitude": origin_geo.longitude,
            },
            "destination": {
                "name": dest_geo.name,
                "admin1": dest_geo.admin1,
                "latitude": dest_geo.latitude,
                "longitude": dest_geo.longitude,
            },
            "route": {
                "total_distance_mi": total_distance_mi,
                "total_distance_km": total_distance_m / 1000,
                "total_duration_seconds": total_duration_s,
                "total_duration_formatted": format_duration(total_duration_s),
                "departure_time": departure_time.isoformat(),
                "arrival_time": (departure_time + total_duration).isoformat(),
                "source": "OSRM/OpenStreetMap",
            },
            "directions": [
                {
                    "maneuver": seg["maneuver"],
                    "road": seg["road"],
                    "distance_mi": seg["distance_mi"],
                    "duration_formatted": format_duration(seg["duration_s"]),
                }
                for seg in consolidated
            ],
            "weather_checkpoints": [
                {
                    "mile": cp["mile"],
                    "road": cp["road"],
                    "arrival_time": cp["arrival_time"].isoformat(),
                    "latitude": cp["lat"],
                    "longitude": cp["lon"],
                    **cp["weather"],
                }
                for cp in checkpoints
            ],
            "units": {
                "temperature": settings.temperature_unit,
                "wind_speed": settings.wind_speed_unit,
            },
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # === DISPLAY ===
    arrival = departure_time + total_duration

    console.print(
        f"\n[bold cyan]Route: {origin_geo.name} \u2192 {dest_geo.name}[/bold cyan]"
    )
    console.print(
        f"[dim]{total_distance_mi:.0f} miles \u00b7 {format_duration(total_duration_s)} \u00b7 "
        f"Depart {departure_time.strftime('%-I:%M %p')} \u2192 Arrive {arrival.strftime('%-I:%M %p')}[/dim]\n"
    )

    # Turn-by-turn directions (unless --brief)
    if not brief:
        console.print("[bold]Turn-by-Turn Directions[/bold]\n")

        dir_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        dir_table.add_column("Mile", style="dim", justify="right", width=6)
        dir_table.add_column("Action", width=20)
        dir_table.add_column("Road", style="cyan")
        dir_table.add_column("Dist", style="dim", justify="right", width=8)

        for seg in consolidated:
            if seg["distance_mi"] < 0.5 and seg["maneuver"] == "Continue":
                continue  # Skip very short "continue" segments

            mile_marker = f"{seg['cumulative_distance_mi']:.0f}"
            maneuver = seg["maneuver"]
            road = seg["road"]
            dist = f"{seg['distance_mi']:.1f} mi"

            # Highlight significant roads (highways)
            if any(x in road for x in ["I-", "US-", "SR-", "Hwy", "Interstate"]):
                road = f"[bold]{road}[/bold]"

            dir_table.add_row(mile_marker, maneuver, road, dist)

        console.print(dir_table)
        console.print()

    # Weather along route
    console.print(
        f"[bold]Weather Along Route[/bold] [dim](every {interval} miles)[/dim]\n"
    )

    weather_table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    weather_table.add_column("Mile", justify="right", width=5)
    weather_table.add_column("ETA", justify="right", width=9)
    weather_table.add_column("Road", style="cyan", max_width=25)
    weather_table.add_column("Weather", width=8)
    weather_table.add_column("Temp", justify="right", width=6)
    weather_table.add_column("Precip", justify="right", width=6)
    weather_table.add_column("Wind", justify="right", width=8)

    temps = []
    precip_probs = []
    weather_concerns = []

    for i, cp in enumerate(checkpoints):
        w = cp["weather"]

        # Mile
        if i == 0:
            mile_str = "[bold]Start[/bold]"
        elif i == len(checkpoints) - 1:
            mile_str = "[bold]End[/bold]"
        else:
            mile_str = f"{cp['mile']:.0f}"

        # ETA
        eta_str = cp["arrival_time"].strftime("%-I:%M %p").lower()

        # Road (truncate if needed)
        road = cp["road"]
        if len(road) > 25:
            road = road[:22] + "..."

        # Weather condition
        code = w.get("weather_code", 0) or 0
        label, color = WEATHER_LABELS.get(code, ("?", "white"))
        weather_str = f"[{color}]{label}[/{color}]"

        # Track concerns
        if code in {71, 73, 75, 65, 82, 95, 96, 99, 45, 48}:
            weather_concerns.append((cp["mile"], cp["road"], label))

        # Temperature
        temp = w.get("temperature")
        temps.append(temp)
        temp_str = f"{temp:.0f}\u00b0{temp_symbol}" if temp is not None else "\u2014"

        # Precipitation
        prob = w.get("precip_prob")
        precip_probs.append(prob)
        if prob and prob > 0:
            if prob >= 70:
                precip_str = f"[bold blue]{prob}%[/bold blue]"
            elif prob >= 40:
                precip_str = f"[blue]{prob}%[/blue]"
            else:
                precip_str = f"{prob}%"
        else:
            precip_str = "[dim]\u2014[/dim]"

        # Wind
        wind = w.get("wind_speed")
        if wind is not None:
            if wind >= 25:
                wind_str = f"[yellow]{wind:.0f} {wind_symbol}[/yellow]"
            else:
                wind_str = f"{wind:.0f} {wind_symbol}"
        else:
            wind_str = "\u2014"

        weather_table.add_row(
            mile_str, eta_str, road, weather_str, temp_str, precip_str, wind_str
        )

    console.print(weather_table)

    # Temperature sparkline
    console.print(f"\n[dim]Temperature trend:[/dim] {sparkline(temps)}")

    # Temp range
    valid_temps = [t for t in temps if t is not None]
    if valid_temps:
        console.print(
            f"[dim]Range: {min(valid_temps):.0f}\u00b0 to {max(valid_temps):.0f}\u00b0{temp_symbol}[/dim]"
        )

    # Weather warnings
    if weather_concerns:
        console.print("\n[bold red]Weather Alerts Along Route:[/bold red]")
        for mile, road, condition in weather_concerns:
            console.print(f"  [red]\u26a0 Mile {mile:.0f} ({road}): {condition}[/red]")

    has_heavy_precip = any(p and p > 70 for p in precip_probs)
    has_precip = any(p and p > 40 for p in precip_probs)

    if has_heavy_precip and not weather_concerns:
        console.print(
            "\n[yellow]Note: High chance of precipitation along parts of route.[/yellow]"
        )
    elif has_precip and not weather_concerns:
        console.print(
            "\n[dim]Tip: Some chance of precipitation - consider rain gear.[/dim]"
        )
