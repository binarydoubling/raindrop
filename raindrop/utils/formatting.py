"""Formatting utilities for raindrop CLI output."""

from datetime import datetime, timedelta

# Sparkline characters (Unicode block elements)
SPARK_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def sparkline(values: list[float | int | None]) -> str:
    """Generate a sparkline string from a list of values."""
    clean_values = [v for v in values if v is not None]
    if not clean_values:
        return ""

    min_val = min(clean_values)
    max_val = max(clean_values)
    val_range = max_val - min_val

    result = []
    for v in values:
        if v is None:
            result.append(" ")
        elif val_range == 0:
            result.append(SPARK_CHARS[3])
        else:
            idx = int((v - min_val) / val_range * 7)
            idx = min(7, max(0, idx))
            result.append(SPARK_CHARS[idx])

    return "".join(result)


def format_duration(td: timedelta) -> str:
    """Format a timedelta as human readable."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def format_visibility(meters: float) -> str:
    """Format visibility in human-readable form."""
    if meters >= 10000:
        return f"{meters / 1000:.0f} km"
    elif meters >= 1000:
        return f"{meters / 1000:.1f} km"
    else:
        return f"{meters:.0f} m"


def format_uv(uv: float) -> str:
    """Format UV index with risk level."""
    if uv < 3:
        return f"{uv:.1f} [green](Low)[/green]"
    elif uv < 6:
        return f"{uv:.1f} [yellow](Moderate)[/yellow]"
    elif uv < 8:
        return f"{uv:.1f} [orange1](High)[/orange1]"
    elif uv < 11:
        return f"{uv:.1f} [red](Very High)[/red]"
    else:
        return f"{uv:.1f} [magenta](Extreme)[/magenta]"


def format_delta(
    current: float, previous: float, unit: str = "", precision: int = 0
) -> str:
    """Format a value with its delta from the previous value."""
    delta = current - previous
    if abs(delta) < 1:
        delta_str = "[dim]\u00b7[/dim]"
    elif delta > 0:
        delta_str = f"[red]\u2191{abs(delta):.0f}[/red]"
    else:
        delta_str = f"[cyan]\u2193{abs(delta):.0f}[/cyan]"
    return f"{current:.{precision}f}{unit} {delta_str}"


def format_precip_chance(chance: int, prev_chance: int) -> str:
    """Format precipitation chance with delta."""
    if chance == 0:
        return "[dim]\u2014[/dim]"

    delta = chance - prev_chance
    if abs(delta) < 1:
        delta_str = " [dim]\u00b7[/dim]"
    elif delta > 0:
        delta_str = f" [red]\u2191{abs(delta)}[/red]"
    else:
        delta_str = f" [cyan]\u2193{abs(delta)}[/cyan]"

    if chance >= 70:
        return f"[bold blue]{chance}%[/bold blue]{delta_str}"
    elif chance >= 40:
        return f"[blue]{chance}%[/blue]{delta_str}"
    else:
        return f"[dim]{chance}%[/dim]{delta_str}"


def format_pollutant(value: float | None, unit: str = "\u03bcg/m\u00b3") -> str:
    """Format a pollutant value."""
    if value is None:
        return "\u2014"
    return f"{value:.1f} {unit}"


# US AQI levels and colors
US_AQI_LEVELS = [
    (50, "Good", "green"),
    (100, "Moderate", "yellow"),
    (150, "Unhealthy (Sensitive)", "orange1"),
    (200, "Unhealthy", "red"),
    (300, "Very Unhealthy", "magenta"),
    (500, "Hazardous", "bold red"),
]


def format_us_aqi(aqi: int | None) -> str:
    """Format US AQI with level and color."""
    if aqi is None:
        return "\u2014"
    for threshold, level, color in US_AQI_LEVELS:
        if aqi <= threshold:
            return f"[{color}]{aqi} ({level})[/{color}]"
    return f"[bold red]{aqi} (Hazardous)[/bold red]"


def format_alert_time(iso_time: str | None) -> str:
    """Format alert time for display."""
    if not iso_time:
        return "\u2014"
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        return dt.strftime("%a %-I:%M %p")
    except Exception:
        return iso_time[:16] if iso_time else "\u2014"


def deg_to_compass(deg: int) -> str:
    """Convert degrees to compass direction."""
    directions = [
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    ]
    idx = round(deg / 22.5) % 16
    return directions[idx]
