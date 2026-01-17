"""Weather-related constants and analysis functions."""

# WMO Weather codes to descriptions
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# Weather code to short labels with colors
WEATHER_LABELS: dict[int, tuple[str, str]] = {
    0: ("Clear", "yellow"),
    1: ("Clear", "yellow"),
    2: ("Cloudy", "white"),
    3: ("Cloudy", "dim white"),
    45: ("Fog", "dim white"),
    48: ("Fog", "dim white"),
    51: ("Drizzle", "cyan"),
    53: ("Drizzle", "cyan"),
    55: ("Drizzle", "blue"),
    61: ("Rain", "cyan"),
    63: ("Rain", "blue"),
    65: ("Rain", "bold blue"),
    71: ("Snow", "white"),
    73: ("Snow", "bold white"),
    75: ("Snow", "bold white"),
    80: ("Showers", "cyan"),
    81: ("Showers", "blue"),
    82: ("Storms", "bold blue"),
    95: ("Storms", "bold magenta"),
    96: ("Storms", "bold magenta"),
    99: ("Storms", "bold magenta"),
}

# Unit symbols
TEMP_SYMBOLS = {"celsius": "C", "fahrenheit": "F"}
WIND_SYMBOLS = {"kmh": "km/h", "ms": "m/s", "mph": "mph", "kn": "kn"}

# US AQI levels (also in formatting.py for format_us_aqi)
US_AQI_LEVELS = [
    (50, "Good", "green"),
    (100, "Moderate", "yellow"),
    (150, "Unhealthy (Sensitive)", "orange1"),
    (200, "Unhealthy", "red"),
    (300, "Very Unhealthy", "magenta"),
    (500, "Hazardous", "bold red"),
]

# Alert severity colors
SEVERITY_COLORS = {
    "Extreme": "bold red",
    "Severe": "red",
    "Moderate": "yellow",
    "Minor": "cyan",
    "Unknown": "dim",
}

URGENCY_COLORS = {
    "Immediate": "bold red",
    "Expected": "yellow",
    "Future": "cyan",
    "Past": "dim",
    "Unknown": "dim",
}


# =============================================================================
# Technical Analysis Functions
# =============================================================================


def ema(values: list[float], period: int) -> list[float | None]:
    """Calculate Exponential Moving Average."""
    if len(values) < period:
        return [None] * len(values)

    result: list[float | None] = [None] * (period - 1)
    multiplier = 2 / (period + 1)

    # First EMA is SMA
    sma = sum(values[:period]) / period
    result.append(sma)

    # Calculate EMA for remaining values
    for i in range(period, len(values)):
        ema_val = (values[i] - result[-1]) * multiplier + result[-1]  # type: ignore
        result.append(ema_val)

    return result


def calc_roc(values: list[float], period: int = 3) -> list[float | None]:
    """
    Calculate Rate of Change over a period.
    Returns the temperature change over the last N days.
    """
    result: list[float | None] = [None] * period
    for i in range(period, len(values)):
        roc = values[i] - values[i - period]
        result.append(roc)
    return result


def calc_volatility(highs: list[float], lows: list[float]) -> list[float]:
    """Calculate daily temperature range (volatility)."""
    return [h - l for h, l in zip(highs, lows)]


def trend_signal(
    value: float, ema_short: float | None, ema_long: float | None
) -> tuple[str, str]:
    """
    Determine trend signal based on EMA crossover.
    Returns (signal, color).
    """
    if ema_short is None or ema_long is None:
        return ("\u2014", "dim")

    diff = ema_short - ema_long
    pct_diff = (diff / ema_long) * 100 if ema_long != 0 else 0

    if pct_diff > 2:
        return ("\u25b2 Hot", "red")
    elif pct_diff > 0.5:
        return ("\u2197 Warming", "yellow")
    elif pct_diff < -2:
        return ("\u25bc Cold", "cyan")
    elif pct_diff < -0.5:
        return ("\u2198 Cooling", "blue")
    else:
        return ("\u2192 Stable", "dim")


def roc_signal(roc: float | None) -> tuple[str, str]:
    """
    Interpret rate of change as a momentum signal.
    Returns (description, color).
    """
    if roc is None:
        return ("\u2014", "dim")

    if roc >= 10:
        return ("Surging", "bold red")
    elif roc >= 5:
        return ("Rising", "red")
    elif roc >= 2:
        return ("Warming", "yellow")
    elif roc <= -10:
        return ("Plunging", "bold cyan")
    elif roc <= -5:
        return ("Falling", "cyan")
    elif roc <= -2:
        return ("Cooling", "blue")
    else:
        return ("Steady", "dim")
