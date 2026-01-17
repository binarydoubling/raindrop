"""Astronomical calculations for sun and moon data."""

import math
from datetime import datetime, timedelta


def julian_day(dt: datetime) -> float:
    """Calculate Julian Day Number from a datetime."""
    year = dt.year
    month = dt.month
    day = dt.day + dt.hour / 24 + dt.minute / 1440 + dt.second / 86400

    if month <= 2:
        year -= 1
        month += 12

    a = int(year / 100)
    b = 2 - a + int(a / 4)

    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
    return jd


def moon_phase(dt: datetime) -> tuple[float, str, str]:
    """
    Calculate moon phase for a given date.

    Returns:
        (phase: 0-1, name: str, emoji: str)
        phase 0 = new moon, 0.5 = full moon
    """
    # Calculate days since known new moon (Jan 6, 2000 at 18:14 UTC)
    known_new_moon = datetime(2000, 1, 6, 18, 14)
    synodic_month = 29.530588853  # days

    days_since = (dt - known_new_moon).total_seconds() / 86400
    cycles = days_since / synodic_month
    phase = cycles % 1  # 0 to 1

    # Phase names and symbols
    if phase < 0.0625:
        name, symbol = "New Moon", "\U0001f311"
    elif phase < 0.1875:
        name, symbol = "Waxing Crescent", "\U0001f312"
    elif phase < 0.3125:
        name, symbol = "First Quarter", "\U0001f313"
    elif phase < 0.4375:
        name, symbol = "Waxing Gibbous", "\U0001f314"
    elif phase < 0.5625:
        name, symbol = "Full Moon", "\U0001f315"
    elif phase < 0.6875:
        name, symbol = "Waning Gibbous", "\U0001f316"
    elif phase < 0.8125:
        name, symbol = "Last Quarter", "\U0001f317"
    elif phase < 0.9375:
        name, symbol = "Waning Crescent", "\U0001f318"
    else:
        name, symbol = "New Moon", "\U0001f311"

    return phase, name, symbol


def moon_illumination(phase: float) -> float:
    """Calculate moon illumination percentage from phase (0-1)."""
    # Use cosine to get illumination
    # At phase 0 (new moon) = 0%, phase 0.5 (full moon) = 100%
    illumination = (1 - math.cos(phase * 2 * math.pi)) / 2 * 100
    return illumination


def next_moon_phase(dt: datetime, target_phase: float) -> datetime:
    """
    Calculate date of next occurrence of a moon phase.

    target_phase: 0 = new moon, 0.25 = first quarter, 0.5 = full, 0.75 = last quarter
    """
    synodic_month = 29.530588853
    known_new_moon = datetime(2000, 1, 6, 18, 14)

    days_since = (dt - known_new_moon).total_seconds() / 86400
    cycles = days_since / synodic_month
    current_phase = cycles % 1

    # Calculate days until target phase
    if target_phase > current_phase:
        days_until = (target_phase - current_phase) * synodic_month
    else:
        days_until = (1 - current_phase + target_phase) * synodic_month

    return dt + timedelta(days=days_until)


def golden_hour(
    sunrise: datetime, sunset: datetime
) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
    """
    Calculate golden hour times.

    Golden hour is roughly the first/last hour after sunrise/before sunset.
    More precisely, when the sun is 6 degrees above horizon or less.
    We approximate this as ~45-60 minutes.

    Returns:
        ((morning_start, morning_end), (evening_start, evening_end))
    """
    duration = timedelta(minutes=50)  # Approximate golden hour duration

    morning_start = sunrise
    morning_end = sunrise + duration

    evening_start = sunset - duration
    evening_end = sunset

    return (morning_start, morning_end), (evening_start, evening_end)


def blue_hour(
    sunrise: datetime, sunset: datetime
) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
    """
    Calculate blue hour times.

    Blue hour occurs when sun is 4-6 degrees below horizon.
    Approximately 20-30 minutes before sunrise and after sunset.

    Returns:
        ((morning_start, morning_end), (evening_start, evening_end))
    """
    duration = timedelta(minutes=25)

    morning_start = sunrise - duration - timedelta(minutes=10)
    morning_end = sunrise - timedelta(minutes=10)

    evening_start = sunset + timedelta(minutes=10)
    evening_end = sunset + duration + timedelta(minutes=10)

    return (morning_start, morning_end), (evening_start, evening_end)


def daylight_duration(sunrise: datetime, sunset: datetime) -> timedelta:
    """Calculate duration of daylight."""
    return sunset - sunrise


def format_daylight_duration(td: timedelta) -> str:
    """Format daylight duration as hours and minutes."""
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}h {minutes}m"


def solar_noon(sunrise: datetime, sunset: datetime) -> datetime:
    """Calculate solar noon (midpoint between sunrise and sunset)."""
    duration = sunset - sunrise
    return sunrise + duration / 2
