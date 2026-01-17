"""Utility functions for raindrop."""

from .formatting import (
    sparkline,
    format_duration,
    format_visibility,
    format_uv,
    format_delta,
    format_precip_chance,
    format_pollutant,
    format_us_aqi,
    format_alert_time,
    deg_to_compass,
)

from .weather import (
    WEATHER_CODES,
    WEATHER_LABELS,
    TEMP_SYMBOLS,
    WIND_SYMBOLS,
    US_AQI_LEVELS,
    SEVERITY_COLORS,
    URGENCY_COLORS,
    ema,
    calc_roc,
    calc_volatility,
    trend_signal,
    roc_signal,
)

__all__ = [
    # formatting
    "sparkline",
    "format_duration",
    "format_visibility",
    "format_uv",
    "format_delta",
    "format_precip_chance",
    "format_pollutant",
    "format_us_aqi",
    "format_alert_time",
    "deg_to_compass",
    # weather
    "WEATHER_CODES",
    "WEATHER_LABELS",
    "TEMP_SYMBOLS",
    "WIND_SYMBOLS",
    "US_AQI_LEVELS",
    "SEVERITY_COLORS",
    "URGENCY_COLORS",
    "ema",
    "calc_roc",
    "calc_volatility",
    "trend_signal",
    "roc_signal",
]
