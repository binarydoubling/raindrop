"""Persistent settings management for raindrop."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, cast

from raindrop.open_meteo import PrecipitationUnit, TemperatureUnit, WindSpeedUnit

TEMPERATURE_UNITS = ("celsius", "fahrenheit")
WIND_SPEED_UNITS = ("kmh", "ms", "mph", "kn")
PRECIPITATION_UNITS = ("mm", "inch")
WEATHER_PROVIDERS = ("auto", "open-meteo", "xweather")
WeatherProviderName = Literal["auto", "open-meteo", "xweather"]


def _default_config_dir() -> Path:
    """Return the platform-friendly config directory for raindrop."""
    override = os.environ.get("RAINDROP_CONFIG_DIR")
    if override:
        return Path(override).expanduser()

    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home).expanduser() / "raindrop"

    return Path.home() / ".config" / "raindrop"


CONFIG_DIR = _default_config_dir()
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class Favorite:
    """A saved location."""

    name: str
    country_code: str | None = None


# Available weather models (from Open-Meteo docs)
# Maps friendly name -> API name
# Uses "_seamless" variants where available (auto-blends global + regional).
AVAILABLE_MODELS: dict[str, str] = {
    # ECMWF
    "ecmwf": "ecmwf_ifs025",
    # US models (NOAA/NCEP)
    "gfs": "gfs_seamless",
    "hrrr": "ncep_hrrr_conus",
    # German (DWD)
    "icon": "icon_seamless",
    "icon_eu": "icon_eu",
    "icon_d2": "icon_d2",
    # French
    "arpege": "arpege_seamless",
    "arome": "arome_seamless",
    # UK
    "ukmo": "ukmo_seamless",
    # Canadian
    "gem": "gem_seamless",
    "gem_hrdps": "gem_hrdps_continental",
    # Japanese
    "jma": "jma_seamless",
    # Norwegian
    "metno": "metno_seamless",
}


def normalize_country_code(country_code: str | None) -> str | None:
    """Validate and normalize an ISO 3166-1 alpha-2 country code."""
    if country_code is None or country_code == "":
        return None

    normalized = country_code.strip().upper()
    if len(normalized) != 2 or not normalized.isalpha():
        raise ValueError("Country code must be a 2-letter ISO code, e.g. US, ES, DE")
    return normalized


@dataclass
class Settings:
    """User settings for raindrop."""

    # Default location
    location: str | None = None
    country_code: str | None = None

    # Units
    temperature_unit: TemperatureUnit = "fahrenheit"
    wind_speed_unit: WindSpeedUnit = "mph"
    precipitation_unit: PrecipitationUnit = "mm"

    # Weather provider and model
    weather_provider: WeatherProviderName = "auto"
    model: str | None = None  # None = let Open-Meteo auto-select

    # Favorites (alias -> Favorite)
    favorites: dict[str, Favorite] = field(default_factory=dict)

    def save(self) -> None:
        """Save settings to config file atomically."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        payload = json.dumps(data, indent=2)
        tmp_file = CONFIG_FILE.with_suffix(".json.tmp")
        tmp_file.write_text(payload)
        tmp_file.replace(CONFIG_FILE)

    @classmethod
    def load(cls) -> Settings:
        """Load settings from config file, or return defaults."""
        if not CONFIG_FILE.exists():
            return cls()

        try:
            raw_data = json.loads(CONFIG_FILE.read_text())
            if not isinstance(raw_data, dict):
                return cls()

            favorites = _load_favorites(raw_data.get("favorites", {}))
            temperature_unit = _load_temperature_unit(raw_data.get("temperature_unit"))
            wind_speed_unit = _load_wind_speed_unit(raw_data.get("wind_speed_unit"))
            precipitation_unit = _load_precipitation_unit(raw_data.get("precipitation_unit"))
            model = raw_data.get("model")
            if not isinstance(model, str) or model not in AVAILABLE_MODELS:
                model = None

            weather_provider = _load_weather_provider(raw_data.get("weather_provider"))

            location = raw_data.get("location")
            if not isinstance(location, str):
                location = None

            raw_country_code = raw_data.get("country_code")
            if isinstance(raw_country_code, str):
                try:
                    country_code = normalize_country_code(raw_country_code)
                except ValueError:
                    country_code = None
            else:
                country_code = None

            return cls(
                location=location,
                country_code=country_code,
                temperature_unit=temperature_unit,
                wind_speed_unit=wind_speed_unit,
                precipitation_unit=precipitation_unit,
                weather_provider=weather_provider,
                model=model,
                favorites=favorites,
            )
        except (json.JSONDecodeError, OSError, TypeError, AttributeError):
            return cls()

    def resolve_location(self, location: str | None) -> tuple[str, str | None]:
        """
        Resolve a location string, checking favorites first.
        Returns (location_name, country_code).
        """
        if location is None:
            # Use default location
            if self.location is None:
                raise ValueError("No location provided and no default set")
            return self.location, self.country_code

        # Check if it's a favorite alias
        if location in self.favorites:
            fav = self.favorites[location]
            return fav.name, fav.country_code

        return location, None


def _load_favorites(value: object) -> dict[str, Favorite]:
    """Parse favorites from untrusted config data."""
    if not isinstance(value, dict):
        return {}

    favorites: dict[str, Favorite] = {}
    for alias, favorite in value.items():
        if not isinstance(alias, str) or not isinstance(favorite, dict):
            continue
        name = favorite.get("name")
        if not isinstance(name, str) or not name:
            continue
        raw_country_code = favorite.get("country_code")
        if isinstance(raw_country_code, str):
            try:
                country_code = normalize_country_code(raw_country_code)
            except ValueError:
                country_code = None
        else:
            country_code = None
        favorites[alias] = Favorite(name=name, country_code=country_code)
    return favorites


def _load_weather_provider(value: object) -> WeatherProviderName:
    """Parse a weather provider from config data."""
    if value in WEATHER_PROVIDERS:
        return cast(WeatherProviderName, value)
    return "auto"


def _load_temperature_unit(value: object) -> TemperatureUnit:
    """Parse a temperature unit from config data."""
    if value in TEMPERATURE_UNITS:
        return cast(TemperatureUnit, value)
    return "fahrenheit"


def _load_wind_speed_unit(value: object) -> WindSpeedUnit:
    """Parse a wind speed unit from config data."""
    if value in WIND_SPEED_UNITS:
        return cast(WindSpeedUnit, value)
    return "mph"


def _load_precipitation_unit(value: object) -> PrecipitationUnit:
    """Parse a precipitation unit from config data."""
    if value in PRECIPITATION_UNITS:
        return cast(PrecipitationUnit, value)
    return "mm"


def resolve_model(
    model_name: str | None, settings: Settings
) -> tuple[str | None, list[str] | None]:
    """Resolve a weather model from CLI flag or settings.

    Args:
        model_name: Model name from CLI ``-m`` flag, or None.
        settings: Current user settings.

    Returns:
        (model_key, models) where model_key is the friendly name (or None
        for auto) and models is the API model list for ``om.forecast()``.

    Raises:
        ValueError: If the model name is not recognized.
    """
    model_key = model_name or settings.model
    if not model_key:
        return None, None
    api_model = AVAILABLE_MODELS.get(model_key)
    if api_model is None:
        available = ", ".join(sorted(AVAILABLE_MODELS))
        raise ValueError(f"Unknown model '{model_key}'. Available models: {available}")
    return model_key, [api_model]


def get_settings() -> Settings:
    """Get current settings."""
    return Settings.load()
