"""Persistent settings management for raindrop."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from open_meteo import TemperatureUnit, WindSpeedUnit, PrecipitationUnit


CONFIG_DIR = Path.home() / ".config" / "raindrop"
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

    # Weather model
    model: str | None = None  # None = let Open-Meteo auto-select

    # Favorites (alias -> Favorite)
    favorites: dict[str, Favorite] = field(default_factory=dict)

    def save(self) -> None:
        """Save settings to config file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        CONFIG_FILE.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from config file, or return defaults."""
        if not CONFIG_FILE.exists():
            return cls()

        try:
            data = json.loads(CONFIG_FILE.read_text())
            # Parse favorites back into Favorite objects
            favorites_data = data.get("favorites", {})
            favorites = {
                alias: Favorite(
                    name=fav["name"],
                    country_code=fav.get("country_code"),
                )
                for alias, fav in favorites_data.items()
            }
            return cls(
                location=data.get("location"),
                country_code=data.get("country_code"),
                temperature_unit=data.get("temperature_unit", "fahrenheit"),
                wind_speed_unit=data.get("wind_speed_unit", "mph"),
                precipitation_unit=data.get("precipitation_unit", "mm"),
                model=data.get("model"),
                favorites=favorites,
            )
        except (json.JSONDecodeError, KeyError):
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
