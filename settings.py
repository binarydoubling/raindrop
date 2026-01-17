"""Persistent settings management for raindrop."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from open_meteo import TemperatureUnit, WindSpeedUnit, PrecipitationUnit


CONFIG_DIR = Path.home() / ".config" / "raindrop"
CONFIG_FILE = CONFIG_DIR / "config.json"


# Available weather models (from Open-Meteo docs)
# Format: (display_name, api_name)
AVAILABLE_MODELS: dict[str, str] = {
    # Best match (default)
    "auto": "best_match",
    # ECMWF
    "ecmwf": "ecmwf_ifs025",
    "ecmwf_aifs": "ecmwf_aifs025",
    # US models (NOAA/NCEP)
    "gfs": "ncep_gfs025",
    "hrrr": "ncep_hrrr_conus",
    # German (DWD)
    "icon": "dwd_icon",
    "icon_eu": "dwd_icon_eu",
    "icon_d2": "dwd_icon_d2",
    # French
    "arpege": "meteofrance_arpege_europe",
    "arome": "meteofrance_arome_france_hd",
    # UK
    "ukmo": "ukmo_global_deterministic_10km",
    # Canadian
    "gem": "gem_global",
    "gem_hrdps": "gem_hrdps_continental",
    # Japanese
    "jma": "jma_gsm",
    # Norwegian
    "metno": "metno_nordic_pp",
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
    model: str | None = None  # None means "best_match" (auto)

    def save(self) -> None:
        """Save settings to config file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from config file, or return defaults."""
        if not CONFIG_FILE.exists():
            return cls()

        try:
            data = json.loads(CONFIG_FILE.read_text())
            return cls(
                location=data.get("location"),
                country_code=data.get("country_code"),
                temperature_unit=data.get("temperature_unit", "fahrenheit"),
                wind_speed_unit=data.get("wind_speed_unit", "mph"),
                precipitation_unit=data.get("precipitation_unit", "mm"),
                model=data.get("model"),
            )
        except (json.JSONDecodeError, KeyError):
            return cls()


def get_settings() -> Settings:
    """Get current settings."""
    return Settings.load()
