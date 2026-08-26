"""Tests for settings loading and validation."""

import json
from pathlib import Path

import pytest

from raindrop import settings as settings_module
from raindrop.settings import Favorite, Settings, normalize_country_code, resolve_model


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep settings tests away from the user's real config."""
    monkeypatch.setattr(settings_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(settings_module, "CONFIG_FILE", tmp_path / "config.json")


def test_settings_round_trip() -> None:
    config = Settings(
        location="Seattle",
        country_code="US",
        temperature_unit="celsius",
        wind_speed_unit="kmh",
        precipitation_unit="mm",
        weather_provider="xweather",
        model="gfs",
        favorites={"home": Favorite(name="Seattle", country_code="US")},
    )

    config.save()
    loaded = Settings.load()

    assert loaded.location == "Seattle"
    assert loaded.country_code == "US"
    assert loaded.temperature_unit == "celsius"
    assert loaded.wind_speed_unit == "kmh"
    assert loaded.weather_provider == "xweather"
    assert loaded.model == "gfs"
    assert loaded.favorites["home"].name == "Seattle"


def test_settings_load_handles_malformed_config() -> None:
    settings_module.CONFIG_FILE.write_text("not json")

    loaded = Settings.load()

    assert loaded == Settings()


def test_settings_load_validates_untrusted_schema() -> None:
    settings_module.CONFIG_FILE.write_text(
        json.dumps(
            {
                "temperature_unit": "kelvin",
                "wind_speed_unit": "warp",
                "precipitation_unit": "buckets",
                "country_code": "USA",
                "weather_provider": "storm-machine",
                "favorites": ["not", "a", "dict"],
            }
        )
    )

    loaded = Settings.load()

    assert loaded.temperature_unit == "fahrenheit"
    assert loaded.wind_speed_unit == "mph"
    assert loaded.precipitation_unit == "mm"
    assert loaded.country_code is None
    assert loaded.weather_provider == "auto"
    assert loaded.favorites == {}


def test_normalize_country_code() -> None:
    assert normalize_country_code("us") == "US"
    assert normalize_country_code(None) is None
    with pytest.raises(ValueError):
        normalize_country_code("USA")


def test_resolve_model() -> None:
    model_key, models = resolve_model("gfs", Settings())

    assert model_key == "gfs"
    assert models == ["gfs_seamless"]
