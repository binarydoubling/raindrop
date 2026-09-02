"""Tests for provider-neutral weather normalization."""

import pytest

from raindrop.open_meteo import OpenMeteo
from raindrop.settings import Settings
from raindrop.weather_provider import XweatherWeatherProvider, select_weather_provider


class FakeCredentialStatus:
    """Minimal credential status for provider-selection tests."""

    configured = True


class FakeXweatherClient:
    """Small Xweather payload source for normalization tests."""

    def conditions_raw(self, latitude: float, longitude: float) -> dict:
        return {
            "success": True,
            "response": [
                {
                    "profile": {
                        "tz": "America/Anchorage",
                        "tzname": "AKDT",
                        "tzoffset": -28800,
                        "elevM": 136,
                    },
                    "periods": [
                        {
                            "dateTimeISO": "2026-08-25T19:37:00-08:00",
                            "tempC": 10,
                            "feelslikeC": 8,
                            "humidity": 97,
                            "dewpointC": 7,
                            "pressureMB": 1005,
                            "spressureMB": 988.8,
                            "windSpeedKPH": 36,
                            "windGustKPH": 54,
                            "windDirDEG": 224,
                            "precipMM": 25.4,
                            "snowCM": 2,
                            "visibilityKM": 16,
                            "sky": 79,
                            "weatherPrimary": "Showers",
                            "weatherPrimaryCoded": "D::RW",
                            "uvi": 2,
                            "isDay": True,
                        }
                    ],
                }
            ],
        }

    def forecast_raw(
        self,
        latitude: float,
        longitude: float,
        *,
        interval: str,
        limit: int,
    ) -> dict:
        if interval == "1hr":
            periods = [
                {
                    "dateTimeISO": "2026-08-25T20:00:00-08:00",
                    "tempC": 12,
                    "feelslikeC": 11,
                    "humidity": 88,
                    "dewpointC": 9,
                    "pop": 70,
                    "precipMM": 1,
                    "snowCM": 0,
                    "windSpeedKPH": 18,
                    "windGustKPH": 30,
                    "windDirDEG": 200,
                    "sky": 90,
                    "visibilityKM": 9,
                    "weatherPrimary": "Mostly Cloudy with Showers",
                    "weatherPrimaryCoded": "D::RW",
                    "uvi": 0,
                    "isDay": True,
                }
            ]
        else:
            periods = [
                {
                    "dateTimeISO": "2026-08-25T07:00:00-08:00",
                    "maxTempC": 15,
                    "minTempC": 5,
                    "avgTempC": 10,
                    "maxFeelslikeC": 14,
                    "minFeelslikeC": 3,
                    "avgFeelslikeC": 8,
                    "pop": 80,
                    "precipMM": 5,
                    "snowCM": 0.5,
                    "windSpeedMaxKPH": 25,
                    "windGustKPH": 40,
                    "windDirDEG": 180,
                    "weatherPrimary": "Cloudy with Showers",
                    "weatherPrimaryCoded": "D::RW",
                    "uvi": 3,
                    "sunriseISO": "2026-08-25T06:10:56-08:00",
                    "sunsetISO": "2026-08-25T21:34:45-08:00",
                }
            ]
        return {
            "success": True,
            "response": [
                {
                    "profile": {
                        "tz": "America/Anchorage",
                        "tzname": "AKDT",
                        "tzoffset": -28800,
                        "elevM": 136,
                    },
                    "periods": periods,
                }
            ],
        }


def test_auto_provider_selects_xweather_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "raindrop.weather_provider.get_xweather_credential_status",
        lambda: FakeCredentialStatus(),
    )

    selection = select_weather_provider(Settings())

    assert selection.name == "xweather"
    assert selection.model_label == "Xweather"


def test_model_selection_keeps_auto_on_open_meteo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "raindrop.weather_provider.get_xweather_credential_status",
        lambda: FakeCredentialStatus(),
    )

    selection = select_weather_provider(Settings(), model_name="gfs")

    assert selection.name == "open-meteo"
    assert selection.model_label == "gfs"
    assert isinstance(selection.client, OpenMeteo)


def test_xweather_normalizes_current_hourly_and_daily_units() -> None:
    provider = XweatherWeatherProvider(client=FakeXweatherClient())  # type: ignore[arg-type]

    result = provider.forecast(
        64.8378,
        -147.7164,
        current=[
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "snowfall",
            "weather_code",
            "cloud_cover",
            "pressure_msl",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "visibility",
            "uv_index",
            "is_day",
        ],
        hourly=["temperature_2m", "precipitation_probability", "weather_code", "wind_speed_10m"],
        daily=[
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "precipitation_sum",
            "snowfall_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
            "sunrise",
            "sunset",
            "weather_code",
        ],
        temperature_unit="fahrenheit",
        wind_speed_unit="mph",
        precipitation_unit="inch",
        forecast_days=1,
    )

    assert result.provider == "xweather"
    assert result.timezone == "America/Anchorage"
    assert result.attribution == "Powered by Vaisala Xweather"
    assert result.current is not None
    assert result.current.temperature_2m == 50
    assert result.current.apparent_temperature == pytest.approx(46.4)
    assert result.current.precipitation == 1
    assert result.current.snowfall == pytest.approx(0.7874)
    assert result.current.wind_speed_10m == pytest.approx(22.369)
    assert result.current.visibility == 16000
    assert result.current.weather_code == 61
    assert result.hourly is not None
    assert result.hourly.precipitation_probability == [70]
    assert result.hourly.weather_code == [61]
    assert result.daily is not None
    assert result.daily.temperature_2m_max == [59]
    assert result.daily.temperature_2m_min == [41]
    assert result.daily.precipitation_sum == pytest.approx([0.1969])
    assert result.daily.sunrise == ["2026-08-25T06:10"]
