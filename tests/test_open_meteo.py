"""Tests for Open-Meteo client helpers."""

import pytest

from raindrop.open_meteo import OpenMeteo, OpenMeteoError, _parse_location


def _geocode_response(admin1: str = "Texas", country: str = "United States") -> dict:
    return {
        "results": [
            {
                "id": 1,
                "name": "Paris",
                "latitude": 33.66,
                "longitude": -95.55,
                "elevation": 183.0,
                "timezone": "America/Chicago",
                "feature_code": "PPL",
                "country_code": "US",
                "country": country,
                "country_id": 6252001,
                "population": 25000,
                "postcodes": [],
                "admin1": admin1,
            }
        ]
    }


def test_parse_location_expands_us_state() -> None:
    assert _parse_location("Fairbanks, AK") == ("Fairbanks", "Alaska")


def test_parse_location_keeps_country_qualifier() -> None:
    assert _parse_location("Munich, Germany") == ("Munich", "Germany")


def test_geocode_filters_matching_qualifier(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenMeteo()
    monkeypatch.setattr(client, "_request", lambda url, ttl=None: _geocode_response())

    results = client.geocode("Paris, TX")

    assert len(results) == 1
    assert results[0].admin1 == "Texas"


def test_geocode_raises_on_unmatched_qualifier(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenMeteo()
    monkeypatch.setattr(
        client, "_request", lambda url, ttl=None: _geocode_response(admin1="Ontario")
    )

    with pytest.raises(OpenMeteoError):
        client.geocode("Paris, TX")


def test_forecast_maps_known_fields_and_ignores_unknown_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenMeteo()
    monkeypatch.setattr(
        client,
        "_request",
        lambda url, ttl=None: {
            "latitude": 1.0,
            "longitude": 2.0,
            "elevation": 3.0,
            "timezone": "UTC",
            "timezone_abbreviation": "UTC",
            "utc_offset_seconds": 0,
            "current": {
                "time": "2026-01-01T00:00",
                "interval": 900,
                "temperature_2m": 4.0,
                "is_day": 0,
                "future_api_field": "ignored",
            },
        },
    )

    result = client.forecast(1.0, 2.0, current=["temperature_2m", "is_day"])

    assert result.current is not None
    assert result.current.temperature_2m == 4.0
    assert result.current.is_day is False


def test_marine_uses_shared_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenMeteo()
    requested: list[tuple[str, int | None]] = []

    def fake_request(url: str, ttl: int | None = None) -> dict:
        requested.append((url, ttl))
        return {"hourly": {}}

    monkeypatch.setattr(client, "_request", fake_request)

    assert client.marine(1.0, 2.0) == {"hourly": {}}
    assert requested[0][0].startswith("https://marine-api.open-meteo.com/v1/marine?")
    assert "wave_height" in requested[0][0]
    assert requested[0][1] == 600
