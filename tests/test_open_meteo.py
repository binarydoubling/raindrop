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
