"""Tests for Xweather station observation integration."""

import importlib
import io
import json
import urllib.error
from datetime import UTC, datetime, timedelta
from email.message import Message
from pathlib import Path

import pytest
from click.testing import CliRunner

from raindrop.cache import Cache, reset_cache
from raindrop.cli import cli
from raindrop.observations import (
    Measurement,
    Station,
    StationObservation,
    filter_observations,
    sort_observations,
)
from raindrop.open_meteo import GeocodingResult
from raindrop.providers.xweather import (
    XweatherClient,
    XweatherCredential,
    XweatherError,
    load_xweather_credential,
)
from raindrop.settings import Settings

stations_module = importlib.import_module("raindrop.commands.stations")

PWS_PAYLOAD = {
    "success": True,
    "error": None,
    "response": [
        {
            "id": "PWS_TEST",
            "dataSource": "PWS",
            "loc": {"lat": 64.89383, "long": -147.87567},
            "place": {"name": "goldstream", "city": "ester", "state": "ak", "country": "us"},
            "profile": {"elevM": 181.0},
            "relativeTo": {"distanceKM": 9.8, "distanceMI": 6.1},
            "ob": {
                "dateTimeISO": "2026-08-06T20:48:00+00:00",
                "recDateTimeISO": "2026-08-06T20:48:52+00:00",
                "tempC": 21.1,
                "dewpointC": 8.0,
                "humidity": 43,
                "windSpeedKPH": 0,
                "windGustKPH": 10,
                "windDirDEG": 229,
                "pressureMB": 1016,
                "precipMM": 0,
                "precipRateMM": None,
                "solradWM2": 422,
                "uvi": 3,
                "QC": "O",
                "QCcode": 10,
                "trustFactor": 100,
                "software": "WiFiLogger22.42",
            },
        }
    ],
}


class FakeResponse:
    """Minimal context manager response for urllib tests."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _observation(
    station_id: str,
    *,
    age_minutes: int,
    distance_km: float,
    qc: str = "pass",
) -> StationObservation:
    now = datetime(2026, 8, 6, 21, 0, tzinfo=UTC)
    observed = now - timedelta(minutes=age_minutes)
    station = Station(
        provider="xweather",
        station_id=station_id,
        name=station_id,
        latitude=64.0,
        longitude=-147.0,
        elevation_m=None,
        distance_km=distance_km,
        kind="personal",
        network_id="PWS",
        network_name="PWSweather",
    )
    return StationObservation(
        station=station,
        measurements={
            "air_temperature": Measurement(
                value_si=20,
                unit_si="degC",
                observed_at=observed,
                qc=qc,  # type: ignore[arg-type]
                qc_flags=[],
            )
        },
        retrieved_at=now,
        receipt_at=observed,
    )


def _geocoding_result() -> GeocodingResult:
    return GeocodingResult(
        id=1,
        name="Fairbanks",
        latitude=64.8378,
        longitude=-147.7164,
        elevation=136,
        timezone="America/Anchorage",
        feature_code="PPL",
        country_code="US",
        country="United States",
        country_id=6252001,
        population=32000,
        postcodes=[],
        admin1="Alaska",
        admin2=None,
        admin3=None,
        admin4=None,
        admin1_id=None,
        admin2_id=None,
        admin3_id=None,
        admin4_id=None,
    )


def test_credential_precedence_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XWEATHER_API_KEY", "env_id_env_secret")

    credential = load_xweather_credential()

    assert credential.source == "environment"
    assert credential.split_client_credentials() == ("env", "id_env_secret")


def test_credential_file_uses_raindrop_config_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XWEATHER_API_KEY", raising=False)
    monkeypatch.setattr("raindrop.settings.CONFIG_DIR", tmp_path)
    (tmp_path / "xweather.json").write_text(json.dumps({"api_key": "file_id_file_secret"}))

    credential = load_xweather_credential()

    assert credential.source == "file"
    assert credential.split_client_credentials() == ("file", "id_file_secret")


def test_parse_pws_payload_normalizes_units_and_qc() -> None:
    client = XweatherClient(credential=XweatherCredential("id_secret", "environment"))

    observations = client._parse_observations(PWS_PAYLOAD)

    assert len(observations) == 1
    observation = observations[0]
    assert observation.station.kind == "personal"
    assert observation.station.network_name == "PWSweather"
    assert observation.station.distance_km == 9.8
    assert observation.station.elevation_m == 181.0
    assert observation.measurements["air_temperature"].value_si == 21.1
    assert observation.measurements["wind_speed"].value_si == 0
    assert observation.measurements["wind_gust"].value_si == pytest.approx(10 / 3.6)
    assert observation.measurements["pressure_msl"].value_si == 101600
    assert observation.measurements["solar_radiation"].value_si == 422
    assert observation.measurements["uv_index"].value_si == 3
    assert observation.worst_qc() == "pass"


def test_parse_official_and_missing_wind_without_conflating_calm() -> None:
    payload = {
        "success": True,
        "response": [
            {
                "id": "PAFA",
                "dataSource": "METAR_NOAA",
                "loc": {"lat": 64.8, "long": -147.85},
                "place": {"name": "fairbanks"},
                "profile": {"elevM": 132},
                "ob": {
                    "dateTimeISO": "2026-08-06T20:00:00+00:00",
                    "tempC": 21.7,
                    "humidity": 53,
                    "QCcode": 10,
                },
            }
        ],
    }
    client = XweatherClient(credential=XweatherCredential("id_secret", "environment"))

    observation = client._parse_observations(payload)[0]

    assert observation.station.kind == "official"
    assert "wind_speed" not in observation.measurements


def test_filter_observations_removes_stale_future_and_failed() -> None:
    now = datetime(2026, 8, 6, 21, 0, tzinfo=UTC)
    fresh = _observation("fresh", age_minutes=5, distance_km=5)
    stale = _observation("stale", age_minutes=90, distance_km=5)
    failed = _observation("failed", age_minutes=5, distance_km=5, qc="fail")
    future = StationObservation(
        station=fresh.station,
        measurements={
            "air_temperature": Measurement(
                value_si=20,
                unit_si="degC",
                observed_at=now + timedelta(minutes=10),
                qc="pass",
                qc_flags=[],
            )
        },
        retrieved_at=now,
        receipt_at=None,
    )

    filtered = filter_observations(
        [fresh, stale, failed, future],
        max_age_minutes=60,
        include_stale=False,
        now=now,
    )

    assert [ob.station.station_id for ob in filtered] == ["fresh"]


def test_best_sort_prioritizes_freshness_before_distance() -> None:
    now = datetime(2026, 8, 6, 21, 0, tzinfo=UTC)
    nearer_stale = _observation("nearer", age_minutes=50, distance_km=1)
    farther_fresh = _observation("farther", age_minutes=5, distance_km=10)

    sorted_observations = sort_observations(
        [nearer_stale, farther_fresh],
        sort="best",
        max_age_minutes=60,
        now=now,
    )

    assert [ob.station.station_id for ob in sorted_observations] == ["farther", "nearer"]


def test_cache_key_does_not_contain_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAINDROP_NO_CACHE", raising=False)
    cache = Cache(cache_dir=tmp_path, default_ttl=60)
    reset_cache(cache)

    def fake_urlopen(request: object, timeout: int = 0) -> FakeResponse:
        assert "client_id=id" in request.full_url  # type: ignore[attr-defined]
        assert "client_secret=secret" in request.full_url  # type: ignore[attr-defined]
        return FakeResponse(PWS_PAYLOAD)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = XweatherClient(credential=XweatherCredential("id_secret", "environment"))

    try:
        client.nearby_observations(64.8378, -147.7164, radius_km=40, limit=1, kind="pws")
        cache_files = list(tmp_path.glob("*.json"))
        assert len(cache_files) == 1
        cache_payload = cache_files[0].read_text()
        assert "id_secret" not in cache_payload
        assert "client_secret" not in cache_payload
        assert "xweather:observations:closest" in cache_payload
    finally:
        reset_cache(None)


def test_provider_errors_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    body = json.dumps(
        {
            "error": {
                "code": "invalid_client",
                "description": "bad id_secret secret",
            }
        }
    ).encode()

    def fake_urlopen(request: object, timeout: int = 0) -> FakeResponse:
        raise urllib.error.HTTPError(
            url="https://redacted.invalid/",
            code=401,
            msg="Unauthorized",
            hdrs=Message(),
            fp=io.BytesIO(body),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = XweatherClient(credential=XweatherCredential("id_secret", "environment"))

    with pytest.raises(XweatherError) as exc:
        client.nearby_raw(64.8378, -147.7164, radius_km=40, limit=1, kind="pws")

    assert "id_secret" not in str(exc.value)
    assert "secret" not in str(exc.value)
    assert "[redacted]" in str(exc.value)


def test_stations_help_is_registered() -> None:
    runner = CliRunner()

    result = runner.invoke(cli, ["stations", "nearby", "--help"])

    assert result.exit_code == 0
    assert "List live nearby measured station observations" in result.output


def test_stations_nearby_json_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    client = XweatherClient(credential=XweatherCredential("id_secret", "environment"))
    observation = client._parse_observations(PWS_PAYLOAD)[0]
    monkeypatch.setattr(stations_module, "get_settings", lambda: Settings())
    monkeypatch.setattr(
        stations_module, "geocode", lambda location, country=None: _geocoding_result()
    )
    monkeypatch.setattr(
        stations_module.XweatherClient,
        "nearby_observations",
        lambda self, latitude, longitude, radius_km, limit=10, kind="pws": [observation],
    )
    runner = CliRunner()

    result = runner.invoke(cli, ["stations", "nearby", "Fairbanks", "--json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["provider"] == "xweather"
    assert data["units"] == "SI"
    assert data["attribution"] == "Powered by Vaisala Xweather"
    assert data["observations"][0]["station"]["kind"] == "personal"
    assert "api_key" not in result.output
    assert "id_secret" not in result.output
