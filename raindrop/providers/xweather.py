"""Xweather station observation provider."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from raindrop import settings as settings_module
from raindrop.cache import cached_request
from raindrop.observations import (
    Measurement,
    QcStatus,
    Station,
    StationKind,
    StationObservation,
    haversine_km,
    parse_datetime,
    utc_now,
)

XWEATHER_BASE_URL = "https://data.api.xweather.com"
XWEATHER_CREDENTIAL_FILE = "xweather.json"
XWEATHER_CREDENTIAL_ENV = "XWEATHER_API_KEY"
XWEATHER_DEFAULT_TTL = 60
XWEATHER_USER_AGENT = "Raindrop/0.1 (+https://github.com/binarydoubling/raindrop)"

StationKindFilter = Literal["pws", "official", "all"]

_OBSERVATION_FIELDS = "id,dataSource,loc,place,profile,ob,relativeTo"
_CONDITION_FIELDS = (
    "loc,place,profile,"
    "periods.timestamp,periods.dateTimeISO,periods.tempC,periods.feelslikeC,"
    "periods.humidity,periods.dewpointC,periods.pressureMB,periods.spressureMB,"
    "periods.altimeterMB,periods.windSpeedKPH,periods.windGustKPH,periods.windDirDEG,"
    "periods.precipMM,periods.precipRateMM,periods.snowCM,periods.snowRateCM,"
    "periods.visibilityKM,periods.sky,periods.weather,periods.weatherPrimary,"
    "periods.weatherPrimaryCoded,periods.uvi,periods.solradWM2,periods.isDay"
)
_FORECAST_FIELDS = (
    "loc,place,profile,"
    "periods.timestamp,periods.dateTimeISO,periods.maxTempC,periods.minTempC,"
    "periods.avgTempC,periods.tempC,periods.feelslikeC,periods.maxFeelslikeC,"
    "periods.minFeelslikeC,periods.avgFeelslikeC,periods.dewpointC,periods.humidity,"
    "periods.pop,periods.precipMM,periods.snowCM,periods.pressureMB,"
    "periods.windSpeedKPH,periods.windSpeedMaxKPH,periods.windGustKPH,"
    "periods.windDirDEG,periods.sky,periods.visibilityKM,periods.weather,"
    "periods.weatherPrimary,periods.weatherPrimaryCoded,periods.uvi,periods.solradWM2,"
    "periods.isDay,periods.sunriseISO,periods.sunsetISO"
)
_AUTH_QUERY_KEYS = {"client_id", "client_secret", "api_key", "apikey", "access_token"}


class XweatherError(Exception):
    """Raised when Xweather returns an error or malformed payload."""


class XweatherCredentialError(XweatherError):
    """Raised when Xweather credentials are missing or unusable."""


@dataclass(frozen=True)
class XweatherCredential:
    """A loaded Xweather credential."""

    api_key: str
    source: Literal["environment", "file"]

    def split_client_credentials(self) -> tuple[str, str]:
        """Return Weather API client id and secret from the single configured token."""
        if "_" not in self.api_key:
            raise XweatherCredentialError(
                "Xweather Weather API credentials must be the combined "
                "client_id_client_secret key issued by Xweather."
            )
        client_id, client_secret = self.api_key.split("_", 1)
        if not client_id or not client_secret:
            raise XweatherCredentialError("Xweather credential is incomplete")
        return client_id, client_secret


@dataclass(frozen=True)
class XweatherCredentialStatus:
    """Non-secret Xweather credential status for display."""

    configured: bool
    source: Literal["environment", "file"] | None
    path: str | None = None


def get_xweather_credential_path() -> Path:
    """Return the configured Xweather credential file path."""
    return settings_module.CONFIG_DIR / XWEATHER_CREDENTIAL_FILE


def get_xweather_credential_status() -> XweatherCredentialStatus:
    """Return whether Xweather credentials are configured without exposing values."""
    if os.environ.get(XWEATHER_CREDENTIAL_ENV):
        return XweatherCredentialStatus(configured=True, source="environment")

    path = get_xweather_credential_path()
    if not path.exists():
        return XweatherCredentialStatus(configured=False, source=None, path=str(path))

    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return XweatherCredentialStatus(configured=False, source=None, path=str(path))

    configured = (
        isinstance(raw, dict)
        and isinstance(raw.get("api_key"), str)
        and bool(raw["api_key"].strip())
    )
    return XweatherCredentialStatus(
        configured=configured,
        source="file" if configured else None,
        path=str(path),
    )


def load_xweather_credential() -> XweatherCredential:
    """Load an Xweather API key from environment or Raindrop's config directory."""
    env_value = os.environ.get(XWEATHER_CREDENTIAL_ENV)
    if env_value and env_value.strip():
        return XweatherCredential(api_key=env_value.strip(), source="environment")

    path = get_xweather_credential_path()
    if not path.exists():
        raise XweatherCredentialError(
            "Xweather is not configured. Set XWEATHER_API_KEY or create "
            f"{path} with {json.dumps({'api_key': '...'})}."
        )

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise XweatherCredentialError("Xweather credential file is malformed JSON") from e
    except OSError as e:
        raise XweatherCredentialError("Could not read Xweather credential file") from e

    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("api_key"), str)
        or not raw["api_key"].strip()
    ):
        raise XweatherCredentialError("Xweather credential file must contain an api_key string")
    return XweatherCredential(api_key=raw["api_key"].strip(), source="file")


class XweatherClient:
    """Read-only client for Xweather station observations."""

    def __init__(
        self,
        base_url: str = XWEATHER_BASE_URL,
        timeout: int = 10,
        credential: XweatherCredential | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._credential = credential

    @property
    def credential(self) -> XweatherCredential:
        """Return the client credential, loading it lazily."""
        if self._credential is None:
            self._credential = load_xweather_credential()
        return self._credential

    def nearby_observations(
        self,
        latitude: float,
        longitude: float,
        *,
        radius_km: float,
        limit: int = 10,
        kind: StationKindFilter = "pws",
    ) -> list[StationObservation]:
        """Return nearby station observations normalized to Raindrop's schema."""
        payload = self.nearby_raw(
            latitude,
            longitude,
            radius_km=radius_km,
            limit=limit,
            kind=kind,
        )
        return self._parse_observations(
            payload,
            requested_latitude=latitude,
            requested_longitude=longitude,
        )

    def nearby_raw(
        self,
        latitude: float,
        longitude: float,
        *,
        radius_km: float,
        limit: int = 10,
        kind: StationKindFilter = "pws",
    ) -> dict[str, Any]:
        """Return Xweather's nearby observation payload without credentials."""
        radius_mi = radius_km * 0.621371
        params = {
            "p": f"{latitude:.5f},{longitude:.5f}",
            "radius": f"{radius_mi:.3f}mi",
            "limit": str(limit),
            "filter": _filter_for_kind(kind),
            "fields": _OBSERVATION_FIELDS,
        }
        cache_key = (
            "xweather:observations:closest:"
            f"{latitude:.5f}:{longitude:.5f}:{radius_mi:.3f}mi:{limit}:{kind}"
        )
        payload = self._request(
            "observations/closest", params, cache_key=cache_key, ttl=XWEATHER_DEFAULT_TTL
        )
        return _sanitize_payload(payload)

    def conditions_raw(self, latitude: float, longitude: float) -> dict[str, Any]:
        """Return Xweather conditions for a coordinate without credentials."""
        params = {"fields": _CONDITION_FIELDS}
        cache_key = f"xweather:conditions:{latitude:.5f}:{longitude:.5f}"
        payload = self._request(
            "conditions/" + _coordinate_id(latitude, longitude), params, cache_key=cache_key, ttl=60
        )
        return _sanitize_payload(payload)

    def forecast_raw(
        self,
        latitude: float,
        longitude: float,
        *,
        interval: str,
        limit: int,
    ) -> dict[str, Any]:
        """Return Xweather forecast periods for a coordinate without credentials."""
        params = {
            "filter": interval,
            "limit": str(limit),
            "fields": _FORECAST_FIELDS,
        }
        cache_key = f"xweather:forecasts:{latitude:.5f}:{longitude:.5f}:{interval}:{limit}"
        payload = self._request(
            "forecasts/" + _coordinate_id(latitude, longitude), params, cache_key=cache_key, ttl=600
        )
        return _sanitize_payload(payload)

    def current_observation(self, station_id: str) -> StationObservation:
        """Return the current observation for a specific station ID."""
        payload = self.current_raw(station_id)
        observations = self._parse_observations(payload)
        if not observations:
            raise XweatherError(f"No current observation returned for station '{station_id}'")
        return observations[0]

    def current_raw(self, station_id: str) -> dict[str, Any]:
        """Return Xweather's current station payload without credentials."""
        safe_id = urllib.parse.quote(station_id, safe="")
        params = {
            "filter": "allstations",
            "fields": _OBSERVATION_FIELDS,
        }
        cache_key = f"xweather:observations:station:{station_id}"
        payload = self._request(
            f"observations/{safe_id}", params, cache_key=cache_key, ttl=XWEATHER_DEFAULT_TTL
        )
        return _sanitize_payload(payload)

    def _request(
        self,
        endpoint: str,
        params: dict[str, str],
        *,
        cache_key: str,
        ttl: int,
    ) -> dict[str, Any]:
        """Make a credentialed Xweather request with a credential-free cache key."""

        client_id, client_secret = self.credential.split_client_credentials()

        def fetch() -> dict[str, Any]:
            auth_params = params | {"client_id": client_id, "client_secret": client_secret}
            url = f"{self.base_url}/{endpoint}?{urllib.parse.urlencode(auth_params)}"
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": XWEATHER_USER_AGENT,
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode())
            except urllib.error.HTTPError as e:
                reason = self._http_error_reason(e, client_id, client_secret)
                raise XweatherError(f"Xweather {endpoint}: HTTP {e.code}: {reason}") from e
            except urllib.error.URLError as e:
                raise XweatherError(f"Xweather {endpoint}: network error") from e
            except TimeoutError as e:
                raise XweatherError(f"Xweather {endpoint}: request timed out") from e
            except json.JSONDecodeError as e:
                raise XweatherError(f"Xweather {endpoint}: invalid JSON response") from e

            if not isinstance(payload, dict):
                raise XweatherError(f"Xweather {endpoint}: malformed response")
            if payload.get("success") is False:
                error = payload.get("error")
                raise XweatherError(f"Xweather {endpoint}: {_provider_error_text(error)}")
            return payload

        return cast(dict[str, Any], cached_request(cache_key, fetch, ttl))

    def _http_error_reason(
        self,
        error: urllib.error.HTTPError,
        client_id: str,
        client_secret: str,
    ) -> str:
        """Return a redacted reason for an HTTP error."""
        try:
            raw_body = error.read().decode("utf-8", "replace")
            body = json.loads(raw_body)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return error.reason
        return _redact(_provider_error_text(body.get("error") or body), client_id, client_secret)

    def _parse_observations(
        self,
        payload: dict[str, Any],
        *,
        requested_latitude: float | None = None,
        requested_longitude: float | None = None,
    ) -> list[StationObservation]:
        """Parse an Xweather response payload."""
        response = payload.get("response")
        if isinstance(response, dict):
            records = [response]
        elif isinstance(response, list):
            records = response
        elif response is None:
            records = []
        else:
            raise XweatherError("Xweather observations: malformed response records")

        retrieved_at = utc_now()
        observations: list[StationObservation] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            observation = _parse_record(
                record,
                retrieved_at=retrieved_at,
                requested_latitude=requested_latitude,
                requested_longitude=requested_longitude,
            )
            if observation is not None:
                observations.append(observation)
        return observations


def _parse_record(
    record: dict[str, Any],
    *,
    retrieved_at: datetime,
    requested_latitude: float | None,
    requested_longitude: float | None,
) -> StationObservation | None:
    station_id = _string_or_none(record.get("id"))
    data_source = _string_or_none(record.get("dataSource"))
    loc = _dict_or_empty(record.get("loc"))
    place = _dict_or_empty(record.get("place"))
    profile = _dict_or_empty(record.get("profile"))
    relative_to = _dict_or_empty(record.get("relativeTo"))
    ob = _dict_or_empty(record.get("ob"))

    latitude = _float_or_none(loc.get("lat"))
    longitude = _float_or_none(loc.get("long"))
    if not station_id or latitude is None or longitude is None:
        return None

    distance_km = _float_or_none(relative_to.get("distanceKM"))
    if distance_km is None and requested_latitude is not None and requested_longitude is not None:
        distance_km = haversine_km(requested_latitude, requested_longitude, latitude, longitude)

    observed_at = parse_datetime(ob.get("dateTimeISO") or ob.get("timestamp"))
    receipt_at = parse_datetime(ob.get("recDateTimeISO") or ob.get("recTimestamp"))
    qc_status = _qc_status(ob)
    qc_flags = _qc_flags(ob)

    station = Station(
        provider="xweather",
        station_id=station_id,
        name=_station_name(station_id, place),
        latitude=latitude,
        longitude=longitude,
        elevation_m=_float_or_none(profile.get("elevM")),
        distance_km=distance_km,
        kind=_station_kind(data_source, station_id),
        network_id=data_source,
        network_name=_network_name(data_source),
    )

    measurements: dict[str, Measurement] = {}
    _add_measurement(
        measurements,
        "air_temperature",
        _float_or_none(ob.get("tempC")),
        "degC",
        observed_at,
        qc_status,
        qc_flags,
    )
    _add_measurement(
        measurements,
        "dew_point",
        _float_or_none(ob.get("dewpointC")),
        "degC",
        observed_at,
        qc_status,
        qc_flags,
    )
    _add_measurement(
        measurements,
        "relative_humidity",
        _float_or_none(ob.get("humidity")),
        "%",
        observed_at,
        qc_status,
        qc_flags,
    )
    wind_kph = _float_or_none(ob.get("windSpeedKPH"))
    if wind_kph is None:
        wind_kph = _float_or_none(ob.get("windKPH"))
    _add_measurement(
        measurements, "wind_speed", _kph_to_ms(wind_kph), "m/s", observed_at, qc_status, qc_flags
    )
    _add_measurement(
        measurements,
        "wind_gust",
        _kph_to_ms(_float_or_none(ob.get("windGustKPH"))),
        "m/s",
        observed_at,
        qc_status,
        qc_flags,
    )
    _add_measurement(
        measurements,
        "wind_direction",
        _float_or_none(ob.get("windDirDEG")),
        "degrees",
        observed_at,
        qc_status,
        qc_flags,
    )
    _add_measurement(
        measurements,
        "pressure_msl",
        _mb_to_pa(_float_or_none(ob.get("pressureMB"))),
        "Pa",
        observed_at,
        qc_status,
        qc_flags,
    )
    _add_measurement(
        measurements,
        "altimeter_setting",
        _mb_to_pa(_float_or_none(ob.get("altimeterMB"))),
        "Pa",
        observed_at,
        qc_status,
        qc_flags,
    )
    _add_measurement(
        measurements,
        "precipitation",
        _float_or_none(ob.get("precipMM")),
        "mm",
        observed_at,
        qc_status,
        qc_flags,
    )
    _add_measurement(
        measurements,
        "precipitation_rate",
        _float_or_none(ob.get("precipRateMM")),
        "mm/h",
        observed_at,
        qc_status,
        qc_flags,
    )
    solar = _float_or_none(ob.get("solradWM2"))
    if solar is None:
        solar = _float_or_none(ob.get("solarWM2"))
    _add_measurement(
        measurements, "solar_radiation", solar, "W/m^2", observed_at, qc_status, qc_flags
    )
    uv_index = _float_or_none(ob.get("uvi"))
    if uv_index is None:
        uv_index = _float_or_none(ob.get("uvIndex"))
    _add_measurement(measurements, "uv_index", uv_index, "index", observed_at, qc_status, qc_flags)

    return StationObservation(
        station=station,
        measurements=measurements,
        retrieved_at=retrieved_at,
        receipt_at=receipt_at,
        raw_qc={
            "QC": ob.get("QC"),
            "QCcode": ob.get("QCcode"),
            "trustFactor": ob.get("trustFactor"),
        },
    )


def _add_measurement(
    measurements: dict[str, Measurement],
    name: str,
    value: float | None,
    unit: str,
    observed_at: Any,
    qc: QcStatus,
    qc_flags: list[str],
) -> None:
    if value is None:
        return
    measurements[name] = Measurement(
        value_si=value,
        unit_si=unit,
        observed_at=observed_at,
        qc=qc,
        qc_flags=list(qc_flags),
    )


def _filter_for_kind(kind: StationKindFilter) -> str:
    return {"pws": "pws", "official": "metar", "all": "allstations"}[kind]


def _coordinate_id(latitude: float, longitude: float) -> str:
    return urllib.parse.quote(f"{latitude:.5f},{longitude:.5f}", safe=",")


def _station_kind(data_source: str | None, station_id: str) -> StationKind:
    source = (data_source or "").upper()
    if source == "PWS" or station_id.upper().startswith("PWS_"):
        return "personal"
    if "METAR" in source:
        return "official"
    if "RWIS" in source or "ROAD" in source:
        return "road"
    if "MADIS" in source or "MESONET" in source:
        return "research"
    return "unknown"


def _network_name(data_source: str | None) -> str | None:
    source = (data_source or "").upper()
    if source == "PWS":
        return "PWSweather"
    if source == "METAR_NOAA":
        return "NOAA METAR"
    if source.startswith("METAR"):
        return "METAR"
    if source.startswith("MADIS"):
        return "MADIS"
    return data_source


def _station_name(station_id: str, place: dict[str, Any]) -> str:
    name = _string_or_none(place.get("name")) or _string_or_none(place.get("city"))
    if name:
        return name
    return station_id


def _qc_status(ob: dict[str, Any]) -> QcStatus:
    qc_code = _float_or_none(ob.get("QCcode"))
    trust = _float_or_none(ob.get("trustFactor"))
    qc_text = _string_or_none(ob.get("QC"))

    if qc_code == 0 or (trust is not None and trust < 50):
        return "fail"
    if qc_code in {1, 3, 7} or (trust is not None and trust < 80):
        return "suspect"
    if qc_code == 10 or qc_text == "O" or (trust is not None and trust >= 80):
        return "pass"
    return "unknown"


def _qc_flags(ob: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for key in ("QC", "QCcode", "trustFactor"):
        value = ob.get(key)
        if value is not None:
            flags.append(f"{key}={value}")
    return flags


def _provider_error_text(error: object) -> str:
    if isinstance(error, dict):
        code = error.get("code")
        description = error.get("description") or error.get("message")
        if code and description:
            return f"{code}: {description}"
        if code:
            return str(code)
        if description:
            return str(description)
    if isinstance(error, str):
        return error
    return "provider error"


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_payload(nested)
            for key, nested in value.items()
            if key.lower() not in _AUTH_QUERY_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    return value


def _redact(text: str, client_id: str, client_secret: str) -> str:
    redacted = text
    for secret in (client_id, client_secret, f"{client_id}_{client_secret}"):
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    return redacted


def _dict_or_empty(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _kph_to_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 3.6


def _mb_to_pa(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100
