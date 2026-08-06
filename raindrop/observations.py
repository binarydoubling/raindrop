"""Provider-neutral station observation models and helpers."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import Any, Literal

StationKind = Literal["personal", "official", "road", "research", "unknown"]
QcStatus = Literal["pass", "suspect", "fail", "unknown"]
StationSort = Literal["best", "distance", "freshness"]

CORE_MEASUREMENTS = {"air_temperature", "dew_point", "relative_humidity", "wind_speed"}


@dataclass(frozen=True)
class Measurement:
    """A single normalized station measurement."""

    value_si: float | None
    unit_si: str | None
    observed_at: datetime | None
    qc: QcStatus = "unknown"
    qc_flags: list[str] | None = None


@dataclass(frozen=True)
class Station:
    """A reporting station."""

    provider: str
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float | None
    distance_km: float | None
    kind: StationKind
    network_id: str | None
    network_name: str | None


@dataclass(frozen=True)
class StationObservation:
    """A normalized observation from a station."""

    station: Station
    measurements: dict[str, Measurement]
    retrieved_at: datetime
    receipt_at: datetime | None
    raw_qc: Any | None = None

    def freshest_observed_at(self, names: set[str] | None = None) -> datetime | None:
        """Return the newest timestamp among selected measurements."""
        timestamps: list[datetime] = []
        for name, measurement in self.measurements.items():
            if names is not None and name not in names:
                continue
            if measurement.observed_at is not None:
                timestamps.append(measurement.observed_at)
        return max(timestamps) if timestamps else None

    def age(self, now: datetime | None = None, names: set[str] | None = None) -> timedelta | None:
        """Return age of the newest selected measurement."""
        observed_at = self.freshest_observed_at(names)
        if observed_at is None:
            return None
        if now is None:
            now = datetime.now(UTC)
        return now - observed_at

    def worst_qc(self) -> QcStatus:
        """Return a conservative aggregate QC status."""
        statuses = [measurement.qc for measurement in self.measurements.values()]
        if "fail" in statuses:
            return "fail"
        if "suspect" in statuses:
            return "suspect"
        if "pass" in statuses:
            return "pass"
        return "unknown"

    def core_completeness(
        self, now: datetime | None = None, max_age_minutes: int | None = None
    ) -> int:
        """Count fresh core measurements."""
        count = 0
        for name in CORE_MEASUREMENTS:
            measurement = self.measurements.get(name)
            if measurement is None or measurement.value_si is None:
                continue
            if max_age_minutes is not None and measurement.observed_at is not None:
                if now is None:
                    now = datetime.now(UTC)
                if now - measurement.observed_at > timedelta(minutes=max_age_minutes):
                    continue
            count += 1
        return count


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(UTC)


def parse_datetime(value: object) -> datetime | None:
    """Parse a provider timestamp into an aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)
    if not isinstance(value, str):
        return None
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance between two coordinates in kilometers."""
    radius_km = 6371.0088
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)

    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    return 2 * radius_km * asin(sqrt(a))


def has_fresh_core_measurement(
    observation: StationObservation,
    *,
    max_age_minutes: int,
    now: datetime | None = None,
    future_tolerance_minutes: int = 5,
) -> bool:
    """Return True when any core measurement is fresh and not materially future-dated."""
    if now is None:
        now = utc_now()
    max_age = timedelta(minutes=max_age_minutes)
    future_tolerance = timedelta(minutes=future_tolerance_minutes)

    for name in CORE_MEASUREMENTS:
        measurement = observation.measurements.get(name)
        if measurement is None or measurement.value_si is None or measurement.observed_at is None:
            continue
        age = now - measurement.observed_at
        if age < -future_tolerance:
            continue
        if age <= max_age:
            return True
    return False


def is_future_dated(
    observation: StationObservation,
    *,
    now: datetime | None = None,
    future_tolerance_minutes: int = 5,
) -> bool:
    """Return True if every timestamped core measurement is materially in the future."""
    if now is None:
        now = utc_now()
    future_tolerance = timedelta(minutes=future_tolerance_minutes)
    timestamps = [
        measurement.observed_at
        for name, measurement in observation.measurements.items()
        if name in CORE_MEASUREMENTS and measurement.observed_at is not None
    ]
    return bool(timestamps) and all(ts - now > future_tolerance for ts in timestamps)


def filter_observations(
    observations: list[StationObservation],
    *,
    radius_km: float | None = None,
    max_age_minutes: int = 60,
    include_stale: bool = False,
    now: datetime | None = None,
) -> list[StationObservation]:
    """Apply radius, freshness, future-skew and failed-QC eligibility filters."""
    if now is None:
        now = utc_now()

    filtered: list[StationObservation] = []
    for observation in observations:
        distance = observation.station.distance_km
        if radius_km is not None and distance is not None and distance > radius_km:
            continue
        if is_future_dated(observation, now=now):
            continue
        if observation.worst_qc() == "fail" and not include_stale:
            continue
        if not include_stale and not has_fresh_core_measurement(
            observation,
            max_age_minutes=max_age_minutes,
            now=now,
        ):
            continue
        filtered.append(observation)
    return filtered


def sort_observations(
    observations: list[StationObservation],
    *,
    sort: StationSort = "best",
    max_age_minutes: int = 60,
    now: datetime | None = None,
) -> list[StationObservation]:
    """Sort station observations according to Raindrop's ranking rules."""
    if now is None:
        now = utc_now()

    if sort == "distance":
        return sorted(
            observations,
            key=lambda ob: ob.station.distance_km if ob.station.distance_km is not None else 1e9,
        )

    if sort == "freshness":
        return sorted(observations, key=lambda ob: _age_minutes_for_sort(ob, now))

    return sorted(
        observations,
        key=lambda ob: (
            _age_minutes_for_sort(ob, now),
            ob.station.distance_km if ob.station.distance_km is not None else 1e9,
            -ob.core_completeness(now=now, max_age_minutes=max_age_minutes),
            _qc_rank(ob.worst_qc()),
            ob.station.elevation_m if ob.station.elevation_m is not None else 1e9,
        ),
    )


def measurement_to_dict(measurement: Measurement) -> dict[str, Any]:
    """Return a JSON-serializable measurement payload."""
    return {
        "value_si": measurement.value_si,
        "unit_si": measurement.unit_si,
        "observed_at": measurement.observed_at.isoformat() if measurement.observed_at else None,
        "qc": measurement.qc,
        "qc_flags": measurement.qc_flags or [],
    }


def observation_to_dict(observation: StationObservation) -> dict[str, Any]:
    """Return a JSON-serializable observation payload."""
    station = observation.station
    return {
        "station": {
            "provider": station.provider,
            "station_id": station.station_id,
            "name": station.name,
            "latitude": station.latitude,
            "longitude": station.longitude,
            "elevation_m": station.elevation_m,
            "distance_km": station.distance_km,
            "kind": station.kind,
            "network_id": station.network_id,
            "network_name": station.network_name,
        },
        "measurements": {
            name: measurement_to_dict(measurement)
            for name, measurement in sorted(observation.measurements.items())
        },
        "retrieved_at": observation.retrieved_at.isoformat(),
        "receipt_at": observation.receipt_at.isoformat() if observation.receipt_at else None,
        "raw_qc": observation.raw_qc,
    }


def _age_minutes_for_sort(observation: StationObservation, now: datetime) -> float:
    age = observation.age(now=now, names=CORE_MEASUREMENTS)
    if age is None:
        return 1e9
    return max(age.total_seconds() / 60, 0)


def _qc_rank(status: QcStatus) -> int:
    return {"pass": 0, "unknown": 1, "suspect": 2, "fail": 3}[status]
