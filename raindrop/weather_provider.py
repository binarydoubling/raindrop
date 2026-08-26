"""Provider-neutral weather forecast selection and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, cast

from raindrop.open_meteo import (
    CellSelection,
    CurrentWeather,
    DailyWeather,
    ForecastResult,
    HourlyWeather,
    OpenMeteo,
    PrecipitationUnit,
    TemperatureUnit,
    WindSpeedUnit,
)
from raindrop.providers.xweather import (
    XweatherClient,
    XweatherError,
    get_xweather_credential_status,
)
from raindrop.settings import Settings, resolve_model

WeatherProviderName = Literal["auto", "open-meteo", "xweather"]
ResolvedWeatherProviderName = Literal["open-meteo", "xweather"]

XWEATHER_MAX_HOURLY_PERIODS = 168


class WeatherProvider(Protocol):
    """Protocol for forecast providers."""

    name: ResolvedWeatherProviderName
    label: str
    attribution: str | None

    def forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        current: list[str] | None = None,
        hourly: list[str] | None = None,
        daily: list[str] | None = None,
        temperature_unit: TemperatureUnit = "celsius",
        wind_speed_unit: WindSpeedUnit = "kmh",
        precipitation_unit: PrecipitationUnit = "mm",
        timezone: str = "auto",
        forecast_days: int = 7,
        past_days: int = 0,
        start_date: str | None = None,
        end_date: str | None = None,
        models: list[str] | None = None,
        cell_selection: CellSelection = "land",
    ) -> ForecastResult:
        """Return normalized forecast data."""
        ...


@dataclass(frozen=True)
class WeatherProviderSelection:
    """A selected provider plus provider-specific forecast options."""

    client: WeatherProvider
    model_key: str | None = None
    models: list[str] | None = None

    @property
    def name(self) -> ResolvedWeatherProviderName:
        """Return selected provider name."""
        return self.client.name

    @property
    def label(self) -> str:
        """Return selected provider display label."""
        return self.client.label

    @property
    def attribution(self) -> str | None:
        """Return selected provider attribution."""
        return self.client.attribution

    @property
    def model_label(self) -> str:
        """Return a display label for the forecast source/model."""
        if self.name == "open-meteo":
            return self.model_key or "auto"
        return self.label

    def forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        current: list[str] | None = None,
        hourly: list[str] | None = None,
        daily: list[str] | None = None,
        temperature_unit: TemperatureUnit = "celsius",
        wind_speed_unit: WindSpeedUnit = "kmh",
        precipitation_unit: PrecipitationUnit = "mm",
        timezone: str = "auto",
        forecast_days: int = 7,
        past_days: int = 0,
        start_date: str | None = None,
        end_date: str | None = None,
        cell_selection: CellSelection = "land",
    ) -> ForecastResult:
        """Fetch forecast data from the selected provider."""
        return self.client.forecast(
            latitude,
            longitude,
            current=current,
            hourly=hourly,
            daily=daily,
            temperature_unit=temperature_unit,
            wind_speed_unit=wind_speed_unit,
            precipitation_unit=precipitation_unit,
            timezone=timezone,
            forecast_days=forecast_days,
            past_days=past_days,
            start_date=start_date,
            end_date=end_date,
            models=self.models,
            cell_selection=cell_selection,
        )


class OpenMeteoWeatherProvider:
    """Open-Meteo forecast provider adapter."""

    name: ResolvedWeatherProviderName = "open-meteo"
    label: str = "Open-Meteo"
    attribution: str | None = None

    def __init__(self, client: OpenMeteo | None = None) -> None:
        self.client = client or OpenMeteo()

    def forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        current: list[str] | None = None,
        hourly: list[str] | None = None,
        daily: list[str] | None = None,
        temperature_unit: TemperatureUnit = "celsius",
        wind_speed_unit: WindSpeedUnit = "kmh",
        precipitation_unit: PrecipitationUnit = "mm",
        timezone: str = "auto",
        forecast_days: int = 7,
        past_days: int = 0,
        start_date: str | None = None,
        end_date: str | None = None,
        models: list[str] | None = None,
        cell_selection: CellSelection = "land",
    ) -> ForecastResult:
        """Fetch Open-Meteo forecast data."""
        result = self.client.forecast(
            latitude,
            longitude,
            current=current,
            hourly=hourly,
            daily=daily,
            temperature_unit=temperature_unit,
            wind_speed_unit=wind_speed_unit,
            precipitation_unit=precipitation_unit,
            timezone=timezone,
            forecast_days=forecast_days,
            past_days=past_days,
            start_date=start_date,
            end_date=end_date,
            models=models,
            cell_selection=cell_selection,
        )
        result.provider = self.name
        result.provider_label = self.label
        result.attribution = self.attribution
        return result


class XweatherWeatherProvider:
    """Xweather forecast provider adapter normalized to Raindrop fields."""

    name: ResolvedWeatherProviderName = "xweather"
    label: str = "Xweather"
    attribution: str | None = "Powered by Vaisala Xweather"

    def __init__(self, client: XweatherClient | None = None) -> None:
        self.client = client or XweatherClient()

    def forecast(
        self,
        latitude: float,
        longitude: float,
        *,
        current: list[str] | None = None,
        hourly: list[str] | None = None,
        daily: list[str] | None = None,
        temperature_unit: TemperatureUnit = "celsius",
        wind_speed_unit: WindSpeedUnit = "kmh",
        precipitation_unit: PrecipitationUnit = "mm",
        timezone: str = "auto",
        forecast_days: int = 7,
        past_days: int = 0,
        start_date: str | None = None,
        end_date: str | None = None,
        models: list[str] | None = None,
        cell_selection: CellSelection = "land",
    ) -> ForecastResult:
        """Fetch Xweather forecast data and normalize it to Raindrop fields."""
        if past_days or start_date or end_date:
            raise XweatherError("Xweather historical forecast mode is not implemented yet")
        if models:
            raise XweatherError("Open-Meteo model selection is not available with Xweather")

        current_weather = None
        hourly_weather = None
        daily_weather = None
        elevation = 0.0
        forecast_timezone = "UTC"
        timezone_abbreviation = "UTC"
        utc_offset_seconds = 0

        if current:
            payload = self.client.conditions_raw(latitude, longitude)
            record = _first_record(payload)
            period = _first_period(record)
            profile = _dict_or_empty(record.get("profile"))
            elevation = _float_or_default(profile.get("elevM"), elevation)
            forecast_timezone = _string_or_default(profile.get("tz"), forecast_timezone)
            timezone_abbreviation = _string_or_default(profile.get("tzname"), timezone_abbreviation)
            utc_offset_seconds = int(_float_or_default(profile.get("tzoffset"), utc_offset_seconds))
            current_weather = _period_to_current(
                period,
                requested=current,
                temperature_unit=temperature_unit,
                wind_speed_unit=wind_speed_unit,
                precipitation_unit=precipitation_unit,
            )

        if hourly:
            limit = min(max(forecast_days * 24, 1), XWEATHER_MAX_HOURLY_PERIODS)
            payload = self.client.forecast_raw(latitude, longitude, interval="1hr", limit=limit)
            record = _first_record(payload)
            periods = _periods(record)
            profile = _dict_or_empty(record.get("profile"))
            elevation = _float_or_default(profile.get("elevM"), elevation)
            forecast_timezone = _string_or_default(profile.get("tz"), forecast_timezone)
            timezone_abbreviation = _string_or_default(profile.get("tzname"), timezone_abbreviation)
            utc_offset_seconds = int(_float_or_default(profile.get("tzoffset"), utc_offset_seconds))
            hourly_weather = _periods_to_hourly(
                periods,
                requested=hourly,
                temperature_unit=temperature_unit,
                wind_speed_unit=wind_speed_unit,
                precipitation_unit=precipitation_unit,
            )

        if daily:
            payload = self.client.forecast_raw(
                latitude, longitude, interval="1day", limit=forecast_days
            )
            record = _first_record(payload)
            periods = _periods(record)
            profile = _dict_or_empty(record.get("profile"))
            elevation = _float_or_default(profile.get("elevM"), elevation)
            forecast_timezone = _string_or_default(profile.get("tz"), forecast_timezone)
            timezone_abbreviation = _string_or_default(profile.get("tzname"), timezone_abbreviation)
            utc_offset_seconds = int(_float_or_default(profile.get("tzoffset"), utc_offset_seconds))
            daily_weather = _periods_to_daily(
                periods,
                requested=daily,
                temperature_unit=temperature_unit,
                wind_speed_unit=wind_speed_unit,
                precipitation_unit=precipitation_unit,
            )

        return ForecastResult(
            latitude=latitude,
            longitude=longitude,
            elevation=elevation,
            timezone=forecast_timezone,
            timezone_abbreviation=timezone_abbreviation,
            utc_offset_seconds=utc_offset_seconds,
            current=current_weather,
            hourly=hourly_weather,
            daily=daily_weather,
            provider=self.name,
            provider_label=self.label,
            attribution=self.attribution,
        )


def select_weather_provider(
    settings: Settings, model_name: str | None = None
) -> WeatherProviderSelection:
    """Select the weather provider and normalize provider-specific model settings."""
    configured_provider = settings.weather_provider
    if configured_provider == "auto":
        if model_name or settings.model:
            provider_name: ResolvedWeatherProviderName = "open-meteo"
        elif get_xweather_credential_status().configured:
            provider_name = "xweather"
        else:
            provider_name = "open-meteo"
    else:
        provider_name = cast(ResolvedWeatherProviderName, configured_provider)

    if provider_name == "open-meteo":
        model_key, models = resolve_model(model_name, settings)
        return WeatherProviderSelection(
            client=OpenMeteoWeatherProvider(),
            model_key=model_key,
            models=models,
        )

    if model_name:
        raise ValueError("--model selects Open-Meteo models and cannot be used with Xweather")
    return WeatherProviderSelection(client=XweatherWeatherProvider())


def provider_source_payload(selection: WeatherProviderSelection) -> dict[str, str | None]:
    """Return a stable JSON source payload for weather commands."""
    return {
        "provider": selection.name,
        "label": selection.label,
        "model": selection.model_label,
        "attribution": selection.attribution,
    }


def _period_to_current(
    period: dict[str, Any],
    *,
    requested: list[str],
    temperature_unit: TemperatureUnit,
    wind_speed_unit: WindSpeedUnit,
    precipitation_unit: PrecipitationUnit,
) -> CurrentWeather:
    weather = CurrentWeather(
        time=_local_iso(period.get("dateTimeISO")),
        interval=0,
    )
    for field in requested:
        if field == "temperature_2m":
            weather.temperature_2m = _convert_temperature(
                _float_or_none(period.get("tempC")), temperature_unit
            )
        elif field == "apparent_temperature":
            weather.apparent_temperature = _convert_temperature(
                _float_or_none(period.get("feelslikeC")), temperature_unit
            )
        elif field == "relative_humidity_2m":
            weather.relative_humidity_2m = _int_or_none(period.get("humidity"))
        elif field == "dew_point_2m":
            weather.dew_point_2m = _convert_temperature(
                _float_or_none(period.get("dewpointC")), temperature_unit
            )
        elif field == "is_day":
            weather.is_day = _bool_or_none(period.get("isDay"))
        elif field in {"precipitation", "rain", "showers"}:
            setattr(
                weather,
                field,
                _convert_precipitation(_float_or_none(period.get("precipMM")), precipitation_unit),
            )
        elif field == "snowfall":
            weather.snowfall = _convert_precipitation(
                _cm_to_mm(_float_or_none(period.get("snowCM"))), precipitation_unit
            )
        elif field == "weather_code":
            weather.weather_code = _weather_code(period)
        elif field == "cloud_cover":
            weather.cloud_cover = _int_or_none(period.get("sky"))
        elif field == "pressure_msl":
            weather.pressure_msl = _float_or_none(period.get("pressureMB"))
        elif field == "surface_pressure":
            weather.surface_pressure = _float_or_none(period.get("spressureMB"))
        elif field == "wind_speed_10m":
            weather.wind_speed_10m = _convert_wind(
                _float_or_none(period.get("windSpeedKPH")), wind_speed_unit
            )
        elif field == "wind_direction_10m":
            weather.wind_direction_10m = _int_or_none(period.get("windDirDEG"))
        elif field == "wind_gusts_10m":
            weather.wind_gusts_10m = _convert_wind(
                _float_or_none(period.get("windGustKPH")), wind_speed_unit
            )
        elif field == "visibility":
            weather.visibility = _km_to_m(_float_or_none(period.get("visibilityKM")))
        elif field == "uv_index":
            weather.uv_index = _float_or_none(period.get("uvi"))
    return weather


def _periods_to_hourly(
    periods: list[dict[str, Any]],
    *,
    requested: list[str],
    temperature_unit: TemperatureUnit,
    wind_speed_unit: WindSpeedUnit,
    precipitation_unit: PrecipitationUnit,
) -> HourlyWeather:
    kwargs: dict[str, Any] = {"time": [_local_iso(period.get("dateTimeISO")) for period in periods]}
    for field in requested:
        if field == "temperature_2m":
            kwargs[field] = [
                _convert_temperature(_period_temperature(period), temperature_unit)
                for period in periods
            ]
        elif field == "apparent_temperature":
            kwargs[field] = [
                _convert_temperature(_float_or_none(period.get("feelslikeC")), temperature_unit)
                for period in periods
            ]
        elif field == "relative_humidity_2m":
            kwargs[field] = [_int_or_none(period.get("humidity")) for period in periods]
        elif field == "dew_point_2m":
            kwargs[field] = [
                _convert_temperature(_float_or_none(period.get("dewpointC")), temperature_unit)
                for period in periods
            ]
        elif field == "pressure_msl":
            kwargs[field] = [_float_or_none(period.get("pressureMB")) for period in periods]
        elif field == "surface_pressure":
            kwargs[field] = [None for _ in periods]
        elif field == "cloud_cover":
            kwargs[field] = [_int_or_none(period.get("sky")) for period in periods]
        elif field in {"cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"}:
            kwargs[field] = None
        elif field == "wind_speed_10m":
            kwargs[field] = [
                _convert_wind(_period_wind_speed(period), wind_speed_unit) for period in periods
            ]
        elif field == "wind_direction_10m":
            kwargs[field] = [_int_or_none(period.get("windDirDEG")) for period in periods]
        elif field == "wind_gusts_10m":
            kwargs[field] = [
                _convert_wind(_float_or_none(period.get("windGustKPH")), wind_speed_unit)
                for period in periods
            ]
        elif field == "shortwave_radiation":
            kwargs[field] = [_float_or_none(period.get("solradWM2")) for period in periods]
        elif field in {"precipitation", "rain", "showers"}:
            kwargs[field] = [
                _convert_precipitation(_float_or_none(period.get("precipMM")), precipitation_unit)
                for period in periods
            ]
        elif field == "snowfall":
            kwargs[field] = [
                _convert_precipitation(
                    _cm_to_mm(_float_or_none(period.get("snowCM"))), precipitation_unit
                )
                for period in periods
            ]
        elif field == "precipitation_probability":
            kwargs[field] = [_int_or_none(period.get("pop")) for period in periods]
        elif field == "weather_code":
            kwargs[field] = [_weather_code(period) for period in periods]
        elif field == "visibility":
            kwargs[field] = [
                _km_to_m(_float_or_none(period.get("visibilityKM"))) for period in periods
            ]
        elif field == "is_day":
            kwargs[field] = [_bool_to_int(period.get("isDay")) for period in periods]
        elif field == "uv_index":
            kwargs[field] = [_float_or_none(period.get("uvi")) for period in periods]
        else:
            kwargs[field] = None
    return HourlyWeather(**kwargs)


def _periods_to_daily(
    periods: list[dict[str, Any]],
    *,
    requested: list[str],
    temperature_unit: TemperatureUnit,
    wind_speed_unit: WindSpeedUnit,
    precipitation_unit: PrecipitationUnit,
) -> DailyWeather:
    kwargs: dict[str, Any] = {"time": [_date_part(period) for period in periods]}
    for field in requested:
        if field == "weather_code":
            kwargs[field] = [_weather_code(period) for period in periods]
        elif field == "temperature_2m_max":
            kwargs[field] = [
                _convert_temperature(_float_or_none(period.get("maxTempC")), temperature_unit)
                for period in periods
            ]
        elif field == "temperature_2m_min":
            kwargs[field] = [
                _convert_temperature(_float_or_none(period.get("minTempC")), temperature_unit)
                for period in periods
            ]
        elif field == "temperature_2m_mean":
            kwargs[field] = [
                _convert_temperature(_float_or_none(period.get("avgTempC")), temperature_unit)
                for period in periods
            ]
        elif field == "apparent_temperature_max":
            kwargs[field] = [
                _convert_temperature(_float_or_none(period.get("maxFeelslikeC")), temperature_unit)
                for period in periods
            ]
        elif field == "apparent_temperature_min":
            kwargs[field] = [
                _convert_temperature(_float_or_none(period.get("minFeelslikeC")), temperature_unit)
                for period in periods
            ]
        elif field == "apparent_temperature_mean":
            kwargs[field] = [
                _convert_temperature(_float_or_none(period.get("avgFeelslikeC")), temperature_unit)
                for period in periods
            ]
        elif field == "sunrise":
            kwargs[field] = [_local_iso(period.get("sunriseISO")) or None for period in periods]
        elif field == "sunset":
            kwargs[field] = [_local_iso(period.get("sunsetISO")) or None for period in periods]
        elif field in {"daylight_duration", "sunshine_duration"}:
            kwargs[field] = None
        elif field == "uv_index_max":
            kwargs[field] = [_float_or_none(period.get("uvi")) for period in periods]
        elif field == "uv_index_clear_sky_max":
            kwargs[field] = None
        elif field in {"precipitation_sum", "rain_sum", "showers_sum"}:
            kwargs[field] = [
                _convert_precipitation(_float_or_none(period.get("precipMM")), precipitation_unit)
                for period in periods
            ]
        elif field == "snowfall_sum":
            kwargs[field] = [
                _convert_precipitation(
                    _cm_to_mm(_float_or_none(period.get("snowCM"))), precipitation_unit
                )
                for period in periods
            ]
        elif field == "precipitation_hours":
            kwargs[field] = None
        elif field in {"precipitation_probability_max", "precipitation_probability_mean"}:
            kwargs[field] = [_int_or_none(period.get("pop")) for period in periods]
        elif field == "precipitation_probability_min":
            kwargs[field] = None
        elif field == "wind_speed_10m_max":
            kwargs[field] = [
                _convert_wind(_period_wind_max(period), wind_speed_unit) for period in periods
            ]
        elif field == "wind_gusts_10m_max":
            kwargs[field] = [
                _convert_wind(_float_or_none(period.get("windGustKPH")), wind_speed_unit)
                for period in periods
            ]
        elif field == "wind_direction_10m_dominant":
            kwargs[field] = [_int_or_none(period.get("windDirDEG")) for period in periods]
        elif field == "shortwave_radiation_sum":
            kwargs[field] = None
        elif field == "et0_fao_evapotranspiration":
            kwargs[field] = None
        else:
            kwargs[field] = None
    return DailyWeather(**kwargs)


def _first_record(payload: dict[str, Any]) -> dict[str, Any]:
    response = payload.get("response")
    if isinstance(response, list) and response and isinstance(response[0], dict):
        return response[0]
    if isinstance(response, dict):
        return response
    raise XweatherError("Xweather returned no forecast data")


def _first_period(record: dict[str, Any]) -> dict[str, Any]:
    periods = _periods(record)
    if not periods:
        ob = record.get("ob")
        if isinstance(ob, dict):
            return ob
        raise XweatherError("Xweather returned no current condition period")
    return periods[0]


def _periods(record: dict[str, Any]) -> list[dict[str, Any]]:
    value = record.get("periods")
    if not isinstance(value, list):
        return []
    return [period for period in value if isinstance(period, dict)]


def _date_part(period: dict[str, Any]) -> str:
    value = _local_iso(period.get("dateTimeISO"))
    return value[:10]


def _local_iso(value: object) -> str:
    """Return provider ISO datetimes as local naive strings for command compatibility."""
    raw = _string_or_none(value)
    if raw is None:
        return ""
    try:
        return datetime.fromisoformat(raw).replace(tzinfo=None).isoformat(timespec="minutes")
    except ValueError:
        return raw


def _period_temperature(period: dict[str, Any]) -> float | None:
    for key in ("tempC", "avgTempC", "maxTempC"):
        value = _float_or_none(period.get(key))
        if value is not None:
            return value
    return None


def _period_wind_speed(period: dict[str, Any]) -> float | None:
    for key in ("windSpeedKPH", "windSpeedMaxKPH"):
        value = _float_or_none(period.get(key))
        if value is not None:
            return value
    return None


def _period_wind_max(period: dict[str, Any]) -> float | None:
    for key in ("windSpeedMaxKPH", "windSpeedKPH"):
        value = _float_or_none(period.get(key))
        if value is not None:
            return value
    return None


def _weather_code(period: dict[str, Any]) -> int:
    coded = _string_or_none(period.get("weatherPrimaryCoded")) or _string_or_none(
        period.get("weatherCoded")
    )
    text = " ".join(
        value
        for value in (
            coded,
            _string_or_none(period.get("weatherPrimary")),
            _string_or_none(period.get("weather")),
        )
        if value
    ).lower()
    if any(term in text for term in ("thunder", ":t", "trw")):
        return 95
    if any(term in text for term in ("heavy snow", "blizzard")):
        return 75
    if "snow" in text:
        return 71
    if any(term in text for term in ("freezing rain", "ice", "sleet")):
        return 71
    if any(term in text for term in ("heavy rain", "downpour")):
        return 65
    if any(term in text for term in ("shower", "rain")):
        return 61
    if "drizzle" in text:
        return 51
    if any(term in text for term in ("fog", "mist", "haze")):
        return 45
    if any(term in text for term in ("overcast", "cloudy", "::ov")):
        return 3
    if any(term in text for term in ("partly", "scattered", "broken", "::sc", "::bk")):
        return 2
    if any(term in text for term in ("mostly clear", "mostly sunny", "fair")):
        return 1
    return 0


def _convert_temperature(value_c: float | None, unit: TemperatureUnit) -> float | None:
    if value_c is None:
        return None
    if unit == "fahrenheit":
        return round(value_c * 9 / 5 + 32, 1)
    return round(value_c, 1)


def _convert_wind(value_kph: float | None, unit: WindSpeedUnit) -> float | None:
    if value_kph is None:
        return None
    if unit == "mph":
        return round(value_kph * 0.621371, 3)
    if unit == "ms":
        return round(value_kph / 3.6, 3)
    if unit == "kn":
        return round(value_kph * 0.539957, 3)
    return round(value_kph, 3)


def _convert_precipitation(value_mm: float | None, unit: PrecipitationUnit) -> float | None:
    if value_mm is None:
        return None
    if unit == "inch":
        return round(value_mm / 25.4, 4)
    return round(value_mm, 3)


def _cm_to_mm(value_cm: float | None) -> float | None:
    if value_cm is None:
        return None
    return value_cm * 10


def _km_to_m(value_km: float | None) -> float | None:
    if value_km is None:
        return None
    return value_km * 1000


def _bool_to_int(value: object) -> int | None:
    parsed = _bool_or_none(value)
    if parsed is None:
        return None
    return 1 if parsed else 0


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return int(value)
    return None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _float_or_default(value: object, default: float) -> float:
    parsed = _float_or_none(value)
    return parsed if parsed is not None else default


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_or_default(value: object, default: str) -> str:
    parsed = _string_or_none(value)
    return parsed if parsed is not None else default


def _dict_or_empty(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
