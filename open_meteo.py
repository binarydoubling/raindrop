import urllib.request
import urllib.parse
import urllib.error
import json
from dataclasses import dataclass
from typing import Any, Literal


FORECAST_BASE_URL = "https://api.open-meteo.com/v1"
GEOCODING_BASE_URL = "https://geocoding-api.open-meteo.com/v1"
AIR_QUALITY_BASE_URL = "https://air-quality-api.open-meteo.com/v1"
HISTORICAL_BASE_URL = "https://archive-api.open-meteo.com/v1"
NWS_API_BASE_URL = "https://api.weather.gov"


class OpenMeteoError(Exception):
    """Raised when the Open-Meteo API returns an error or request fails."""

    pass


# =============================================================================
# Geocoding Types
# =============================================================================


@dataclass
class GeocodingResult:
    """A geocoding search result."""

    id: int
    name: str
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    feature_code: str
    country_code: str
    country: str
    country_id: int
    population: int
    postcodes: list[str]
    admin1: str | None
    admin2: str | None
    admin3: str | None
    admin4: str | None
    admin1_id: int | None
    admin2_id: int | None
    admin3_id: int | None
    admin4_id: int | None


# =============================================================================
# Weather Forecast Types
# =============================================================================

TemperatureUnit = Literal["celsius", "fahrenheit"]
WindSpeedUnit = Literal["kmh", "ms", "mph", "kn"]
PrecipitationUnit = Literal["mm", "inch"]
TimeFormat = Literal["iso8601", "unixtime"]
CellSelection = Literal["land", "sea", "nearest"]

# Current weather variables
CurrentVariable = Literal[
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "is_day",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "pressure_msl",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

# Hourly weather variables (subset of most common ones)
HourlyVariable = Literal[
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_direction_10m",
    "wind_direction_80m",
    "wind_direction_120m",
    "wind_direction_180m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "direct_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "global_tilted_irradiance",
    "vapour_pressure_deficit",
    "cape",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
    "precipitation",
    "snowfall",
    "precipitation_probability",
    "rain",
    "showers",
    "weather_code",
    "snow_depth",
    "freezing_level_height",
    "visibility",
    "soil_temperature_0cm",
    "soil_temperature_6cm",
    "soil_temperature_18cm",
    "soil_temperature_54cm",
    "soil_moisture_0_to_1cm",
    "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm",
    "soil_moisture_9_to_27cm",
    "soil_moisture_27_to_81cm",
    "is_day",
    "sunshine_duration",
    "uv_index",
    "uv_index_clear_sky",
]

# Daily weather variables
DailyVariable = Literal[
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "sunrise",
    "sunset",
    "daylight_duration",
    "sunshine_duration",
    "uv_index_max",
    "uv_index_clear_sky_max",
    "precipitation_sum",
    "rain_sum",
    "showers_sum",
    "snowfall_sum",
    "precipitation_hours",
    "precipitation_probability_max",
    "precipitation_probability_min",
    "precipitation_probability_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]


@dataclass
class CurrentWeather:
    """Current weather conditions. All requested fields are guaranteed present."""

    time: str
    interval: int
    temperature_2m: float | None = None
    relative_humidity_2m: int | None = None
    apparent_temperature: float | None = None
    dew_point_2m: float | None = None
    is_day: bool | None = None
    precipitation: float | None = None
    rain: float | None = None
    showers: float | None = None
    snowfall: float | None = None
    weather_code: int | None = None
    cloud_cover: int | None = None
    pressure_msl: float | None = None
    surface_pressure: float | None = None
    wind_speed_10m: float | None = None
    wind_direction_10m: int | None = None
    wind_gusts_10m: float | None = None
    visibility: float | None = None
    uv_index: float | None = None


@dataclass
class HourlyWeather:
    """Hourly weather forecast. All requested fields are guaranteed present."""

    time: list[str]
    temperature_2m: list[float] | None = None
    relative_humidity_2m: list[int] | None = None
    dew_point_2m: list[float] | None = None
    apparent_temperature: list[float] | None = None
    pressure_msl: list[float] | None = None
    surface_pressure: list[float] | None = None
    cloud_cover: list[int] | None = None
    cloud_cover_low: list[int] | None = None
    cloud_cover_mid: list[int] | None = None
    cloud_cover_high: list[int] | None = None
    wind_speed_10m: list[float] | None = None
    wind_speed_80m: list[float] | None = None
    wind_speed_120m: list[float] | None = None
    wind_speed_180m: list[float] | None = None
    wind_direction_10m: list[int] | None = None
    wind_direction_80m: list[int] | None = None
    wind_direction_120m: list[int] | None = None
    wind_direction_180m: list[int] | None = None
    wind_gusts_10m: list[float] | None = None
    shortwave_radiation: list[float] | None = None
    direct_radiation: list[float] | None = None
    direct_normal_irradiance: list[float] | None = None
    diffuse_radiation: list[float] | None = None
    global_tilted_irradiance: list[float] | None = None
    vapour_pressure_deficit: list[float] | None = None
    cape: list[float] | None = None
    evapotranspiration: list[float] | None = None
    et0_fao_evapotranspiration: list[float] | None = None
    precipitation: list[float] | None = None
    snowfall: list[float] | None = None
    precipitation_probability: list[int] | None = None
    rain: list[float] | None = None
    showers: list[float] | None = None
    weather_code: list[int] | None = None
    snow_depth: list[float] | None = None
    freezing_level_height: list[float] | None = None
    visibility: list[float] | None = None
    soil_temperature_0cm: list[float] | None = None
    soil_temperature_6cm: list[float] | None = None
    soil_temperature_18cm: list[float] | None = None
    soil_temperature_54cm: list[float] | None = None
    soil_moisture_0_to_1cm: list[float] | None = None
    soil_moisture_1_to_3cm: list[float] | None = None
    soil_moisture_3_to_9cm: list[float] | None = None
    soil_moisture_9_to_27cm: list[float] | None = None
    soil_moisture_27_to_81cm: list[float] | None = None
    is_day: list[int] | None = None
    sunshine_duration: list[float] | None = None
    uv_index: list[float] | None = None
    uv_index_clear_sky: list[float] | None = None


@dataclass
class DailyWeather:
    """Daily weather forecast. All requested fields are guaranteed present."""

    time: list[str]
    weather_code: list[int] | None = None
    temperature_2m_max: list[float] | None = None
    temperature_2m_min: list[float] | None = None
    temperature_2m_mean: list[float] | None = None
    apparent_temperature_max: list[float] | None = None
    apparent_temperature_min: list[float] | None = None
    apparent_temperature_mean: list[float] | None = None
    sunrise: list[str] | None = None
    sunset: list[str] | None = None
    daylight_duration: list[float] | None = None
    sunshine_duration: list[float] | None = None
    uv_index_max: list[float] | None = None
    uv_index_clear_sky_max: list[float] | None = None
    precipitation_sum: list[float] | None = None
    rain_sum: list[float] | None = None
    showers_sum: list[float] | None = None
    snowfall_sum: list[float] | None = None
    precipitation_hours: list[float] | None = None
    precipitation_probability_max: list[int] | None = None
    precipitation_probability_min: list[int] | None = None
    precipitation_probability_mean: list[int] | None = None
    wind_speed_10m_max: list[float] | None = None
    wind_gusts_10m_max: list[float] | None = None
    wind_direction_10m_dominant: list[int] | None = None
    shortwave_radiation_sum: list[float] | None = None
    et0_fao_evapotranspiration: list[float] | None = None


@dataclass
class ForecastResult:
    """Weather forecast response. All fields are guaranteed present."""

    latitude: float
    longitude: float
    elevation: float
    timezone: str
    timezone_abbreviation: str
    utc_offset_seconds: int
    current: CurrentWeather | None = None
    hourly: HourlyWeather | None = None
    daily: DailyWeather | None = None


# =============================================================================
# Air Quality Types
# =============================================================================


@dataclass
class CurrentAirQuality:
    """Current air quality conditions."""

    time: str
    interval: int
    us_aqi: int | None = None
    european_aqi: int | None = None
    pm10: float | None = None
    pm2_5: float | None = None
    carbon_monoxide: float | None = None
    nitrogen_dioxide: float | None = None
    sulphur_dioxide: float | None = None
    ozone: float | None = None
    dust: float | None = None
    uv_index: float | None = None
    uv_index_clear_sky: float | None = None
    ammonia: float | None = None
    alder_pollen: float | None = None
    birch_pollen: float | None = None
    grass_pollen: float | None = None
    mugwort_pollen: float | None = None
    olive_pollen: float | None = None
    ragweed_pollen: float | None = None


@dataclass
class HourlyAirQuality:
    """Hourly air quality forecast."""

    time: list[str]
    us_aqi: list[int] | None = None
    european_aqi: list[int] | None = None
    pm10: list[float] | None = None
    pm2_5: list[float] | None = None
    carbon_monoxide: list[float] | None = None
    nitrogen_dioxide: list[float] | None = None
    sulphur_dioxide: list[float] | None = None
    ozone: list[float] | None = None
    dust: list[float] | None = None
    uv_index: list[float] | None = None
    uv_index_clear_sky: list[float] | None = None
    ammonia: list[float] | None = None
    alder_pollen: list[float] | None = None
    birch_pollen: list[float] | None = None
    grass_pollen: list[float] | None = None
    mugwort_pollen: list[float] | None = None
    olive_pollen: list[float] | None = None
    ragweed_pollen: list[float] | None = None


@dataclass
class AirQualityResult:
    """Air quality response."""

    latitude: float
    longitude: float
    elevation: float
    timezone: str
    timezone_abbreviation: str
    utc_offset_seconds: int
    current: CurrentAirQuality | None = None
    hourly: HourlyAirQuality | None = None


# =============================================================================
# API Client
# =============================================================================


class OpenMeteo:
    """Client for the Open-Meteo weather API."""

    def __init__(
        self,
        forecast_base_url: str = FORECAST_BASE_URL,
        geocoding_base_url: str = GEOCODING_BASE_URL,
        air_quality_base_url: str = AIR_QUALITY_BASE_URL,
        historical_base_url: str = HISTORICAL_BASE_URL,
        timeout: int = 10,
    ):
        self.forecast_base_url = forecast_base_url
        self.geocoding_base_url = geocoding_base_url
        self.air_quality_base_url = air_quality_base_url
        self.historical_base_url = historical_base_url
        self.timeout = timeout

    def _request(self, url: str) -> dict[str, Any]:
        """Make HTTP request. Returns JSON data or raises OpenMeteoError."""
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode())
                reason = body.get("reason", str(e))
            except (json.JSONDecodeError, AttributeError):
                reason = f"HTTP {e.code}: {e.reason}"
            raise OpenMeteoError(reason)
        except urllib.error.URLError as e:
            raise OpenMeteoError(f"Network error: {e.reason}")
        except TimeoutError:
            raise OpenMeteoError("Request timed out")
        except json.JSONDecodeError as e:
            raise OpenMeteoError(f"Invalid JSON response: {e}")

    def _build_query(self, params: dict[str, Any]) -> str:
        """Build URL query string, filtering None values and formatting lists."""
        filtered: dict[str, str] = {}
        for k, v in params.items():
            if v is None:
                continue
            if isinstance(v, list):
                filtered[k] = ",".join(str(x) for x in v)
            elif isinstance(v, bool):
                filtered[k] = str(v).lower()
            else:
                filtered[k] = str(v)
        return urllib.parse.urlencode(filtered)

    def geocode(
        self,
        name: str,
        *,
        count: int = 10,
        language: str = "en",
        country_code: str | None = None,
    ) -> list[GeocodingResult]:
        """
        Search for locations by name.

        Args:
            name: Location name or postal code to search for
            count: Maximum number of results (1-100)
            language: Language for results (e.g., "en", "de", "fr")
            country_code: ISO 3166-1 alpha-2 country code to filter results (e.g., "US", "DE", "ES")

        Returns:
            List of matching locations

        Raises:
            OpenMeteoError: If request fails or no results found
        """
        query = self._build_query(
            {
                "name": name,
                "count": count,
                "language": language,
                "countryCode": country_code,
            }
        )
        url = f"{self.geocoding_base_url}/search?{query}"

        data = self._request(url)
        results = data.get("results")

        if not results:
            raise OpenMeteoError(f"No locations found for '{name}'")

        return [
            GeocodingResult(
                id=r["id"],
                name=r["name"],
                latitude=r["latitude"],
                longitude=r["longitude"],
                elevation=r.get("elevation", 0.0),
                timezone=r.get("timezone", "UTC"),
                feature_code=r.get("feature_code", ""),
                country_code=r.get("country_code", ""),
                country=r.get("country", ""),
                country_id=r.get("country_id", 0),
                population=r.get("population", 0),
                postcodes=r.get("postcodes", []),
                admin1=r.get("admin1"),
                admin2=r.get("admin2"),
                admin3=r.get("admin3"),
                admin4=r.get("admin4"),
                admin1_id=r.get("admin1_id"),
                admin2_id=r.get("admin2_id"),
                admin3_id=r.get("admin3_id"),
                admin4_id=r.get("admin4_id"),
            )
            for r in results
        ]

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
        """
        Get weather forecast for a location.

        Args:
            latitude: Latitude (-90 to 90)
            longitude: Longitude (-180 to 180)
            current: Current weather variables to include
            hourly: Hourly forecast variables to include
            daily: Daily forecast variables to include
            temperature_unit: Temperature unit ("celsius" or "fahrenheit")
            wind_speed_unit: Wind speed unit ("kmh", "ms", "mph", "kn")
            precipitation_unit: Precipitation unit ("mm" or "inch")
            timezone: Timezone for times (e.g., "America/New_York", "auto")
            forecast_days: Number of forecast days (1-16)
            past_days: Number of past days to include (0-92)
            start_date: Start date (YYYY-MM-DD) for custom range
            end_date: End date (YYYY-MM-DD) for custom range
            models: Specific weather models to use
            cell_selection: Grid cell selection method

        Returns:
            ForecastResult with requested weather data

        Raises:
            OpenMeteoError: If request fails
        """
        query = self._build_query(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": current,
                "hourly": hourly,
                "daily": daily,
                "temperature_unit": temperature_unit,
                "wind_speed_unit": wind_speed_unit,
                "precipitation_unit": precipitation_unit,
                "timezone": timezone,
                "forecast_days": forecast_days,
                "past_days": past_days,
                "start_date": start_date,
                "end_date": end_date,
                "models": models,
                "cell_selection": cell_selection,
            }
        )
        url = f"{self.forecast_base_url}/forecast?{query}"

        data = self._request(url)

        # Parse current weather
        current_weather = None
        if "current" in data:
            c = data["current"]
            current_weather = CurrentWeather(
                time=c["time"],
                interval=c["interval"],
                temperature_2m=c.get("temperature_2m"),
                relative_humidity_2m=c.get("relative_humidity_2m"),
                apparent_temperature=c.get("apparent_temperature"),
                dew_point_2m=c.get("dew_point_2m"),
                is_day=bool(c["is_day"]) if "is_day" in c else None,
                precipitation=c.get("precipitation"),
                rain=c.get("rain"),
                showers=c.get("showers"),
                snowfall=c.get("snowfall"),
                weather_code=c.get("weather_code"),
                cloud_cover=c.get("cloud_cover"),
                pressure_msl=c.get("pressure_msl"),
                surface_pressure=c.get("surface_pressure"),
                wind_speed_10m=c.get("wind_speed_10m"),
                wind_direction_10m=c.get("wind_direction_10m"),
                wind_gusts_10m=c.get("wind_gusts_10m"),
                visibility=c.get("visibility"),
                uv_index=c.get("uv_index"),
            )

        # Parse hourly forecast
        hourly_weather = None
        if "hourly" in data:
            h = data["hourly"]
            hourly_weather = HourlyWeather(
                time=h["time"],
                temperature_2m=h.get("temperature_2m"),
                relative_humidity_2m=h.get("relative_humidity_2m"),
                dew_point_2m=h.get("dew_point_2m"),
                apparent_temperature=h.get("apparent_temperature"),
                pressure_msl=h.get("pressure_msl"),
                surface_pressure=h.get("surface_pressure"),
                cloud_cover=h.get("cloud_cover"),
                cloud_cover_low=h.get("cloud_cover_low"),
                cloud_cover_mid=h.get("cloud_cover_mid"),
                cloud_cover_high=h.get("cloud_cover_high"),
                wind_speed_10m=h.get("wind_speed_10m"),
                wind_speed_80m=h.get("wind_speed_80m"),
                wind_speed_120m=h.get("wind_speed_120m"),
                wind_speed_180m=h.get("wind_speed_180m"),
                wind_direction_10m=h.get("wind_direction_10m"),
                wind_direction_80m=h.get("wind_direction_80m"),
                wind_direction_120m=h.get("wind_direction_120m"),
                wind_direction_180m=h.get("wind_direction_180m"),
                wind_gusts_10m=h.get("wind_gusts_10m"),
                shortwave_radiation=h.get("shortwave_radiation"),
                direct_radiation=h.get("direct_radiation"),
                direct_normal_irradiance=h.get("direct_normal_irradiance"),
                diffuse_radiation=h.get("diffuse_radiation"),
                global_tilted_irradiance=h.get("global_tilted_irradiance"),
                vapour_pressure_deficit=h.get("vapour_pressure_deficit"),
                cape=h.get("cape"),
                evapotranspiration=h.get("evapotranspiration"),
                et0_fao_evapotranspiration=h.get("et0_fao_evapotranspiration"),
                precipitation=h.get("precipitation"),
                snowfall=h.get("snowfall"),
                precipitation_probability=h.get("precipitation_probability"),
                rain=h.get("rain"),
                showers=h.get("showers"),
                weather_code=h.get("weather_code"),
                snow_depth=h.get("snow_depth"),
                freezing_level_height=h.get("freezing_level_height"),
                visibility=h.get("visibility"),
                soil_temperature_0cm=h.get("soil_temperature_0cm"),
                soil_temperature_6cm=h.get("soil_temperature_6cm"),
                soil_temperature_18cm=h.get("soil_temperature_18cm"),
                soil_temperature_54cm=h.get("soil_temperature_54cm"),
                soil_moisture_0_to_1cm=h.get("soil_moisture_0_to_1cm"),
                soil_moisture_1_to_3cm=h.get("soil_moisture_1_to_3cm"),
                soil_moisture_3_to_9cm=h.get("soil_moisture_3_to_9cm"),
                soil_moisture_9_to_27cm=h.get("soil_moisture_9_to_27cm"),
                soil_moisture_27_to_81cm=h.get("soil_moisture_27_to_81cm"),
                is_day=h.get("is_day"),
                sunshine_duration=h.get("sunshine_duration"),
                uv_index=h.get("uv_index"),
                uv_index_clear_sky=h.get("uv_index_clear_sky"),
            )

        # Parse daily forecast
        daily_weather = None
        if "daily" in data:
            d = data["daily"]
            daily_weather = DailyWeather(
                time=d["time"],
                weather_code=d.get("weather_code"),
                temperature_2m_max=d.get("temperature_2m_max"),
                temperature_2m_min=d.get("temperature_2m_min"),
                temperature_2m_mean=d.get("temperature_2m_mean"),
                apparent_temperature_max=d.get("apparent_temperature_max"),
                apparent_temperature_min=d.get("apparent_temperature_min"),
                apparent_temperature_mean=d.get("apparent_temperature_mean"),
                sunrise=d.get("sunrise"),
                sunset=d.get("sunset"),
                daylight_duration=d.get("daylight_duration"),
                sunshine_duration=d.get("sunshine_duration"),
                uv_index_max=d.get("uv_index_max"),
                uv_index_clear_sky_max=d.get("uv_index_clear_sky_max"),
                precipitation_sum=d.get("precipitation_sum"),
                rain_sum=d.get("rain_sum"),
                showers_sum=d.get("showers_sum"),
                snowfall_sum=d.get("snowfall_sum"),
                precipitation_hours=d.get("precipitation_hours"),
                precipitation_probability_max=d.get("precipitation_probability_max"),
                precipitation_probability_min=d.get("precipitation_probability_min"),
                precipitation_probability_mean=d.get("precipitation_probability_mean"),
                wind_speed_10m_max=d.get("wind_speed_10m_max"),
                wind_gusts_10m_max=d.get("wind_gusts_10m_max"),
                wind_direction_10m_dominant=d.get("wind_direction_10m_dominant"),
                shortwave_radiation_sum=d.get("shortwave_radiation_sum"),
                et0_fao_evapotranspiration=d.get("et0_fao_evapotranspiration"),
            )

        return ForecastResult(
            latitude=data["latitude"],
            longitude=data["longitude"],
            elevation=data["elevation"],
            timezone=data["timezone"],
            timezone_abbreviation=data["timezone_abbreviation"],
            utc_offset_seconds=data["utc_offset_seconds"],
            current=current_weather,
            hourly=hourly_weather,
            daily=daily_weather,
        )

    def air_quality(
        self,
        latitude: float,
        longitude: float,
        *,
        current: list[str] | None = None,
        hourly: list[str] | None = None,
        timezone: str = "auto",
        forecast_days: int = 5,
        past_days: int = 0,
        domains: str = "auto",
    ) -> AirQualityResult:
        """
        Get air quality forecast for a location.

        Args:
            latitude: Latitude (-90 to 90)
            longitude: Longitude (-180 to 180)
            current: Current air quality variables to include
            hourly: Hourly air quality variables to include
            timezone: Timezone for times (e.g., "America/New_York", "auto")
            forecast_days: Number of forecast days (1-7)
            past_days: Number of past days to include (0-92)
            domains: Domain selection ("auto", "cams_europe", "cams_global")

        Returns:
            AirQualityResult with requested air quality data

        Raises:
            OpenMeteoError: If request fails
        """
        query = self._build_query(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": current,
                "hourly": hourly,
                "timezone": timezone,
                "forecast_days": forecast_days,
                "past_days": past_days,
                "domains": domains,
            }
        )
        url = f"{self.air_quality_base_url}/air-quality?{query}"

        data = self._request(url)

        # Parse current air quality
        current_aq = None
        if "current" in data:
            c = data["current"]
            current_aq = CurrentAirQuality(
                time=c["time"],
                interval=c["interval"],
                us_aqi=c.get("us_aqi"),
                european_aqi=c.get("european_aqi"),
                pm10=c.get("pm10"),
                pm2_5=c.get("pm2_5"),
                carbon_monoxide=c.get("carbon_monoxide"),
                nitrogen_dioxide=c.get("nitrogen_dioxide"),
                sulphur_dioxide=c.get("sulphur_dioxide"),
                ozone=c.get("ozone"),
                dust=c.get("dust"),
                uv_index=c.get("uv_index"),
                uv_index_clear_sky=c.get("uv_index_clear_sky"),
                ammonia=c.get("ammonia"),
                alder_pollen=c.get("alder_pollen"),
                birch_pollen=c.get("birch_pollen"),
                grass_pollen=c.get("grass_pollen"),
                mugwort_pollen=c.get("mugwort_pollen"),
                olive_pollen=c.get("olive_pollen"),
                ragweed_pollen=c.get("ragweed_pollen"),
            )

        # Parse hourly air quality
        hourly_aq = None
        if "hourly" in data:
            h = data["hourly"]
            hourly_aq = HourlyAirQuality(
                time=h["time"],
                us_aqi=h.get("us_aqi"),
                european_aqi=h.get("european_aqi"),
                pm10=h.get("pm10"),
                pm2_5=h.get("pm2_5"),
                carbon_monoxide=h.get("carbon_monoxide"),
                nitrogen_dioxide=h.get("nitrogen_dioxide"),
                sulphur_dioxide=h.get("sulphur_dioxide"),
                ozone=h.get("ozone"),
                dust=h.get("dust"),
                uv_index=h.get("uv_index"),
                uv_index_clear_sky=h.get("uv_index_clear_sky"),
                ammonia=h.get("ammonia"),
                alder_pollen=h.get("alder_pollen"),
                birch_pollen=h.get("birch_pollen"),
                grass_pollen=h.get("grass_pollen"),
                mugwort_pollen=h.get("mugwort_pollen"),
                olive_pollen=h.get("olive_pollen"),
                ragweed_pollen=h.get("ragweed_pollen"),
            )

        return AirQualityResult(
            latitude=data["latitude"],
            longitude=data["longitude"],
            elevation=data.get("elevation", 0),
            timezone=data["timezone"],
            timezone_abbreviation=data["timezone_abbreviation"],
            utc_offset_seconds=data["utc_offset_seconds"],
            current=current_aq,
            hourly=hourly_aq,
        )

    def historical(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        *,
        daily: list[str] | None = None,
        hourly: list[str] | None = None,
        temperature_unit: TemperatureUnit = "celsius",
        wind_speed_unit: WindSpeedUnit = "kmh",
        precipitation_unit: PrecipitationUnit = "mm",
        timezone: str = "auto",
    ) -> dict[str, Any]:
        """
        Get historical weather data for a location.

        Args:
            latitude: Latitude (-90 to 90)
            longitude: Longitude (-180 to 180)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            daily: Daily weather variables to include
            hourly: Hourly weather variables to include
            temperature_unit: Temperature unit
            wind_speed_unit: Wind speed unit
            precipitation_unit: Precipitation unit
            timezone: Timezone for times

        Returns:
            Raw API response dict with daily/hourly data

        Raises:
            OpenMeteoError: If request fails
        """
        query = self._build_query(
            {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
                "daily": daily,
                "hourly": hourly,
                "temperature_unit": temperature_unit,
                "wind_speed_unit": wind_speed_unit,
                "precipitation_unit": precipitation_unit,
                "timezone": timezone,
            }
        )
        url = f"{self.historical_base_url}/archive?{query}"
        return self._request(url)


# =============================================================================
# NWS API Client (for forecast discussions)
# =============================================================================


@dataclass
class NWSOffice:
    """NWS Weather Forecast Office info."""

    id: str  # e.g., "MTR"
    name: str  # e.g., "San Francisco Bay Area"


@dataclass
class ForecastDiscussion:
    """NWS Area Forecast Discussion."""

    id: str
    office: str
    issuance_time: str
    product_text: str


@dataclass
class WeatherAlert:
    """NWS Weather Alert."""

    id: str
    event: str  # e.g., "Winter Storm Warning"
    severity: str  # Minor, Moderate, Severe, Extreme, Unknown
    certainty: str  # Observed, Likely, Possible, Unlikely, Unknown
    urgency: str  # Immediate, Expected, Future, Past, Unknown
    headline: str
    description: str
    instruction: str | None
    onset: str | None  # ISO datetime
    expires: str | None  # ISO datetime
    sender_name: str
    areas: list[str]  # Affected area names


class NWSClient:
    """Client for NWS API (forecast discussions, alerts, etc.)."""

    def __init__(
        self,
        base_url: str = NWS_API_BASE_URL,
        timeout: int = 10,
        user_agent: str = "raindrop-weather-cli (github.com/binarydoubling/raindrop)",
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.user_agent = user_agent

    def _request(self, url: str) -> dict[str, Any]:
        """Make HTTP request with proper User-Agent header."""
        req = urllib.request.Request(url)
        req.add_header("User-Agent", self.user_agent)
        req.add_header("Accept", "application/geo+json")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode())
                reason = body.get("detail", str(e))
            except (json.JSONDecodeError, AttributeError):
                reason = f"HTTP {e.code}: {e.reason}"
            raise OpenMeteoError(f"NWS API error: {reason}")
        except urllib.error.URLError as e:
            raise OpenMeteoError(f"Network error: {e.reason}")
        except TimeoutError:
            raise OpenMeteoError("Request timed out")
        except json.JSONDecodeError as e:
            raise OpenMeteoError(f"Invalid JSON response: {e}")

    def get_office_for_point(self, latitude: float, longitude: float) -> NWSOffice:
        """Get the NWS forecast office for a location."""
        url = f"{self.base_url}/points/{latitude},{longitude}"
        data = self._request(url)

        props = data.get("properties", {})
        office_url = props.get("forecastOffice", "")
        # Extract office ID from URL like "https://api.weather.gov/offices/MTR"
        office_id = office_url.split("/")[-1] if office_url else ""

        # Get office name
        office_name = (
            props.get("relativeLocation", {}).get("properties", {}).get("city", "")
        )
        if not office_name:
            office_name = f"NWS {office_id}"

        return NWSOffice(id=office_id, name=office_name)

    def get_latest_discussion(self, office_id: str) -> ForecastDiscussion:
        """Get the latest Area Forecast Discussion for an NWS office."""
        # Get list of AFD products for this office
        url = f"{self.base_url}/products/types/AFD/locations/{office_id}"
        data = self._request(url)

        products = data.get("@graph", [])
        if not products:
            raise OpenMeteoError(
                f"No forecast discussions found for office {office_id}"
            )

        # Get the latest one (first in list)
        latest = products[0]
        product_id = latest.get("id")

        # Fetch the full product
        product_url = f"{self.base_url}/products/{product_id}"
        product_data = self._request(product_url)

        return ForecastDiscussion(
            id=product_id,
            office=office_id,
            issuance_time=product_data.get("issuanceTime", ""),
            product_text=product_data.get("productText", ""),
        )

    def get_alerts(
        self, latitude: float, longitude: float, *, active_only: bool = True
    ) -> list[WeatherAlert]:
        """Get weather alerts for a location."""
        # Use point-based alerts endpoint
        url = f"{self.base_url}/alerts"
        params = [f"point={latitude},{longitude}"]
        if active_only:
            params.append("status=actual")
        url = f"{url}?{'&'.join(params)}"

        data = self._request(url)

        alerts = []
        features = data.get("features", [])
        for feature in features:
            props = feature.get("properties", {})
            alerts.append(
                WeatherAlert(
                    id=props.get("id", ""),
                    event=props.get("event", "Unknown"),
                    severity=props.get("severity", "Unknown"),
                    certainty=props.get("certainty", "Unknown"),
                    urgency=props.get("urgency", "Unknown"),
                    headline=props.get("headline", ""),
                    description=props.get("description", ""),
                    instruction=props.get("instruction"),
                    onset=props.get("onset"),
                    expires=props.get("expires"),
                    sender_name=props.get("senderName", ""),
                    areas=props.get("areaDesc", "").split("; "),
                )
            )

        return alerts

    def get_alerts_by_zone(self, zone_id: str) -> list[WeatherAlert]:
        """Get alerts for a specific NWS zone."""
        url = f"{self.base_url}/alerts/active/zone/{zone_id}"
        data = self._request(url)

        alerts = []
        features = data.get("features", [])
        for feature in features:
            props = feature.get("properties", {})
            alerts.append(
                WeatherAlert(
                    id=props.get("id", ""),
                    event=props.get("event", "Unknown"),
                    severity=props.get("severity", "Unknown"),
                    certainty=props.get("certainty", "Unknown"),
                    urgency=props.get("urgency", "Unknown"),
                    headline=props.get("headline", ""),
                    description=props.get("description", ""),
                    instruction=props.get("instruction"),
                    onset=props.get("onset"),
                    expires=props.get("expires"),
                    sender_name=props.get("senderName", ""),
                    areas=props.get("areaDesc", "").split("; "),
                )
            )

        return alerts
