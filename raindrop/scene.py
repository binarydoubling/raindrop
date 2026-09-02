"""Translate weather variables into a first-person outdoor scene."""

from dataclasses import asdict, dataclass

from raindrop.open_meteo import PrecipitationUnit, TemperatureUnit, WindSpeedUnit
from raindrop.utils import WEATHER_CODES, deg_to_compass

RAIN_CODES = {51, 53, 55, 61, 63, 65, 80, 81, 82}
SNOW_CODES = {71, 73, 75}
FOG_CODES = {45, 48}
STORM_CODES = {95, 96, 99}


@dataclass(frozen=True)
class SceneReading:
    """Normalized inputs for a weather scene description."""

    temperature: float | None
    apparent_temperature: float | None
    humidity: int | None
    dew_point: float | None
    weather_code: int | None
    cloud_cover: int | None
    wind_speed: float | None
    wind_gusts: float | None
    wind_direction: int | None
    pressure_msl: float | None
    visibility: float | None
    uv_index: float | None
    precipitation: float | None
    rain: float | None
    showers: float | None
    snowfall: float | None
    is_day: bool | None
    temperature_unit: TemperatureUnit
    wind_speed_unit: WindSpeedUnit
    precipitation_unit: PrecipitationUnit
    precipitation_probability: int | None = None
    low_cloud_cover: int | None = None
    mid_cloud_cover: int | None = None
    high_cloud_cover: int | None = None


@dataclass(frozen=True)
class SceneReport:
    """A generated scene report."""

    summary: str
    sections: dict[str, str]
    cues: list[str]

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


def build_scene_report(reading: SceneReading) -> SceneReport:
    """Build a first-person scene description from weather variables."""
    sections = {
        "Air": describe_air(reading),
        "Sky": describe_sky(reading),
        "Light": describe_light(reading),
        "Motion": describe_motion(reading),
        "Ground": describe_ground(reading),
        "Distance": describe_distance(reading),
    }
    cues = scene_cues(reading)
    summary = describe_summary(reading, cues)
    return SceneReport(summary=summary, sections=sections, cues=cues)


def describe_air(reading: SceneReading) -> str:
    """Describe how the air would feel on skin and breath."""
    temp_f = _to_fahrenheit(
        reading.apparent_temperature or reading.temperature, reading.temperature_unit
    )
    actual_f = _to_fahrenheit(reading.temperature, reading.temperature_unit)
    dew_f = _to_fahrenheit(reading.dew_point, reading.temperature_unit)
    humidity = reading.humidity
    wind_mph = _wind_to_mph(reading.wind_speed, reading.wind_speed_unit)

    thermal = _thermal_phrase(temp_f)
    moisture = _moisture_phrase(humidity, dew_f)
    feels_offset = None if temp_f is None or actual_f is None else temp_f - actual_f

    parts = [thermal]
    if moisture:
        parts.append(moisture)

    if feels_offset is not None and abs(feels_offset) >= 4:
        if feels_offset < 0:
            parts.append(
                "wind or evaporative cooling would make it read colder than the thermometer"
            )
        else:
            parts.append("humidity and sun would make it feel warmer than the number suggests")
    elif humidity is not None and humidity >= 80 and wind_mph is not None and wind_mph < 4:
        parts.append("the air would sit close and slow, with little help from wind")
    elif humidity is not None and humidity <= 35:
        parts.append("the dryness would sharpen the edges of the air")

    return _sentence(parts)


def describe_sky(reading: SceneReading) -> str:
    """Describe sky structure and cloud texture."""
    code = reading.weather_code or 0
    cloud = reading.cloud_cover
    low = reading.low_cloud_cover
    high = reading.high_cloud_cover

    if code in FOG_CODES:
        return "The sky would be less a ceiling than a pale wall of fog, with cloud and ground blending together."
    if code in STORM_CODES:
        return "The sky would feel active and unstable, with darker convective cloud and a charged, unsettled look."
    if code in SNOW_CODES:
        return "Cloud would likely look thick and low, softening the sky into a bright gray backdrop for falling snow."
    if code in RAIN_CODES:
        if cloud is not None and cloud >= 85:
            return "A sealed gray overcast would press down overhead, with rain texture hanging through it."
        return "Cloud would be broken but wet-looking, with darker patches where showers are passing through."
    if cloud is None:
        return WEATHER_CODES.get(code, "The sky state is unclear from the returned data.")
    if cloud >= 90:
        if low is not None and low >= 70:
            return "The sky would read as a low lid of cloud — compressed, close, and evenly gray."
        if high is not None and high >= 70:
            return (
                "High cloud would veil the sky, making it bright but milky rather than openly blue."
            )
        return "The sky would be fully overcast, more continuous sheet than distinct cloud shapes."
    if cloud >= 60:
        return (
            "Cloud would dominate, but with enough breaks to give the sky some texture and depth."
        )
    if cloud >= 25:
        if high is not None and low is not None and high > low:
            return "A partial veil of higher cloud would soften the blue without fully closing the sky."
        return "Patches of cloud would move through an otherwise open sky, changing the feel moment to moment."
    return "The sky would be mostly open, with cloud playing only a minor role in the scene."


def describe_light(reading: SceneReading) -> str:
    """Describe light quality."""
    cloud = reading.cloud_cover
    uv = reading.uv_index
    code = reading.weather_code or 0

    if reading.is_day is False:
        if code in RAIN_CODES:
            return "At night, wet surfaces would catch streetlights and turn the scene reflective."
        if code in SNOW_CODES:
            return "At night, snow would lift the darkness and make the ground glow softly."
        return "The light would be artificial and pooled, with weather mostly visible around lamps and windows."

    if code in FOG_CODES:
        return (
            "Daylight would scatter in every direction, flattening shadows and bleaching distance."
        )
    if code in STORM_CODES:
        return "Light would be contrasty and uneven, switching between dull shadow and sudden bright breaks."
    if cloud is not None and cloud >= 85:
        return "The light would be flat and silvery, with weak shadows and muted color."
    if cloud is not None and cloud >= 45:
        return "Light would pulse as cloud passes — sometimes bright, sometimes briefly dimmed."
    if uv is not None and uv >= 7:
        return "Sunlight would feel hard and direct; exposed surfaces would look high-contrast."
    if uv is not None and uv >= 3:
        return (
            "The light would be clear enough for defined shadows without feeling especially harsh."
        )
    return "The light would be gentle, low-contrast, and easy on the eyes."


def describe_motion(reading: SceneReading) -> str:
    """Describe wind and movement in the scene."""
    wind = _wind_to_mph(reading.wind_speed, reading.wind_speed_unit)
    gust = _wind_to_mph(reading.wind_gusts, reading.wind_speed_unit)
    direction = (
        deg_to_compass(reading.wind_direction) if reading.wind_direction is not None else None
    )

    if wind is None:
        return "The movement of the air is hard to read from the returned data."

    direction_text = f" from the {direction}" if direction else ""
    if wind < 3:
        base = f"Air would feel nearly still{direction_text}; smoke, mist, or breath would linger close by."
    elif wind < 8:
        base = f"A light breeze{direction_text} would add just enough movement to leaves, hair, and loose fabric."
    elif wind < 16:
        base = f"A steady breeze{direction_text} would be part of the place, noticeable on exposed skin."
    elif wind < 26:
        base = f"Wind{direction_text} would tug at clothing and make corners or open ground feel more exposed."
    else:
        base = f"Strong wind{direction_text} would dominate the scene, pushing sound and weather sideways."

    if gust is not None and gust - wind >= 8:
        base += " Gusts would arrive as separate pushes rather than a smooth flow."
    return base


def describe_ground(reading: SceneReading) -> str:
    """Describe likely ground and surface conditions."""
    code = reading.weather_code or 0
    precip = _precip_total(reading)
    temp_f = _to_fahrenheit(reading.temperature, reading.temperature_unit)
    humidity = reading.humidity

    if code in SNOW_CODES or (reading.snowfall is not None and reading.snowfall > 0):
        if temp_f is not None and temp_f > 34:
            return "Snow would likely be wet at the edges, collecting as slush on warmer surfaces."
        return "Surfaces would look softened and muted, with snow dulling sharp edges and footfall sound."
    if code in RAIN_CODES or precip > 0:
        if temp_f is not None and temp_f <= 32:
            return "Any wet pavement could be slick or icy where water has a chance to freeze."
        return "Pavement and stone would likely look dark, damp, and slightly reflective."
    if code in FOG_CODES or (humidity is not None and humidity >= 92):
        return "Even without obvious rain, surfaces may feel cool and damp to the touch."
    if humidity is not None and humidity <= 30:
        return "Exposed ground and pavement would tend to look dry and matte rather than glossy."
    return "Ground conditions would likely be ordinary and dry unless sheltered surfaces are holding older moisture."


def describe_distance(reading: SceneReading) -> str:
    """Describe visibility, horizon, and depth."""
    visibility = reading.visibility
    code = reading.weather_code or 0
    humidity = reading.humidity

    if code in FOG_CODES:
        return "Distance would collapse quickly; the world would reveal itself in short layers."
    if visibility is None:
        if humidity is not None and humidity >= 85:
            return "The far view would probably be softened by moisture even if nearby objects stay clear."
        return "The horizon is hard to infer because visibility was not returned."

    if visibility < 500:
        return "Nearby objects would be clear, but the background would disappear into opaque air."
    if visibility < 2_000:
        return "The horizon would feel close, with buildings, trees, or hills fading after a short distance."
    if visibility < 8_000:
        return "Distance would be readable but softened, as if the air has a faint veil in it."
    if humidity is not None and humidity >= 85:
        return "Visibility is technically good, but humidity would still blur the farthest edges."
    return "The horizon would be crisp, with far-off objects holding their shape."


def describe_summary(reading: SceneReading, cues: list[str]) -> str:
    """Build a concise overall impression."""
    code = reading.weather_code or 0
    temp_f = _to_fahrenheit(
        reading.apparent_temperature or reading.temperature, reading.temperature_unit
    )
    wind = _wind_to_mph(reading.wind_speed, reading.wind_speed_unit)
    humidity = reading.humidity

    weather = WEATHER_CODES.get(code, "unknown weather")
    thermal = _thermal_label(temp_f)

    if code in FOG_CODES:
        return "You would be standing inside a softened, close-range world: muted, damp, and visually compressed."
    if code in STORM_CODES:
        return "The place would feel charged and unsettled, with the sky, wind, and light all in motion."
    if code in SNOW_CODES:
        return "The scene would feel muffled and pale, with snow changing both sound and texture."
    if code in RAIN_CODES:
        if wind is not None and wind >= 15:
            return "It would feel wet and exposed — rain plus wind turning the weather directional."
        return "It would feel wet, enclosed, and reflective, with the weather written clearly on surfaces."
    if humidity is not None and humidity >= 80 and wind is not None and wind < 5:
        return f"A {thermal}, humid stillness would define the scene more than the basic code of {weather.lower()}."
    if cues:
        return f"A {thermal} scene: {', '.join(cues[:3])}."
    return f"A {thermal} outdoor scene under {weather.lower()}."


def scene_cues(reading: SceneReading) -> list[str]:
    """Return concise diagnostic cues behind the prose."""
    cues: list[str] = []
    temp_f = _to_fahrenheit(
        reading.apparent_temperature or reading.temperature, reading.temperature_unit
    )
    wind = _wind_to_mph(reading.wind_speed, reading.wind_speed_unit)
    gust = _wind_to_mph(reading.wind_gusts, reading.wind_speed_unit)
    dew_f = _to_fahrenheit(reading.dew_point, reading.temperature_unit)
    code = reading.weather_code or 0

    cues.append(_thermal_label(temp_f))
    if reading.humidity is not None:
        if reading.humidity >= 85:
            cues.append("moist air")
        elif reading.humidity <= 35:
            cues.append("dry air")
    if dew_f is not None and dew_f >= 65:
        cues.append("high dew point")
    if wind is not None:
        if wind < 3:
            cues.append("near-still wind")
        elif wind >= 20:
            cues.append("wind-dominated")
    if gust is not None and wind is not None and gust - wind >= 8:
        cues.append("distinct gusts")
    if reading.cloud_cover is not None and reading.cloud_cover >= 85:
        cues.append("sealed cloud cover")
    if code in RAIN_CODES:
        cues.append("wet surfaces")
    elif code in SNOW_CODES:
        cues.append("muffled snow")
    elif code in FOG_CODES:
        cues.append("compressed visibility")
    if reading.visibility is not None and reading.visibility < 2_000:
        cues.append("short horizon")
    if reading.precipitation_probability is not None and reading.precipitation_probability >= 60:
        cues.append("precipitation likely nearby")
    return _unique(cues)


def _sentence(parts: list[str]) -> str:
    clean = [part for part in parts if part]
    if not clean:
        return "The feel of the air is hard to infer from the returned data."
    sentence = "; ".join(clean)
    return sentence[0].upper() + sentence[1:] + "."


def _thermal_label(temp_f: float | None) -> str:
    if temp_f is None:
        return "unknown-temperature"
    if temp_f < 15:
        return "arctic"
    if temp_f < 32:
        return "freezing"
    if temp_f < 45:
        return "cold"
    if temp_f < 58:
        return "cool"
    if temp_f < 72:
        return "mild"
    if temp_f < 84:
        return "warm"
    if temp_f < 95:
        return "hot"
    return "oppressive heat"


def _thermal_phrase(temp_f: float | None) -> str:
    label = _thermal_label(temp_f)
    return {
        "arctic": "the air would bite immediately, the kind of cold that narrows attention",
        "freezing": "the air would feel freezing and hard against exposed skin",
        "cold": "the air would feel cold and bracing",
        "cool": "the air would feel cool, noticeable but not harsh",
        "mild": "the air would feel mild and easy to stand in",
        "warm": "the air would feel warm on skin",
        "hot": "the air would feel hot, with heat radiating back from exposed surfaces",
        "oppressive heat": "the air would feel oppressive, with heat becoming the main fact of being outside",
        "unknown-temperature": "the temperature feel is hard to infer",
    }[label]


def _moisture_phrase(humidity: int | None, dew_f: float | None) -> str:
    if dew_f is not None:
        if dew_f >= 72:
            return "the dew point suggests a tropical, sweat-sticking heaviness"
        if dew_f >= 65:
            return "the moisture would feel present and muggy rather than just humid on paper"
        if dew_f <= 25:
            return "the low dew point would make the air feel crisp and dry"
    if humidity is None:
        return ""
    if humidity >= 90:
        return "moisture would be obvious, softening breath and surfaces"
    if humidity >= 75:
        return "humidity would give the air a damp, close texture"
    if humidity <= 30:
        return "humidity is low enough for a dry, clean edge"
    return ""


def _to_fahrenheit(value: float | None, unit: TemperatureUnit) -> float | None:
    if value is None:
        return None
    if unit == "celsius":
        return value * 9 / 5 + 32
    return value


def _wind_to_mph(value: float | None, unit: WindSpeedUnit) -> float | None:
    if value is None:
        return None
    if unit == "kmh":
        return value * 0.621371
    if unit == "ms":
        return value * 2.23694
    if unit == "kn":
        return value * 1.15078
    return value


def _precip_total(reading: SceneReading) -> float:
    values = [reading.precipitation, reading.rain, reading.showers, reading.snowfall]
    return sum(value for value in values if value is not None)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
