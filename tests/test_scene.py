"""Tests for weather scene generation."""

from raindrop.scene import SceneReading, build_scene_report, describe_distance, describe_motion


def _reading(**overrides) -> SceneReading:
    base = {
        "temperature": 68.0,
        "apparent_temperature": 70.0,
        "humidity": 55,
        "dew_point": 50.0,
        "weather_code": 2,
        "cloud_cover": 45,
        "wind_speed": 6.0,
        "wind_gusts": 8.0,
        "wind_direction": 180,
        "pressure_msl": 1013.0,
        "visibility": 10_000.0,
        "uv_index": 4.0,
        "precipitation": 0.0,
        "rain": 0.0,
        "showers": 0.0,
        "snowfall": 0.0,
        "is_day": True,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "mm",
        "precipitation_probability": 10,
        "low_cloud_cover": 20,
        "mid_cloud_cover": 40,
        "high_cloud_cover": 60,
    }
    base.update(overrides)
    return SceneReading(**base)


def test_scene_report_contains_sensory_sections() -> None:
    report = build_scene_report(_reading())

    assert report.summary
    assert set(report.sections) == {"Air", "Sky", "Light", "Motion", "Ground", "Distance"}
    assert "mild" in report.cues


def test_fog_collapses_distance() -> None:
    reading = _reading(weather_code=45, visibility=300.0, humidity=98)

    assert "collapse" in describe_distance(reading)


def test_gusts_are_described_as_separate_pushes() -> None:
    reading = _reading(wind_speed=10.0, wind_gusts=24.0, wind_direction=270)

    assert "Gusts" in describe_motion(reading)
    assert "W" in describe_motion(reading)


def test_rainy_scene_mentions_reflective_surfaces() -> None:
    report = build_scene_report(_reading(weather_code=61, precipitation=0.8, cloud_cover=95))

    assert "reflective" in report.summary or "reflective" in report.sections["Ground"]
