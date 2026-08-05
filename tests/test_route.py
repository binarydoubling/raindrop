"""Tests for route helper functions."""

import pytest

from raindrop.commands.route import (
    get_weather_checkpoints,
    parse_maneuver,
    parse_road_name,
)


def test_parse_road_name_prefers_ref_and_name() -> None:
    assert parse_road_name({"ref": "I-5", "name": "Pacific Highway"}) == "I-5 (Pacific Highway)"
    assert parse_road_name({}) == "Local road"


def test_parse_maneuver_turns_modifier_into_instruction() -> None:
    step = {"maneuver": {"type": "turn", "modifier": "left"}}

    assert parse_maneuver(step) == "Turn left"


def test_get_weather_checkpoints_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError):
        get_weather_checkpoints([], total_distance_mi=10, interval_mi=0)


def test_get_weather_checkpoints_includes_final_destination() -> None:
    segments = [
        {
            "cumulative_distance_mi": 0,
            "distance_mi": 12,
            "cumulative_duration_s": 0,
            "duration_s": 600,
            "geometry": [[-122.0, 47.0], [-122.1, 47.1]],
            "start_coord": [-122.0, 47.0],
            "end_coord": [-122.1, 47.1],
            "road": "I-5",
        }
    ]

    checkpoints = get_weather_checkpoints(segments, total_distance_mi=12, interval_mi=10)

    assert [checkpoint["mile"] for checkpoint in checkpoints] == [0, 10, 12]
    assert checkpoints[-1]["lat"] == 47.1
    assert checkpoints[-1]["lon"] == -122.1
