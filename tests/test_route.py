"""Tests for route helper functions."""

import pytest

from raindrop.commands.route import (
    build_route_segments,
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


def test_build_route_segments_consolidates_continuations() -> None:
    route_data = {
        "legs": [
            {
                "steps": [
                    {
                        "name": "Pacific Highway",
                        "ref": "I-5",
                        "distance": 1000,
                        "duration": 60,
                        "maneuver": {"type": "depart"},
                        "geometry": {"coordinates": [[-122.0, 47.0], [-122.1, 47.1]]},
                    },
                    {
                        "name": "Pacific Highway",
                        "ref": "I-5",
                        "distance": 500,
                        "duration": 30,
                        "maneuver": {"type": "new name"},
                        "geometry": {"coordinates": [[-122.1, 47.1], [-122.2, 47.2]]},
                    },
                    {
                        "name": "Main Street",
                        "distance": 250,
                        "duration": 20,
                        "maneuver": {"type": "turn", "modifier": "right"},
                        "geometry": {"coordinates": [[-122.2, 47.2], [-122.3, 47.3]]},
                    },
                ]
            }
        ]
    }

    segments = build_route_segments(route_data)

    assert len(segments) == 2
    assert segments[0]["distance_m"] == 1500
    assert segments[1]["cumulative_distance_m"] == 1500
    assert segments[1]["maneuver"] == "Turn right"


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
