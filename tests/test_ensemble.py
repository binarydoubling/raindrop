"""Tests for ensemble forecast analysis."""

import pytest

from raindrop.ensemble import (
    distribution_summary,
    evaluate_member,
    member_series,
    percentile,
    rank_members,
    values_at,
)


def test_member_series_extracts_control_and_numbered_members() -> None:
    dataset = {
        "time": ["2026-09-02T00:00"],
        "temperature_2m": [10.0],
        "temperature_2m_member01": [11.0],
        "temperature_2m_member12": [12.0],
        "precipitation_member01": [2.0],
    }

    assert member_series(dataset, "temperature_2m") == {
        0: [10.0],
        1: [11.0],
        12: [12.0],
    }


def test_values_at_drops_missing_and_non_finite_values() -> None:
    members = {0: [1.0], 1: [None], 2: [float("nan")], 3: [4.0]}

    assert values_at(members, 0) == {0: 1.0, 3: 4.0}


def test_continuous_summary_includes_probability_and_outliers() -> None:
    values = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 100.0}

    summary = distribution_summary(
        values,
        variable="precipitation",
        unit="mm",
        threshold=10,
    )

    assert summary["median"] == 0
    assert summary["threshold_probability"] == pytest.approx(0.2)
    assert summary["outlier_members"] == [4]
    assert percentile(list(values.values()), 90) == pytest.approx(60)


def test_weather_codes_use_consensus_instead_of_numeric_mean() -> None:
    summary = distribution_summary(
        {0: 3.0, 1: 61.0, 2: 61.0},
        variable="weather_code",
        unit="wmo code",
    )

    assert summary["kind"] == "categorical"
    assert summary["mode"] == 61
    assert summary["agreement"] == pytest.approx(2 / 3)


def test_wind_direction_uses_circular_mean() -> None:
    summary = distribution_summary(
        {0: 350.0, 1: 10.0},
        variable="wind_direction_10m",
        unit="°",
    )

    assert summary["kind"] == "circular"
    assert summary["mean"] == pytest.approx(0)
    assert summary["concentration"] > 0.98


def test_member_evaluation_and_ranking_measure_consensus_departure() -> None:
    values = {0: 10.0, 1: 11.0, 2: 30.0}
    evaluation = evaluate_member(
        values,
        2,
        variable="temperature_2m",
        unit="°C",
    )
    ranking = rank_members(
        {0: [10.0, 10.0], 1: [11.0, 11.0], 2: [30.0, 30.0]},
        variable="temperature_2m",
        unit="°C",
        steps=2,
    )

    assert evaluation is not None
    assert evaluation["difference"] == 19
    assert ranking[-1]["member"] == 2
