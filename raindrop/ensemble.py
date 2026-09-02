"""Analysis helpers for Open-Meteo ensemble forecasts."""

import math
import re
from collections import Counter
from statistics import fmean, median, pstdev
from typing import Any

_MEMBER_SUFFIX = re.compile(r"^(?P<variable>.+)_member(?P<member>\d+)$")

ENSEMBLE_MODELS: dict[str, tuple[str, str, int, str]] = {
    "icon_seamless_eps": ("DWD ICON EPS Seamless", "Best available ICON domain", 40, "7.5d"),
    "icon_global_eps": ("DWD ICON Global EPS", "Global", 40, "7.5d"),
    "icon_eu_eps": ("DWD ICON EU EPS", "Europe", 40, "5d"),
    "icon_d2_eps": ("DWD ICON D2 EPS", "Central Europe", 20, "2d"),
    "ncep_gefs_seamless": ("NOAA GFS Ensemble Seamless", "Global", 31, "35d"),
    "ncep_gefs025": ("NOAA GFS Ensemble 0.25°", "Global", 31, "10d"),
    "ncep_gefs05": ("NOAA GFS Ensemble 0.5°", "Global", 31, "35d"),
    "ncep_aigefs025": ("NOAA AI GEFS 0.25°", "Global", 31, "16d"),
    "ecmwf_ifs025_ensemble": ("ECMWF IFS 0.25° Ensemble", "Global", 51, "15d"),
    "ecmwf_ifs_europe_ensemble": ("ECMWF IFS 9 km Ensemble", "Europe", 51, "15d"),
    "ecmwf_aifs025_ensemble": ("ECMWF AIFS 0.25° Ensemble", "Global", 51, "15d"),
    "ecmwf_aifs_europe_ensemble": ("ECMWF AIFS 31 km Ensemble", "Europe", 51, "15d"),
    "gem_global_ensemble": ("Canadian GEM Global Ensemble", "Global", 21, "16d"),
    "bom_access_global_ensemble": ("BOM ACCESS Global Ensemble", "Global", 18, "10d"),
    "ukmo_global_ensemble_20km": ("UKMO Global Ensemble", "Global", 18, "8d"),
    "ukmo_uk_ensemble_2km": ("UKMO UK Ensemble", "United Kingdom", 3, "5d"),
    "meteoswiss_icon_ch1_ensemble": ("MeteoSwiss ICON CH1", "Central Europe", 11, "33h"),
    "meteoswiss_icon_ch2_ensemble": ("MeteoSwiss ICON CH2", "Central Europe", 21, "12h"),
    "google_weathernext2_ensemble": ("Google WeatherNext 2", "Global", 64, "15d"),
}


def member_series(dataset: dict[str, Any], variable: str) -> dict[int, list[float | None]]:
    """Return control/member series for a variable, keyed by member number."""
    members: dict[int, list[float | None]] = {}
    base = dataset.get(variable)
    if isinstance(base, list):
        members[0] = base

    for key, values in dataset.items():
        match = _MEMBER_SUFFIX.fullmatch(key)
        if match and match.group("variable") == variable and isinstance(values, list):
            members[int(match.group("member"))] = values

    if not members:
        raise ValueError(f"No ensemble data returned for variable '{variable}'")
    return dict(sorted(members.items()))


def values_at(members: dict[int, list[float | None]], index: int) -> dict[int, float]:
    """Return finite numeric member values at one forecast step."""
    values: dict[int, float] = {}
    for member, series in members.items():
        if index >= len(series):
            continue
        value = series[index]
        if isinstance(value, (int, float)) and math.isfinite(value):
            values[member] = float(value)
    return values


def variable_kind(variable: str, unit: str) -> str:
    """Classify values that need non-linear ensemble summaries."""
    if variable == "is_day" or "weather_code" in variable or "wmo code" in unit.lower():
        return "categorical"
    if "wind_direction" in variable:
        return "circular"
    return "continuous"


def percentile(values: list[float], percent: float) -> float:
    """Return a linearly interpolated percentile."""
    if not values:
        raise ValueError("Cannot calculate a percentile without values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution_summary(
    values: dict[int, float],
    *,
    variable: str,
    unit: str,
    percentiles: tuple[float, ...] = (10, 25, 50, 75, 90),
    threshold: float | None = None,
    operator: str = "above",
) -> dict[str, Any]:
    """Summarize one forecast step across all available members."""
    series = list(values.values())
    if not series:
        return {"kind": variable_kind(variable, unit), "count": 0}

    kind = variable_kind(variable, unit)
    summary: dict[str, Any] = {
        "kind": kind,
        "count": len(series),
        "control": values.get(0),
    }

    if kind == "categorical":
        counts = Counter(series)
        mode, mode_count = counts.most_common(1)[0]
        summary.update(
            {
                "mode": mode,
                "agreement": mode_count / len(series),
                "distinct": len(counts),
                "frequencies": {str(int(k) if k.is_integer() else k): v for k, v in counts.items()},
            }
        )
    elif kind == "circular":
        radians = [math.radians(value) for value in series]
        x = fmean(math.cos(value) for value in radians)
        y = fmean(math.sin(value) for value in radians)
        direction = math.degrees(math.atan2(y, x)) % 360
        summary.update(
            {
                "mean": 0.0 if math.isclose(direction, 360) else direction,
                "concentration": math.hypot(x, y),
            }
        )
    else:
        q1 = percentile(series, 25)
        q3 = percentile(series, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        deviation = pstdev(series)
        center = fmean(series)
        summary.update(
            {
                "mean": center,
                "median": median(series),
                "stdev": deviation,
                "minimum": min(series),
                "maximum": max(series),
                "spread": max(series) - min(series),
                "percentiles": {
                    str(p).removesuffix(".0"): percentile(series, p) for p in percentiles
                },
                "within_one_stdev": (
                    sum(abs(value - center) <= deviation for value in series) / len(series)
                    if deviation
                    else 1.0
                ),
                "outlier_members": [
                    member for member, value in values.items() if value < lower or value > upper
                ],
            }
        )

    if threshold is not None:
        matches = sum(value > threshold for value in series)
        if operator == "below":
            matches = sum(value < threshold for value in series)
        summary["threshold"] = threshold
        summary["operator"] = operator
        summary["threshold_probability"] = matches / len(series)

    return summary


def evaluate_member(
    values: dict[int, float],
    member: int,
    *,
    variable: str,
    unit: str,
) -> dict[str, Any] | None:
    """Evaluate one member against the same-step ensemble consensus."""
    if member not in values:
        return None

    value = values[member]
    kind = variable_kind(variable, unit)
    summary = distribution_summary(values, variable=variable, unit=unit)
    evaluation: dict[str, Any] = {
        "value": value,
        "control": values.get(0),
        "kind": kind,
    }

    if kind == "categorical":
        mode = summary["mode"]
        evaluation.update(
            {
                "consensus": mode,
                "matches_consensus": value == mode,
                "consensus_support": summary["agreement"],
            }
        )
    elif kind == "circular":
        center = summary["mean"]
        evaluation.update(
            {
                "consensus": center,
                "angular_difference": (value - center + 180) % 360 - 180,
                "directional_agreement": summary["concentration"],
            }
        )
    else:
        series = list(values.values())
        center = summary["median"]
        deviation = summary["stdev"]
        q1 = percentile(series, 25)
        q3 = percentile(series, 75)
        iqr = q3 - q1
        evaluation.update(
            {
                "consensus": center,
                "difference": value - center,
                "percentile_rank": (
                    sum(candidate < value for candidate in series)
                    + 0.5 * sum(candidate == value for candidate in series)
                )
                / len(series),
                "z_score": (value - summary["mean"]) / deviation if deviation else 0.0,
                "outlier": value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr,
            }
        )

    return evaluation


def rank_members(
    members: dict[int, list[float | None]],
    *,
    variable: str,
    unit: str,
    steps: int,
) -> list[dict[str, Any]]:
    """Rank members by departure from the ensemble consensus, not forecast skill."""
    kind = variable_kind(variable, unit)
    samples: dict[int, list[tuple[float, bool]]] = {member: [] for member in members}

    for index in range(steps):
        values = values_at(members, index)
        if not values:
            continue
        summary = distribution_summary(values, variable=variable, unit=unit)
        for member, value in values.items():
            evaluation = evaluate_member(values, member, variable=variable, unit=unit)
            if evaluation is None:
                continue
            if kind == "categorical":
                delta = 0.0 if evaluation["matches_consensus"] else 1.0
                outlier = bool(delta)
            elif kind == "circular":
                delta = float(evaluation["angular_difference"])
                outlier = abs(delta) > 90
            else:
                delta = value - float(summary["median"])
                outlier = bool(evaluation["outlier"])
            samples[member].append((delta, outlier))

    ranked = []
    for member, rows in samples.items():
        if not rows:
            continue
        deltas = [row[0] for row in rows]
        ranked.append(
            {
                "member": member,
                "samples": len(rows),
                "bias_from_consensus": fmean(deltas),
                "mean_absolute_departure": fmean(abs(delta) for delta in deltas),
                "rmse_from_consensus": math.sqrt(fmean(delta * delta for delta in deltas)),
                "outlier_rate": sum(row[1] for row in rows) / len(rows),
            }
        )

    return sorted(ranked, key=lambda row: (row["rmse_from_consensus"], row["member"]))
