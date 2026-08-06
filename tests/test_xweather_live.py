"""Opt-in live checks for Xweather station coverage."""

import os

import pytest

from raindrop.observations import filter_observations
from raindrop.providers.xweather import (
    XweatherClient,
    XweatherError,
    get_xweather_credential_status,
)


@pytest.mark.skipif(
    os.environ.get("RAINDROP_LIVE_XWEATHER") != "1",
    reason="set RAINDROP_LIVE_XWEATHER=1 to run live Xweather checks",
)
def test_live_xweather_fairbanks_pws_probe() -> None:
    """Check that the configured key can retrieve fresh Fairbanks PWS observations."""
    status = get_xweather_credential_status()
    if not status.configured:
        pytest.skip("Xweather credentials are not configured")

    client = XweatherClient()
    try:
        observations = client.nearby_observations(
            64.8378,
            -147.7164,
            radius_km=80.4672,
            limit=5,
            kind="pws",
        )
    except XweatherError as e:
        pytest.fail(f"live Xweather probe failed without credential details: {e}")

    fresh = filter_observations(observations, radius_km=80.4672, max_age_minutes=60)
    assert any(observation.station.kind == "personal" for observation in fresh)
