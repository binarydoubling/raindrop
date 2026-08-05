"""Tests for file-backed response caching."""

from pathlib import Path

from raindrop.cache import Cache, cached_request, reset_cache


def test_cache_set_get_and_clear(tmp_path: Path) -> None:
    cache = Cache(cache_dir=tmp_path, default_ttl=60)

    cache.set("key", {"value": 1})

    assert cache.get("key") == {"value": 1}
    assert cache.clear() == 1
    assert cache.get("key") is None


def test_cache_expired_entry_returns_none(tmp_path: Path) -> None:
    cache = Cache(cache_dir=tmp_path, default_ttl=60)

    cache.set("key", "value", ttl=-1)

    assert cache.get("key") is None


def test_cached_request_uses_cached_value(tmp_path: Path) -> None:
    cache = Cache(cache_dir=tmp_path, default_ttl=60)
    reset_cache(cache)
    calls = 0

    def fetch() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"calls": calls}

    try:
        assert cached_request("request", fetch) == {"calls": 1}
        assert cached_request("request", fetch) == {"calls": 1}
        assert calls == 1
    finally:
        reset_cache(None)
