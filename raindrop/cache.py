"""Simple file-based caching for API responses."""

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _default_cache_dir() -> Path:
    """Return the platform-friendly cache directory for raindrop."""
    override = os.environ.get("RAINDROP_CACHE_DIR")
    if override:
        return Path(override).expanduser()

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "raindrop"

    return Path.home() / ".cache" / "raindrop"


# Default cache directory - use cache directory for better compatibility
CACHE_DIR = _default_cache_dir()

# Default TTL (time to live) in seconds
DEFAULT_TTL = 300  # 5 minutes


class Cache:
    """Simple file-based cache for API responses."""

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        default_ttl: int = DEFAULT_TTL,
        enabled: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.enabled = enabled and not _no_cache_requested()

        if self.enabled:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                # Disable caching if we can't create the directory
                self.enabled = False

    def _get_cache_key(self, key: str) -> str:
        """Generate a hash-based filename for the cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _get_cache_path(self, key: str) -> Path:
        """Get the full path for a cache entry."""
        return self.cache_dir / f"{self._get_cache_key(key)}.json"

    def get(self, key: str) -> Any | None:
        """
        Get a value from the cache.

        Returns None if the key doesn't exist or has expired.
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with cache_path.open("r") as f:
                entry = json.load(f)

            # Check if expired
            if time.time() > entry.get("expires", 0):
                cache_path.unlink(missing_ok=True)
                return None

            return entry.get("data")
        except (json.JSONDecodeError, OSError, AttributeError):
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to store (must be JSON-serializable)
            ttl: Time to live in seconds (uses default if not specified)
        """
        if not self.enabled:
            return

        if ttl is None:
            ttl = self.default_ttl

        cache_path = self._get_cache_path(key)
        entry = {
            "key": key,
            "data": value,
            "created": time.time(),
            "expires": time.time() + ttl,
        }

        try:
            tmp_path = cache_path.with_suffix(".json.tmp")
            with tmp_path.open("w") as f:
                json.dump(entry, f)
            tmp_path.replace(cache_path)
        except (TypeError, OSError):
            pass  # Silently fail on write errors

    def delete(self, key: str) -> bool:
        """Delete a cache entry. Returns True if deleted."""
        if not self.enabled:
            return False

        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear(self) -> int:
        """Clear all cache entries. Returns number of entries cleared."""
        if not self.enabled or not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass
        return count

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self.enabled or not self.cache_dir.exists():
            return {"enabled": self.enabled, "entries": 0, "size_bytes": 0}

        entries = list(self.cache_dir.glob("*.json"))
        total_size = 0
        for entry in entries:
            try:
                total_size += entry.stat().st_size
            except OSError:
                pass

        # Count valid vs expired
        now = time.time()
        valid = 0
        expired = 0

        for entry_path in entries:
            try:
                with entry_path.open("r") as f:
                    entry = json.load(f)
                if entry.get("expires", 0) > now:
                    valid += 1
                else:
                    expired += 1
            except (json.JSONDecodeError, OSError, AttributeError):
                expired += 1

        return {
            "enabled": self.enabled,
            "entries": len(entries),
            "valid": valid,
            "expired": expired,
            "size_bytes": total_size,
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance
_cache: Cache | None = None


def _no_cache_requested() -> bool:
    """Return True when caching is disabled for this process."""
    return os.environ.get("RAINDROP_NO_CACHE", "").lower() in {"1", "true", "yes"}


def reset_cache(cache: Cache | None = None) -> None:
    """Reset the global cache instance, optionally replacing it."""
    global _cache
    _cache = cache


def get_cache() -> Cache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


def set_cache_enabled(enabled: bool) -> None:
    """Enable or disable the process-wide cache."""
    cache = get_cache()
    cache.enabled = enabled


def cached_request[ValueT](
    key: str, fetch_func: Callable[[], ValueT], ttl: int | None = None
) -> ValueT:
    """
    Decorator-style helper for caching API requests.

    Args:
        key: Unique cache key for this request
        fetch_func: Function that fetches the data if not cached
        ttl: Optional TTL override

    Returns:
        Cached data or freshly fetched data
    """
    cache = get_cache()

    # Try to get from cache
    cached = cache.get(key)
    if cached is not None:
        return cached

    # Fetch fresh data
    data = fetch_func()

    # Store in cache
    cache.set(key, data, ttl)

    return data
