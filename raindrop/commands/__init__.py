"""CLI commands for raindrop."""

from .current import current
from .hourly import hourly
from .daily import daily
from .aqi import aqi
from .alerts import alerts
from .discussion import discussion
from .precip import precip
from .compare import compare
from .history import history
from .config import config
from .favorites import fav
from .astro import astro
from .clothing import clothing
from .route import route
from .completions import completions
from .dashboard import dashboard
from .marine import marine

__all__ = [
    "current",
    "hourly",
    "daily",
    "aqi",
    "alerts",
    "discussion",
    "precip",
    "compare",
    "history",
    "config",
    "fav",
    "astro",
    "clothing",
    "route",
    "completions",
    "dashboard",
    "marine",
]
