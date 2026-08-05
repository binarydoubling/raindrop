"""CLI commands for raindrop."""

from .alerts import alerts
from .aqi import aqi
from .astro import astro
from .clothing import clothing
from .compare import compare
from .completions import completions
from .config import config
from .current import current
from .daily import daily
from .dashboard import dashboard
from .discussion import discussion
from .favorites import fav
from .history import history
from .hourly import hourly
from .marine import marine
from .precip import precip
from .route import route
from .window import window

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
    "window",
]
