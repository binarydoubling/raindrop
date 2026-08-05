"""Raindrop - A beautiful weather CLI tool."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rdrop")
except PackageNotFoundError:
    __version__ = "0.1.0"
