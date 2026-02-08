# AGENTS.md — Raindrop Weather CLI

## Project Overview

Raindrop (`rdrop`) is a Python 3.12+ CLI weather tool built with Click and Rich.
It has only **2 runtime dependencies** (click, rich) and uses `urllib` for HTTP.
Entry point: `raindrop.cli:main`. Package manager: `uv`. Build backend: Hatchling.

## Project Layout

```
raindrop/              # Main package
  cli.py               # Click group, command registration
  __init__.py           # __version__
  commands/             # 17 CLI subcommand modules (current.py, hourly.py, etc.)
  utils/                # Shared utilities (formatting.py, weather.py, astro.py)
open_meteo.py          # API client (Open-Meteo, NWS) — lives at project root
settings.py            # Persistent settings/config — lives at project root
scripts/capture.py     # Screenshot capture for README SVGs
pyproject.toml         # Project metadata, deps, build config
```

Note: `open_meteo.py` and `settings.py` live at the project root (outside the
`raindrop/` package) and are force-included in the wheel via Hatch config.

## Build / Install / Run

```bash
uv sync                          # Install deps (creates .venv)
uv run raindrop current Seattle  # Run via uv
pip install -e ".[dev]"          # Editable install with dev deps
raindrop current Seattle         # Run after install
uv build                         # Build wheel + sdist into dist/
```

## Type Checking

```bash
uv run pyright                   # Run Pyright (only configured dev tool)
pyright                          # If installed globally
pyright raindrop/commands/current.py  # Check a single file
```

## Linting / Formatting

No linter or formatter is configured in pyproject.toml. A `.ruff_cache/` exists,
suggesting Ruff has been used. If adding tooling, prefer Ruff for both linting
and formatting as it aligns with the existing style.

## Testing

**No tests exist yet.** The README mentions pytest for contributing.
If/when tests are added:

```bash
uv run pytest                              # Run all tests
uv run pytest tests/test_cache.py          # Single test file
uv run pytest tests/test_cache.py::test_x  # Single test function
uv run pytest -k "test_geocode"            # Tests matching pattern
```

## Code Style

### Imports

Order imports in three groups separated by blank lines:

1. Standard library (`datetime`, `json`, `math`, `pathlib`, etc.)
2. Third-party (`click`, `rich.*`)
3. Local (`open_meteo`, `settings`, `raindrop.*`)

```python
"""Module docstring."""

from datetime import datetime
import json as json_lib

import click
from rich.console import Console
from rich.table import Table
from rich import box

from open_meteo import OpenMeteo
from settings import get_settings, AVAILABLE_MODELS
from raindrop.utils import WEATHER_CODES, sparkline
```

- Use **absolute imports** everywhere except in `__init__.py` files, which use
  relative imports to re-export submodules (e.g., `from .formatting import sparkline`).
- No path aliases are configured.

### Naming Conventions

| Element     | Convention         | Examples                                    |
|-------------|--------------------|---------------------------------------------|
| Files       | `snake_case.py`    | `open_meteo.py`, `formatting.py`            |
| Directories | lowercase          | `commands/`, `utils/`                        |
| Functions   | `snake_case`       | `get_cache()`, `deg_to_compass()`            |
| Variables   | `snake_case`       | `temp_symbol`, `total_distance_mi`           |
| Constants   | `UPPER_SNAKE_CASE` | `WEATHER_CODES`, `CACHE_DIR`, `DEFAULT_TTL`  |
| Classes     | `PascalCase`       | `OpenMeteo`, `GeocodingResult`, `Settings`   |
| Type aliases| `PascalCase`       | `TemperatureUnit`, `WindSpeedUnit`           |
| CLI commands| match function name| `def current(...)`, `def hourly(...)`        |

Short abbreviations are used for common local variables:
`c` (current), `d` (daily), `h` (hourly), `om` (OpenMeteo), `nws` (NWSClient).

### Type Annotations

- **All function signatures must have type annotations.**
- Use modern union syntax: `str | None`, `float | int | None` (not `Optional`).
- Use lowercase generics: `list[str]`, `dict[str, str]`, `tuple[str, str]`.
- Use `Literal` for constrained string types.
- Dataclass fields use `float | None = None` for optional API response fields.

### Formatting

- f-strings exclusively — never `.format()` or `%` formatting.
- 2 blank lines between top-level definitions.
- 1 blank line between methods inside a class.
- Rich console markup for colored output: `"[bold cyan]text[/bold cyan]"`.
- Strings use double quotes.

### Module Structure

Every command module follows this pattern:

```python
"""Command docstring."""

# 1. Standard library imports
# 2. Third-party imports (click, rich)
# 3. Local imports (open_meteo, settings, raindrop.utils)

om = OpenMeteo()           # Module-level singleton
console = Console()        # Module-level singleton

def geocode(location: str, country: str | None = None):
    """Geocode helper (duplicated per command module)."""
    results = om.geocode(location, country_code=country)
    return results[0]

@click.command()
@click.argument("location", required=False)
@click.option("-c", "--country", help="...")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def command_name(location: str | None, country: str | None, as_json: bool):
    """Help text shown in CLI."""
    settings = get_settings()
    # Resolve location, fetch data, render output
```

### Error Handling

- **User-facing errors**: Raise `click.ClickException(message)` — Click handles
  display and exit code.
- **API errors**: Caught in `open_meteo.py` and re-raised as `OpenMeteoError`.
  The `_request()` methods catch `HTTPError`, `URLError`, `TimeoutError`, and
  `JSONDecodeError`, wrapping them in `OpenMeteoError`.
- **Non-critical failures**: Use `except OSError: pass` (e.g., cache writes).
- **Missing data guards**: Use `or []` defaults and bounds checking:
  ```python
  temps = h.temperature_2m or []
  temp = temps[i] if i < len(temps) else 0
  ```
- **Location resolution**: Wrap in try/except ValueError, raise ClickException.

### Data Modeling

- Use `@dataclass` for all data models (API responses, settings, etc.).
- All API response dataclass fields should be `type | None = None` for optional
  data that may not be returned by every endpoint.

### Docstrings

- Every file has a module-level docstring: `"""Current weather command."""`
- Every class has a docstring.
- API client methods use `Args/Returns` format.
- Click command docstrings double as CLI help text.

## Key Architecture Notes

- Only 2 runtime deps: `click>=8.3.1`, `rich>=14.2.0`.
- HTTP via `urllib.request` — no requests/httpx.
- Pure-Python astronomy (no astropy): `raindrop/utils/astro.py`.
- File-based cache with SHA256 keys: `raindrop/cache.py`.
- External APIs: Open-Meteo (weather/geocoding), OSRM (routes), NWS (alerts/discussions).
- The `geocode()` helper is duplicated in every command module rather than shared.
- `--json` flag on every command outputs `json.dumps(data, indent=2)`.
- `--country` / `-c` flag on every command for country filtering.

## Common Pitfalls

- `open_meteo.py` and `settings.py` are at the project root, not inside `raindrop/`.
  Import them as `from open_meteo import OpenMeteo`, not `from raindrop.open_meteo`.
- When adding a new command, register it in both `raindrop/commands/__init__.py`
  (re-export) and `raindrop/cli.py` (`cli.add_command(new_cmd)`).
- The project targets Python 3.12+ — use modern syntax (`X | Y`, `list[T]`).
- No test suite exists yet. If writing tests, use pytest and consider mocking
  API calls since all commands hit external APIs.
