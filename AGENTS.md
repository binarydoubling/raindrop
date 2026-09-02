# AGENTS.md — Raindrop Weather CLI

## Project Overview

Raindrop is a Python 3.12+ weather CLI built with Click and Rich. It uses
`urllib` for HTTP and has only two runtime dependencies. The package distribution
is `rdrop`; the command entry point is `raindrop.cli:main`. Package manager: `uv`.
Build backend: Hatchling.

## Project Layout

```
raindrop/
  cli.py               # Click group and command registration
  open_meteo.py        # Open-Meteo and NWS clients and response models
  ensemble.py          # Ensemble-member extraction and statistics
  settings.py          # Persistent configuration
  weather_provider.py  # Forecast provider selection and normalization
  commands/            # CLI command modules and shared command helpers
  providers/           # Credentialed provider clients
  utils/               # Formatting, weather, and astronomy helpers
scripts/capture.py     # README screenshot capture
pyproject.toml         # Project metadata and tooling
```

## Build / Install / Run

```bash
uv sync
uv run raindrop current Seattle
pip install -e ".[dev]"
uv build
```

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

Run a focused test with:

```bash
uv run pytest tests/test_cache.py
uv run pytest tests/test_cache.py::test_name
uv run pytest -k "geocode"
```

## Code Style

### Imports

Order imports in three groups separated by blank lines:

1. Standard library
2. Third-party (`click`, `rich.*`)
3. Local (`raindrop.*`)

Use absolute imports except for deliberate package re-exports. Ruff owns import
sorting and formatting.

### Naming

| Element | Convention | Example |
|---------|------------|---------|
| Files/functions/variables | `snake_case` | `open_meteo.py`, `get_cache()` |
| Constants | `UPPER_SNAKE_CASE` | `WEATHER_CODES` |
| Classes/types | `PascalCase` | `OpenMeteo`, `TemperatureUnit` |
| CLI commands | function name | `def current(...)` |

### Types and Formatting

- Annotate every function signature.
- Use `X | None` and built-in generics such as `list[str]`.
- Use `Literal` for constrained strings.
- Use double-quoted strings and f-strings.
- Leave two blank lines between top-level definitions.
- Use Rich markup for styled human output.

### Command Modules

Command modules reuse `console`, `geocode`, `echo_json`, location resolution,
and payload helpers from `raindrop.commands.common`. Each Click command generally:

1. loads settings;
2. resolves a location;
3. fetches provider-neutral data where applicable;
4. emits JSON with `echo_json` or renders Rich output.

Register new commands directly in `raindrop/cli.py`.

### Error Handling

- Raise `click.ClickException` for user-facing errors.
- API clients wrap transport/JSON failures in `OpenMeteoError` or their provider error.
- Ignore only explicitly non-critical filesystem failures.
- Guard optional API arrays with `or []` and bounds checks.

### Data Models

Use `@dataclass` for API and settings models. API fields that may be omitted use
`type | None = None`. Keep forecast commands provider-neutral; do not blend model
forecasts with measured station observations without explicit provenance.

### Documentation

Every Python file has a module docstring. Classes and public helpers have concise
docstrings. Click command docstrings are user-facing help.

## Architecture Notes

- Runtime dependencies: Click and Rich only.
- HTTP: `urllib.request`; no requests/httpx.
- Astronomy: pure Python.
- Cache: JSON files with SHA256-derived keys.
- APIs: Open-Meteo Forecast/Ensemble, OSRM, NWS, and optional credentialed Xweather.
- `--json` command output uses the shared indented serializer.
- Xweather credentials must never be printed, serialized into output, or included in cache keys.
- `--model` selects Open-Meteo; provider `auto` otherwise prefers configured Xweather.

## Common Pitfalls

- Import API/config modules as `raindrop.open_meteo` and `raindrop.settings`.
- Register commands directly in `raindrop/cli.py`; `raindrop.commands` is not a barrel.
- The project targets Python 3.12+.
- Mock external APIs in ordinary tests; use documented CLI commands for manual live probes.
