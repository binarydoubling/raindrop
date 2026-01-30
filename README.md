<p align="center">
  <img src="assets/logo.svg" alt="Raindrop" width="120">
</p>

<h1 align="center">Raindrop</h1>

<p align="center">
  <strong>A beautiful, feature-rich weather CLI for your terminal.</strong><br>
  Sparklines, route planning, live dashboards, marine forecasts, and more — all without an API key.
</p>

<p align="center">
  <a href="#installation"><img src="https://img.shields.io/badge/python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.12+"></a>
  <a href="#commands"><img src="https://img.shields.io/badge/commands-17-22c55e?style=flat-square" alt="17 Commands"></a>
  <a href="https://open-meteo.com/"><img src="https://img.shields.io/badge/API-Open--Meteo-f97316?style=flat-square" alt="Open-Meteo"></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT-a855f7?style=flat-square" alt="MIT License"></a>
  <a href="#"><img src="https://img.shields.io/badge/dependencies-2-64748b?style=flat-square" alt="2 Dependencies"></a>
</p>

<br>

<p align="center">
  <img src="assets/current.svg" alt="raindrop current Seattle" width="700">
</p>

---

## Why Raindrop?

Most weather CLIs give you temperature and a condition. Raindrop gives you **everything**:

- **Sparkline graphs** — `▁▂▃▅▇█▅▃` temperature and precipitation trends at a glance
- **Technical analysis** — EMA crossovers, rate-of-change, and volatility on forecasts
- **Real driving routes** — weather checkpoints along actual roads via OpenStreetMap
- **Full-screen dashboard** — live TUI with auto-refresh
- **Astronomical data** — moon phases, golden hour, blue hour, daylight tracking
- **Marine forecasts** — wave height, swell, water temp for coastal trips
- **Zero API keys** — entirely free, open APIs

---

## Installation

**Requirements:** Python 3.12+

```bash
# pip
pip install rdrop

# From source
git clone https://github.com/binarydoubling/raindrop.git
cd raindrop
pip install -e .
```

---

## Quick Start

```bash
raindrop current Seattle                          # current conditions
raindrop hourly "New York" --spark --hours 12     # sparkline forecast
raindrop daily Portland                           # 10-day with trend analysis
raindrop dashboard Seattle --refresh 300          # live full-screen TUI
raindrop route "Seattle" "San Francisco" -i 100   # road trip weather
raindrop aqi Beijing                              # air quality index
raindrop favorites add home "Seattle, WA"         # save a location
raindrop current home                             # use it anywhere
```

---

## Features

### Current Weather

<img src="assets/current.svg" alt="raindrop current Seattle" width="700">

### Hourly Forecast with Sparklines

<img src="assets/hourly.svg" alt="raindrop hourly Seattle" width="700">

### 10-Day Forecast with Technical Analysis

EMA crossovers, rate-of-change indicators, and trend detection applied to weather data.

<img src="assets/daily.svg" alt="raindrop daily Seattle" width="700">

### Route Weather Planning

Plan road trips with weather at every checkpoint along the actual driving route.

<img src="assets/route.svg" alt="raindrop route Seattle San Francisco" width="700">

### Compare Multiple Locations

<img src="assets/compare.svg" alt="raindrop compare" width="700">

### Air Quality Index

<img src="assets/aqi.svg" alt="raindrop aqi Seattle" width="700">

### Astronomical Data

Moon phases, golden hour, blue hour, and weekly daylight tracking.

<img src="assets/astro.svg" alt="raindrop astro Seattle" width="700">

### Marine Forecasts

<img src="assets/marine.svg" alt="raindrop marine San Diego" width="700">

### Clothing Recommendations

<img src="assets/clothing.svg" alt="raindrop clothing Seattle" width="700">

### Live Dashboard

Full-screen TUI powered by Rich with automatic refresh.

```bash
raindrop dashboard Seattle --refresh 300
```

---

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `current` | Current conditions | `raindrop current Seattle` |
| `hourly` | Hourly forecast (48h) | `raindrop hourly Seattle --spark` |
| `daily` | 10-day forecast with trends | `raindrop daily Seattle` |
| `dashboard` | Full-screen live TUI | `raindrop dashboard Seattle` |
| `route` | Weather along a driving route | `raindrop route "A" "B" -i 50` |
| `compare` | Compare multiple locations | `raindrop compare NYC LA Chicago` |
| `alerts` | NWS weather alerts (US) | `raindrop alerts Seattle` |
| `aqi` | Air quality index | `raindrop aqi Seattle` |
| `astro` | Moon phase, golden hour | `raindrop astro Seattle` |
| `marine` | Ocean/wave forecasts | `raindrop marine "San Diego"` |
| `clothing` | What to wear | `raindrop clothing Seattle` |
| `history` | Compare with past years | `raindrop history Seattle` |
| `discussion` | NWS forecast discussion | `raindrop discussion Seattle` |
| `precip` | Precipitation totals | `raindrop precip Seattle --days 7` |
| `favorites` | Manage saved locations | `raindrop favorites list` |
| `config` | View/edit settings | `raindrop config show` |
| `completions` | Shell completions | `raindrop completions bash` |

### Global Options

```
--units metric|imperial    Set temperature and distance units (default: imperial)
--no-cache                 Bypass cache for fresh data
--json                     Output raw JSON for scripting
--help                     Show help for any command
```

---

## Configuration

Settings live at `~/.config/raindrop/config.json`.

```bash
raindrop config show                  # view current settings
raindrop config set units metric      # switch to metric
raindrop config set location "NYC"    # set default location
raindrop config cache                 # view cache stats
raindrop config cache --clear         # clear cache
```

| Setting | Values | Default | Description |
|---------|--------|---------|-------------|
| `units` | `imperial`, `metric` | `imperial` | Temperature and distance units |
| `location` | any string | none | Default location |
| `cache_ttl` | seconds | `600` | Cache lifetime |

---

## Shell Completions

```bash
# Bash
raindrop completions bash >> ~/.bashrc && source ~/.bashrc

# Zsh
raindrop completions zsh >> ~/.zshrc && source ~/.zshrc

# Fish
raindrop completions fish > ~/.config/fish/completions/raindrop.fish
```

---

## How It Works

Raindrop combines several free, open APIs — no keys required:

| API | Purpose |
|-----|---------|
| [Open-Meteo](https://open-meteo.com/) | Forecasts, historical data, air quality, geocoding |
| [OSRM](http://project-osrm.org/) | Real driving routes via OpenStreetMap |
| [NWS](https://www.weather.gov/documentation/services-web-api) | Weather alerts and forecast discussions (US) |

The entire project has only **2 runtime dependencies** (Click and Rich). HTTP, caching, geocoding, and astronomical calculations are all handled with Python's standard library.

### Technical Highlights

- **Sparklines** via Unicode block characters (`▁▂▃▄▅▆▇█`)
- **EMA crossovers** and rate-of-change analysis on temperature data
- **Pure-Python astronomy** — moon phases, Julian day, daylight duration with no external libs
- **Haversine sampling** along OSRM polylines for route weather checkpoints
- **File-based cache** with SHA256 keys and configurable TTL
- **14 weather models** selectable: ECMWF, GFS, HRRR, ICON, ARPEGE, AROME, UKMO, GEM, JMA, MetNo, and more

---

## Regenerating Screenshots

The feature screenshots in this README are real CLI output captured as SVGs:

```bash
python scripts/capture.py            # capture all commands
python scripts/capture.py current    # capture a specific command
```

---

## Contributing

```bash
git clone https://github.com/binarydoubling/raindrop.git
cd raindrop
pip install -e ".[dev]"
pytest
```

---

## License

[MIT](LICENSE)

---

<p align="center">
  <sub>Built with coffee and curiosity in the Pacific Northwest — where checking the weather is a lifestyle.</sub>
</p>
