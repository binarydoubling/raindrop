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
  <img src="assets/banner.svg" alt="Raindrop demo" width="680">
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
pip install raindrop-weather

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

```
╭─ Seattle, Washington, United States ───────────────────────────╮
│                                                                │
│   Partly Cloudy                                       58°F     │
│                                                                │
│   Feels Like    55°F        Humidity      62%                  │
│   Wind          12 mph NW   Pressure      1018 hPa             │
│   UV Index      4 (Mod)     Visibility    10 mi                │
│   Dew Point     45°F        Cloud Cover   35%                  │
│                                                                │
╰────────────────────────────────────────────────────────────────╯
```

Compact mode for shell prompts:

```bash
$ raindrop current Seattle --compact
Seattle: 58°F ↑62° ↓49° | Partly Cloudy | 62% | 12mph NW
```

### Hourly Forecast with Sparklines

```
╭─ Hourly Forecast ─ Seattle ────────────────────────────────────╮
│                                                                │
│  Temperature:   ▁▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁▁▂▂▃▄▅▆                     │
│                 52° ─────────────────── 67° ──────► 58°        │
│                                                                │
│  Precipitation: ▁▁▁▁▁▁▁▁▁▁▂▃▄▃▂▁▁▂▃▅▆▄▂▁                     │
│                 0% ──────────────────── 45% ─────► 15%         │
│                                                                │
│  Hour   Temp   Precip   Wind      Condition                    │
│  12pm   58°F     0%     8 mph     Partly Cloudy                │
│   1pm   61°F     0%    10 mph     Partly Cloudy                │
│   2pm   64°F     5%    12 mph     Mostly Cloudy                │
│   3pm   67°F    15%    14 mph     Cloudy                       │
│   4pm   65°F    25%    12 mph     Light Rain                   │
│   ...                                                          │
╰────────────────────────────────────────────────────────────────╯
```

### 10-Day Forecast with Technical Analysis

EMA crossovers, rate-of-change indicators, and trend detection applied to weather data.

```
╭─ 10-Day Forecast ─ Seattle ────────────────────────────────────╮
│                                                                │
│  Trend: Warming (+2.3°F/day) │ EMA: Bullish crossover Day 3   │
│                                                                │
│  Day        High   Low    Precip   Condition                   │
│  Today      62°F   48°F    15%     Partly Cloudy               │
│  Tomorrow   65°F   50°F    10%     Mostly Sunny     ↑ +3°      │
│  Wednesday  68°F   52°F     5%     Sunny            ↑ +3°      │
│  Thursday   71°F   54°F     0%     Sunny            ↑ +3° ★EMA │
│  Friday     69°F   53°F    20%     Partly Cloudy    ↓ -2°      │
│  Saturday   64°F   51°F    45%     Rain             ↓ -5°      │
│  Sunday     61°F   49°F    60%     Rain             ↓ -3°      │
│  Monday     63°F   50°F    30%     Showers          ↑ +2°      │
│  Tuesday    66°F   51°F    15%     Partly Cloudy    ↑ +3°      │
│  Wednesday  68°F   52°F    10%     Mostly Sunny     ↑ +2°      │
│                                                                │
╰────────────────────────────────────────────────────────────────╯
```

### Route Weather Planning

Plan road trips with weather at every checkpoint along the actual driving route.

```
╭─ Route Weather: Seattle → San Francisco ───────────────────────╮
│                                                                │
│  Distance: 807 mi │ Est. Time: 12h 15m │ Checkpoints: 9       │
│                                                                │
│  Temperature: ▃▃▄▅▆▆▇█▇▆  (52°F → 71°F)                      │
│                                                                │
│  Mile   Location              Temp   Condition                 │
│    0    Seattle, WA           52°F   Cloudy                    │
│  100    Centralia, WA         55°F   Overcast                  │
│  200    Portland, OR          58°F   Partly Cloudy             │
│  300    Salem, OR             61°F   Mostly Sunny              │
│  400    Eugene, OR            63°F   Sunny                     │
│  500    Grants Pass, OR       67°F   Sunny                     │
│  600    Redding, CA           71°F   Clear                     │
│  700    Sacramento, CA        69°F   Partly Cloudy             │
│  807    San Francisco, CA     62°F   Foggy                     │
│                                                                │
╰────────────────────────────────────────────────────────────────╯
```

### Live Dashboard

Full-screen TUI powered by Rich with automatic refresh.

```bash
raindrop dashboard Seattle --refresh 300
```

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  RAINDROP DASHBOARD                       Updated: 2:34:15 PM  ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                ┃
┃  SEATTLE, WA                                                   ┃
┃  Partly Cloudy              58°F  (feels like 55°F)            ┃
┃                                                                ┃
┃  ┌─ HOURLY ─────────────────────────────────────────────────┐  ┃
┃  │ Temp:   ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁▂▃▄▅  (52° → 67° → 58°)      │  ┃
┃  │ Precip: ▁▁▁▁▂▃▄▃▂▁▁▂▃▅▆▄▂▁▁▁  (0% → 45% → 10%)       │  ┃
┃  └──────────────────────────────────────────────────────────┘  ┃
┃                                                                ┃
┃  ┌─ 5-DAY ──────────────────────────────────────────────────┐  ┃
┃  │ Today     62/48°F  15%  Partly Cloudy                    │  ┃
┃  │ Tomorrow  65/50°F  10%  Mostly Sunny                     │  ┃
┃  │ Wed       68/52°F   5%  Sunny                            │  ┃
┃  │ Thu       71/54°F   0%  Sunny                            │  ┃
┃  │ Fri       69/53°F  20%  Partly Cloudy                    │  ┃
┃  └──────────────────────────────────────────────────────────┘  ┃
┃                                                                ┃
┃  Moon: Waxing Gibbous (78%)  │  Sunset: 8:42 PM               ┃
┃                                                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### And Much More

| Command | What it does |
|---------|-------------|
| `astro` | Moon phases, golden/blue hour, weekly daylight chart |
| `marine` | Wave height, swell period/direction, water temp |
| `alerts` | NWS weather alerts and advisories (US) |
| `aqi` | Air quality index with pollutant breakdown |
| `clothing` | What-to-wear recommendations based on conditions |
| `history` | Compare today's weather against historical years |
| `discussion` | NWS Area Forecast Discussion text (US) |
| `precip` | Precipitation totals and breakdowns |
| `compare` | Side-by-side weather for multiple locations |

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
