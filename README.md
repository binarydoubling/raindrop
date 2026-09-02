<p align="center">
  <img src="assets/logo.svg" alt="Raindrop" width="120">
</p>

<h1 align="center">Raindrop</h1>

<p align="center">
  <strong>A beautiful, feature-rich weather CLI for your terminal.</strong><br>
  Sparklines, route planning, live dashboards, marine forecasts, measured stations, and more — core forecasts need no API key.
</p>

<p align="center">
  <a href="#installation"><img src="https://img.shields.io/badge/python-3.12+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.12+"></a>
  <a href="#commands"><img src="https://img.shields.io/badge/commands-19-22c55e?style=flat-square" alt="19 Commands"></a>
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
- **Ensemble lab** — inspect every model member, uncertainty bands, outliers, thresholds, and consensus departure
- **Real driving routes** — weather checkpoints along actual roads via OpenStreetMap
- **Full-screen dashboard** — live TUI with auto-refresh
- **Astronomical data** — moon phases, golden hour, blue hour, daylight tracking
- **Marine forecasts** — wave height, swell, water temp for coastal trips
- **Zero-key core weather** — forecasts, dashboards, routes, air quality, alerts, and astronomy use free/open APIs
- **Optional measured stations** — nearby personal/official observations via Xweather when you configure a key

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
raindrop window Seattle                           # sensory weather scene
raindrop ensemble forecast Seattle -n 48           # member spread and percentiles
raindrop stations nearby Fairbanks                 # nearby PWS observations (optional Xweather key)
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

### Weather Window

Translate weather variables into the feel of standing outside: air texture, sky, light, motion, surfaces, and distance.

```bash
raindrop window Seattle
raindrop outside Tokyo --compact
```

### Current Weather

<img src="assets/current.svg" alt="raindrop current Seattle" width="700">

### Ensemble Forecast Lab

Ensembles run the same forecast model with multiple plausible initial conditions. Agreement among members suggests greater confidence; disagreement exposes uncertainty that a single deterministic forecast hides. Raindrop analyzes the Open-Meteo Ensemble API directly and requires no API key.

> [!IMPORTANT]
> Ensemble consensus is not historical forecast skill. `member` and `rank` compare members with the same run's consensus, not with later observations, so the closest member is not necessarily the most accurate one.

#### Quick start

```bash
# Default: NOAA GEFS, hourly temperature, next 24 hours
raindrop ensemble forecast Fairbanks

# Temperature uncertainty over the next 72 hours
raindrop ensemble forecast Fairbanks -v temperature_2m -n 72

# Inspect one perturbed member against the ensemble consensus
raindrop ensemble member Fairbanks --member 12 -v temperature_2m -n 72

# Find the 10 members closest to consensus over the requested horizon
raindrop ensemble rank Fairbanks -v temperature_2m -n 72 --top 10

# Discover model identifiers
raindrop ensemble models
```

The default model is `ncep_gefs_seamless`. Every invocation analyzes exactly one model so member identities and consensus statistics remain unambiguous.

#### Forecast distributions and uncertainty

`ensemble forecast` summarizes all available members at each forecast step. Human output shows the control value, distribution center, uncertainty interval, and number of members returned:

```bash
raindrop ensemble forecast Fairbanks -n 48
```

Request several variables by repeating `-v/--variable`:

```bash
raindrop ensemble forecast Fairbanks \
  -v temperature_2m \
  -v precipitation \
  -v wind_speed_10m \
  -n 24
```

Variable names are passed directly to Open-Meteo rather than restricted by a Raindrop allow-list. Common hourly variables include:

```text
temperature_2m
relative_humidity_2m
precipitation
weather_code
wind_speed_10m
wind_gusts_10m
wind_direction_10m
```

Model and regional availability varies. If Open-Meteo does not provide a requested variable for the selected model, Raindrop reports the missing ensemble data.

Continuous variables include mean, median, population standard deviation, minimum, maximum, full spread, requested percentiles, members outside the 1.5-IQR bounds, and the fraction within one standard deviation. Raindrop avoids misleading linear statistics for special variable types:

- `weather_code` and `is_day` use categorical mode, frequency, and agreement.
- `wind_direction_*` uses a circular mean and directional concentration, so 359° and 1° average near north rather than south.

The default percentile set is `10,25,50,75,90`. Supply any finite percentiles from 0 through 100:

```bash
raindrop ensemble forecast Fairbanks \
  -v temperature_2m \
  --percentiles 5,25,50,75,95 \
  -n 48
```

The terminal uncertainty column spans the lowest through highest requested percentile. JSON contains every requested percentile and the complete member values.

#### Threshold probabilities

For one variable, calculate the fraction of available members above or below a threshold:

```bash
# Probability that temperature falls below 32°F with imperial settings
raindrop ensemble forecast Fairbanks \
  -v temperature_2m \
  --threshold 32 \
  --operator below \
  -n 48

# Probability that hourly precipitation exceeds 0.1 inches
raindrop ensemble forecast Fairbanks \
  -v precipitation \
  --threshold 0.1 \
  --operator above \
  -n 24
```

Thresholds use the units in `raindrop config show`; they are not automatically interpreted as Celsius, Fahrenheit, millimeters, or inches independently of that configuration. Threshold analysis accepts exactly one `--variable` so the probability has one clear meaning. Comparisons are strict (`>` or `<`), not inclusive.

#### Daily ensembles

Add `--daily` to request daily rather than hourly variables:

```bash
raindrop ensemble forecast Fairbanks \
  --daily \
  -v temperature_2m_max \
  -v temperature_2m_min \
  -v precipitation_sum \
  -n 14
```

Without `-v`, daily mode defaults to `temperature_2m_mean`. Daily requests support up to 36 periods:

```bash
raindrop ensemble forecast Fairbanks --daily -n 36
```

Hourly requests default to 24 periods and support up to 864. The selected model may have a shorter forecast horizon than the command-level limit.

#### Evaluate an individual member

`ensemble member` compares one member with the same-step ensemble consensus throughout the requested horizon:

```bash
# Member 0 is the unsuffixed control forecast when the model returns one
raindrop ensemble member Fairbanks \
  --member 0 \
  -v temperature_2m \
  -n 48

# Evaluate the same perturbed member for two variables
raindrop ensemble member Fairbanks \
  --member 12 \
  -v temperature_2m \
  -v precipitation \
  -n 72
```

For continuous variables, the evaluation includes the member value, median consensus, signed departure, percentile position, z-score, and outlier status; human tables show the compact value, consensus, departure, and position fields. Categorical variables report whether the member matches the mode; wind direction reports angular departure and directional agreement. Horizon summaries include bias, mean absolute departure, RMSE from consensus, and outlier rate.

Member IDs are discovered from the returned columns rather than assumed from a fixed local list. Raindrop reports an error if the requested member is unavailable for any selected variable.

#### Rank members by consensus proximity

`ensemble rank` accepts one variable and ranks members by RMSE from the same-run consensus across the requested horizon:

```bash
raindrop ensemble rank Fairbanks \
  -v temperature_2m \
  -n 72 \
  --top 10

raindrop ensemble rank Fairbanks \
  -v precipitation \
  -n 48 \
  --top 15
```

The ranking includes sample count, signed bias, mean absolute departure, RMSE, and outlier rate. For categorical variables, departure means disagreement with the mode; for wind direction, it is angular departure. This ranks conformity, not real-world accuracy.

#### Select an ensemble model

List the known model identifiers and their advertised regions, member counts, and horizons:

```bash
raindrop ensemble models
raindrop ensemble models --json
```

Examples using a non-default model:

```bash
# ECMWF's 51-member global IFS ensemble
raindrop ensemble forecast Fairbanks \
  -m ecmwf_ifs025_ensemble \
  -v temperature_2m \
  -n 72

# Google's 64-member WeatherNext 2 ensemble
raindrop ensemble forecast Fairbanks \
  -m google_weathernext2_ensemble \
  -v temperature_2m \
  -n 72
```

Known identifiers are:

| Identifier | Model | Region | Members | Horizon |
|------------|-------|--------|--------:|--------:|
| `icon_seamless_eps` | DWD ICON EPS Seamless | Best available ICON domain | 40 | 7.5d |
| `icon_global_eps` | DWD ICON Global EPS | Global | 40 | 7.5d |
| `icon_eu_eps` | DWD ICON EU EPS | Europe | 40 | 5d |
| `icon_d2_eps` | DWD ICON D2 EPS | Central Europe | 20 | 2d |
| `ncep_gefs_seamless` | NOAA GFS Ensemble Seamless | Global | 31 | 35d |
| `ncep_gefs025` | NOAA GFS Ensemble 0.25° | Global | 31 | 10d |
| `ncep_gefs05` | NOAA GFS Ensemble 0.5° | Global | 31 | 35d |
| `ncep_aigefs025` | NOAA AI GEFS 0.25° | Global | 31 | 16d |
| `ecmwf_ifs025_ensemble` | ECMWF IFS 0.25° Ensemble | Global | 51 | 15d |
| `ecmwf_ifs_europe_ensemble` | ECMWF IFS 9 km Ensemble | Europe | 51 | 15d |
| `ecmwf_aifs025_ensemble` | ECMWF AIFS 0.25° Ensemble | Global | 51 | 15d |
| `ecmwf_aifs_europe_ensemble` | ECMWF AIFS 31 km Ensemble | Europe | 51 | 15d |
| `gem_global_ensemble` | Canadian GEM Global Ensemble | Global | 21 | 16d |
| `bom_access_global_ensemble` | BOM ACCESS Global Ensemble | Global | 18 | 10d |
| `ukmo_global_ensemble_20km` | UKMO Global Ensemble | Global | 18 | 8d |
| `ukmo_uk_ensemble_2km` | UKMO UK Ensemble | United Kingdom | 3 | 5d |
| `meteoswiss_icon_ch1_ensemble` | MeteoSwiss ICON CH1 | Central Europe | 11 | 33h |
| `meteoswiss_icon_ch2_ensemble` | MeteoSwiss ICON CH2 | Central Europe | 21 | 12h |
| `google_weathernext2_ensemble` | Google WeatherNext 2 | Global | 64 | 15d |

These are convenient known identifiers, not a guarantee of current Open-Meteo availability. Supported variables, actual member count, update timing, region, and usable horizon remain provider-controlled.

#### Temporal resolution

Hourly commands default to an hourly time grid. Select another Open-Meteo ensemble resolution when the model and variable support it:

```bash
raindrop ensemble forecast Fairbanks \
  -v temperature_2m \
  --temporal-resolution hourly_6 \
  -n 40
```

Choices are `native`, `hourly`, `hourly_3`, and `hourly_6`. `-n/--periods` counts returned time steps: with `hourly_6`, for example, 40 periods represent 240 hours.

#### JSON and scripting

Every ensemble subcommand supports JSON output:

```bash
raindrop ensemble forecast Fairbanks \
  -v temperature_2m \
  -n 24 \
  --json

raindrop ensemble member Fairbanks \
  --member 12 \
  -v wind_speed_10m \
  -n 48 \
  --json

raindrop ensemble rank Fairbanks \
  -v temperature_2m \
  -n 72 \
  --json
```

With `jq`, extract the five members closest to consensus:

```bash
raindrop ensemble rank Fairbanks \
  -v temperature_2m \
  -n 72 \
  --json | jq '.ranking[:5]'
```

Forecast JSON includes location and source metadata, interval, units, every member value, and the complete per-step analysis. Member JSON includes per-step evaluations and horizon summaries. Rank JSON labels its metric explicitly as departure from ensemble consensus rather than forecast skill.

All commands accept `-c/--country` to disambiguate geocoding. If a default location is configured, omit the location argument:

```bash
raindrop config set location Fairbanks
raindrop ensemble forecast -n 48
```

Use `raindrop --no-cache ensemble ...` to bypass cached API responses, and append `--help` to any level for the authoritative option list:

```bash
raindrop ensemble --help
raindrop ensemble forecast --help
raindrop ensemble member --help
raindrop ensemble rank --help
```

### Measured Station Observations

Nearby personal weather station and official station observations through Xweather. These are measured station readings with station IDs, source classification, timestamps, QC/trust status, and freshness — separate from model/gridded conditions returned by `raindrop current`.

```bash
raindrop stations nearby Fairbanks
raindrop stations nearby Fairbanks --kind pws
raindrop stations current PWS_SM9110
```

Xweather is optional and requires credentials. Human output includes the required attribution: Powered by Vaisala Xweather.

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
| `window` / `outside` | First-person weather scene | `raindrop window Seattle` |
| `current` | Current conditions | `raindrop current Seattle` |
| `ensemble` | Ensemble members and uncertainty | `raindrop ensemble forecast Seattle` |
| `stations` | Measured station observations | `raindrop stations nearby Fairbanks` |
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
| `fav` / `favorites` | Manage saved locations | `raindrop favorites list` |
| `config` | View/edit settings | `raindrop config show` |

### Global Options

```
--no-cache                 Bypass the API response cache for fresh data
--version                  Show the installed version
--help                     Show help for any command
```

Most data commands also support `--json` for scripting and `-c/--country` for ISO country filtering.
Use `raindrop config set units metric|imperial` to switch unit presets.

---

## Configuration

Settings live at `$RAINDROP_CONFIG_DIR/config.json`, `$XDG_CONFIG_HOME/raindrop/config.json`, or `~/.config/raindrop/config.json`.
API responses are cached under `$RAINDROP_CACHE_DIR`, `$XDG_CACHE_HOME/raindrop`, or `~/.cache/raindrop`.

Optional Xweather credentials for Xweather-backed forecasts and `raindrop stations` can be supplied with `XWEATHER_API_KEY` or a mode-0600 JSON file at the same config directory, e.g. `~/.config/raindrop/xweather.json`:

```json
{"api_key":"<your combined Xweather API key>"}
```

Existing forecast/current commands remain keyless through Open-Meteo fallback. With `weather_provider=auto`, Raindrop uses Xweather when credentials are configured unless an Open-Meteo model is selected. `raindrop config show` only reports Xweather as configured/not configured and never prints the key.

```bash
raindrop config show                  # view current settings
raindrop config set units metric      # switch to metric
raindrop config set weather_provider xweather  # prefer Xweather explicitly
raindrop config set location "NYC"    # set default location
raindrop config cache                 # view cache stats
raindrop config cache --clear         # clear cache
```

| Setting | Values | Default | Description |
|---------|--------|---------|-------------|
| `units` | `imperial`, `metric` | `imperial` | Shortcut that updates the unit fields below |
| `temperature_unit` | `fahrenheit`, `celsius` | `fahrenheit` | Temperature display unit |
| `wind_speed_unit` | `mph`, `kmh`, `ms`, `kn` | `mph` | Wind speed display unit |
| `precipitation_unit` | `mm`, `inch` | `mm` | Precipitation display unit |
| `location` | any string | none | Default location |
| `country_code` | ISO 3166-1 alpha-2 | none | Default country filter |
| `weather_provider` | `auto`, `open-meteo`, `xweather` | `auto` | Forecast provider; auto prefers configured Xweather, otherwise Open-Meteo |
| `model` | `auto` or `raindrop config models` value | `auto` | Open-Meteo forecast model |

---

## Shell Completions

Click provides completions directly from the installed command:

```bash
# Bash: add to ~/.bashrc
eval "$(_RAINDROP_COMPLETE=bash_source raindrop)"

# Zsh: add to ~/.zshrc
eval "$(_RAINDROP_COMPLETE=zsh_source raindrop)"

# Fish: add to ~/.config/fish/config.fish
_RAINDROP_COMPLETE=fish_source raindrop | source
```

---

## How It Works

Raindrop combines free/open APIs with optional credentialed Xweather integration:

| API | Purpose |
|-----|---------|
| [Open-Meteo](https://open-meteo.com/) | Forecasts, ensemble members, historical data, air quality, geocoding |
| [OSRM](http://project-osrm.org/) | Real driving routes via OpenStreetMap |
| [NWS](https://www.weather.gov/documentation/services-web-api) | Weather alerts and forecast discussions (US) |
| [Xweather](https://www.xweather.com/) | Optional forecasts/current conditions and measured personal/official station observations |

The entire project has only **2 runtime dependencies** (Click and Rich). HTTP, caching, geocoding, credentials, and astronomical calculations are all handled with Python's standard library.

### Technical Highlights

- **Weather window scene engine** that infers air texture, sky, light, motion, surface, and horizon from combined variables
- **Provider-neutral weather commands** normalized across Open-Meteo and Xweather
- **Ensemble-member analysis** with continuous, categorical, and circular statistics
- **Measured station observations** with provider/source identity, timestamps, QC/trust labels, freshness filtering, and safe credential-free cache keys
- **Sparklines** via Unicode block characters (`▁▂▃▄▅▆▇█`)
- **EMA crossovers** and rate-of-change analysis on temperature data
- **Pure-Python astronomy** — moon phases and daylight calculations with no external libs
- **Distance sampling** along OSRM routes for weather checkpoints
- **File-based cache** with SHA256 keys and endpoint-specific TTLs
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
uv sync
uv run pytest
uv run ruff check
uv run pyright
```

---

## License

[MIT](LICENSE)

---

<p align="center">
  <sub>Built with coffee and curiosity in the Pacific Northwest — where checking the weather is a lifestyle.</sub>
</p>
