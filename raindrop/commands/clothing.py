"""Clothing suggestions based on weather conditions."""

import json as json_lib

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from open_meteo import OpenMeteo
from settings import get_settings
from raindrop.utils import (
    WEATHER_CODES,
    TEMP_SYMBOLS,
)

om = OpenMeteo()
console = Console()


def geocode(location: str, country: str | None = None):
    results = om.geocode(location, country_code=country)
    return results[0]


def get_temp_category(temp: float, unit: str) -> str:
    """Categorize temperature for clothing recommendations."""
    # Convert to Fahrenheit for consistent logic
    if unit == "celsius":
        temp_f = temp * 9 / 5 + 32
    else:
        temp_f = temp

    if temp_f >= 85:
        return "hot"
    elif temp_f >= 75:
        return "warm"
    elif temp_f >= 65:
        return "mild"
    elif temp_f >= 55:
        return "cool"
    elif temp_f >= 45:
        return "cold"
    elif temp_f >= 32:
        return "freezing"
    else:
        return "arctic"


def get_clothing_recommendations(
    temp: float,
    feels_like: float,
    weather_code: int,
    wind_speed: float,
    humidity: int,
    uv_index: float | None,
    unit: str,
) -> dict:
    """Generate clothing recommendations based on conditions."""

    temp_cat = get_temp_category(feels_like, unit)
    is_rainy = weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]
    is_snowy = weather_code in [71, 73, 75]
    is_stormy = weather_code in [95, 96, 99]
    is_foggy = weather_code in [45, 48]
    is_sunny = weather_code in [0, 1]

    # Convert wind to mph for logic
    if unit == "celsius":
        # Assume wind is in km/h
        wind_mph = wind_speed * 0.621371
    else:
        wind_mph = wind_speed

    is_windy = wind_mph > 15
    is_very_windy = wind_mph > 25
    is_humid = humidity > 70

    recommendations = {
        "layers": [],
        "accessories": [],
        "footwear": [],
        "tips": [],
    }

    # Base layers by temperature
    if temp_cat == "arctic":
        recommendations["layers"] = [
            "Thermal base layer",
            "Insulating mid-layer (fleece/down)",
            "Heavy winter coat",
            "Insulated pants or snow pants",
        ]
        recommendations["accessories"] = [
            "Warm hat covering ears",
            "Insulated gloves or mittens",
            "Scarf or neck gaiter",
            "Face covering for extreme cold",
        ]
        recommendations["footwear"] = ["Insulated winter boots"]
        recommendations["tips"].append("Limit time outdoors; frostbite risk")

    elif temp_cat == "freezing":
        recommendations["layers"] = [
            "Thermal or warm base layer",
            "Sweater or fleece",
            "Winter coat",
        ]
        recommendations["accessories"] = [
            "Warm hat",
            "Gloves",
            "Scarf",
        ]
        recommendations["footwear"] = ["Winter boots or insulated shoes"]

    elif temp_cat == "cold":
        recommendations["layers"] = [
            "Long-sleeve shirt or thermal",
            "Sweater or light fleece",
            "Medium-weight jacket",
        ]
        recommendations["accessories"] = [
            "Light hat or beanie",
            "Light gloves (optional)",
        ]
        recommendations["footwear"] = ["Closed-toe shoes or boots"]

    elif temp_cat == "cool":
        recommendations["layers"] = [
            "Long-sleeve shirt",
            "Light jacket or cardigan",
        ]
        recommendations["footwear"] = ["Comfortable closed-toe shoes"]

    elif temp_cat == "mild":
        recommendations["layers"] = [
            "T-shirt or light long-sleeve",
            "Optional light layer for evening",
        ]
        recommendations["footwear"] = ["Comfortable shoes or sneakers"]

    elif temp_cat == "warm":
        recommendations["layers"] = [
            "Light t-shirt or short-sleeve shirt",
            "Shorts or light pants",
        ]
        recommendations["footwear"] = ["Breathable shoes or sandals"]
        recommendations["tips"].append("Stay hydrated")

    else:  # hot
        recommendations["layers"] = [
            "Light, breathable clothing",
            "Light colors to reflect heat",
            "Loose-fitting garments",
        ]
        recommendations["footwear"] = ["Sandals or breathable shoes"]
        recommendations["tips"].append("Stay hydrated and seek shade")
        recommendations["tips"].append("Avoid peak sun hours (10am-4pm)")

    # Weather modifiers
    if is_rainy or is_stormy:
        recommendations["layers"].append("Waterproof rain jacket")
        recommendations["accessories"].append("Umbrella")
        recommendations["footwear"] = ["Waterproof boots or shoes"]
        if is_stormy:
            recommendations["tips"].append("Seek shelter during lightning")

    if is_snowy:
        recommendations["layers"].append("Waterproof outer layer")
        recommendations["footwear"] = ["Waterproof winter boots with traction"]
        recommendations["tips"].append("Watch for slippery surfaces")

    if is_windy:
        recommendations["accessories"].append("Windbreaker or wind-resistant layer")
        if is_very_windy:
            recommendations["tips"].append("Secure loose items; strong winds")

    if is_foggy:
        recommendations["tips"].append("Wear bright/reflective clothing for visibility")

    # UV protection
    if uv_index is not None:
        if uv_index >= 6:
            recommendations["accessories"].append("Sunglasses")
            recommendations["accessories"].append("Wide-brimmed hat")
            recommendations["tips"].append(
                f"UV Index {uv_index:.0f}: Apply SPF 30+ sunscreen"
            )
        elif uv_index >= 3:
            recommendations["accessories"].append("Sunglasses")
            recommendations["tips"].append(
                f"UV Index {uv_index:.0f}: Consider sunscreen"
            )

    # Humidity considerations
    if is_humid and temp_cat in ["warm", "hot"]:
        recommendations["tips"].append("High humidity: wear moisture-wicking fabrics")

    return recommendations


@click.command()
@click.argument("location", required=False)
@click.option(
    "-c", "--country", help="ISO 3166-1 alpha-2 country code (e.g., US, ES, DE)"
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def clothing(location: str | None, country: str | None, as_json: bool):
    """Get clothing suggestions based on current weather.

    Provides practical recommendations for what to wear based on
    temperature, precipitation, wind, and UV conditions.

    LOCATION can be a city name or a favorite alias (see 'raindrop fav list').
    """
    settings = get_settings()

    # Resolve location (favorites, defaults)
    try:
        resolved_location, resolved_country = settings.resolve_location(location)
    except ValueError:
        raise click.ClickException(
            "No location provided. Use 'raindrop clothing <location>' or set a default with 'raindrop config set location <name>'"
        )

    # CLI country flag overrides resolved country
    if country is not None:
        resolved_country = country

    location = resolved_location
    country = resolved_country

    result = geocode(location, country)

    weather = om.forecast(
        result.latitude,
        result.longitude,
        current=[
            "temperature_2m",
            "apparent_temperature",
            "weather_code",
            "wind_speed_10m",
            "relative_humidity_2m",
            "uv_index",
            "precipitation",
        ],
        temperature_unit=settings.temperature_unit,
        wind_speed_unit=settings.wind_speed_unit,
        forecast_days=1,
    )

    c = weather.current
    if c is None:
        raise click.ClickException("No weather data returned")

    temp_symbol = TEMP_SYMBOLS[settings.temperature_unit]
    weather_desc = WEATHER_CODES.get(c.weather_code or 0, "Unknown")

    recommendations = get_clothing_recommendations(
        temp=c.temperature_2m or 0,
        feels_like=c.apparent_temperature or c.temperature_2m or 0,
        weather_code=c.weather_code or 0,
        wind_speed=c.wind_speed_10m or 0,
        humidity=c.relative_humidity_2m or 0,
        uv_index=c.uv_index,
        unit=settings.temperature_unit,
    )

    # JSON output
    if as_json:
        data = {
            "location": {
                "name": result.name,
                "admin1": result.admin1,
                "country": result.country,
            },
            "conditions": {
                "temperature": c.temperature_2m,
                "feels_like": c.apparent_temperature,
                "weather_code": c.weather_code,
                "weather_description": weather_desc,
                "wind_speed": c.wind_speed_10m,
                "humidity": c.relative_humidity_2m,
                "uv_index": c.uv_index,
            },
            "recommendations": recommendations,
        }
        click.echo(json_lib.dumps(data, indent=2))
        return

    # Display
    console.print(f"\n[bold cyan]{result.name}, {result.admin1}[/bold cyan]")
    console.print(
        f"[dim]{c.temperature_2m:.0f}\u00b0{temp_symbol} (feels like {c.apparent_temperature:.0f}\u00b0{temp_symbol}) \u00b7 {weather_desc}[/dim]\n"
    )

    # Clothing panel
    console.print("[bold]What to Wear[/bold]\n")

    if recommendations["layers"]:
        console.print("[yellow]Clothing:[/yellow]")
        for item in recommendations["layers"]:
            console.print(f"  \u2022 {item}")
        console.print()

    if recommendations["accessories"]:
        console.print("[yellow]Accessories:[/yellow]")
        for item in recommendations["accessories"]:
            console.print(f"  \u2022 {item}")
        console.print()

    if recommendations["footwear"]:
        console.print("[yellow]Footwear:[/yellow]")
        for item in recommendations["footwear"]:
            console.print(f"  \u2022 {item}")
        console.print()

    if recommendations["tips"]:
        console.print("[bold cyan]Tips:[/bold cyan]")
        for tip in recommendations["tips"]:
            console.print(f"  [dim]\u2192[/dim] {tip}")
        console.print()
