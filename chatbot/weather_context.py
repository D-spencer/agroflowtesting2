"""
AgroFlow AI Weather Context

Converts structured weather data into a compact,
agriculture-focused context for the AgroFlow AI.
"""

from chatbot.logger import logger
from chatbot.weather_codes import get_weather_condition


def _get_condition(data, weather_code=None):

    condition = data.get("condition")

    if condition:
        return condition

    if weather_code is not None:

        return get_weather_condition(
            weather_code
        )

    return "Unknown"


def _safe_number(value):

    return isinstance(
        value,
        (int, float)
    )


def _build_current_interpretation(current):

    observations = []

    temperature = current.get(
        "temperature"
    )

    humidity = current.get(
        "humidity"
    )

    rain = current.get(
        "rain"
    )

    precipitation = current.get(
        "precipitation"
    )

    wind_speed = current.get(
        "wind_speed"
    )

    condition = current.get(
        "condition",
        "Unknown"
    )

    if _safe_number(rain) and rain > 0:

        observations.append(
            "Rain is currently occurring or has recently occurred."
        )

    elif (
        _safe_number(precipitation)
        and precipitation > 0
    ):

        observations.append(
            "Measurable precipitation has been recorded."
        )

    if _safe_number(wind_speed):

        if wind_speed >= 20:

            observations.append(
                "Wind is relatively strong and may increase "
                "spray drift risk."
            )

        elif wind_speed >= 10:

            observations.append(
                "Moderate wind is present and should be "
                "considered for spraying."
            )

        else:

            observations.append(
                "Wind is relatively light."
            )

    if _safe_number(humidity):

        if humidity >= 85:

            observations.append(
                "Humidity is high."
            )

        elif humidity <= 40:

            observations.append(
                "Humidity is relatively low."
            )

    if _safe_number(temperature):

        if temperature >= 32:

            observations.append(
                "Temperature is high and may increase "
                "evaporation and heat stress."
            )

        elif temperature <= 15:

            observations.append(
                "Temperature is relatively low."
            )

    if condition:

        observations.append(
            f"Current condition: {condition}."
        )

    return observations


def _build_daily_interpretation(day):

    observations = []

    condition = day.get(
        "condition",
        "Unknown"
    )

    temperature_max = day.get(
        "temperature_max"
    )

    temperature_min = day.get(
        "temperature_min"
    )

    precipitation = day.get(
        "precipitation"
    )

    rain = day.get(
        "rain"
    )

    rain_probability = day.get(
        "rain_probability"
    )

    max_wind_speed = day.get(
        "max_wind_speed"
    )

    if _safe_number(rain_probability):

        if rain_probability >= 80:

            observations.append(
                "High probability of precipitation."
            )

        elif rain_probability >= 60:

            observations.append(
                "Moderate-to-high probability of precipitation."
            )

        elif rain_probability >= 30:

            observations.append(
                "Some precipitation is possible."
            )

        else:

            observations.append(
                "Low precipitation probability."
            )

    if _safe_number(rain):

        if rain >= 10:

            observations.append(
                "Forecast rain amount is substantial."
            )

        elif rain > 0:

            observations.append(
                "Rain is expected."
            )

    elif _safe_number(precipitation):

        if precipitation >= 10:

            observations.append(
                "Forecast precipitation is substantial."
            )

        elif precipitation > 0:

            observations.append(
                "Some precipitation is expected."
            )

    if _safe_number(max_wind_speed):

        if max_wind_speed >= 20:

            observations.append(
                "Strong winds may increase spray drift risk."
            )

        elif max_wind_speed >= 10:

            observations.append(
                "Moderate winds should be considered "
                "when planning spraying."
            )

        else:

            observations.append(
                "Wind is expected to remain relatively light."
            )

    if _safe_number(temperature_max):

        if temperature_max >= 32:

            observations.append(
                "High daytime temperature may increase "
                "evaporation and heat stress."
            )

    if _safe_number(temperature_min):

        if temperature_min <= 15:

            observations.append(
                "Low overnight temperature is expected."
            )

    if condition:

        observations.append(
            f"Expected condition: {condition}."
        )

    return observations


def build_weather_context(
    weather_data
):

    if not weather_data:

        logger.warning(
            "No weather data available."
        )

        return ""

    location = weather_data.get(
        "location",
        {}
    )

    latitude = location.get(
        "latitude",
        "Unknown"
    )

    longitude = location.get(
        "longitude",
        "Unknown"
    )

    timezone = location.get(
        "timezone",
        "Unknown"
    )

    current = weather_data.get(
        "current",
        {}
    )

    current_time = current.get(
        "time",
        "Unknown"
    )

    temperature = current.get(
        "temperature",
        "Unknown"
    )

    feels_like = current.get(
        "feels_like",
        "Unknown"
    )

    humidity = current.get(
        "humidity",
        "Unknown"
    )

    precipitation = current.get(
        "precipitation",
        "Unknown"
    )

    rain = current.get(
        "rain",
        "Unknown"
    )

    weather_code = current.get(
        "weather_code"
    )

    condition = _get_condition(
        current,
        weather_code
    )

    wind_speed = current.get(
        "wind_speed",
        "Unknown"
    )

    context = f"""
AGROFLOW WEATHER INFORMATION

LOCATION

Latitude: {latitude}
Longitude: {longitude}
Timezone: {timezone}


CURRENT WEATHER

Time: {current_time}
Temperature: {temperature}°C
Feels like: {feels_like}°C
Humidity: {humidity}%
Condition: {condition}
Precipitation: {precipitation} mm
Rain: {rain} mm
Wind speed: {wind_speed} km/h


CURRENT AGRICULTURAL OBSERVATIONS
"""

    current_observations = _build_current_interpretation(
        current
    )

    if current_observations:

        for observation in current_observations:

            context += (
                f"- {observation}\n"
            )

    else:

        context += (
            "- No specific current weather risk identified.\n"
        )

    context += """

7-DAY WEATHER FORECAST
"""

    daily_forecast = weather_data.get(
        "daily",
        []
    )

    for day in daily_forecast:

        date = day.get(
            "date",
            "Unknown"
        )

        weather_code = day.get(
            "weather_code"
        )

        condition = _get_condition(
            day,
            weather_code
        )

        temperature_max = day.get(
            "temperature_max",
            "Unknown"
        )

        temperature_min = day.get(
            "temperature_min",
            "Unknown"
        )

        precipitation = day.get(
            "precipitation",
            "Unknown"
        )

        rain = day.get(
            "rain",
            "Unknown"
        )

        rain_probability = day.get(
            "rain_probability",
            "Unknown"
        )

        max_wind_speed = day.get(
            "max_wind_speed",
            "Unknown"
        )

        context += f"""

{date}

Condition: {condition}
Temperature: {temperature_min}°C - {temperature_max}°C
Precipitation: {precipitation} mm
Rain: {rain} mm
Rain probability: {rain_probability}%
Maximum wind speed: {max_wind_speed} km/h

Agricultural observations:
"""

        daily_observations = _build_daily_interpretation(
            day
        )

        if daily_observations:

            for observation in daily_observations:

                context += (
                    f"- {observation}\n"
                )

        else:

            context += (
                "- No specific weather risk identified.\n"
            )

    context += """

AGRICULTURAL WEATHER INTERPRETATION GUIDE

SPRAYING

- Rain occurring shortly after spraying can reduce
  product effectiveness or wash products from plant
  surfaces.
- Higher wind speeds can increase spray drift.
- Temperature and humidity can affect evaporation
  and spray performance.
- Actual product requirements must follow the
  product label.
- Do not determine spraying suitability from a
  single weather variable alone.
- Consider rainfall probability, expected rain,
  wind, temperature, humidity, and timing.
- When the user asks whether spraying is suitable,
  use the forecast for the specific requested date.

PLANTING

- Rainfall and soil moisture can affect planting
  conditions.
- Excessive rainfall can contribute to waterlogged
  fields.
- Weather conditions should also be considered
  alongside soil condition and field accessibility.

IRRIGATION

- Expected rainfall should be considered before
  irrigation.
- Higher expected rainfall may reduce the need
  for irrigation.
- Do not assume rainfall will completely replace
  irrigation without considering crop and soil needs.

HARVESTING

- Rain can interfere with harvesting operations.
- Consecutive wet days can make field access
  more difficult.
- Crop-specific harvest requirements should be
  considered separately.

DRYING

- Rain and high humidity can slow crop drying.
- Several consecutive wet days may make outdoor
  drying more difficult.

GENERAL RULES

- Use the actual forecast values supplied above.
- Do not invent weather conditions.
- Do not invent forecast values.
- Distinguish forecast information from agricultural
  interpretation.
- Do not claim certainty when weather forecasts are
  uncertain.
- Do not make product-specific recommendations
  without the product label or appropriate guidance.
- Weather alone may not be sufficient to make a
  final agricultural decision.
"""

    logger.info(
        "Smart agricultural weather context "
        "built successfully."
    )

    return context.strip()