"""
AgroFlow AI Weather Service

Retrieves weather data from Open-Meteo and converts it
into structured, agriculture-focused weather information.
"""

import httpx

from chatbot.logger import logger
from chatbot.weather_codes import get_weather_condition


OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


def get_weather(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
):

    logger.info(
        "Requesting weather data for "
        "latitude=%s, longitude=%s",
        latitude,
        longitude
    )

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "rain,"
            "weather_code,"
            "wind_speed_10m"
        ),

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation_probability,"
            "precipitation,"
            "rain,"
            "weather_code,"
            "wind_speed_10m"
        ),

        "daily": (
            "weather_code,"
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "rain_sum,"
            "precipitation_probability_max,"
            "wind_speed_10m_max"
        ),

        "forecast_days": forecast_days,

        "timezone": "auto"
    }

    try:

        response = httpx.get(
            OPEN_METEO_URL,
            params=params,
            timeout=10.0
        )

        response.raise_for_status()

        weather_data = response.json()

        logger.info(
            "Weather data retrieved successfully."
        )

        return weather_data

    except httpx.HTTPError:

        logger.exception(
            "Weather API request failed."
        )

        return None

    except Exception:

        logger.exception(
            "Unexpected error while retrieving weather."
        )

        return None


def format_current_weather(
    weather_data
):

    if not weather_data:

        return None

    current = weather_data.get(
        "current",
        {}
    )

    weather_code = current.get(
        "weather_code"
    )

    return {

        "time": current.get(
            "time"
        ),

        "temperature": current.get(
            "temperature_2m"
        ),

        "feels_like": current.get(
            "apparent_temperature"
        ),

        "humidity": current.get(
            "relative_humidity_2m"
        ),

        "precipitation": current.get(
            "precipitation"
        ),

        "rain": current.get(
            "rain"
        ),

        "weather_code": weather_code,

        "condition": get_weather_condition(
            weather_code
        ),

        "wind_speed": current.get(
            "wind_speed_10m"
        )
    }


def format_daily_forecast(
    weather_data
):

    if not weather_data:

        return []

    daily = weather_data.get(
        "daily",
        {}
    )

    dates = daily.get(
        "time",
        []
    )

    weather_codes = daily.get(
        "weather_code",
        []
    )

    max_temperatures = daily.get(
        "temperature_2m_max",
        []
    )

    min_temperatures = daily.get(
        "temperature_2m_min",
        []
    )

    precipitation = daily.get(
        "precipitation_sum",
        []
    )

    rain = daily.get(
        "rain_sum",
        []
    )

    precipitation_probability = daily.get(
        "precipitation_probability_max",
        []
    )

    wind_speed = daily.get(
        "wind_speed_10m_max",
        []
    )

    forecast = []

    for index, date in enumerate(dates):

        weather_code = (
            weather_codes[index]
            if index < len(weather_codes)
            else None
        )

        forecast.append({

            "date": date,

            "weather_code": weather_code,

            "condition": get_weather_condition(
                weather_code
            ),

            "temperature_max": (
                max_temperatures[index]
                if index < len(max_temperatures)
                else None
            ),

            "temperature_min": (
                min_temperatures[index]
                if index < len(min_temperatures)
                else None
            ),

            "precipitation": (
                precipitation[index]
                if index < len(precipitation)
                else None
            ),

            "rain": (
                rain[index]
                if index < len(rain)
                else None
            ),

            "rain_probability": (
                precipitation_probability[index]
                if index < len(
                    precipitation_probability
                )
                else None
            ),

            "max_wind_speed": (
                wind_speed[index]
                if index < len(wind_speed)
                else None
            )
        })

    return forecast


def format_hourly_forecast(
    weather_data
):

    if not weather_data:

        return []

    hourly = weather_data.get(
        "hourly",
        {}
    )

    times = hourly.get(
        "time",
        []
    )

    temperatures = hourly.get(
        "temperature_2m",
        []
    )

    humidity = hourly.get(
        "relative_humidity_2m",
        []
    )

    precipitation_probability = hourly.get(
        "precipitation_probability",
        []
    )

    precipitation = hourly.get(
        "precipitation",
        []
    )

    rain = hourly.get(
        "rain",
        []
    )

    weather_codes = hourly.get(
        "weather_code",
        []
    )

    wind_speed = hourly.get(
        "wind_speed_10m",
        []
    )

    forecast = []

    for index, time in enumerate(times):

        weather_code = (
            weather_codes[index]
            if index < len(weather_codes)
            else None
        )

        forecast.append({

            "time": time,

            "temperature": (
                temperatures[index]
                if index < len(temperatures)
                else None
            ),

            "humidity": (
                humidity[index]
                if index < len(humidity)
                else None
            ),

            "rain_probability": (
                precipitation_probability[index]
                if index < len(
                    precipitation_probability
                )
                else None
            ),

            "precipitation": (
                precipitation[index]
                if index < len(precipitation)
                else None
            ),

            "rain": (
                rain[index]
                if index < len(rain)
                else None
            ),

            "weather_code": weather_code,

            "condition": get_weather_condition(
                weather_code
            ),

            "wind_speed": (
                wind_speed[index]
                if index < len(wind_speed)
                else None
            )
        })

    return forecast


def analyze_agricultural_weather(
    weather_data
):

    if not weather_data:

        return None

    current = format_current_weather(
        weather_data
    )

    daily = format_daily_forecast(
        weather_data
    )

    if not current:

        return None

    temperature = current.get(
        "temperature"
    )

    humidity = current.get(
        "humidity"
    )

    rain = current.get(
        "rain"
    ) or 0

    wind_speed = current.get(
        "wind_speed"
    ) or 0

    condition = current.get(
        "condition",
        "Unknown"
    )

    if rain > 5:

        rain_status = "Heavy rain"

    elif rain > 0:

        rain_status = "Rain occurring"

    else:

        rain_status = "No rain currently"

    if wind_speed >= 30:

        wind_status = "Very strong winds"

    elif wind_speed >= 20:

        wind_status = "Strong winds"

    elif wind_speed >= 10:

        wind_status = "Moderate winds"

    else:

        wind_status = "Light winds"

    if temperature is None:

        temperature_status = "Unknown"

    elif temperature >= 35:

        temperature_status = "Very hot"

    elif temperature >= 30:

        temperature_status = "Hot"

    elif temperature >= 20:

        temperature_status = "Moderate"

    elif temperature >= 10:

        temperature_status = "Cool"

    else:

        temperature_status = "Cold"

    if humidity is None:

        humidity_status = "Unknown"

    elif humidity >= 85:

        humidity_status = "Very humid"

    elif humidity >= 70:

        humidity_status = "Humid"

    elif humidity >= 50:

        humidity_status = "Moderate humidity"

    else:

        humidity_status = "Low humidity"

    field_work_status = "Suitable"

    field_work_reasons = []

    if rain > 0:

        field_work_status = "Less suitable"

        field_work_reasons.append(
            "Rain is occurring."
        )

    if wind_speed >= 20:

        field_work_status = "Less suitable"

        field_work_reasons.append(
            "Winds are strong."
        )

    forecast_rain_probability = []

    heavy_rain_days = []

    for day in daily:

        probability = day.get(
            "rain_probability"
        )

        precipitation = day.get(
            "precipitation"
        )

        if probability is not None:

            forecast_rain_probability.append(
                probability
            )

        if (
            precipitation is not None
            and precipitation >= 10
        ):

            heavy_rain_days.append(
                day.get("date")
            )

    if forecast_rain_probability:

        maximum_rain_probability = max(
            forecast_rain_probability
        )

    else:

        maximum_rain_probability = None

    if heavy_rain_days:

        rainfall_outlook = (
            "Significant rainfall is forecast "
            "on some days."
        )

    elif (
        maximum_rain_probability is not None
        and maximum_rain_probability >= 70
    ):

        rainfall_outlook = (
            "High probability of rainfall "
            "during the forecast period."
        )

    else:

        rainfall_outlook = (
            "No major rainfall signal detected "
            "during the forecast period."
        )

    alerts = []

    weather_code = current.get(
        "weather_code"
    )

    if weather_code in {
        95,
        96,
        99
    }:

        alerts.append(
            "Thunderstorm conditions are currently "
            "present or indicated."
        )

    if wind_speed >= 30:

        alerts.append(
            "Very strong winds may affect outdoor "
            "farm activities."
        )

    if (
        temperature is not None
        and temperature >= 35
    ):

        alerts.append(
            "Very high temperatures may increase "
            "heat and water stress."
        )

    if (
        humidity is not None
        and humidity >= 90
    ):

        alerts.append(
            "Very high humidity may increase the "
            "risk of some moisture-related crop diseases."
        )

    return {

        "current_condition": condition,

        "temperature_status": temperature_status,

        "humidity_status": humidity_status,

        "wind_status": wind_status,

        "rain_status": rain_status,

        "field_work_status": field_work_status,

        "field_work_reasons": field_work_reasons,

        "rainfall_outlook": rainfall_outlook,

        "maximum_forecast_rain_probability": (
            maximum_rain_probability
        ),

        "heavy_rain_days": heavy_rain_days,

        "alerts": alerts
    }


def build_weather_response(
    weather_data
):

    if not weather_data:

        return None

    return {

        "location": {

            "latitude": weather_data.get(
                "latitude"
            ),

            "longitude": weather_data.get(
                "longitude"
            ),

            "timezone": weather_data.get(
                "timezone"
            )
        },

        "current": format_current_weather(
            weather_data
        ),

        "daily": format_daily_forecast(
            weather_data
        ),

        "hourly": format_hourly_forecast(
            weather_data
        ),

        "agricultural_analysis": (
            analyze_agricultural_weather(
                weather_data
            )
        )
    }


def get_formatted_weather(
    latitude: float,
    longitude: float,
    forecast_days: int = 7
):

    weather_data = get_weather(
        latitude=latitude,
        longitude=longitude,
        forecast_days=forecast_days
    )

    if not weather_data:

        return None

    return build_weather_response(
        weather_data
    )