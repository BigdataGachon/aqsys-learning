"""
Air Quality Training Data Generator
Based on: UCI Beijing PM2.5 dataset structure + Korea AirKorea feature conventions
Reference: https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data
           https://www.airkorea.or.kr

Features modeled after real Korean air quality patterns:
- Seasonal: yellow dust (황사) in spring, heating pollution in winter
- Daily: morning/evening traffic peaks
- Meteorological: temperature, humidity, wind, pressure
- Other pollutants: NO2, SO2, CO, O3 (correlated with PM)
"""

import random
import math
import csv
import json
from datetime import datetime, timedelta

random.seed(42)

# Seoul districts (구) with rough geographic characteristics
DISTRICTS = [
    {"name": "강남구", "lat": 37.5172, "lon": 127.0473, "altitude": 35, "traffic": 1.3},
    {"name": "종로구", "lat": 37.5735, "lon": 126.9788, "altitude": 42, "traffic": 1.1},
    {"name": "마포구", "lat": 37.5663, "lon": 126.9019, "altitude": 28, "traffic": 1.0},
    {"name": "강서구", "lat": 37.5509, "lon": 126.8495, "altitude": 20, "traffic": 1.1},
    {"name": "노원구", "lat": 37.6542, "lon": 127.0568, "altitude": 55, "traffic": 0.9},
]

# Wind directions (16-point compass encoded as degrees)
WIND_DIRS = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
             180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]


def season(month):
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:
        return "fall"


def season_index(month):
    return {"winter": 0, "spring": 1, "summer": 2, "fall": 3}[season(month)]


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def gauss(mean, std, lo=None, hi=None):
    v = random.gauss(mean, std)
    if lo is not None or hi is not None:
        v = clamp(v, lo if lo is not None else -1e9, hi if hi is not None else 1e9)
    return round(v, 2)


def temperature(month, hour):
    """Seasonal + diurnal temperature (°C), Seoul pattern."""
    season_mean = {1: -2, 2: 1, 3: 7, 4: 14, 5: 20, 6: 24,
                   7: 27, 8: 28, 9: 22, 10: 15, 11: 7, 12: 0}
    base = season_mean[month]
    # Diurnal swing: coolest at 5am, warmest at 2pm
    diurnal = 5 * math.sin(math.pi * (hour - 5) / 12) if 5 <= hour <= 17 else -2
    return round(base + diurnal + random.gauss(0, 1.5), 1)


def humidity(temp, season_name, hour):
    """Relative humidity (%), higher in summer, lower in winter."""
    base = {"winter": 55, "spring": 60, "summer": 78, "fall": 65}[season_name]
    # Humidity inversely correlated with temp
    h = base - 0.5 * temp + random.gauss(0, 5)
    return round(clamp(h, 20, 99), 1)


def wind_speed(season_name, hour):
    """Wind speed (m/s). Spring windiest (yellow dust), summer calmer."""
    base = {"winter": 3.2, "spring": 3.8, "summer": 2.5, "fall": 2.9}[season_name]
    # Slightly windier midday
    diurnal = 0.5 * math.sin(math.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0
    return round(max(0.1, random.gauss(base + diurnal, 1.2)), 1)


def pressure(season_name):
    """Atmospheric pressure (hPa). Higher in winter."""
    base = {"winter": 1022, "spring": 1015, "summer": 1008, "fall": 1016}[season_name]
    return round(random.gauss(base, 4), 1)


def dew_point(temp, rh):
    """Magnus formula approximation."""
    a, b = 17.27, 237.7
    gamma = (a * temp / (b + temp)) + math.log(rh / 100.0)
    dp = (b * gamma) / (a - gamma)
    return round(dp, 1)


def is_rush_hour(hour):
    return hour in (7, 8, 9, 17, 18, 19)


def is_weekend(weekday):  # weekday: 0=Mon, 6=Sun
    return weekday >= 5


def yellow_dust_factor(month, day):
    """Extra PM10 boost during 황사 season (March–May, random events)."""
    if month not in (3, 4, 5):
        return 1.0
    # ~15% chance of a yellow dust day in spring
    seed_val = month * 31 + day
    rng = random.Random(seed_val)
    if rng.random() < 0.15:
        return random.uniform(2.5, 5.0)  # yellow dust multiplier
    return 1.0


def pm_base(season_name, hour, weekday, district_traffic, yd_factor):
    """
    Base PM10 (μg/m³) before weather adjustment.
    Korean seasonal pattern:
      winter > spring > fall > summer
    """
    season_base = {"winter": 55, "spring": 50, "summer": 32, "fall": 40}[season_name]
    rush = 1.3 if is_rush_hour(hour) else 1.0
    weekend = 0.85 if is_weekend(weekday) else 1.0
    night_clean = 0.7 if hour in range(1, 6) else 1.0
    return season_base * rush * weekend * night_clean * district_traffic * yd_factor


def weather_pm_modifier(wind_spd, humidity_val, season_name):
    """Wind disperses PM; humidity captures fine particles."""
    wind_factor = max(0.4, 1.0 - 0.08 * wind_spd)
    if season_name == "summer":
        # Summer rain washes PM — high humidity = lower PM
        hum_factor = max(0.5, 1.0 - 0.004 * humidity_val)
    else:
        # Winter/spring: high humidity traps PM
        hum_factor = 0.8 + 0.004 * (humidity_val - 50)
    return wind_factor * hum_factor


def generate_pm(base_pm10, wind_spd, humidity_val, season_name):
    modifier = weather_pm_modifier(wind_spd, humidity_val, season_name)
    pm10 = max(5.0, base_pm10 * modifier + random.gauss(0, 5))
    # PM2.5 is roughly 50–65% of PM10 in Korean urban areas
    ratio = random.uniform(0.50, 0.65)
    pm25 = max(2.0, pm10 * ratio + random.gauss(0, 2))
    return round(pm10, 1), round(pm25, 1)


def o3(hour, temp, season_name):
    """O3 (ppm). Photochemical — peaks midday, higher in summer."""
    if hour < 6 or hour > 20:
        return round(random.gauss(0.01, 0.003), 4)
    solar_intensity = math.sin(math.pi * (hour - 6) / 14)
    season_factor = {"winter": 0.5, "spring": 0.8, "summer": 1.2, "fall": 0.9}[season_name]
    temp_factor = max(0.5, 0.7 + 0.02 * temp)
    val = 0.03 * solar_intensity * season_factor * temp_factor + random.gauss(0, 0.005)
    return round(clamp(val, 0.001, 0.25), 4)


def no2(hour, weekday, traffic_factor, season_name):
    """NO2 (ppm). Traffic-related."""
    season_factor = {"winter": 1.2, "spring": 1.0, "summer": 0.8, "fall": 1.0}[season_name]
    rush = 1.5 if is_rush_hour(hour) else 1.0
    weekend = 0.7 if is_weekend(weekday) else 1.0
    val = 0.02 * rush * weekend * traffic_factor * season_factor + random.gauss(0, 0.004)
    return round(clamp(val, 0.001, 0.2), 4)


def so2(season_name):
    """SO2 (ppm). Higher in winter (heating)."""
    base = {"winter": 0.006, "spring": 0.004, "summer": 0.003, "fall": 0.004}[season_name]
    return round(clamp(random.gauss(base, 0.002), 0.001, 0.05), 4)


def co(hour, weekday, traffic_factor, season_name):
    """CO (ppm). Traffic and heating."""
    season_factor = {"winter": 1.4, "spring": 1.0, "summer": 0.7, "fall": 1.0}[season_name]
    rush = 1.4 if is_rush_hour(hour) else 1.0
    weekend = 0.75 if is_weekend(weekday) else 1.0
    val = 0.4 * rush * weekend * traffic_factor * season_factor + random.gauss(0, 0.08)
    return round(clamp(val, 0.1, 5.0), 3)


def air_quality_level(pm10_val, pm25_val):
    """Korea AirKorea standard levels."""
    if pm10_val <= 30 and pm25_val <= 15:
        return "green"
    elif pm10_val <= 80 and pm25_val <= 35:
        return "yellow"
    elif pm10_val <= 150 and pm25_val <= 75:
        return "red"
    else:
        return "purple"


def generate_dataset(start_date, end_date, output_csv="air_quality_train.csv",
                     output_json="air_quality_train_sample.json"):
    """Generate hourly air quality records for all districts between two dates."""

    fieldnames = [
        "date", "year", "month", "day", "hour", "day_of_week", "season",
        "district", "lat", "lon", "altitude",
        "temperature", "humidity", "dew_point",
        "wind_speed", "wind_direction", "pressure",
        "no2", "so2", "co", "o3",
        "pm10", "pm25",
        "level",
        # Lag features (previous-hour values — filled as 0 initially; real pipeline uses sliding window)
        "pm10_lag1", "pm25_lag1",
        "pm10_lag3", "pm25_lag3",
        "pm10_lag24", "pm25_lag24",
    ]

    rows = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Per-district rolling history for lag features  {district_name: deque of (pm10, pm25)}
    history = {d["name"]: [] for d in DISTRICTS}

    print(f"Generating from {start_date} to {end_date}...")

    while current <= end:
        month = current.month
        day = current.day
        year = current.year
        weekday = current.weekday()  # 0=Mon
        season_name = season(month)
        yd = yellow_dust_factor(month, day)

        for hour in range(24):
            for district in DISTRICTS:
                dname = district["name"]
                traffic = district["traffic"]

                temp = temperature(month, hour)
                hum = humidity(temp, season_name, hour)
                dp = dew_point(temp, hum)
                ws = wind_speed(season_name, hour)
                wd = random.choice(WIND_DIRS)
                pres = pressure(season_name)

                base = pm_base(season_name, hour, weekday, traffic, yd)
                pm10_val, pm25_val = generate_pm(base, ws, hum, season_name)

                o3_val = o3(hour, temp, season_name)
                no2_val = no2(hour, weekday, traffic, season_name)
                so2_val = so2(season_name)
                co_val = co(hour, weekday, traffic, season_name)

                level = air_quality_level(pm10_val, pm25_val)

                hist = history[dname]

                def lag(n, idx):  # idx 0=pm10, 1=pm25
                    if len(hist) >= n:
                        return hist[-n][idx]
                    return 0.0

                row = {
                    "date": current.strftime("%Y-%m-%d"),
                    "year": year,
                    "month": month,
                    "day": day,
                    "hour": hour,
                    "day_of_week": weekday,
                    "season": season_index(month),
                    "district": dname,
                    "lat": district["lat"],
                    "lon": district["lon"],
                    "altitude": district["altitude"],
                    "temperature": temp,
                    "humidity": hum,
                    "dew_point": dp,
                    "wind_speed": ws,
                    "wind_direction": wd,
                    "pressure": pres,
                    "no2": no2_val,
                    "so2": so2_val,
                    "co": co_val,
                    "o3": o3_val,
                    "pm10": pm10_val,
                    "pm25": pm25_val,
                    "level": level,
                    "pm10_lag1": lag(1, 0),
                    "pm25_lag1": lag(1, 1),
                    "pm10_lag3": lag(3, 0),
                    "pm25_lag3": lag(3, 1),
                    "pm10_lag24": lag(24, 0),
                    "pm25_lag24": lag(24, 1),
                }
                rows.append(row)
                hist.append((pm10_val, pm25_val))
                if len(hist) > 48:  # keep 48h rolling window
                    hist.pop(0)

        current += timedelta(days=1)

    # Write CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Write a small JSON sample (first 72 rows = 3 days × 1 district)
    sample = [r for r in rows if r["district"] == "강남구"][:72]
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    total_days = (datetime.strptime(end_date, "%Y-%m-%d") - datetime.strptime(start_date, "%Y-%m-%d")).days + 1
    print(f"Done. {len(rows):,} records ({total_days} days × 24 hours × {len(DISTRICTS)} districts)")
    print(f"  → {output_csv}")
    print(f"  → {output_json} (sample: 72 rows, 강남구)")

    # Print feature summary
    pm10_vals = [r["pm10"] for r in rows]
    pm25_vals = [r["pm25"] for r in rows]
    print(f"\nPM10  range: {min(pm10_vals):.1f} ~ {max(pm10_vals):.1f}  μg/m³  "
          f"(mean {sum(pm10_vals)/len(pm10_vals):.1f})")
    print(f"PM25  range: {min(pm25_vals):.1f} ~ {max(pm25_vals):.1f}  μg/m³  "
          f"(mean {sum(pm25_vals)/len(pm25_vals):.1f})")
    levels = {}
    for r in rows:
        levels[r["level"]] = levels.get(r["level"], 0) + 1
    print(f"\nLevel distribution: {levels}")
    print(f"\nFeature columns ({len(fieldnames)}):")
    for fn in fieldnames:
        print(f"  {fn}")


if __name__ == "__main__":
    generate_dataset(
        start_date="2024-01-01",
        end_date="2025-12-31",   # 2 years
        output_csv="air_quality_train.csv",
        output_json="air_quality_train_sample.json",
    )
