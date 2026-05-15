"""
live_logger.py
--------------
Logs one row of real network measurements to Raw/starlink_data.csv
every time it runs. Schedule this with cron every 15 minutes.

It measures:
  - Ping latency, jitter, packet loss  (to 1.1.1.1)
  - Download and upload speed          (via speedtest-cli)
  - Weather context                    (via Open-Meteo, no API key needed)

After logging, it automatically re-runs the cleaning pipeline so the
dashboard always reflects the latest data.
"""

import subprocess
import csv
import os
import re
import sys
import json
import datetime
import urllib.request

# ======================
# CONFIGURATION
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root is two levels up from src/collection/
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))

# Raw data file
RAW_FILE = os.path.join(PROJECT_ROOT, "Raw", "experiment_A", "starlink_data.csv")

# Retrain queue file
RETRAIN_QUEUE_FILE = os.path.join(
    PROJECT_ROOT,
    "Cleaned",
    "state",
    "starlink_retrain_queue.csv"
)

# Portable speedtest-cli resolution
SPEEDTEST_CMD = os.path.join(os.path.dirname(sys.executable), "speedtest-cli")
if not os.path.exists(SPEEDTEST_CMD):
    SPEEDTEST_CMD = "speedtest-cli"

PING_TARGET = "1.1.1.1"
PING_COUNT = 20
NETWORK_TYPE = "Starlink"
ISP_NAME = "SpaceX Starlink"
LOCATION = "Muscat"
LATITUDE = 23.5880
LONGITUDE = 58.3829

# ======================
# STEP 1: PING
# ======================
def run_ping():
    try:
        result = subprocess.run(
            ["/sbin/ping", "-c", str(PING_COUNT), PING_TARGET],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout

        loss_match = re.search(r"(\d+\.?\d*)% packet loss", output)
        packet_loss = float(loss_match.group(1)) if loss_match else 0.0

        rtt_match = re.search(
            r"min/avg/max/stddev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)",
            output
        )

        if rtt_match:
            ping_min = float(rtt_match.group(1))
            ping_avg = float(rtt_match.group(2))
            ping_max = float(rtt_match.group(3))
            ping_jitter = float(rtt_match.group(4))
        else:
            ping_min = ping_avg = ping_max = ping_jitter = None

        return {
            "ping_target": PING_TARGET,
            "ping_min_rtt_ms": ping_min,
            "ping_avg_rtt_ms": ping_avg,
            "ping_max_rtt_ms": ping_max,
            "ping_jitter_ms": ping_jitter,
            "packet_loss_pct": packet_loss
        }

    except Exception as e:
        print(f"Ping error: {e}")
        return {
            "ping_target": PING_TARGET,
            "ping_min_rtt_ms": None,
            "ping_avg_rtt_ms": None,
            "ping_max_rtt_ms": None,
            "ping_jitter_ms": None,
            "packet_loss_pct": None
        }

# ======================
# STEP 2: SPEED TEST
# ======================
def run_speedtest():
    try:
        result = subprocess.run(
            [SPEEDTEST_CMD, "--json"],
            capture_output=True,
            text=True,
            timeout=120
        )

        data = json.loads(result.stdout)

        download_mbps = round(data["download"] / 1_000_000, 2)
        upload_mbps = round(data["upload"] / 1_000_000, 2)
        speedtest_ping_ms = data.get("ping", None)

        isp = data.get("client", {}).get("isp", ISP_NAME)
        server_name = data.get("server", {}).get("name", LOCATION)

        return {
            "speedtest_ping_ms": speedtest_ping_ms,
            "download_mbps": download_mbps,
            "upload_mbps": upload_mbps,
            "isp": isp,
            "server_name": server_name
        }

    except Exception as e:
        print(f"Speedtest error: {e}")
        return {
            "speedtest_ping_ms": None,
            "download_mbps": None,
            "upload_mbps": None,
            "isp": ISP_NAME,
            "server_name": LOCATION
        }

# ======================
# STEP 3: WEATHER
# ======================
def get_weather():
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={LATITUDE}&longitude={LONGITUDE}"
            f"&current_weather=true"
            f"&hourly=relativehumidity_2m"
            f"&forecast_days=1"
        )

        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read())

        current = data.get("current_weather", {})
        temp = current.get("temperature", None)
        wind = current.get("windspeed", None)
        weather_code = current.get("weathercode", None)

        hourly = data.get("hourly", {})
        humidity_list = hourly.get("relativehumidity_2m", [None])
        humidity = humidity_list[0] if humidity_list else None

        return {
            "temperature_c": temp,
            "humidity_percent": humidity,
            "wind_speed_mps": round(wind / 3.6, 2) if wind else None,
            "weather_code": weather_code
        }

    except Exception as e:
        print(f"Weather error: {e}")
        return {
            "temperature_c": None,
            "humidity_percent": None,
            "wind_speed_mps": None,
            "weather_code": None
        }

# ======================
# STEP 4: WRITE ROW
# ======================
def write_row(timestamp, ping_data, speed_data, weather_data):
    fieldnames = [
        "timestamp",
        "network_type",
        "ping_target",
        "ping_avg_rtt_ms",
        "ping_jitter_ms",
        "packet_loss_percent",
        "speedtest_ping_ms",
        "download_mbps",
        "upload_mbps",
        "speedtest_server_name",
        "speedtest_server_location",
        "temperature_C",
        "humidity_percent",
        "wind_speed_mps",
        "precipitation_mm",
        "weather_code"
    ]

    row = {
        "timestamp": timestamp,
        "network_type": NETWORK_TYPE,
        "ping_target": ping_data.get("ping_target", PING_TARGET),
        "ping_avg_rtt_ms": ping_data.get("ping_avg_rtt_ms"),
        "ping_jitter_ms": ping_data.get("ping_jitter_ms"),
        "packet_loss_percent": ping_data.get("packet_loss_pct"),
        "speedtest_ping_ms": speed_data.get("speedtest_ping_ms"),
        "download_mbps": speed_data.get("download_mbps"),
        "upload_mbps": speed_data.get("upload_mbps"),
        "speedtest_server_name": speed_data.get("server_name"),
        "speedtest_server_location": speed_data.get("isp"),
        "temperature_C": weather_data.get("temperature_c"),
        "humidity_percent": weather_data.get("humidity_percent"),
        "wind_speed_mps": weather_data.get("wind_speed_mps"),
        "precipitation_mm": None,
        "weather_code": weather_data.get("weather_code")
    }

    # Write to raw master file
    os.makedirs(os.path.dirname(RAW_FILE), exist_ok=True)
    raw_exists = os.path.isfile(RAW_FILE)

    with open(RAW_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not raw_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged: {timestamp}")

    # Write the same new row to retrain queue
    os.makedirs(os.path.dirname(RETRAIN_QUEUE_FILE), exist_ok=True)
    queue_exists = os.path.isfile(RETRAIN_QUEUE_FILE)

    with open(RETRAIN_QUEUE_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not queue_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Queued for retraining: {timestamp}")

# ======================
# STEP 5: RUN PIPELINE
# ======================
def run_pipeline():
    print("Running pipeline...")

    scripts = [
        os.path.join("archive", "cleaning", "clean_data_fixed.py")
    ]

    for script in scripts:
        path = os.path.join(PROJECT_ROOT, script)

        if os.path.exists(path):
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )

            if result.returncode != 0:
                print(f"Error in {script}: {result.stderr}")
            else:
                print(f"Done: {script}")
        else:
            print(f"Skipped (not found): {script}")

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    now = datetime.datetime.now()

    # Round down to nearest 15-minute interval
    minutes = (now.minute // 15) * 15
    timestamp = now.replace(minute=minutes, second=0, microsecond=0)
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n--- Starting log at {timestamp_str} ---")

    ping_data = run_ping()
    print(f"Ping avg: {ping_data['ping_avg_rtt_ms']} ms")

    speed_data = run_speedtest()
    print(
        f"Download: {speed_data['download_mbps']} Mbps | "
        f"Upload: {speed_data['upload_mbps']} Mbps"
    )

    weather_data = get_weather()
    print(
        f"Temp: {weather_data['temperature_c']} C | "
        f"Weather code: {weather_data['weather_code']}"
    )

    write_row(timestamp_str, ping_data, speed_data, weather_data)

    run_pipeline()

    print("--- Done ---\n")