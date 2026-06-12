# Recreated logger variant for thesis repository. Same structure as Starlink logger.
import csv
import json
import os
import re
import subprocess
from datetime import datetime
import requests

# =======================
# OMANTEL CONFIG
# =======================
NETWORK_TYPE = "Omantel"
PING_TARGET = "1.1.1.1"
PING_COUNT = 50

LOG_FILE = r"C:\network_logger\data\omantel_data.csv"

# Muscat weather (Open-Meteo, free, no key)
LATITUDE = 23.5880
LONGITUDE = 58.3829


def run_cmd(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_windows_ping(text):
    rtts = [float(m.group(1)) for m in re.finditer(r"time[=<]\s*(\d+)\s*ms", text)]
    m_loss = re.search(r"\((\d+)%\s*loss\)", text)
    loss = float(m_loss.group(1)) if m_loss else None

    if not rtts:
        return None, None, loss if loss is not None else 100.0

    avg = sum(rtts) / len(rtts)
    jitter = (sum(abs(rtts[i] - rtts[i - 1]) for i in range(1, len(rtts))) / (len(rtts) - 1)) if len(rtts) > 1 else 0.0

    return round(avg, 2), round(jitter, 2), round(loss if loss is not None else 0.0, 2)


def run_ping():
    res = run_cmd(["ping", PING_TARGET, "-n", str(PING_COUNT)])
    return parse_windows_ping(res.stdout)


def run_speedtest():
    res = run_cmd(["speedtest", "-f", "json", "--accept-license", "--accept-gdpr", "--progress=no"])
    if res.returncode != 0 or not res.stdout.strip().startswith("{"):
        return None, None, None, None, None

    data = json.loads(res.stdout)
    st_ping = data.get("ping", {}).get("latency", None)

    down_bw = data.get("download", {}).get("bandwidth", None)
    up_bw = data.get("upload", {}).get("bandwidth", None)

    down_mbps = (down_bw * 8 / 1_000_000) if down_bw else None
    up_mbps = (up_bw * 8 / 1_000_000) if up_bw else None

    server = data.get("server", {})
    return (
        None if st_ping is None else round(float(st_ping), 2),
        None if down_mbps is None else round(float(down_mbps), 2),
        None if up_mbps is None else round(float(up_mbps), 2),
        server.get("name", None),
        server.get("location", None)
    )


def get_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code",
        "timezone": "Asia/Muscat",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        cur = r.json().get("current", {})

        temp = cur.get("temperature_2m", None)
        hum = cur.get("relative_humidity_2m", None)
        wind_kmh = cur.get("wind_speed_10m", None)
        precip = cur.get("precipitation", None)
        code = cur.get("weather_code", None)

        wind_mps = (float(wind_kmh) / 3.6) if wind_kmh is not None else None

        def r2(x):
            return None if x is None else round(float(x), 2)

        return r2(temp), r2(hum), r2(wind_mps), r2(precip), code
    except Exception:
        return None, None, None, None, None


def ensure_header(path, header):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def append_row(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def main():
    ts = datetime.now().isoformat(timespec="seconds")

    ping_avg, ping_jitter, loss = run_ping()
    st_ping, down, up, server_name, server_loc = run_speedtest()
    temp, hum, wind, precip, wcode = get_weather()

    header = [
        "timestamp","network_type","ping_target","ping_avg_rtt_ms","ping_jitter_ms","packet_loss_percent",
        "speedtest_ping_ms","download_mbps","upload_mbps","speedtest_server_name","speedtest_server_location",
        "temperature_C","humidity_percent","wind_speed_mps","precipitation_mm","weather_code"
    ]

    row = [
        ts, NETWORK_TYPE, PING_TARGET, ping_avg, ping_jitter, loss,
        st_ping, down, up, server_name, server_loc,
        temp, hum, wind, precip, wcode
    ]

    ensure_header(LOG_FILE, header)
    append_row(LOG_FILE, row)

    print("Logged:", ts)


if __name__ == "__main__":
    main()