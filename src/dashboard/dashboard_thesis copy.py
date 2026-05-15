import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Starlink Thesis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

refresh_seconds = 60
st_autorefresh(interval=refresh_seconds * 1000, key="thesis_refresh")

# ============================================================
# LOGIN
# ============================================================
from security_config import USERS

def check_login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.role = None
    if not st.session_state.authenticated:
        st.markdown("## Starlink Thesis Dashboard")
        st.markdown("Please log in to continue.")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in USERS and USERS[username]["password"] == password:
                st.session_state.authenticated = True
                st.session_state.role = USERS[username]["role"]
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.stop()

check_login()

# ============================================================
# FILE PATHS
# ============================================================
# -- Clean measurement files (one per experiment period) --
PERIOD1_CLEAN_FILE    = "Cleaned/experiment_A/starlink_clean_FIXED.csv"
PERIOD2_CLEAN_FILE    = "Cleaned/experiment_B/Starlink_2_cleaned.csv"
PERIOD1_CLEAN_FALLBACK = "starlink_clean_FIXED.csv"
PERIOD2_CLEAN_FALLBACK = "Starlink_2_cleaned.csv"

# -- Feature-engineered forecast files --
THESIS_FORECAST_FILE       = "Cleaned/combined/starlink_forecast_combined.csv"
FORECAST_PERIOD1_FILE      = "Cleaned/experiment_A/starlink_forecast_v2.csv"
FORECAST_PERIOD2_FILE      = "Cleaned/experiment_B/starlink_2_forecast.csv"
FORECAST_PERIOD1_FALLBACK  = "starlink_forecast_v2.csv"
FORECAST_PERIOD2_FALLBACK  = "starlink_2_forecast.csv"
FORECAST_COMBINED_FALLBACK = "starlink_forecast_combined.csv"

# -- Comparison ISP files --
OMANTEL_FILE = "Cleaned/experiment_A/omantel_clean.csv"
OMANTEL_FALLBACK = "omantel_clean.csv"
AWASR_FILE   = "Cleaned/experiment_B/Awasr_cleaned.csv"
AWASR_FALLBACK = "Awasr_cleaned.csv"

# -- Live / retrain files --
RAW_FILE              = "Raw/experiment_A/starlink_data.csv"
STARLINK_RETRAIN_FILE = "Cleaned/state/starlink_retrain_queue.csv"
STARLINK_RETRAIN_FALLBACK = "starlink_retrain_queue.csv"
LIVE_MONITOR_FILE     = "Cleaned/state/live_monitor_only.csv"

# -- Validated research windows (from actual data inspection) --
PERIOD1_START = pd.Timestamp("2026-03-07 01:00:00")
PERIOD1_END   = pd.Timestamp("2026-03-28 11:00:00")
PERIOD2_START = pd.Timestamp("2026-04-20 11:45:00")
PERIOD2_END   = pd.Timestamp("2026-05-12 07:45:00")

# Backwards-compatible single window for legacy callers
THESIS_START = PERIOD1_START
THESIS_END   = PERIOD2_END

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;700;800&family=Space+Grotesk:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, .main, .block-container {
    background-color: #05080f !important;
    color: #d1d9e6 !important;
    font-family: 'Syne', sans-serif !important;
}

section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #080c17 0%, #05080f 100%) !important;
    border-right: 1px solid #0e1f3a !important;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #091628 0%, #060d1e 100%) !important;
    border: 1px solid #112240 !important;
    border-top: 3px solid #2d6af0 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
    box-shadow: 0 0 18px rgba(45,106,240,0.12) !important;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #4a6080 !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] div,
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8f0fe !important;
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #22c55e !important;
    font-weight: 700 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #080c17 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid #0e1f3a !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #4a6080 !important;
    font-weight: 700 !important;
    border-radius: 7px !important;
    padding: 7px 18px !important;
    font-size: 0.82rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.3px !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #93c5fd !important; background: rgba(45,106,240,0.08) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #4338ca) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 14px rgba(29,78,216,0.35) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

.card {
    background: linear-gradient(135deg, #091628 0%, #060d1e 100%);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #112240;
    box-shadow: 0 0 18px rgba(45,106,240,0.08);
    margin-bottom: 0.5rem;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #4a6080;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
}
.period-tag {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.5px;
}

hr { border-color: #0e1f3a !important; }

[data-testid="stSidebar"] label {
    color: #4a6080 !important; font-size: 0.72rem !important;
    font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.8px !important; font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stSidebar"] strong { color: #d1d9e6 !important; }
div[data-baseweb="select"] > div {
    background: #080c17 !important; border: 1px solid #0e1f3a !important;
    border-radius: 7px !important; color: #d1d9e6 !important;
}
p, li, span { color: #d1d9e6; }
h1, h2, h3 { color: #e8f0fe !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# PLOT THEME
# ============================================================
PLOT_BG   = "rgba(9,22,40,0.85)"
PAPER_BG  = "rgba(0,0,0,0)"
GRID_CLR  = "rgba(255,255,255,0.04)"
AXIS_CLR  = "#4a6080"
FONT_CLR  = "#d1d9e6"

P1_COLOR  = "#2d6af0"   # blue  – Period 1
P2_COLOR  = "#22c55e"   # green – Period 2
P1_FILL   = "rgba(45,106,240,0.08)"
P2_FILL   = "rgba(34,197,94,0.08)"

BASE_LAYOUT = dict(
    plot_bgcolor=PLOT_BG,
    paper_bgcolor=PAPER_BG,
    font=dict(color=FONT_CLR, family="IBM Plex Mono"),
    margin=dict(l=10, r=10, t=42, b=10),
    xaxis=dict(gridcolor=GRID_CLR, color=AXIS_CLR, showline=False),
    yaxis=dict(gridcolor=GRID_CLR, color=AXIS_CLR, showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=FONT_CLR)),
)

# ============================================================
# HELPERS, FORMATTING
# ============================================================
def fmt_temp(v):
    try:    return f"{float(v):.1f} °C" if pd.notna(v) else "—"
    except: return "—"
def fmt_humidity(v):
    try:    return f"{float(v):.0f}%" if pd.notna(v) else "—"
    except: return "—"
def fmt_wind(v):
    try:    return f"{float(v):.2f} m/s" if pd.notna(v) else "—"
    except: return "—"
def fmt_code(v):
    desc = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Cloudy",
            45:"Foggy",48:"Icy fog",51:"Light drizzle",53:"Drizzle",
            55:"Heavy drizzle",61:"Slight rain",63:"Rain",65:"Heavy rain",
            71:"Slight snow",73:"Snow",75:"Heavy snow",80:"Rain showers",
            81:"Heavy showers",82:"Violent showers",95:"Thunderstorm",
            96:"Thunderstorm+hail",99:"Heavy thunderstorm+hail"}
    try:    return desc.get(int(float(v)), f"Code {int(float(v))}")
    except: return "—"
def safe_float(row, col):
    try:    return float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0
    except: return 0.0

# ============================================================
# HELPERS, BUSINESS LOGIC
# ============================================================
def compute_health(latency, jitter, download, upload):
    score = 100
    if latency  > 80:  score -= 30
    elif latency > 50: score -= 15
    if jitter   > 15:  score -= 25
    elif jitter  > 8:  score -= 10
    if download  < 20: score -= 20
    elif download < 50: score -= 10
    if upload    < 10: score -= 15
    elif upload  < 20: score -= 8
    score = max(0, min(score, 100))
    label = ("Excellent" if score >= 85 else "Good" if score >= 70
             else "Fair" if score >= 50 else "Poor")
    return score, label

def detect_outage(latency, download, upload):
    if latency  > 100: return "High Latency Outage"
    if download < 5:   return "Severe Download Drop"
    if upload   < 2:   return "Upload Failure Risk"
    return "No Outage Detected"

def generate_alerts(latency_f, jitter_now, download_f, upload_f):
    alerts = []
    if latency_f  > 50: alerts.append(("warning", "Latency forecast to increase in the next 15 minutes."))
    if jitter_now > 10: alerts.append(("warning", "Current jitter is high. Real-time applications may be affected."))
    if download_f < 50: alerts.append(("error",   "Download speed forecast is low."))
    if upload_f   < 10: alerts.append(("error",   "Upload speed forecast is low."))
    if not alerts:      alerts.append(("success",  "No immediate network issue forecast."))
    return alerts

def usage_recommendation(latency, jitter, download, upload):
    recs = []
    recs.append("Suitable for video calls and online meetings."
                if latency < 40 and jitter < 8
                else "Use caution for video calls and real-time meetings.")
    recs.append("Suitable for streaming and large downloads."
                if download > 100 else
                "General browsing and normal streaming should work well."
                if download > 40 else
                "Heavy streaming or large downloads may be affected.")
    recs.append("Suitable for file uploads and cloud syncing."
                if upload > 20 else
                "Normal uploads should work, but large uploads may be slower."
                if upload > 10 else
                "Large uploads may be unreliable at this time.")
    return recs

def ai_insight(latency, jitter, download, upload):
    if latency < 35 and jitter < 8 and download > 100 and upload > 20:
        return ("Starlink is in a strong operating state and appears suitable "
                "for most academic and administrative activities.")
    if latency > 50 or jitter > 10:
        return ("The network may support general use, but latency-sensitive "
                "activities such as live meetings may be affected.")
    if download < 50 or upload < 10:
        return ("Bandwidth-intensive activities may experience reduced quality. "
                "File transfers and streaming should be scheduled with caution.")
    return "Current conditions appear acceptable for standard university use."

def health_badge_html(score):
    if score >= 85:
        return ('<span class="badge" style="background:#052e16;color:#22c55e;'
                'border:1px solid #22c55e33;">Excellent</span>')
    elif score >= 70:
        return ('<span class="badge" style="background:#1c1a08;color:#eab308;'
                'border:1px solid #eab30833;">Good</span>')
    elif score >= 50:
        return ('<span class="badge" style="background:#1c0f05;color:#f97316;'
                'border:1px solid #f9731633;">Fair</span>')
    return ('<span class="badge" style="background:#1c0505;color:#ef4444;'
            'border:1px solid #ef444433;">Poor</span>')

# ============================================================
# HELPERS, KPI CARD
# ============================================================
def kpi_card(label, value, unit, color, icon):
    return f"""
<div style="
    background:linear-gradient(135deg,{color}1a 0%,{color}0d 100%);
    border:1px solid {color}33; border-top:3px solid {color};
    border-radius:10px; padding:1rem 1.1rem;
    box-shadow:0 4px 18px {color}1a;
">
    <div style="color:{color};font-size:1.2rem;margin-bottom:0.15rem;">{icon}</div>
    <div style="color:#4a6080;font-size:0.66rem;font-weight:700;text-transform:uppercase;
                letter-spacing:1.4px;font-family:'IBM Plex Mono',monospace;">{label}</div>
    <div style="color:#e8f0fe;font-size:1.8rem;font-weight:900;line-height:1.1;
                margin-top:0.15rem;font-family:'IBM Plex Mono',monospace;">
        {value}<span style="font-size:0.82rem;color:#4a6080;font-weight:500;
        margin-left:3px;">{unit}</span>
    </div>
</div>"""

# ============================================================
# HELPERS, CHARTS
# ============================================================
def _apply_base(fig, title, height=360):
    layout = dict(BASE_LAYOUT)
    layout["title"] = dict(text=title, font=dict(color="#e8f0fe", size=13))
    layout["height"] = height
    fig.update_layout(**layout)
    return fig

def make_dual_period_chart(df1, df2, col, title, y_label, height=370):
    """
    Two-panel subplot: Period 1 on the left, Period 2 on the right.
    Both panels share the same y-axis scale so values are directly comparable.
    A styled divider column in the middle labels the gap period.
    This avoids the 35%-wide empty dead zone that a single shared time axis
    creates when two 21-day periods are separated by a 23-day gap.
    """
    # Compute shared y-range across both periods so scales match exactly
    combined = pd.concat([df1[col].dropna(), df2[col].dropna()])
    y_min = combined.min() * 0.92
    y_max = combined.max() * 1.08

    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.46, 0.08, 0.46],
        shared_yaxes=True,
        horizontal_spacing=0.0,
        subplot_titles=["", "", ""]   # we draw our own labels
    )

    # --- Period 1 (left panel) ---
    fig.add_trace(go.Scatter(
        x=df1["timestamp"], y=df1[col],
        mode="lines", name="Period 1  Mar 7–28",
        line=dict(color=P1_COLOR, width=1.6),
        fill="tozeroy", fillcolor="rgba(45,106,240,0.10)",
        showlegend=True
    ), row=1, col=1)

    # --- Period 2 (right panel) ---
    fig.add_trace(go.Scatter(
        x=df2["timestamp"], y=df2[col],
        mode="lines", name="Period 2  Apr 20 – May 12",
        line=dict(color=P2_COLOR, width=1.6),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.09)",
        showlegend=True
    ), row=1, col=3)

    # --- Middle divider: single invisible scatter to anchor the panel ---
    fig.add_trace(go.Scatter(
        x=[0.5], y=[(y_min + y_max) / 2],
        mode="text",
        text=["23-day<br>gap"],
        textfont=dict(color="#4a6080", size=9, family="IBM Plex Mono"),
        showlegend=False,
        xaxis="x2", yaxis="y"
    ), row=1, col=2)

    # Shared y-axis range applied to both outer panels
    fig.update_yaxes(range=[y_min, y_max], gridcolor=GRID_CLR,
                     color=AXIS_CLR, showline=False, title_text=y_label, row=1, col=1)
    fig.update_yaxes(range=[y_min, y_max], gridcolor=GRID_CLR,
                     color=AXIS_CLR, showline=False, showticklabels=False, row=1, col=3)
    fig.update_yaxes(visible=False, row=1, col=2)

    fig.update_xaxes(gridcolor=GRID_CLR, color=AXIS_CLR, showline=False, row=1, col=1)
    fig.update_xaxes(gridcolor=GRID_CLR, color=AXIS_CLR, showline=False, row=1, col=3)
    fig.update_xaxes(visible=False, row=1, col=2)

    # Period label annotations above each panel
    fig.add_annotation(
        text="<b>Period 1</b>  Mar 7 – 28",
        xref="paper", yref="paper",
        x=0.22, y=1.06, showarrow=False,
        font=dict(color=P1_COLOR, size=11, family="IBM Plex Mono"),
        xanchor="center"
    )
    fig.add_annotation(
        text="<b>Period 2</b>  Apr 20 – May 12",
        xref="paper", yref="paper",
        x=0.78, y=1.06, showarrow=False,
        font=dict(color=P2_COLOR, size=11, family="IBM Plex Mono"),
        xanchor="center"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#e8f0fe", size=13,
                                         family="IBM Plex Mono")),
        height=height,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_CLR, family="IBM Plex Mono"),
        margin=dict(l=10, r=10, t=65, b=90),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=FONT_CLR, size=11),
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="center",
            x=0.5
        )
    )
    return fig

def make_single_period_chart(df, col, title, y_label, color, height=360):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df[col], mode="lines", name=col,
        line=dict(color=color, width=1.8),
        fill="tozeroy",
        fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba")
                  if "rgb" in color else f"rgba(45,106,240,0.08)"
    ))
    _apply_base(fig, title, height)
    fig.update_layout(yaxis_title=y_label)
    return fig

def make_actual_vs_predicted_chart(y_test, y_pred, points, title):
    actual    = pd.Series(y_test).reset_index(drop=True)
    predicted = pd.Series(y_pred).reset_index(drop=True)
    if points < len(actual):
        actual    = actual.iloc[-points:]
        predicted = predicted.iloc[-points:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual,    mode="lines", name="Actual",
                             line=dict(color="#38bdf8", width=2)))
    fig.add_trace(go.Scatter(y=predicted, mode="lines", name="Predicted",
                             line=dict(color="#f472b6", width=2, dash="dot")))
    _apply_base(fig, title, 400)
    fig.update_layout(xaxis_title="Test Time Steps", yaxis_title="Latency (ms)")
    return fig

def make_health_gauge(score):
    bar_color = ("#22c55e" if score >= 85 else "#eab308" if score >= 70
                 else "#f97316" if score >= 50 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 26, "color": bar_color,
                                            "family": "IBM Plex Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1e3a5f",
                     "tickfont": {"color": "#4a6080", "size": 9}},
            "bar":  {"color": bar_color, "thickness": 0.38},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0,  50], "color": "rgba(239,68,68,0.10)"},
                {"range": [50, 70], "color": "rgba(249,115,22,0.10)"},
                {"range": [70, 85], "color": "rgba(234,179,8,0.10)"},
                {"range": [85,100], "color": "rgba(34,197,94,0.10)"},
            ],
            "threshold": {"line": {"color": bar_color, "width": 3},
                          "thickness": 0.8, "value": score}
        }
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=28, b=8),
                      paper_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#d1d9e6", "family": "IBM Plex Mono"})
    return fig

def make_isp_comparison_chart(df_sl, df_isp, isp_name, col, title, y_label):
    """Side-by-side comparison of Starlink vs a terrestrial ISP for the same period."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sl["timestamp"], y=df_sl[col], mode="lines", name="Starlink",
        fill="tozeroy", fillcolor="rgba(45,106,240,0.12)"
    ))
    fig.add_trace(go.Scatter(
        x=df_isp["timestamp"], y=df_isp[col], mode="lines", name=isp_name,
        line=dict(color="#a78bfa", width=1.6), fill="tozeroy",
        fillcolor="rgba(167,139,250,0.07)"
    ))
    _apply_base(fig, title, 340)
    fig.update_layout(yaxis_title=y_label, xaxis_title="Time")
    return fig

def make_scatter_weather(df, weather_col, metric_col, title):
    """Scatter of a weather variable vs a network metric, coloured by period."""
    fig = go.Figure()
    # We may get a combined df with a 'period' column, or just one df
    if "period" in df.columns:
        for period, color, label in [(1, P1_COLOR, "Period 1"), (2, P2_COLOR, "Period 2")]:
            sub = df[df["period"] == period]
            fig.add_trace(go.Scatter(
                x=sub[weather_col], y=sub[metric_col], mode="markers", name=label,
                marker=dict(color=color, size=4, opacity=0.6)
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[weather_col], y=df[metric_col], mode="markers",
            marker=dict(color=P1_COLOR, size=4, opacity=0.6)
        ))
    _apply_base(fig, title, 320)
    fig.update_layout(xaxis_title=weather_col, yaxis_title=metric_col)
    return fig

def make_box_comparison(df1, df2, col, title, y_label):
    """Box plot comparing distributions for both periods."""
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df1[col].dropna(), name="Period 1  Mar 7–28",
        marker_color=P1_COLOR, line_color=P1_COLOR,
        fillcolor=P1_FILL, boxmean=True
    ))
    fig.add_trace(go.Box(
        y=df2[col].dropna(), name="Period 2  Apr 20 – May 12",
        marker_color=P2_COLOR, line_color=P2_COLOR,
        fillcolor=P2_FILL, boxmean=True
    ))
    _apply_base(fig, title, 360)
    fig.update_layout(yaxis_title=y_label, showlegend=True)
    return fig

def make_residual_chart(y_test, y_pred, title):
    residuals = np.array(y_test) - np.array(y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=residuals, mode="lines", name="Residual",
        line=dict(color="#f472b6", width=1.4)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#4a6080", line_width=1)
    _apply_base(fig, title, 300)
    fig.update_layout(yaxis_title="Residual (ms)", xaxis_title="Test Steps")
    return fig

# ============================================================
# LIVE DATA ROUTING
# ============================================================
def classify_and_route_new_rows():
    """Route new rows from the raw file into retrain/monitor queues."""
    if not os.path.exists(RAW_FILE):
        return None
    raw = pd.read_csv(RAW_FILE)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], dayfirst=False, errors="coerce")
    new_rows = raw[raw["timestamp"] > THESIS_END].copy()
    if len(new_rows) == 0:
        return None
    starlink_new = new_rows[new_rows["network_type"].str.strip().str.lower() == "starlink"]
    other_new    = new_rows[new_rows["network_type"].str.strip().str.lower() != "starlink"]
    if len(starlink_new) > 0:
        _append_dedup(STARLINK_RETRAIN_FILE, starlink_new)
    if len(other_new) > 0:
        _append_dedup(LIVE_MONITOR_FILE, other_new)
    return new_rows.sort_values("timestamp").iloc[-1]

def load_latest_live_row():
    """
    Return (row, source) for the live KPI overlay.
    Priority: raw file post-thesis rows -> retrain queue -> (None, None).
    """
    # 1. Try raw file
    if os.path.exists(RAW_FILE):
        try:
            raw = pd.read_csv(RAW_FILE)
            raw.columns = [c.lower() for c in raw.columns]
            raw["timestamp"] = pd.to_datetime(raw["timestamp"], dayfirst=False, errors="coerce")
            post = raw[raw["timestamp"] > THESIS_END].dropna(subset=["timestamp"])
            if len(post) > 0:
                return post.sort_values("timestamp").iloc[-1], "raw"
        except Exception:
            pass
    # 2. Fall back to retrain queue
    rpath = STARLINK_RETRAIN_FILE if os.path.exists(STARLINK_RETRAIN_FILE) else (
            STARLINK_RETRAIN_FALLBACK if os.path.exists(STARLINK_RETRAIN_FALLBACK) else None)
    if rpath:
        try:
            df = pd.read_csv(rpath)
            df.columns = [c.lower() for c in df.columns]
            df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            if len(df) > 0:
                # Prefer most recent row with valid speedtest data (download > 0)
                valid = df[df["download_mbps"].notna() & (df["download_mbps"] > 0)]
                best = valid.iloc[-1] if len(valid) > 0 else df.iloc[-1]
                return best, "retrain_queue"
        except Exception:
            pass
    return None, None

def _append_dedup(path, new_df):
    if os.path.exists(path):
        existing = pd.read_csv(path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], dayfirst=False, errors="coerce")
        combined = pd.concat([existing, new_df]).drop_duplicates(subset=["timestamp"])
    else:
        combined = new_df
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    combined.to_csv(path, index=False)

def retrain_queue_status():
    path = STARLINK_RETRAIN_FILE if os.path.exists(STARLINK_RETRAIN_FILE) else (
           STARLINK_RETRAIN_FALLBACK if os.path.exists(STARLINK_RETRAIN_FALLBACK) else None)
    if path is None:
        return 0, None
    df = pd.read_csv(path)
    return len(df), pd.to_datetime(df["timestamp"], format="mixed", errors="coerce").max()

# ============================================================
# FILE PATH RESOLVER
# ============================================================
def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data(ttl=120)
def load_period1():
    path = first_existing(PERIOD1_CLEAN_FILE, PERIOD1_CLEAN_FALLBACK)
    if path is None:
        st.error("Cannot find Period 1 clean file (starlink_clean_FIXED.csv). "
                 "Place it at Cleaned/experiment_A/ or beside this script.")
        st.stop()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]  # normalise temperature_C -> temperature_c
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, errors="coerce")
    df = df[(df["timestamp"] >= PERIOD1_START) & (df["timestamp"] <= PERIOD1_END)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

@st.cache_data(ttl=120)
def load_period2():
    path = first_existing(PERIOD2_CLEAN_FILE, PERIOD2_CLEAN_FALLBACK)
    if path is None:
        st.error("Cannot find Period 2 clean file (Starlink_2_cleaned.csv). "
                 "Place it at Cleaned/experiment_B/ or beside this script.")
        st.stop()
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, errors="coerce")
    df = df[(df["timestamp"] >= PERIOD2_START) & (df["timestamp"] <= PERIOD2_END)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

@st.cache_data(ttl=120)
def load_omantel():
    path = first_existing(OMANTEL_FILE, OMANTEL_FALLBACK)
    if path is None:
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)

@st.cache_data(ttl=120)
def load_awasr():
    path = first_existing(AWASR_FILE, AWASR_FALLBACK)
    if path is None:
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=False, errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)

@st.cache_data(ttl=120)
def load_forecast_combined():
    """
    Loads the combined feature-engineered forecast dataset.
    Falls back to auto-merging the two period forecast files if combined is missing.
    """
    path = first_existing(THESIS_FORECAST_FILE, FORECAST_COMBINED_FALLBACK)
    if path:
        df = pd.read_csv(path)
        return _clean_forecast_df(df), "combined_existing"

    p1 = first_existing(FORECAST_PERIOD1_FILE, FORECAST_PERIOD1_FALLBACK)
    p2 = first_existing(FORECAST_PERIOD2_FILE, FORECAST_PERIOD2_FALLBACK)
    if p1 is None or p2 is None:
        missing = []
        if p1 is None: missing.append("starlink_forecast_v2.csv (Period 1)")
        if p2 is None: missing.append("starlink_2_forecast.csv (Period 2)")
        st.error("Missing forecast dataset(s): " + ", ".join(missing))
        st.stop()
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)
    common = [c for c in df1.columns if c in set(df2.columns)]
    combined = pd.concat([df1[common], df2[common]], ignore_index=True)
    return _clean_forecast_df(combined), "combined_auto_merged"

def _clean_forecast_df(df):
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    if "target" not in df.columns:
        st.error("Forecast dataset must contain a 'target' column.")
        st.stop()
    if len(df) < 10:
        st.error("Forecast dataset too small after cleaning.")
        st.stop()
    return df

# ============================================================
# ML
# ============================================================
def prepare_xy(df):
    df = df.copy()
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])
    df = df.select_dtypes(include=["number"])
    return df.drop(columns=["target"]), df["target"]

def train_model(df, model_name):
    X, y = prepare_xy(df)
    split = int(len(df) * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte  = y.iloc[:split], y.iloc[split:]

    if model_name == "Naive Baseline":
        lags   = [c for c in Xte.columns if c.endswith("_lag_1")]
        y_pred = Xte[lags[0]].values
        model  = None
    elif model_name == "Linear Regression":
        model = LinearRegression(); model.fit(Xtr, ytr); y_pred = model.predict(Xte)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(Xtr, ytr); y_pred = model.predict(Xte)
    elif model_name == "XGBoost":
        if not XGB_AVAILABLE: raise ValueError("XGBoost not installed.")
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(Xtr, ytr); y_pred = model.predict(Xte)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return {"model": model, "X_test": Xte, "y_test": yte, "y_pred": y_pred,
            "mae":  float(mean_absolute_error(yte, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(yte, y_pred)))}

def predict_next(df, model_name):
    X, y = prepare_xy(df)
    if len(X) == 0: return 0.0
    latest = X.iloc[[-1]]
    if model_name == "Naive Baseline":
        lags = [c for c in latest.columns if c.endswith("_lag_1")]
        return float(latest[lags[0]].iloc[0])
    if model_name == "Linear Regression":  m = LinearRegression()
    elif model_name == "Random Forest":    m = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":          m = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    else:                                  return 0.0
    m.fit(X, y)
    return float(m.predict(latest)[0])

def build_download_upload_features(clean_df, target_col):
    df = clean_df.copy()
    if "was_estimated_row" in df.columns:
        df = df[df["was_estimated_row"] == False]
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in [target_col, "ping_avg_rtt_ms", "ping_jitter_ms",
              "weather_code", "temperature_c", "humidity_percent", "wind_speed_mps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df[target_col] > 0].copy()
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    for lag in range(1, 5):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    if "ping_avg_rtt_ms" in df.columns: df["latency_lag_1"]  = df["ping_avg_rtt_ms"].shift(1)
    if "ping_jitter_ms"  in df.columns: df["jitter_lag_1"]   = df["ping_jitter_ms"].shift(1)
    df["target"] = df[target_col].shift(-1)
    df = df.dropna().reset_index(drop=True)
    features = ["hour", "day_of_week", "is_weekend",
                f"{target_col}_lag_1", f"{target_col}_lag_2",
                f"{target_col}_lag_3", f"{target_col}_lag_4",
                "latency_lag_1", "jitter_lag_1", "weather_code",
                "temperature_c", "humidity_percent", "wind_speed_mps"]
    features = [c for c in features if c in df.columns]
    return df[features + ["target"]].copy()

def make_model_comparison_table(df):
    names = ["Naive Baseline", "Linear Regression", "Random Forest"]
    if XGB_AVAILABLE: names.append("XGBoost")
    rows = []
    for m in names:
        r = train_model(df, m)
        rows.append({"Model": m, "MAE (ms)": round(r["mae"], 3),
                     "RMSE (ms)": round(r["rmse"], 3)})
    cmp_df = pd.DataFrame(rows)
    best_mae = cmp_df["MAE (ms)"].idxmin()
    cmp_df["Best"] = ""
    cmp_df.loc[best_mae, "Best"] = "Selected"
    return cmp_df

# ============================================================
# LOAD DATA
# ============================================================
p1_df = load_period1()   # Period 1 clean
p2_df = load_period2()   # Period 2 clean
omantel_df = load_omantel()
awasr_df   = load_awasr()
forecast_df, forecast_merge_status = load_forecast_combined()

# Combined clean (for any legacy callers that need one df)
combined_clean = pd.concat([p1_df, p2_df], ignore_index=True).sort_values("timestamp")

# Build per-period support datasets for download/upload forecasting (use combined)
dl_feat = build_download_upload_features(combined_clean, "download_mbps")
ul_feat = build_download_upload_features(combined_clean, "upload_mbps")

# Route any new raw rows into queues (best-effort)
try:
    classify_and_route_new_rows()
except Exception:
    pass
# Load most recent available row for live overlay
latest_live_row, latest_live_source = load_latest_live_row()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Controls")
    model_options = ["Naive Baseline", "Linear Regression", "Random Forest"]
    if XGB_AVAILABLE: model_options.append("XGBoost")
    selected_model = st.selectbox("Forecast model", model_options, index=1)
    chart_points   = st.slider("Prediction chart points", 50, 500, 200, 25)

    st.markdown("---")
    show_live = st.toggle(
        "Show live KPI overlay", value=False,
        help="Displays the latest logged measurement on KPI cards. "
             "Does not affect the model, charts, or research data."
    )

    st.markdown("---")
    st.markdown("**Dataset periods**")
    st.markdown(
        f'<span class="period-tag" style="background:#0d2348;color:{P1_COLOR};'
        f'border:1px solid {P1_COLOR}44;">P1</span> '
        f'<span style="color:#4a6080;font-size:0.78rem;">Mar 7 – Mar 28, 2026</span>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<span class="period-tag" style="background:#0a2318;color:{P2_COLOR};'
        f'border:1px solid {P2_COLOR}44;">P2</span> '
        f'<span style="color:#4a6080;font-size:0.78rem;">Apr 20 – May 12, 2026</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("**GUtech AI Thesis · 2026**")
    st.markdown("Starlink Performance Analysis")
    st.markdown("Muscat, Oman")

    q_count, q_latest = retrain_queue_status()
    st.markdown("---")
    if q_count > 0:
        st.markdown(f"**Retrain queue:** {q_count} rows")
        st.markdown(f"Latest: `{q_latest}`")
        st.caption("New Starlink-only rows eligible for future retraining.")
    else:
        st.caption("No new Starlink data queued for retraining.")

# ============================================================
# TRAIN MODELS
# ============================================================
latency_result    = train_model(forecast_df, selected_model)
latency_forecast  = predict_next(forecast_df, selected_model)
download_forecast = predict_next(dl_feat, "Linear Regression")
upload_forecast   = predict_next(ul_feat, "Linear Regression")

# Fallback if feature builder returns 0
_last_p2 = p2_df.iloc[-1]
if download_forecast <= 0:
    download_forecast = safe_float(_last_p2, "download_mbps")
if upload_forecast <= 0:
    upload_forecast   = safe_float(_last_p2, "upload_mbps")

# ============================================================
# DECIDE DISPLAY ROW
# ============================================================
if show_live:
    if latest_live_row is not None:
        display_row = latest_live_row
        _net = str(latest_live_row.get("network_type", "Starlink")
                   if hasattr(latest_live_row, "get")
                   else getattr(latest_live_row, "network_type", "Starlink"))
        _ts  = str(latest_live_row.get("timestamp", "")
                   if hasattr(latest_live_row, "get")
                   else getattr(latest_live_row, "timestamp", ""))[:16]
        if latest_live_source == "raw":
            kpi_note = f"Live data (raw file), {_net} · {_ts}"
            kpi_note_color = "#22c55e"
        else:
            kpi_note = f"Live overlay, retrain queue · most recent row · {_ts}"
            kpi_note_color = "#eab308"
    else:
        display_row    = p2_df.iloc[-1]
        kpi_note       = "Live overlay: no post-thesis data found, showing last Period 2 row"
        kpi_note_color = "#ef4444"
else:
    display_row    = p2_df.iloc[-1]
    kpi_note       = ("Showing last observation from Period 2 "
                      "(most recent validated clean dataset, May 12, 2026)")
    kpi_note_color = "#4a6080"

current_latency  = safe_float(display_row, "ping_avg_rtt_ms")
current_jitter   = safe_float(display_row, "ping_jitter_ms")
current_download = safe_float(display_row, "download_mbps")
current_upload   = safe_float(display_row, "upload_mbps")

health_score, health_label = compute_health(
    latency_forecast, current_jitter, download_forecast, upload_forecast)
alerts          = generate_alerts(latency_forecast, current_jitter, download_forecast, upload_forecast)
recommendations = usage_recommendation(latency_forecast, current_jitter, download_forecast, upload_forecast)
system_insight  = ai_insight(latency_forecast, current_jitter, download_forecast, upload_forecast)
outage_status   = detect_outage(latency_forecast, download_forecast, upload_forecast)

latest_time   = combined_clean["timestamp"].max()
forecast_time = latest_time + pd.Timedelta(minutes=15)

# Weather from display row
def _get(row, col):
    try:    return row[col] if col in row.index else np.nan
    except: return np.nan

weather_code = _get(display_row, "weather_code")
temp_val     = _get(display_row, "temperature_c")
hum_val      = _get(display_row, "humidity_percent")
wind_val     = _get(display_row, "wind_speed_mps")

# ============================================================
# HEADER
# ============================================================
st.markdown(
    f'<div style="margin-bottom:-1.4rem;'
    f'margin-top:-2rem;color:#4a6080;'
    f'font-size:0.72rem;font-family:\'IBM Plex Mono\',monospace;">'
    f'Last refresh: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}'
    f'</div>',
    unsafe_allow_html=True
)

_now_str = pd.Timestamp.now().strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(
    f'<div style="background:linear-gradient(135deg,#0d1f3c 0%,#111830 40%,#0a1830 100%);'
    f'border-radius:16px;padding:1.2rem 2.4rem 1.6rem 2.4rem;margin-bottom:0.5rem;'
    f'border:1px solid rgba(45,106,240,0.3);'
    f'box-shadow:0 0 40px rgba(45,106,240,0.10);'
    f'position:relative;overflow:hidden;">'

    f'<div style="position:absolute;inset:0;'
    f'background-image:linear-gradient(rgba(61,127,255,0.04) 1px,transparent 1px),'
    f'linear-gradient(90deg,rgba(61,127,255,0.04) 1px,transparent 1px);'
    f'background-size:40px 40px;pointer-events:none;"></div>'

    f'<div style="position:absolute;top:0;left:0;right:0;height:1px;'
    f'background:linear-gradient(90deg,transparent,rgba(61,127,255,0.55),rgba(52,211,153,0.35),transparent);'
    f'pointer-events:none;"></div>'

    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
    f'position:relative;z-index:1;">'

    f'<div>'
    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;'
    f'color:#4d8fff;letter-spacing:2.5px;font-weight:600;margin-bottom:0.6rem;'
    f'display:flex;align-items:center;gap:0.45rem;">'
    f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
    f'background:#22c55e;box-shadow:0 0 8px #22c55e;"></span>'
    f'STARLINK INTELLIGENCE PLATFORM</div>'

    f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:2rem;'
    f'font-weight:800;line-height:1.1;letter-spacing:-0.8px;margin-bottom:0.5rem;">'
    f'<span style="color:#e2e8f8;">Performance </span>'
    f'<span style="background:linear-gradient(90deg,#60a5fa,#34d399);'
    f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
    f'background-clip:text;">Forecast System</span></div>'

    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.75rem;'
    f'color:#4a6080;letter-spacing:0.3px;">'
    f'Network Monitoring &nbsp;&middot;&nbsp; Latency Forecasting &nbsp;&middot;&nbsp;'
    f'Outage Detection &nbsp;&middot;&nbsp; Infrastructure Intelligence</div>'
    f'</div>'

    f'<div style="text-align:right;flex-shrink:0;margin-left:2rem;">'
    f'<div style="background:linear-gradient(135deg,rgba(29,78,216,0.5),rgba(67,56,202,0.4));'
    f'border:1px solid rgba(29,78,216,0.4);'
    f'padding:0.32rem 0.9rem;border-radius:6px;'
    f'font-size:0.68rem;font-weight:700;color:#93c5fd;'
    f'letter-spacing:0.8px;margin-bottom:0.5rem;'
    f'font-family:\'JetBrains Mono\',monospace;display:inline-block;">'
    f'GUtech AI Thesis Prototype</div>'
    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;'
    f'color:#4a6080;margin-bottom:0.2rem;">Starlink · Muscat, Oman · 2026</div>'
    f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.65rem;'
    f'color:#2a4060;">{_now_str}</div>'
    f'</div>'

    f'</div></div>',
    unsafe_allow_html=True
)

# Dataset banner
st.markdown(f"""
<div style="
    background:linear-gradient(90deg,#04200f,#062a14);
    border:1px solid #22c55e22; border-left:4px solid #22c55e;
    border-radius:9px; padding:0.65rem 1.1rem; margin-bottom:0.7rem;
    display:flex; align-items:center; gap:0.8rem;
">
    <span style="font-size:1.1rem;">🛰️</span>
    <div>
        <span style="color:#22c55e;font-weight:700;font-size:0.82rem;
                     font-family:'IBM Plex Mono',monospace;">
            VALIDATED RESEARCH DATASET
        </span>
        <span style="color:#4a6080;font-size:0.8rem;margin-left:0.7rem;">
            Period 1: Mar 7 – Mar 28 2026 &nbsp;|&nbsp;
            Period 2: Apr 20 – May 12 2026 &nbsp;|&nbsp;
            {len(p1_df):,} + {len(p2_df):,} = {len(p1_df)+len(p2_df):,} total rows &nbsp;|&nbsp;
            Forecast dataset: {len(forecast_df):,} rows ({forecast_merge_status.replace("_"," ")})
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# KPI source note
st.markdown(f"""
<div style="
    background:rgba(255,255,255,0.025);border:1px solid #0e1f3a;
    border-radius:7px;padding:0.45rem 1rem;margin-bottom:0.7rem;
    font-size:0.78rem;color:{kpi_note_color};
    font-family:'IBM Plex Mono',monospace;
">{kpi_note}</div>
""", unsafe_allow_html=True)

# ============================================================
# KPI CARDS
# ============================================================
c1, c2, c3, c4, c5 = st.columns(5)
hcolor = ("#22c55e" if health_score >= 85 else "#eab308" if health_score >= 70
          else "#f97316" if health_score >= 50 else "#ef4444")

with c1: st.markdown(kpi_card("Latency",      f"{current_latency:.1f}",  "ms",   "#2d6af0", "📡"), unsafe_allow_html=True)
with c2: st.markdown(kpi_card("Jitter",       f"{current_jitter:.1f}",   "ms",   "#f472b6", "〰️"), unsafe_allow_html=True)
with c3: st.markdown(kpi_card("Download",     f"{current_download:.1f}", "Mbps", "#22c55e", "⬇"), unsafe_allow_html=True)
with c4: st.markdown(kpi_card("Upload",       f"{current_upload:.1f}",   "Mbps", "#f97316", "⬆"), unsafe_allow_html=True)
with c5: st.markdown(kpi_card("Health Score", f"{health_score}",         "/100", hcolor,    "💚"), unsafe_allow_html=True)

st.markdown(
    f'<div style="margin:0.5rem 0 0.9rem;">'
    f'{health_badge_html(health_score)}'
    f'&nbsp;&nbsp;<span style="color:#4a6080;font-size:0.88rem;'
    f'font-family:\'IBM Plex Mono\',monospace;">'
    f'Network status: <b style="color:#d1d9e6;">{outage_status}</b></span></div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Historical Trends",
    "ISP Comparison",
    "Forecasting",
    "Alerts & Recs",
    "Model Evaluation"
])

# ============================================================
# TAB 1, OVERVIEW
# ============================================================
with tab1:
    st.markdown('<div class="card-title">Network Overview</div>', unsafe_allow_html=True)

    left, right = st.columns([2.2, 1])
    with left:
        # Dual-period latency chart, the signature chart of the thesis
        st.plotly_chart(
            make_dual_period_chart(p1_df, p2_df, "ping_avg_rtt_ms",
                                   "Latency, Both Collection Periods",
                                   "Latency (ms)", height=430),
            use_container_width=True
        )

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Next 15-Minute Forecast</div>', unsafe_allow_html=True)
        st.metric("Latency",  f"{latency_forecast:.2f} ms")
        st.metric("Download", f"{download_forecast:.2f} Mbps")
        st.metric("Upload",   f"{upload_forecast:.2f} Mbps")
        st.markdown(
            f'<div style="color:#4a6080;font-size:0.75rem;margin-top:0.5rem;'
            f'font-family:\'IBM Plex Mono\',monospace;">'
            f'Forecast for: {forecast_time.strftime("%Y-%m-%d %H:%M")}<br>'
            f'Model: {selected_model}</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1, 1.1, 1])
    with col_a:
        st.plotly_chart(make_health_gauge(health_score), use_container_width=True)
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Recommended Usage</div>', unsafe_allow_html=True)
        for rec in recommendations:
            st.markdown(f"- {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Latest Weather Context</div>', unsafe_allow_html=True)
        st.write(f"Condition: **{fmt_code(weather_code)}**")
        st.write(f"Temperature: **{fmt_temp(temp_val)}**")
        st.write(f"Humidity: **{fmt_humidity(hum_val)}**")
        st.write(f"Wind: **{fmt_wind(wind_val)}**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">System Insight</div>', unsafe_allow_html=True)
    st.write(system_insight)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2, HISTORICAL TRENDS
# ============================================================
with tab2:
    st.markdown('<div class="card-title">Historical Performance, Both Experiment Periods</div>',
                unsafe_allow_html=True)

    st.markdown(
        f'<div style="color:#4a6080;font-size:0.78rem;margin-bottom:0.8rem;'
        f'font-family:\'IBM Plex Mono\',monospace;">'
        f'<span style="color:{P1_COLOR};">Blue = Period 1 (Mar 7–28, 2026)</span>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<span style="color:{P2_COLOR};">Green = Period 2 (Apr 20 – May 12, 2026)</span>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'The shaded region between them marks the 23-day gap where no data was collected.</div>',
        unsafe_allow_html=True
    )

    # Jitter and download side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            make_dual_period_chart(p1_df, p2_df, "ping_jitter_ms",
                                   "Jitter, Both Periods", "Jitter (ms)"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            make_dual_period_chart(p1_df, p2_df, "download_mbps",
                                   "Download Speed, Both Periods", "Download (Mbps)"),
            use_container_width=True
        )

    # Upload and packet loss
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(
            make_dual_period_chart(p1_df, p2_df, "upload_mbps",
                                   "Upload Speed, Both Periods", "Upload (Mbps)"),
            use_container_width=True
        )
    with col4:
        st.plotly_chart(
            make_dual_period_chart(p1_df, p2_df, "packet_loss_percent",
                                   "Packet Loss, Both Periods", "Packet Loss (%)"),
            use_container_width=True
        )

    # Distribution comparison (box plots)
    st.markdown('<div class="card-title" style="margin-top:0.5rem;">Distribution Comparison, Period 1 vs Period 2</div>',
                unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.plotly_chart(
            make_box_comparison(p1_df, p2_df, "ping_avg_rtt_ms",
                                "Latency Distribution", "Latency (ms)"),
            use_container_width=True
        )
    with b2:
        st.plotly_chart(
            make_box_comparison(p1_df, p2_df, "download_mbps",
                                "Download Distribution", "Download (Mbps)"),
            use_container_width=True
        )

    # Per-period summary stats
    st.markdown('<div class="card-title" style="margin-top:0.5rem;">Period Summary Statistics</div>',
                unsafe_allow_html=True)
    cols_of_interest = ["ping_avg_rtt_ms", "ping_jitter_ms", "download_mbps",
                        "upload_mbps", "packet_loss_percent"]
    stats_rows = []
    for col in cols_of_interest:
        for period_label, df in [("Period 1 (Mar 7–28)", p1_df),
                                  ("Period 2 (Apr 20 – May 12)", p2_df)]:
            if col in df.columns:
                s = df[col].dropna()
                stats_rows.append({
                    "Metric": col, "Period": period_label,
                    "Mean": round(s.mean(), 2), "Median": round(s.median(), 2),
                    "Std Dev": round(s.std(), 2), "Min": round(s.min(), 2),
                    "Max": round(s.max(), 2), "Rows": len(s)
                })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

# ============================================================
# TAB 3, ISP COMPARISON
# ============================================================
with tab3:
    st.markdown('<div class="card-title">ISP Comparison, Starlink vs Terrestrial Providers</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div style="color:#4a6080;font-size:0.78rem;margin-bottom:0.8rem;'
        'font-family:\'IBM Plex Mono\',monospace;">'
        'Period 1 (Mar 7–28): Starlink vs OmanTel &nbsp;|&nbsp; '
        'Period 2 (Apr 20 – May 12): Starlink vs Awasr. '
        'Each comparison uses data from the same time window, so conditions are comparable.</div>',
        unsafe_allow_html=True
    )

    # --- PERIOD 1: Starlink vs OmanTel ---
    st.markdown(
        f'<span class="period-tag" style="background:#0d2348;color:{P1_COLOR};'
        f'border:1px solid {P1_COLOR}44;">Period 1</span>'
        f'<span style="color:#d1d9e6;margin-left:0.6rem;font-weight:700;">'
        f'Starlink vs OmanTel, Mar 7 to Mar 28, 2026</span>',
        unsafe_allow_html=True
    )

    if omantel_df is not None:
        om_p1 = omantel_df[(omantel_df["timestamp"] >= PERIOD1_START) &
                            (omantel_df["timestamp"] <= PERIOD1_END)]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                make_isp_comparison_chart(p1_df, om_p1, "OmanTel", "ping_avg_rtt_ms",
                                          "Latency: Starlink vs OmanTel", "Latency (ms)"),
                use_container_width=True
            )
        with c2:
            st.plotly_chart(
                make_isp_comparison_chart(p1_df, om_p1, "OmanTel", "download_mbps",
                                          "Download: Starlink vs OmanTel", "Download (Mbps)"),
                use_container_width=True
            )
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                make_isp_comparison_chart(p1_df, om_p1, "OmanTel", "upload_mbps",
                                          "Upload: Starlink vs OmanTel", "Upload (Mbps)"),
                use_container_width=True
            )
        with c4:
            st.plotly_chart(
                make_isp_comparison_chart(p1_df, om_p1, "OmanTel", "ping_jitter_ms",
                                          "Jitter: Starlink vs OmanTel", "Jitter (ms)"),
                use_container_width=True
            )

        # Summary table P1
        isp_rows = []
        for metric in ["ping_avg_rtt_ms", "download_mbps", "upload_mbps", "ping_jitter_ms"]:
            if metric in p1_df.columns and metric in om_p1.columns:
                isp_rows.append({
                    "Metric": metric,
                    "Starlink Mean": round(p1_df[metric].mean(), 2),
                    "OmanTel Mean":  round(om_p1[metric].mean(), 2),
                    "Difference":    round(p1_df[metric].mean() - om_p1[metric].mean(), 2)
                })
        if isp_rows:
            st.dataframe(pd.DataFrame(isp_rows), use_container_width=True)
    else:
        st.info("OmanTel data not found. Place omantel_clean.csv at Cleaned/experiment_A/ or beside this script.")

    st.markdown("---")

    # --- PERIOD 2: Starlink vs Awasr ---
    st.markdown(
        f'<span class="period-tag" style="background:#0a2318;color:{P2_COLOR};'
        f'border:1px solid {P2_COLOR}44;">Period 2</span>'
        f'<span style="color:#d1d9e6;margin-left:0.6rem;font-weight:700;">'
        f'Starlink vs Awasr, Apr 20 to May 12, 2026</span>',
        unsafe_allow_html=True
    )

    if awasr_df is not None:
        aw_p2 = awasr_df[(awasr_df["timestamp"] >= PERIOD2_START) &
                          (awasr_df["timestamp"] <= PERIOD2_END)]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                make_isp_comparison_chart(p2_df, aw_p2, "Awasr", "ping_avg_rtt_ms",
                                          "Latency: Starlink vs Awasr", "Latency (ms)"),
                use_container_width=True
            )
        with c2:
            st.plotly_chart(
                make_isp_comparison_chart(p2_df, aw_p2, "Awasr", "download_mbps",
                                          "Download: Starlink vs Awasr", "Download (Mbps)"),
                use_container_width=True
            )
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                make_isp_comparison_chart(p2_df, aw_p2, "Awasr", "upload_mbps",
                                          "Upload: Starlink vs Awasr", "Upload (Mbps)"),
                use_container_width=True
            )
        with c4:
            st.plotly_chart(
                make_isp_comparison_chart(p2_df, aw_p2, "Awasr", "ping_jitter_ms",
                                          "Jitter: Starlink vs Awasr", "Jitter (ms)"),
                use_container_width=True
            )

        isp_rows2 = []
        for metric in ["ping_avg_rtt_ms", "download_mbps", "upload_mbps", "ping_jitter_ms"]:
            if metric in p2_df.columns and metric in aw_p2.columns:
                isp_rows2.append({
                    "Metric": metric,
                    "Starlink Mean": round(p2_df[metric].mean(), 2),
                    "Awasr Mean":    round(aw_p2[metric].mean(), 2),
                    "Difference":    round(p2_df[metric].mean() - aw_p2[metric].mean(), 2)
                })
        if isp_rows2:
            st.dataframe(pd.DataFrame(isp_rows2), use_container_width=True)
    else:
        st.info("Awasr data not found. Place Awasr_cleaned.csv at Cleaned/experiment_B/ or beside this script.")

# ============================================================
# TAB 4, FORECASTING
# ============================================================
with tab4:
    st.markdown('<div class="card-title">Forecasting, Model trained on combined validated dataset</div>',
                unsafe_allow_html=True)

    st.plotly_chart(
        make_actual_vs_predicted_chart(
            latency_result["y_test"], latency_result["y_pred"],
            chart_points, f"Actual vs Predicted Latency ({selected_model})"),
        use_container_width=True
    )

    f1, f2, f3 = st.columns(3)
    f1.metric("Forecast Latency",  f"{latency_forecast:.2f} ms")
    f2.metric("Forecast Download", f"{download_forecast:.2f} Mbps")
    f3.metric("Forecast Upload",   f"{upload_forecast:.2f} Mbps")

    # Residuals
    st.plotly_chart(
        make_residual_chart(
            latency_result["y_test"], latency_result["y_pred"],
            f"Prediction Residuals ({selected_model})"
        ),
        use_container_width=True
    )

    st.markdown(
        '<div style="color:#4a6080;font-size:0.8rem;margin-top:0.4rem;'
        'font-family:\'IBM Plex Mono\',monospace;">'
        'Latency is the primary forecasting target. Download and upload forecasts '
        'are supporting operational estimates. All predictions use the combined validated '
        'dataset (Period 1 + Period 2).</div>',
        unsafe_allow_html=True
    )

    q_count, q_latest = retrain_queue_status()
    if q_count > 0:
        st.markdown("---")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">New Starlink Data Available for Retraining</div>',
                    unsafe_allow_html=True)
        st.write(f"**{q_count}** new Starlink rows collected after the thesis window.")
        st.write(f"Latest timestamp: **{q_latest}**")
        st.write("These rows are in `starlink_retrain_queue.csv` and can be merged "
                 "to retrain the model when ready.")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 5, ALERTS & RECOMMENDATIONS
# ============================================================
with tab5:
    st.markdown('<div class="card-title">Alerts and Recommendations</div>', unsafe_allow_html=True)

    for level, message in alerts:
        if level == "success":  st.success(message)
        elif level == "warning": st.warning(message)
        else:                    st.error(message)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Operational Interpretation</div>', unsafe_allow_html=True)
        st.write(
            f"Forecast-based health status is **{health_label}** with a score of "
            f"**{health_score}/100**. This combines predicted latency, forecast bandwidth, "
            f"and current jitter."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Suggested Use Cases Right Now</div>', unsafe_allow_html=True)
        for rec in recommendations:
            st.write(f"- {rec}")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 6, MODEL EVALUATION
# ============================================================
with tab6:
    st.markdown('<div class="card-title">Model Evaluation, Combined Forecast Dataset</div>',
                unsafe_allow_html=True)

    model_table = make_model_comparison_table(forecast_df)
    st.dataframe(model_table, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric(f"{selected_model}, MAE",  f"{latency_result['mae']:.3f} ms")
    m2.metric(f"{selected_model}, RMSE", f"{latency_result['rmse']:.3f} ms")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Research Notes</div>', unsafe_allow_html=True)
    st.write(
        "Models were trained and evaluated on the combined validated Starlink dataset "
        "(Period 1: Mar 7–28 2026 + Period 2: Apr 20 – May 12 2026, "
        f"total {len(forecast_df):,} feature-engineered rows). "
        "The naive baseline copies the most recent lag as the prediction, serving as a "
        "minimum benchmark. Linear Regression was selected as the primary model for its "
        "balance of accuracy, stability, and interpretability. More complex models showed "
        "no consistent improvement, indicating that Starlink latency follows primarily "
        "linear patterns driven by recent history and time-of-day. Sudden spike events "
        "caused by satellite handoffs remain difficult to predict from available features "
        "across all models."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance (Random Forest only, since it's available without XGB)
    with st.expander("Random Forest Feature Importance"):
        try:
            X_fi, y_fi = prepare_xy(forecast_df)
            rf_fi = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_fi.fit(X_fi, y_fi)
            fi_df = pd.DataFrame({
                "Feature":   X_fi.columns,
                "Importance": rf_fi.feature_importances_
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
            fig_fi = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h", marker_color=P1_COLOR
            ))
            _apply_base(fig_fi, "Feature Importance (Random Forest)", 380)
            fig_fi.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.caption(f"Feature importance unavailable: {e}")

    st.markdown("---")
    st.markdown("**Export Results**")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button(
            "Download Model Comparison CSV",
            model_table.to_csv(index=False).encode("utf-8"),
            "model_comparison_thesis.csv", "text/csv"
        )
    with col_e2:
        pred_df = pd.DataFrame({
            "actual":    np.array(latency_result["y_test"]),
            "predicted": np.array(latency_result["y_pred"])
        })
        st.download_button(
            "Download Latency Predictions CSV",
            pred_df.to_csv(index=False).encode("utf-8"),
            "latency_predictions_thesis.csv", "text/csv"
        )