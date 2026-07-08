import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="AEMO Grid Demand Forecaster", layout="wide", page_icon="⚡")

# ---------------- Design tokens (matches climate-economic-risk-explorer) ----------------
BG_MAIN = "#0a120e"
BG_CARD = "#10201a"
BORDER = "#1f3a2e"
ACCENT = "#64FFDA"
ACCENT2 = "#FFD166"
ACCENT3 = "#FF6B6B"
TEXT = "#e8f5ee"
SUBTEXT = "#8fae9d"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG_MAIN}; font-family: {FONT}; }}
    h1, h2, h3, h4 {{ font-family: {FONT} !important; color: {TEXT} !important; font-weight: 800 !important; letter-spacing: -0.3px; }}
    p, span, label, .stMarkdown {{ font-family: {FONT} !important; color: {TEXT}; }}
    [data-testid="stCaptionContainer"] {{ color: {SUBTEXT} !important; }}

    [data-testid="stMetric"] {{
        background-color: {BG_CARD};
        border: 1px solid {BORDER};
        border-left: 3px solid {ACCENT};
        border-radius: 10px;
        padding: 16px 18px;
    }}
    [data-testid="stMetricLabel"] {{ color: {SUBTEXT} !important; font-size: 12.5px !important; }}
    [data-testid="stMetricValue"] {{ color: {ACCENT} !important; font-weight: 800 !important; }}
    [data-testid="stMetricDelta"] {{ font-weight: 600 !important; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        color: {SUBTEXT}; font-family: {FONT}; font-weight: 600; font-size: 13.5px;
        background-color: transparent; border-radius: 8px 8px 0 0; padding: 10px 16px;
    }}
    .stTabs [aria-selected="true"] {{ color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important; }}

    [data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {BG_CARD}; border: 1px solid {BORDER}; border-radius: 12px;
    }}
    hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "temp_max", "temp_min", "temp_mean", "precip_sum", "wind_max",
    "dow", "month", "is_weekend", "doy", "year", "is_holiday",
    "demand_mean_lag1", "demand_mean_lag7", "temp_mean_lag1", "cdd", "hdd",
]
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def style_fig(fig, height=420, title=None):
    layout_kwargs = dict(
        paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
        font=dict(color=TEXT, family=FONT, size=12),
        height=height,
        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, l=10, r=10, b=10),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(size=14, color=TEXT))
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig

@st.cache_data
def load_daily():
    return pd.read_csv("data/model_ready.csv", parse_dates=["date"])

@st.cache_data
def load_raw():
    df = pd.read_csv("data/nem_price_demand.csv", parse_dates=["SETTLEMENTDATE"])
    df["time_of_day"] = df["SETTLEMENTDATE"].dt.hour + df["SETTLEMENTDATE"].dt.minute / 60
    df["dow"] = df["SETTLEMENTDATE"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

@st.cache_data
def load_diagnostics():
    return pd.read_csv("data/diagnostics.csv", parse_dates=["date"])

@st.cache_resource
def load_model():
    return joblib.load("models/lgbm_final.pkl")

@st.cache_data
def load_results():
    with open("models/walk_forward_results.json") as f:
        return json.load(f)

df = load_daily()
raw = load_raw()
diag = load_diagnostics()
model = load_model()
results = load_results()

st.title("⚡ AEMO NSW Grid Demand Forecaster")
st.caption("LightGBM demand model trained on 3 years of AEMO NEM half-hourly data + Open-Meteo Sydney weather. "
           "Walk-forward validated: 3.24% MAPE (5-fold avg) · 2.47% MAPE on a clean 90-day holdout.")

tabs = st.tabs([
    "📈 History & Forecast", "🗓️ Seasonal Patterns", "⏰ Intraday Profile",
    "📉 Load Duration & Peaks", "💰 Price & Demand", "🔗 Correlations & Diagnostics",
    "🏆 Model Comparison", "🌡️ What-If Simulator",
])

with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest daily demand (MW avg)", f"{df['demand_mean'].iloc[-1]:,.0f}")
    col2.metric("Latest temp mean (°C)", f"{df['temp_mean'].iloc[-1]:.1f}")
    col3.metric("Walk-forward MAPE (5-fold avg)", "3.24%")
    col4.metric("Holdout MAPE (90-day, clean)", f"{diag['abs_pct_error'].mean():.2f}%")

    st.write("")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["demand_mean"], name="Actual Demand (MW avg)", line=dict(color=ACCENT)))
    fig.add_trace(go.Scatter(x=df["date"], y=df["temp_mean"] * 30, name="Temp (scaled, °C)", line=dict(color=ACCENT3, dash="dot"), yaxis="y2"))
    fig.update_layout(yaxis=dict(title="Demand (MW)"), yaxis2=dict(title="Temp (scaled)", overlaying="y", side="right", showgrid=False))
    st.plotly_chart(style_fig(fig, 450), use_container_width=True)

    st.subheader("30-day rolling average demand")
    df_roll = df.copy()
    df_roll["rolling_30d"] = df_roll["demand_mean"].rolling(30).mean()
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=df_roll["date"], y=df_roll["demand_mean"], name="Daily", line=dict(color=BORDER, width=1)))
    fig_roll.add_trace(go.Scatter(x=df_roll["date"], y=df_roll["rolling_30d"], name="30-day avg", line=dict(color=ACCENT, width=3)))
    fig_roll.update_layout(yaxis_title="Demand (MW)")
    st.plotly_chart(style_fig(fig_roll, 350), use_container_width=True)

with tabs[1]:
    st.subheader("Demand distribution by month")
    df_m = df.copy()
    df_m["month_name"] = df_m["month"].apply(lambda m: MONTH_NAMES[m-1])
    fig_box = px.box(df_m, x="month_name", y="demand_mean", category_orders={"month_name": MONTH_NAMES},
                      color_discrete_sequence=[ACCENT])
    fig_box.update_layout(xaxis_title="Month", yaxis_title="Daily avg demand (MW)")
    st.plotly_chart(style_fig(fig_box), use_container_width=True)
    st.caption("Twin peaks confirm classic NEM seasonality: summer aircon load (Dec-Feb) and winter heating load (Jun-Aug), "
               "with shoulder-season troughs in Apr-May and Oct-Nov.")

    st.subheader("Day-of-week x Month heatmap")
    pivot = df_m.pivot_table(index="dow", columns="month_name", values="demand_mean", aggfunc="mean")
    pivot = pivot.reindex(index=range(7), columns=MONTH_NAMES)
    fig_hm = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=[DOW_NAMES[i] for i in pivot.index],
                                   colorscale=[[0, BG_CARD], [0.5, "#2d6a53"], [1, ACCENT]], colorbar=dict(title="MW")))
    st.plotly_chart(style_fig(fig_hm, 400), use_container_width=True)

with tabs[2]:
    st.subheader("Average half-hourly load shape")
    profile = raw.groupby(["time_of_day", "is_weekend"])["TOTALDEMAND"].mean().reset_index()
    profile["day_type"] = profile["is_weekend"].map({0: "Weekday", 1: "Weekend"})
    fig_profile = px.line(profile, x="time_of_day", y="TOTALDEMAND", color="day_type",
                           color_discrete_map={"Weekday": ACCENT, "Weekend": ACCENT2})
    fig_profile.update_layout(xaxis_title="Hour of day", yaxis_title="Demand (MW)")
    st.plotly_chart(style_fig(fig_profile, 450), use_container_width=True)
    st.caption("Weekday profile shows the classic double-hump (morning + evening peak); weekend demand is flatter "
               "and shifted later, with no sharp morning ramp.")

with tabs[3]:
    st.subheader("Load duration curve")
    sorted_demand = raw["TOTALDEMAND"].sort_values(ascending=False).reset_index(drop=True)
    pct_time = (sorted_demand.index + 1) / len(sorted_demand) * 100
    fig_ldc = go.Figure(go.Scatter(x=pct_time, y=sorted_demand, line=dict(color=ACCENT)))
    fig_ldc.update_layout(xaxis_title="% of time demand is at or above this level", yaxis_title="Demand (MW)")
    st.plotly_chart(style_fig(fig_ldc), use_container_width=True)
    st.caption("Standard power-industry chart: the steepness of the left-hand tail shows how much capacity exists "
                "only to serve rare peak events — a key driver of grid infrastructure cost.")

    st.subheader("Top 10 peak demand days")
    peaks = df.nlargest(10, "demand_max")[["date", "demand_max", "demand_mean", "temp_max", "temp_min"]]
    peaks.columns = ["Date", "Peak Demand (MW)", "Avg Demand (MW)", "Max Temp (°C)", "Min Temp (°C)"]
    st.dataframe(peaks.style.format({"Peak Demand (MW)": "{:,.0f}", "Avg Demand (MW)": "{:,.0f}",
                                       "Max Temp (°C)": "{:.1f}", "Min Temp (°C)": "{:.1f}"}),
                 use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Price vs demand relationship")
    sample = raw.sample(min(20000, len(raw)), random_state=42)
    fig_pd = px.scatter(sample, x="TOTALDEMAND", y="RRP", opacity=0.3, color_discrete_sequence=[ACCENT])
    fig_pd.update_layout(xaxis_title="Demand (MW)", yaxis_title="Price ($/MWh)")
    st.plotly_chart(style_fig(fig_pd, 450), use_container_width=True)

    neg_price_pct = (raw["RRP"] < 0).mean() * 100
    spike_price_pct = (raw["RRP"] > 300).mean() * 100
    col1, col2 = st.columns(2)
    col1.metric("Intervals with negative price", f"{neg_price_pct:.2f}%")
    col2.metric("Intervals with price > $300/MWh", f"{spike_price_pct:.2f}%")
    st.caption("Negative prices happen when renewable supply outpaces demand and generators pay to keep running; "
               "price spikes cluster at the high-demand tail, visible as the upward scatter at the right edge.")

with tabs[5]:
    st.subheader("Feature correlation with demand")
    corr_cols = ["demand_mean", "temp_mean", "temp_max", "temp_min", "cdd", "hdd",
                 "precip_sum", "wind_max", "is_weekend", "is_holiday", "demand_mean_lag1", "demand_mean_lag7"]
    corr = df[corr_cols].corr()
    fig_corr = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                     colorscale=[[0, ACCENT3], [0.5, BG_CARD], [1, ACCENT]], zmid=0, colorbar=dict(title="r")))
    st.plotly_chart(style_fig(fig_corr, 500), use_container_width=True)

    st.subheader("Model diagnostics — clean 90-day holdout (not seen during training)")
    fig_diag = px.scatter(diag, x="demand_mean", y="predicted", opacity=0.6, color_discrete_sequence=[ACCENT])
    min_v, max_v = diag["demand_mean"].min(), diag["demand_mean"].max()
    fig_diag.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(color=ACCENT3, dash="dash"), name="Perfect fit"))
    fig_diag.update_layout(xaxis_title="Actual (MW)", yaxis_title="Predicted (MW)")
    st.plotly_chart(style_fig(fig_diag), use_container_width=True)

    fig_resid = go.Figure(go.Scatter(x=diag["date"], y=diag["residual"], mode="markers", marker=dict(color=ACCENT, size=6)))
    fig_resid.add_hline(y=0, line_dash="dash", line_color=ACCENT3)
    fig_resid.update_layout(xaxis_title="Date", yaxis_title="Residual (MW)")
    st.plotly_chart(style_fig(fig_resid, 350, title="Residuals over the holdout window"), use_container_width=True)
    st.metric("Holdout MAPE", f"{diag['abs_pct_error'].mean():.2f}%")

with tabs[6]:
    res_df = pd.DataFrame(results)
    st.dataframe(res_df[["fold", "test_start", "test_end", "lightgbm_mape", "ets_mape", "naive_mape"]]
                 .rename(columns={"lightgbm_mape": "LightGBM MAPE %", "ets_mape": "ETS MAPE %", "naive_mape": "Naive MAPE %"}),
                 use_container_width=True, hide_index=True)

    avg = res_df[["lightgbm_mape", "ets_mape", "naive_mape"]].mean()
    fig2 = go.Figure(go.Bar(x=["LightGBM", "ETS (Holt-Winters)", "Naive (lag-7)"],
                             y=[avg["lightgbm_mape"], avg["ets_mape"], avg["naive_mape"]],
                             marker_color=[ACCENT, ACCENT2, ACCENT3]))
    fig2.update_layout(yaxis_title="Avg MAPE %")
    st.plotly_chart(style_fig(fig2, 400, title="5-Fold Walk-Forward Comparison"), use_container_width=True)

    st.subheader("Feature importance (LightGBM gain)")
    imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_}).sort_values("importance", ascending=True)
    fig3 = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h", marker_color=ACCENT))
    st.plotly_chart(style_fig(fig3, 500), use_container_width=True)

with tabs[7]:
    st.write("Adjust tomorrow's forecast conditions to see the predicted demand shift.")
    last = df.iloc[-1]
    c1, c2 = st.columns(2)
    temp_mean = c1.slider("Forecast temp mean (°C)", 5.0, 45.0, float(last["temp_mean"]))
    is_weekend = c2.selectbox("Day type", ["Weekday", "Weekend"], index=int(last["is_weekend"]))

    row = last.copy()
    row["temp_mean"] = temp_mean
    row["temp_max"] = temp_mean + 4
    row["temp_min"] = temp_mean - 4
    row["is_weekend"] = 1 if is_weekend == "Weekend" else 0
    row["cdd"] = max(temp_mean - 18, 0)
    row["hdd"] = max(18 - temp_mean, 0)

    X = pd.DataFrame([row[FEATURES]])
    pred = model.predict(X)[0]
    st.metric("Predicted daily avg demand (MW)", f"{pred:,.0f}", delta=f"{pred - last['demand_mean']:,.0f} vs last observed")