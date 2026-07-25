"""
Urban Heat Island (UHI) Simulation Dashboard — Bengaluru
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌡️ Bengaluru UHI Simulation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #ff6b6b; }
    .metric-box {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        border-radius: 12px; padding: 16px; text-align: center;
        border: 1px solid #334;
    }
    .metric-val { font-size: 2.2rem; font-weight: 700; color: #ff6b6b; }
    .metric-lbl { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .zone-card {
        background: #1a1a2e; border-radius: 10px; padding: 12px;
        border-left: 4px solid #ff6b6b; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ─── DATA LAYER ──────────────────────────────────────────────────────────────
@st.cache_data
def get_zone_data():
    """Real-world-inspired temperature & green cover data for Bengaluru zones."""
    zones = {
        "Whitefield":       {"base_temp": 38.5, "green_cover": 8,  "lat": 12.969, "lon": 77.750, "type": "IT/Industrial"},
        "Electronic City":  {"base_temp": 37.8, "green_cover": 10, "lat": 12.839, "lon": 77.676, "type": "IT/Industrial"},
        "Marathahalli":     {"base_temp": 37.2, "green_cover": 12, "lat": 12.959, "lon": 77.701, "type": "Residential"},
        "Koramangala":      {"base_temp": 36.5, "green_cover": 14, "lat": 12.934, "lon": 77.623, "type": "Residential"},
        "MG Road":          {"base_temp": 36.0, "green_cover": 11, "lat": 12.975, "lon": 77.607, "type": "Commercial"},
        "Yeshwanthpur":     {"base_temp": 35.5, "green_cover": 15, "lat": 13.022, "lon": 77.553, "type": "Mixed"},
        "Rajajinagar":      {"base_temp": 35.0, "green_cover": 18, "lat": 13.001, "lon": 77.556, "type": "Residential"},
        "Malleswaram":      {"base_temp": 34.5, "green_cover": 22, "lat": 13.004, "lon": 77.574, "type": "Heritage"},
        "Cubbon Park Area": {"base_temp": 31.5, "green_cover": 45, "lat": 12.976, "lon": 77.593, "type": "Green Zone"},
        "Lalbagh Area":     {"base_temp": 30.8, "green_cover": 55, "lat": 12.950, "lon": 77.585, "type": "Green Zone"},
        "Yelahanka":        {"base_temp": 33.5, "green_cover": 28, "lat": 13.100, "lon": 77.594, "type": "Suburban"},
        "Bannerghatta":     {"base_temp": 32.0, "green_cover": 38, "lat": 12.800, "lon": 77.576, "type": "Forest Fringe"},
    }
    df = pd.DataFrame(zones).T.reset_index()
    df.columns = ["Zone", "base_temp", "green_cover", "lat", "lon", "type"]
    return df

@st.cache_data
def get_historical_data():
    """Bengaluru historical temp vs green cover loss (1970–2024)."""
    years  = list(range(1970, 2025, 2))
    green  = [68, 65, 62, 58, 54, 50, 46, 42, 38, 34, 30, 27, 24, 21, 19, 17, 15, 14, 13, 12, 11, 10, 9, 8, 8, 7, 6, 6]
    temp   = [26.5, 26.7, 26.9, 27.1, 27.4, 27.7, 28.0, 28.4, 28.9, 29.3, 29.8, 30.2, 30.6, 31.0, 31.4, 31.8, 32.1, 32.4, 32.6, 32.9, 33.1, 33.3, 33.5, 33.7, 33.8, 33.9, 34.1, 34.2]
    return pd.DataFrame({"Year": years, "GreenCover_pct": green, "AvgTemp_C": temp})

def simulate_temp(base_temp, green_cover, cool_roofs_pct, water_bodies):
    """Physics-inspired temperature model."""
    green_effect  = (green_cover - 10) * 0.08   # Each % green cover over baseline lowers temp
    roof_effect   = cool_roofs_pct * 0.04        # Cool roofs reflection
    water_effect  = water_bodies * 0.6           # Evaporative cooling
    return max(base_temp - green_effect - roof_effect - water_effect, 22.0)

def make_heatmap_grid(df, green_boost=0, cool_roofs=0, water_bodies=0):
    """Create a 2D temperature grid interpolated over Bengaluru's lat/lon extent."""
    lats = np.linspace(12.78, 13.12, 60)
    lons = np.linspace(77.50, 77.78, 60)
    grid = np.zeros((60, 60))

    for _, row in df.iterrows():
        gc   = min(row["green_cover"] + green_boost, 100)
        temp = simulate_temp(row["base_temp"], gc, cool_roofs, water_bodies)
        i = int((row["lat"] - 12.78) / (13.12 - 12.78) * 59)
        j = int((row["lon"] - 77.50) / (77.78 - 77.50) * 59)
        i, j = np.clip(i, 0, 59), np.clip(j, 0, 59)
        for di in range(-7, 8):
            for dj in range(-7, 8):
                ni, nj = i+di, j+dj
                if 0 <= ni < 60 and 0 <= nj < 60:
                    dist  = np.sqrt(di**2 + dj**2) + 0.1
                    grid[ni][nj] += temp / dist

    # Normalize
    counts = np.zeros((60, 60))
    for _, row in df.iterrows():
        i = int((row["lat"] - 12.78) / (13.12 - 12.78) * 59)
        j = int((row["lon"] - 77.50) / (77.78 - 77.50) * 59)
        i, j = np.clip(i, 0, 59), np.clip(j, 0, 59)
        for di in range(-7, 8):
            for dj in range(-7, 8):
                ni, nj = i+di, j+dj
                if 0 <= ni < 60 and 0 <= nj < 60:
                    dist = np.sqrt(di**2 + dj**2) + 0.1
                    counts[ni][nj] += 1 / dist
    counts[counts == 0] = 1
    grid = grid / counts
    # Fill zeros with ambient
    grid[grid == 0] = 29.0
    return lats, lons, grid

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛠️ Intervention Controls")
st.sidebar.markdown("Simulate how urban greening reduces heat:")

green_boost  = st.sidebar.slider("🌳 Extra Green Cover (%)",    0, 40, 0, 1)
cool_roofs   = st.sidebar.slider("🏠 Cool Roofs Installed (%)", 0, 100, 0, 5)
water_bodies = st.sidebar.slider("💧 New Water Bodies (count)", 0, 10, 0, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Focus Zone")
df_zones = get_zone_data()
selected_zone = st.sidebar.selectbox("Select zone to inspect", df_zones["Zone"].tolist(), index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Made for:** Smart Cities Hackathon 2025\n\n**Tech:** Python · NumPy · Matplotlib · Streamlit")

# ─── HEADER ─────────────────────────────────────────────────────────────────
st.markdown("# 🌡️ Bengaluru Urban Heat Island (UHI) Simulation Tool")
st.markdown("> *Visualizing the invisible crisis: how concrete replaced canopy, and how we can fix it.*")
st.markdown("---")

# ─── SECTION 1: METRICS ─────────────────────────────────────────────────────
df = get_zone_data()
df["sim_temp"] = df.apply(
    lambda r: simulate_temp(r["base_temp"], min(r["green_cover"] + green_boost, 100), cool_roofs, water_bodies), axis=1
)
avg_before = df["base_temp"].mean()
avg_after  = df["sim_temp"].mean()
max_zone   = df.loc[df["sim_temp"].idxmax(), "Zone"]
min_zone   = df.loc[df["sim_temp"].idxmin(), "Zone"]
delta      = avg_before - avg_after

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{avg_before:.1f}°C</div><div class="metric-lbl">Avg Temp (Baseline)</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{avg_after:.1f}°C</div><div class="metric-lbl">Avg Temp (After Intervention)</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-box"><div class="metric-val">↓ {delta:.2f}°C</div><div class="metric-lbl">Temperature Reduction</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-box"><div class="metric-val">{max_zone}</div><div class="metric-lbl">Hottest Zone</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ─── SECTION 2: HEATMAPS ────────────────────────────────────────────────────
st.markdown("## 🗺️ Zone Temperature Heatmap")
st.markdown("Left: **Baseline (today)**. Right: **After your interventions** (adjust sliders on left).")

col_a, col_b = st.columns(2)

def plot_heatmap(ax, df, green_boost, cool_roofs, water_bodies, title):
    lats, lons, grid = make_heatmap_grid(df, green_boost, cool_roofs, water_bodies)
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(grid, cmap=cmap, vmin=28, vmax=40,
                   extent=[lons[0], lons[-1], lats[0], lats[-1]],
                   origin="lower", aspect="auto", alpha=0.85)
    # Plot zone dots
    for _, row in df.iterrows():
        gc   = min(row["green_cover"] + green_boost, 100)
        temp = simulate_temp(row["base_temp"], gc, cool_roofs, water_bodies)
        color = "lime" if temp < 33 else ("orange" if temp < 36 else "red")
        ax.scatter(row["lon"], row["lat"], c=color, s=80, zorder=5, edgecolors="white", linewidths=0.5)
        ax.annotate(row["Zone"].split()[0], (row["lon"], row["lat"]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6, color="white", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Surface Temp (°C)", shrink=0.8)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude", color="#aaa", fontsize=8)
    ax.set_ylabel("Latitude", color="#aaa", fontsize=8)
    ax.tick_params(colors="#aaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_facecolor("#1a1a2e")

with col_a:
    fig1, ax1 = plt.subplots(figsize=(5.5, 5), facecolor="#0e1117")
    plot_heatmap(ax1, df, 0, 0, 0, "🔴 Baseline — Current Urban Heat")
    st.pyplot(fig1)
    plt.close()

with col_b:
    fig2, ax2 = plt.subplots(figsize=(5.5, 5), facecolor="#0e1117")
    plot_heatmap(ax2, df, green_boost, cool_roofs, water_bodies, "🟢 After Intervention")
    st.pyplot(fig2)
    plt.close()

st.markdown("---")

# ─── SECTION 3: HISTORICAL TREND ────────────────────────────────────────────
st.markdown("## 📉 The Story: Green Cover Loss → Temperature Rise (1970–2024)")

hist = get_historical_data()
fig3, ax3 = plt.subplots(figsize=(11, 4), facecolor="#0e1117")
ax3.set_facecolor("#0e1117")

color_temp  = "#ff6b6b"
color_green = "#51cf66"

ax_r = ax3.twinx()
l1, = ax3.plot(hist["Year"], hist["AvgTemp_C"], color=color_temp, linewidth=2.5, label="Avg Temp (°C)")
ax3.fill_between(hist["Year"], hist["AvgTemp_C"], alpha=0.15, color=color_temp)
l2, = ax_r.plot(hist["Year"], hist["GreenCover_pct"], color=color_green, linewidth=2.5, linestyle="--", label="Green Cover (%)")
ax_r.fill_between(hist["Year"], hist["GreenCover_pct"], alpha=0.1, color=color_green)

# Annotate key events
events = {1985: "IT Boom begins", 2000: "Rapid urbanization", 2010: "Lake encroachments", 2020: "COVID pause"}
for yr, label in events.items():
    ax3.axvline(yr, color="#555", linestyle=":", alpha=0.7)
    ax3.text(yr + 0.3, 27.0, label, color="#888", fontsize=7, rotation=90, va="bottom")

ax3.set_xlabel("Year", color="#aaa")
ax3.set_ylabel("Average Temperature (°C)", color=color_temp)
ax_r.set_ylabel("Green Cover (%)", color=color_green)
ax3.tick_params(colors="#aaa")
ax_r.tick_params(colors="#aaa")
for spine in ax3.spines.values(): spine.set_edgecolor("#333")
for spine in ax_r.spines.values(): spine.set_edgecolor("#333")
ax3.set_title("Bengaluru: 54 Years of Canopy Loss & Temperature Rise", color="white", fontsize=12, fontweight="bold")
ax3.legend(handles=[l1, l2], facecolor="#1a1a2e", labelcolor="white", loc="upper left")
plt.tight_layout()
st.pyplot(fig3)
plt.close()

st.markdown("---")

# ─── SECTION 4: ZONE DETAIL ─────────────────────────────────────────────────
st.markdown("## 🔍 Zone Deep Dive")

col_x, col_y = st.columns([1, 2])

with col_x:
    zone_row  = df[df["Zone"] == selected_zone].iloc[0]
    gc_after  = min(zone_row["green_cover"] + green_boost, 100)
    temp_aft  = simulate_temp(zone_row["base_temp"], gc_after, cool_roofs, water_bodies)
    reduction = zone_row["base_temp"] - temp_aft

    st.markdown(f"### {selected_zone}")
    st.markdown(f"""
<div class="zone-card">
<b>Zone Type:</b> {zone_row['type']}<br>
<b>Baseline Temp:</b> {zone_row['base_temp']}°C<br>
<b>Green Cover (now):</b> {zone_row['green_cover']}%<br>
<b>Green Cover (after):</b> {gc_after}%<br>
<b>Simulated Temp:</b> {temp_aft:.1f}°C<br>
<b>Reduction Achieved:</b> ↓ {reduction:.2f}°C
</div>
""", unsafe_allow_html=True)

    # Gauge-style progress
    st.markdown(f"**Intervention Effectiveness**")
    effectiveness = min(reduction / 6 * 100, 100)
    st.progress(int(effectiveness))
    st.caption(f"{effectiveness:.0f}% of maximum possible cooling")

with col_y:
    # Sensitivity chart — how temp drops as green cover increases
    gc_range = np.arange(zone_row["green_cover"], 100, 2)
    temps_gc = [simulate_temp(zone_row["base_temp"], gc, cool_roofs, water_bodies) for gc in gc_range]

    fig4, ax4 = plt.subplots(figsize=(6, 3.5), facecolor="#0e1117")
    ax4.set_facecolor("#0e1117")
    ax4.plot(gc_range, temps_gc, color="#51cf66", linewidth=2.5)
    ax4.fill_between(gc_range, temps_gc, min(temps_gc)-0.5, alpha=0.2, color="#51cf66")
    ax4.axvline(gc_after, color="#ff6b6b", linestyle="--", label=f"Current setting ({gc_after}%)")
    ax4.axhline(temp_aft, color="#ffd43b", linestyle=":", alpha=0.6)
    ax4.scatter([gc_after], [temp_aft], color="#ff6b6b", s=80, zorder=5)
    ax4.set_xlabel("Green Cover (%)", color="#aaa")
    ax4.set_ylabel("Surface Temperature (°C)", color="#aaa")
    ax4.set_title(f"Temperature Sensitivity to Green Cover — {selected_zone}", color="white", fontsize=10)
    ax4.tick_params(colors="#aaa")
    ax4.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    for spine in ax4.spines.values(): spine.set_edgecolor("#333")
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

st.markdown("---")

# ─── SECTION 5: COMPARISON BAR CHART ────────────────────────────────────────
st.markdown("## 📊 All Zones — Before vs After Comparison")

fig5, ax5 = plt.subplots(figsize=(12, 5), facecolor="#0e1117")
ax5.set_facecolor("#0e1117")

x = np.arange(len(df))
w = 0.38
bars1 = ax5.bar(x - w/2, df["base_temp"], width=w, color="#ff6b6b", label="Baseline Temp", alpha=0.85, edgecolor="#0e1117")
bars2 = ax5.bar(x + w/2, df["sim_temp"],  width=w, color="#51cf66", label="After Intervention", alpha=0.85, edgecolor="#0e1117")

# Value labels
for bar in bars1:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{bar.get_height():.1f}", ha="center", va="bottom", color="#ff6b6b", fontsize=7)
for bar in bars2:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{bar.get_height():.1f}", ha="center", va="bottom", color="#51cf66", fontsize=7)

ax5.set_xticks(x)
ax5.set_xticklabels(df["Zone"], rotation=35, ha="right", color="#aaa", fontsize=8)
ax5.set_ylabel("Surface Temperature (°C)", color="#aaa")
ax5.set_title("Bengaluru Zones: Baseline vs Simulated Temperature After Intervention", color="white", fontsize=12, fontweight="bold")
ax5.tick_params(colors="#aaa")
ax5.legend(facecolor="#1a1a2e", labelcolor="white")
ax5.set_ylim(25, 42)
for spine in ax5.spines.values(): spine.set_edgecolor("#333")
plt.tight_layout()
st.pyplot(fig5)
plt.close()

st.markdown("---")

# ─── SECTION 6: IMPACT CALCULATOR ───────────────────────────────────────────
st.markdown("## 💡 Real-World Impact Calculator")

col_p, col_q = st.columns(2)

with col_p:
    pop_affected = st.number_input("Estimated population in hot zones (millions)", min_value=0.5, max_value=15.0, value=4.5, step=0.5)
    energy_per_deg = st.number_input("Energy saved per °C cooling per person (kWh/yr)", min_value=5, max_value=100, value=35)

with col_q:
    total_cooling = delta
    energy_saved = total_cooling * energy_per_deg * pop_affected * 1e6 / 1e9  # TWh
    co2_saved    = energy_saved * 0.82  # India grid factor ~0.82 kg CO2/kWh → MtCO2
    cost_saved   = energy_saved * 1e9 * 8.5 / 1e7  # ₹8.5/kWh → crore

    st.markdown(f"""
<div class="metric-box" style="margin-bottom:10px">
  <div class="metric-val">{energy_saved:.2f} TWh</div>
  <div class="metric-lbl">Energy Saved Annually</div>
</div>
<div class="metric-box" style="margin-bottom:10px">
  <div class="metric-val">{co2_saved:.2f} Mt CO₂</div>
  <div class="metric-lbl">Carbon Emissions Avoided</div>
</div>
<div class="metric-box">
  <div class="metric-val">₹{cost_saved:.0f} Cr</div>
  <div class="metric-lbl">Annual Electricity Cost Saved</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<center>
🌱 <b>Bengaluru Urban Heat Island Simulation Tool-Bengaluru</b> &nbsp;|&nbsp; 
Built with Python · Streamlit · NumPy · Matplotlib &nbsp;|&nbsp;
<i>Data shaped around real Bengaluru geography and BBMP/IMD reports</i>
</center>
""", unsafe_allow_html=True)