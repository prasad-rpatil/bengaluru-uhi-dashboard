"""
Urban Heat Island (UHI) Simulation — Bengaluru
Standalone version: runs with just matplotlib, numpy, pandas
Usage: python standalone_demo.py

Generates 3 figures:
  1. Temperature heatmap (baseline vs intervention)
  2. Historical green cover loss vs temperature rise (1970–2024)
  3. All-zones before/after bar chart + impact summary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor":  "#0e1117",
    "axes.facecolor":    "#0e1117",
    "text.color":        "white",
    "axes.labelcolor":   "#aaa",
    "xtick.color":       "#aaa",
    "ytick.color":       "#aaa",
    "axes.edgecolor":    "#333",
    "grid.color":        "#222",
})

# ─── ZONE DATA ───────────────────────────────────────────────────────────────
ZONES = {
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
df = pd.DataFrame(ZONES).T.reset_index()
df.columns = ["Zone", "base_temp", "green_cover", "lat", "lon", "type"]
df["base_temp"]   = df["base_temp"].astype(float)
df["green_cover"] = df["green_cover"].astype(float)
df["lat"]         = df["lat"].astype(float)
df["lon"]         = df["lon"].astype(float)

HIST_YEARS  = list(range(1970, 2025, 2))
HIST_GREEN  = [68, 65, 62, 58, 54, 50, 46, 42, 38, 34, 30, 27, 24, 21, 19, 17, 15, 14, 13, 12, 11, 10, 9, 8, 8, 7, 6, 6]
HIST_TEMP   = [26.5, 26.7, 26.9, 27.1, 27.4, 27.7, 28.0, 28.4, 28.9, 29.3, 29.8, 30.2, 30.6, 31.0, 31.4, 31.8, 32.1, 32.4, 32.6, 32.9, 33.1, 33.3, 33.5, 33.7, 33.8, 33.9, 34.1, 34.2]

# ─── PHYSICS MODEL ───────────────────────────────────────────────────────────
def simulate_temp(base_temp, green_cover, cool_roofs_pct=0, water_bodies=0):
    green_effect = (green_cover - 10) * 0.08
    roof_effect  = cool_roofs_pct * 0.04
    water_effect = water_bodies * 0.6
    return max(base_temp - green_effect - roof_effect - water_effect, 22.0)

def make_grid(df, green_boost=0, cool_roofs=0, water_bodies=0):
    grid   = np.full((60, 60), 29.0)
    counts = np.zeros((60, 60))
    for _, row in df.iterrows():
        gc   = min(row["green_cover"] + green_boost, 100)
        temp = simulate_temp(row["base_temp"], gc, cool_roofs, water_bodies)
        i = int((row["lat"] - 12.78) / (13.12 - 12.78) * 59)
        j = int((row["lon"] - 77.50) / (77.78 - 77.50) * 59)
        i, j = np.clip(i, 0, 59), np.clip(j, 0, 59)
        for di in range(-8, 9):
            for dj in range(-8, 9):
                ni, nj = i+di, j+dj
                if 0 <= ni < 60 and 0 <= nj < 60:
                    dist = np.sqrt(di**2 + dj**2) + 0.1
                    grid[ni][nj]   += temp / dist
                    counts[ni][nj] += 1 / dist
    counts[counts == 0] = 1
    return grid / counts

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — HEATMAPS (Baseline vs After Intervention)
# ═══════════════════════════════════════════════════════════════════════════
GREEN_BOOST  = 20   # ← CHANGE THIS to simulate different levels
COOL_ROOFS   = 40
WATER_BODIES = 3

print("⏳ Generating Figure 1: Heatmaps...")
grid_before = make_grid(df, 0, 0, 0)
grid_after  = make_grid(df, GREEN_BOOST, COOL_ROOFS, WATER_BODIES)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("🌡️ Urban Heat Island Simulation — Bengaluru", fontsize=15, fontweight="bold", color="white", y=1.01)

extent = [77.50, 77.78, 12.78, 13.12]
cmap   = plt.cm.RdYlGn_r

for ax, grid, title, subtitle in [
    (ax1, grid_before, "🔴 Baseline — Current Heat Map", "No interventions"),
    (ax2, grid_after,  "🟢 After Greening Interventions", f"+{GREEN_BOOST}% green, {COOL_ROOFS}% cool roofs, {WATER_BODIES} water bodies"),
]:
    im = ax.imshow(grid, cmap=cmap, vmin=28, vmax=40, extent=extent, origin="lower", aspect="auto", alpha=0.88)
    for _, row in df.iterrows():
        gc   = row["green_cover"] if ax == ax1 else min(row["green_cover"] + GREEN_BOOST, 100)
        temp = simulate_temp(row["base_temp"], gc, 0 if ax==ax1 else COOL_ROOFS, 0 if ax==ax1 else WATER_BODIES)
        c    = "#00ff88" if temp < 33 else ("#ffaa00" if temp < 36 else "#ff4444")
        ax.scatter(row["lon"], row["lat"], c=c, s=90, zorder=5, edgecolors="white", linewidths=0.7)
        ax.annotate(row["Zone"].split()[0], (row["lon"], row["lat"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7, color="white", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Surface Temp (°C)", shrink=0.85)
    ax.set_title(f"{title}\n{subtitle}", color="white", fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.tight_layout()
plt.savefig("figure1_heatmaps.png", dpi=150, bbox_inches="tight", facecolor="#0e1117")
print("   ✅ Saved: figure1_heatmaps.png")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — HISTORICAL TREND (1970–2024)
# ═══════════════════════════════════════════════════════════════════════════
print("⏳ Generating Figure 2: Historical trend...")

fig2, ax_t = plt.subplots(figsize=(13, 5))
ax_g = ax_t.twinx()

l1, = ax_t.plot(HIST_YEARS, HIST_TEMP,  color="#ff6b6b", linewidth=2.5, label="Avg Surface Temp (°C)")
ax_t.fill_between(HIST_YEARS, HIST_TEMP, min(HIST_TEMP)-0.5, alpha=0.15, color="#ff6b6b")

l2, = ax_g.plot(HIST_YEARS, HIST_GREEN, color="#51cf66", linewidth=2.5, linestyle="--", label="Green Cover (%)")
ax_g.fill_between(HIST_YEARS, HIST_GREEN, 0, alpha=0.1, color="#51cf66")

# Annotate key events
events = {1985: "IT Boom\nbegins", 2000: "Rapid\nurbanization", 2010: "Lake\nencroachments", 2020: "COVID\npause"}
for yr, lbl in events.items():
    ax_t.axvline(yr, color="#555", linestyle=":", alpha=0.6)
    ax_t.text(yr+0.4, 27.2, lbl, color="#aaa", fontsize=7.5, rotation=0, va="bottom",
              bbox=dict(facecolor="#1a1a2e", edgecolor="#444", boxstyle="round,pad=0.2", alpha=0.85))

ax_t.set_xlabel("Year", fontsize=11)
ax_t.set_ylabel("Average Temperature (°C)", color="#ff6b6b", fontsize=11)
ax_g.set_ylabel("Green Cover (%)", color="#51cf66", fontsize=11)
ax_t.tick_params(axis="y", labelcolor="#ff6b6b")
ax_g.tick_params(axis="y", labelcolor="#51cf66")
fig2.suptitle("Bengaluru: 54 Years of Canopy Loss & Temperature Rise (1970–2024)",
              fontsize=13, fontweight="bold", color="white")
ax_t.legend(handles=[l1, l2], facecolor="#1a1a2e", labelcolor="white", loc="upper left", fontsize=10)

# Correlation annotation
corr = np.corrcoef(HIST_GREEN, HIST_TEMP)[0, 1]
ax_t.text(0.98, 0.05, f"Correlation: r = {corr:.3f}", transform=ax_t.transAxes,
          color="white", fontsize=9, ha="right",
          bbox=dict(facecolor="#ff6b6b", alpha=0.8, boxstyle="round,pad=0.4"))

plt.tight_layout()
plt.savefig("figure2_historical.png", dpi=150, bbox_inches="tight", facecolor="#0e1117")
print("   ✅ Saved: figure2_historical.png")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — ZONE COMPARISON + IMPACT SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("⏳ Generating Figure 3: Zone comparison & impact...")

df["sim_temp"] = df.apply(
    lambda r: simulate_temp(r["base_temp"], min(r["green_cover"] + GREEN_BOOST, 100), COOL_ROOFS, WATER_BODIES), axis=1)

avg_b   = df["base_temp"].mean()
avg_a   = df["sim_temp"].mean()
delta   = avg_b - avg_a

fig3 = plt.figure(figsize=(14, 8))
gs   = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.45, wspace=0.35)

# ── Bar chart ─────────────────────────────────────────────────────────────
ax_bar = fig3.add_subplot(gs[0, :])
x = np.arange(len(df))
w = 0.38
b1 = ax_bar.bar(x - w/2, df["base_temp"], width=w, color="#ff6b6b", label="Baseline Temp (°C)", alpha=0.88, edgecolor="#0e1117")
b2 = ax_bar.bar(x + w/2, df["sim_temp"],  width=w, color="#51cf66", label="After Intervention (°C)", alpha=0.88, edgecolor="#0e1117")

for bar in b1:
    ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{bar.get_height():.1f}",
                ha="center", va="bottom", color="#ff6b6b", fontsize=7)
for bar in b2:
    ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{bar.get_height():.1f}",
                ha="center", va="bottom", color="#51cf66", fontsize=7)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(df["Zone"], rotation=30, ha="right", fontsize=8)
ax_bar.set_ylabel("Surface Temperature (°C)")
ax_bar.set_title("All Bengaluru Zones: Baseline vs Post-Intervention Temperature", fontsize=12, fontweight="bold", color="white")
ax_bar.legend(facecolor="#1a1a2e", labelcolor="white")
ax_bar.set_ylim(25, 42)
ax_bar.axhline(avg_b, color="#ff6b6b", linestyle="--", alpha=0.5, linewidth=1)
ax_bar.axhline(avg_a, color="#51cf66", linestyle="--", alpha=0.5, linewidth=1)
ax_bar.text(len(df)-0.5, avg_b + 0.2, f"Avg {avg_b:.1f}°C", color="#ff6b6b", fontsize=8)
ax_bar.text(len(df)-0.5, avg_a + 0.2, f"Avg {avg_a:.1f}°C", color="#51cf66", fontsize=8)

# ── Scatter: Green cover vs Temp ──────────────────────────────────────────
ax_sc = fig3.add_subplot(gs[1, 0])
sc    = ax_sc.scatter(df["green_cover"], df["base_temp"], c=df["base_temp"],
                       cmap="RdYlGn_r", s=120, edgecolors="white", linewidths=0.5, zorder=5)
for _, row in df.iterrows():
    ax_sc.annotate(row["Zone"].split()[0], (row["green_cover"], row["base_temp"]),
                   textcoords="offset points", xytext=(5, 3), fontsize=6.5, color="#ccc")
# Regression line
m, b_r = np.polyfit(df["green_cover"], df["base_temp"], 1)
xr = np.linspace(df["green_cover"].min(), df["green_cover"].max(), 100)
ax_sc.plot(xr, m*xr+b_r, color="#ffd43b", linewidth=1.5, linestyle="--", label=f"Trend (slope={m:.2f}°C/%)")
ax_sc.set_xlabel("Green Cover (%)")
ax_sc.set_ylabel("Surface Temp (°C)")
ax_sc.set_title("Green Cover vs Temperature", fontsize=10, fontweight="bold", color="white")
ax_sc.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

# ── Impact summary text box ────────────────────────────────────────────────
ax_imp = fig3.add_subplot(gs[1, 1])
ax_imp.axis("off")

pop       = 4.5e6
kwh_deg   = 35
e_saved   = delta * kwh_deg * pop / 1e9   # TWh
co2_saved = e_saved * 0.82                 # MtCO2
cost_cr   = e_saved * 1e9 * 8.5 / 1e7     # crore INR
trees_eq  = delta * 0.08 / 0.001 * len(df) / 1000  # thousand trees equivalent

summary = f"""
  INTERVENTION SUMMARY
  ─────────────────────────────────
  🌳 Extra Green Cover:   +{GREEN_BOOST}%
  🏠 Cool Roofs:          {COOL_ROOFS}%
  💧 Water Bodies Added:  {WATER_BODIES}

  TEMPERATURE IMPACT
  ─────────────────────────────────
  Avg Before:  {avg_b:.2f} °C
  Avg After:   {avg_a:.2f} °C
  Reduction:   ↓ {delta:.2f} °C

  REAL-WORLD IMPACT (4.5M residents)
  ─────────────────────────────────
  ⚡ Energy Saved:   {e_saved:.2f} TWh/yr
  🌿 CO₂ Avoided:   {co2_saved:.2f} Mt/yr
  💰 Cost Saved:    ₹{cost_cr:.0f} Crore/yr
"""
ax_imp.text(0.05, 0.95, summary, transform=ax_imp.transAxes,
            fontsize=9, va="top", fontfamily="monospace", color="#e0e0e0",
            bbox=dict(facecolor="#1a1a2e", edgecolor="#51cf66", boxstyle="round,pad=0.8", linewidth=1.5))
ax_imp.set_title("Impact Summary", fontsize=10, fontweight="bold", color="white")

fig3.patch.set_facecolor("#0e1117")
plt.savefig("figure3_impact.png", dpi=150, bbox_inches="tight", facecolor="#0e1117")
print("   ✅ Saved: figure3_impact.png")
plt.show()

print("\n🎉 All 3 figures generated successfully!")
print("   figure1_heatmaps.png  — UHI map before/after")
print("   figure2_historical.png — 54 year trend")
print("   figure3_impact.png    — Zone comparison + impact")
print(f"\n📊 Key result: {delta:.2f}°C average cooling with your intervention settings")
print(f"   Hottest zone: {df.loc[df['base_temp'].idxmax(), 'Zone']} at {df['base_temp'].max():.1f}°C baseline")
print(f"   Coolest zone: {df.loc[df['base_temp'].idxmin(), 'Zone']} at {df['base_temp'].min():.1f}°C baseline")