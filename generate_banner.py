import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import pandas as pd

np.random.seed(42)

# ── Simulate daily returns ────────────────────────────────────────────────────
start = datetime(2022, 1, 3)
end   = datetime(2024, 12, 31)
dates = pd.bdate_range(start, end)
n     = len(dates)

# Portfolio: annualized ~18%, Sharpe ~1.4, max drawdown ~-12%
ann_ret_p  = 0.18
ann_vol_p  = ann_ret_p / 1.4          # ≈ 0.1286
daily_mu_p = ann_ret_p / 252
daily_sd_p = ann_vol_p / np.sqrt(252)

# Benchmark (IPC): annualized ~9%, higher vol
ann_ret_b  = 0.09
ann_vol_b  = 0.20
daily_mu_b = ann_ret_b / 252
daily_sd_b = ann_vol_b / np.sqrt(252)

# Correlated shocks
corr   = 0.45
cov    = [[daily_sd_p**2, corr*daily_sd_p*daily_sd_b],
          [corr*daily_sd_p*daily_sd_b, daily_sd_b**2]]
shocks = np.random.multivariate_normal([daily_mu_p, daily_mu_b], cov, n)

port_ret  = shocks[:, 0]
bench_ret = shocks[:, 1]

port_cum  = (1 + port_ret).cumprod() - 1
bench_cum = (1 + bench_ret).cumprod() - 1

# Drawdown
port_wealth = (1 + port_ret).cumprod()
rolling_max = np.maximum.accumulate(port_wealth)
drawdown    = (port_wealth - rolling_max) / rolling_max  # negative values

# ── Canvas ────────────────────────────────────────────────────────────────────
DPI       = 600
WIDTH_IN  = 1584 / 150   # physical size in inches (baseline 150 DPI reference)
HEIGHT_IN = 396  / 150   # at 600 DPI this renders to 6336×1584 px
fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN), dpi=DPI, facecolor="#0A0A0A")

gs = GridSpec(
    2, 1,
    figure=fig,
    height_ratios=[6.5, 1],   # drawdown ≈ 13% of plot area
    hspace=0.0,
    left=0.06, right=0.98,
    top=0.82,  bottom=0.18,
)

ax_main = fig.add_subplot(gs[0])
ax_dd   = fig.add_subplot(gs[1], sharex=ax_main)

# ── Styling helpers ───────────────────────────────────────────────────────────
TEAL    = "#00D4AA"
GRAY    = "#555555"
RED     = "#FF4444"
BG      = "#0A0A0A"
LGRAY   = "#888888"
GRID_C  = "#FFFFFF"

for ax in (ax_main, ax_dd):
    ax.set_facecolor(BG)
    ax.tick_params(colors=LGRAY, labelsize=5.5, length=2, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color(LGRAY)
    ax.yaxis.label.set_color(LGRAY)

# ── Main equity curve ─────────────────────────────────────────────────────────
x = dates

ax_main.plot(x, bench_cum * 100, color=GRAY,  lw=1.5, zorder=2)
ax_main.plot(x, port_cum  * 100, color=TEAL,  lw=2.5, zorder=3)

# Shaded alpha area between curves
ax_main.fill_between(
    x,
    bench_cum * 100,
    port_cum  * 100,
    where=(port_cum >= bench_cum),
    color=TEAL, alpha=0.15, zorder=1,
)
ax_main.fill_between(
    x,
    bench_cum * 100,
    port_cum  * 100,
    where=(port_cum < bench_cum),
    color=RED, alpha=0.10, zorder=1,
)

# Extremely subtle grid
ax_main.yaxis.grid(True, color=GRID_C, alpha=0.06, lw=0.4, linestyle="--")
ax_main.set_axisbelow(True)
ax_main.xaxis.grid(False)

# Y-axis formatting
ax_main.yaxis.set_tick_params(labelcolor=LGRAY)
ax_main.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%")
)
ax_main.tick_params(axis="y", labelsize=5.5)
plt.setp(ax_main.get_xticklabels(), visible=False)

# ── Drawdown panel ────────────────────────────────────────────────────────────
ax_dd.fill_between(x, drawdown * 100, 0, color=RED, alpha=0.40, zorder=2)
ax_dd.axhline(0, color=GRAY, lw=0.5, alpha=0.4)
ax_dd.yaxis.grid(True, color=GRID_C, alpha=0.05, lw=0.4, linestyle="--")
ax_dd.set_axisbelow(True)
ax_dd.xaxis.grid(False)

ax_dd.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
)
ax_dd.tick_params(axis="y", labelsize=5.0, labelcolor=LGRAY)

# X-axis: year labels
import matplotlib.dates as mdates
ax_dd.xaxis.set_major_locator(mdates.YearLocator())
ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_dd.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
ax_dd.tick_params(axis="x", which="major", labelsize=5.5, labelcolor=LGRAY)
ax_dd.tick_params(axis="x", which="minor", length=0)
ax_dd.set_xlim(
    pd.Timestamp("2022-01-01"),
    pd.Timestamp("2024-12-31"),
)

# ── Legend (top-left) ─────────────────────────────────────────────────────────
leg_x  = 0.062
leg_y  = 0.95
gap    = 0.11   # tight vertical gap between the two legend rows
legend_props = dict(transform=fig.transFigure, clip_on=False)

fig.text(leg_x,         leg_y,       "▬", color=TEAL, fontsize=7, va="top", **legend_props)
fig.text(leg_x + 0.013, leg_y,       "FMIA Portfolio",  color=TEAL, fontsize=6.0,
         va="top", fontweight="bold", **legend_props)

fig.text(leg_x,         leg_y - gap, "▬", color=GRAY, fontsize=7, va="top", **legend_props)
fig.text(leg_x + 0.013, leg_y - gap, "IPC Benchmark",   color=GRAY, fontsize=6.0,
         va="top", **legend_props)

# ── Bottom-right attribution ───────────────────────────────────────────────────
fig.text(0.97, 0.045,
         "FMIA — Systematic Equity Trading Pipeline",
         color="white", fontsize=6.0, ha="right", va="bottom",
         transform=fig.transFigure, alpha=0.85)
fig.text(0.97, 0.005,
         "Maximiliano Hidalgo León",
         color=LGRAY, fontsize=5.0, ha="right", va="bottom",
         transform=fig.transFigure, alpha=0.70)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "/Users/hidalgo/Documents/Repos/Fondo Mexico-VS/linkedin_banner.png"
fig.savefig(out, dpi=DPI, facecolor=BG)   # no bbox_inches — canvas stays exactly 1584x396
print(f"Saved → {out}")

from PIL import Image
img = Image.open(out)
print(f"Canvas: {img.size[0]}x{img.size[1]} px")
