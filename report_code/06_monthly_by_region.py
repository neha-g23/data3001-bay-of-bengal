"""Monthly switch split by region A-F (appendix figure): same measure as
05_monthly_switch.py, for each region. Months with fewer than 10 drifters or
under 10% of the region measured are marked as indicative only.

Needs regions.npy from 03_region_table.py.
Outputs: monthly_regions.csv, monthly_switch_by_region.png (six panels with 95% ranges),
monthly_switch_regions_combined.png (all regions on one chart, no ranges).
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from importlib import import_module
import reversal_figures as rf
from common import load, OUT, NB, SEED, NAVY, redraw

mm = import_module("05_monthly_switch")
TAGS = list("ABCDEF")
NAMES = {"A": "A India coast", "B": "B Sri Lanka", "C": "C South", "D": "D Central-west", "E": "E Central-east", "F": "F North"}
MONTHS = [rf.MONTH_NAMES[m-1] for m in mm.ORDER]

df = load(); grid = rf.Grid(0.2, df)
M = np.load(OUT + "regions.npy")
Wsw, Wm, pm, allid = mm.monthly_weights(df, grid)
full = mm.monthly_shares(Wsw, Wm, np.ones(len(allid)), list(M), grid)
rng = np.random.default_rng(SEED)
B = np.array([mm.monthly_shares(Wsw, Wm, redraw(rng, len(allid)), list(M), grid) for _ in range(NB)])
lo, hi = np.nanpercentile(B, 2.5, axis=0), np.nanpercentile(B, 97.5, axis=0)

# drifters and measured share of each region, per month
ii = np.clip(((pm.lat - 5) / 0.2).astype(int), 0, grid.LAT.shape[0] - 1)
jj = np.clip(((pm.lon - 78) / 0.2).astype(int), 0, grid.LAT.shape[1] - 1)
ve0, vn0 = Wsw.field(np.ones(len(allid))); s0 = np.hypot(ve0, vn0)
rows = []
for i, m in enumerate(mm.ORDER):
    ve, vn = Wm[m].field(np.ones(len(allid))); s = np.hypot(ve, vn)
    ok = np.isfinite(s) & np.isfinite(s0) & (s >= .1) & (s0 >= .1)
    for k, t in enumerate(TAGS):
        val = lambda a: round(a[i, k]) if np.isfinite(a[i, k]) else None
        rows.append(dict(month=rf.MONTH_NAMES[m-1], region=t,
                         drifters=pm[(pm.month == m).values & M[k][ii, jj]].drifter_id.nunique(),
                         covered_pct=round((ok & M[k]).sum() / max(1, (M[k] & ~grid.never).sum()) * 100),
                         turned=val(full), lo=val(lo), hi=val(hi)))
R = pd.DataFrame(rows); R.to_csv(OUT + "monthly_regions.csv", index=False)
print(R.pivot(index="month", columns="region", values="turned").reindex(MONTHS).to_string())


def region_data(t):
    d = R[R.region == t].set_index("month").reindex(MONTHS)
    ok = (d.drifters >= 10) & (d.covered_pct >= 10) & d.turned.notna()
    return d, ok


x = np.arange(12)
# ---- six panels with 95% ranges
COL = "#c2410c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
fig.subplots_adjust(left=.07, right=.98, top=.86, bottom=.15, hspace=.28, wspace=.08)
for ax, t in zip(axes.ravel(), TAGS):
    d, ok = region_data(t)
    ax.axvspan(-.5, 3.5, color="#fdf1e7", zorder=0); ax.axvspan(5.5, 8.5, color="#eaf1f9", zorder=0)
    ax.fill_between(x, d.lo.where(ok), d.hi.where(ok), color=COL, alpha=.18, lw=0)
    ax.plot(x, d.turned, color=COL, lw=1.5, ls=":", alpha=.6)
    ax.plot(x, d.turned.where(ok), color=COL, lw=3, marker="o", ms=8)
    weak = ~ok & d.turned.notna()
    ax.scatter(x[weak.values], d.turned[weak], s=60, facecolors="white", edgecolors=COL, lw=2, zorder=3)
    ax.axhline(50, color="grey", ls=":", lw=1)
    ax.set_title(NAMES[t], fontsize=18, fontweight="bold", color=NAVY, loc="left")
    ax.set_ylim(0, 105); ax.set_yticks([0, 50, 100]); ax.set_xlim(-.5, 11.5)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in MONTHS])
for ax in axes[:, 0]: ax.set_ylabel("% turned > 90°")
fig.suptitle("Monthly switch by region", fontsize=26, fontweight="bold", color=NAVY, y=.97)
fig.legend(handles=[Line2D([], [], color=COL, lw=3, marker="o", ms=8, label="Enough data (≥10 drifters, ≥10% of area)"),
                    Line2D([], [], color=COL, lw=0, marker="o", ms=8, mfc="white", mew=2, label="Too few data (indicative only)"),
                    plt.Rectangle((0, 0), 1, 1, color=COL, alpha=.18, label="95% range"),
                    plt.Rectangle((0, 0), 1, 1, color="#fdf1e7", label="Southwest monsoon"),
                    plt.Rectangle((0, 0), 1, 1, color="#eaf1f9", label="Northeast monsoon")],
           loc="lower center", ncol=5, frameon=False, fontsize=14)
fig.savefig(OUT + "monthly_switch_by_region.png", dpi=150, facecolor="white"); plt.close(fig)

# ---- all regions on one chart (months with enough data only)
cols = {"A": "#7a0177", "B": "#d95f02", "C": "#1b9e77", "D": "#e7298a", "E": "#2b6cb0", "F": "#66a61e"}
mk = {"A": "o", "B": "s", "C": "^", "D": "D", "E": "v", "F": "P"}
plt.rcParams.update({"font.size": 18})
fig, ax = plt.subplots(figsize=(15, 8.8)); fig.subplots_adjust(left=.1, right=.75, top=.86, bottom=.12)
ax.axvspan(-.5, 3.5, color="#fdf1e7", zorder=0); ax.axvspan(5.5, 8.5, color="#eaf1f9", zorder=0)
ax.text(1.5, 107, "Southwest monsoon", ha="center", fontsize=17, color="#a04000", fontweight="bold")
ax.text(7, 107, "Northeast monsoon", ha="center", fontsize=17, color="#1f4e8c", fontweight="bold")
for t in TAGS:
    d, ok = region_data(t)
    ax.plot(x, d.turned.where(ok), color=cols[t], lw=3, marker=mk[t], ms=10, label=NAMES[t])
ax.axhline(50, color="grey", ls=":", lw=1.5)
ax.set_xticks(x); ax.set_xticklabels(MONTHS); ax.set_xlim(-.5, 11.5)
ax.set_ylim(0, 104); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("% of region turned more than 90°")
ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), frameon=False, fontsize=17)
fig.text(.1, .02, "Only months with ≥10 drifters and ≥10% of the region measured are shown; gaps = too few data.", fontsize=14, color="#555")
ax.set_title("Monthly switch by region", fontsize=28, fontweight="bold", color=NAVY, pad=40)
fig.savefig(OUT + "monthly_switch_regions_combined.png", dpi=150, facecolor="white")
print("saved monthly_switch_by_region.png, monthly_switch_regions_combined.png")
