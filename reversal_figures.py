"""
Monsoon reversal figures for the Bay of Bengal (DATA3001).

Builds every reversal figure from the cached Global Drifter Program subset
(bob_subset.nc): seasonal and monthly speed maps, east-west maps,
direction-change maps and charts, reversal vs speed, reversal vs distance
from the coast, and the coastal-current (EICC) timeline. Each map can be
made from drogued drifters only or from drogued + undrogued drifters.

Method in one paragraph
-----------------------
Hourly observations are cleaned (speeds above 3 m/s removed), labelled by
monsoon (Southwest = Jun-Sep, Northeast = Dec-Feb) and averaged within each
drifter in each 1-degree cell (at least 12 hourly observations), so one
drifter cannot dominate. These per-drifter means are then smoothed onto a
fine grid with a Gaussian kernel on great-circle distance (75 km for seasons,
100 km for single months, both chosen by leave-drifters-out cross-validation
in the main project code). A location is only shown when at least 3
different drifters lie within 100 km of it.

Usage
-----
    python reversal_figures.py --data bob_subset.nc --out figures
    python reversal_figures.py --drifters drogued      # main figures only
    python reversal_figures.py --drifters all          # robustness versions only

Requirements: numpy, pandas, matplotlib, xarray, netCDF4 (see requirements.txt).
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

# ============================================================
# Settings
# ============================================================
EARTH_RADIUS_KM = 6371.0
REGION = dict(lat=(5.0, 23.0), lon=(78.0, 99.0))
EXTENT = [78, 99, 5, 23]

MAX_SPEED = 3.0              # m/s, implausible above this
MIN_OBS_PER_DRIFTER = 12     # hourly obs a drifter needs in a cell to count
MIN_DRIFTERS = 3             # distinct drifters within SUPPORT_KM to show a location
SUPPORT_KM = 100             # radius for the coverage rule (independent of bandwidth)
BANDWIDTH_SEASON_KM = 75     # from cross-validation (seasonal maps)
BANDWIDTH_MONTH_KM = 100     # from cross-validation (monthly maps)
KERNEL_CUTOFF = 3.0          # ignore data beyond 3 bandwidths
MIN_SPEED_DIRECTION = 0.10   # m/s; slower currents have no meaningful direction
NEVER_SAMPLED_KM = 40        # no observation within this distance = land / never visited

SEASON_MONTHS = {"Southwest": [6, 7, 8, 9], "Northeast": [12, 1, 2]}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
REVERSAL_COLOURS = ["#fef0d9", "#fdcc8a", "#fc8d59", "#b30000"]   # 0-45-90-135-180 degrees

# Approximate western boundary (lon, lat): India's east coast, then Sri Lanka
INDIA_COAST = np.array([(87.0, 21.0), (86.7, 20.3), (84.9, 19.3), (83.3, 17.7), (82.3, 16.6),
                        (81.2, 16.3), (80.2, 15.5), (80.3, 13.1), (79.9, 10.3)])
LANKA_COAST = np.array([(79.9, 9.8), (80.4, 9.5), (81.2, 8.6), (81.9, 7.3),
                        (81.8, 6.5), (80.6, 5.9)])


# ============================================================
# Data
# ============================================================
def load_observations(path):
    """Hourly observations as a DataFrame with a drifter id on every row."""
    import xarray as xr
    ds = xr.open_dataset(path)
    df = pd.DataFrame({
        "time": pd.to_datetime(ds["time"].values),
        "lat": ds["lat"].values,
        "lon": ds["lon"].values,
        "ve": ds["ve"].values,
        "vn": ds["vn"].values,
        "drogue": ds["drogue_status"].values.astype(bool),
        "drifter_id": np.repeat(ds["id"].values, ds["rowsize"].values.astype(int)),
    })
    return clean_observations(df)


def clean_observations(df):
    """Drop missing values and implausible speeds; add month, season and 1-degree cell."""
    df = df.dropna(subset=["time", "lat", "lon", "ve", "vn"]).copy()
    df = df[np.hypot(df.ve, df.vn) <= MAX_SPEED]
    df["month"] = pd.to_datetime(df["time"]).dt.month
    df["season"] = "Inter"
    for name, months in SEASON_MONTHS.items():
        df.loc[df.month.isin(months), "season"] = name
    df["lat_bin"] = np.floor(df.lat).astype(int)
    df["lon_bin"] = np.floor(df.lon).astype(int)
    return df


def select_drifters(df, which):
    """which = 'drogued' (main analysis) or 'all' (drogued + undrogued)."""
    return df[df.drogue].copy() if which == "drogued" else df.copy()


def per_drifter_means(df, group):
    """Average within each drifter in each 1-degree cell (drifter weighting).

    group: 'season' or 'month'. Returns one row per (group, cell, drifter)
    with the drifter's mean velocity and mean position in that cell.
    """
    keys = [group, "lat_bin", "lon_bin", "drifter_id"]
    out = (df.groupby(keys)
             .agg(ve=("ve", "mean"), vn=("vn", "mean"),
                  lat=("lat", "mean"), lon=("lon", "mean"), n=("ve", "size"))
             .reset_index())
    return out[out.n >= MIN_OBS_PER_DRIFTER]


# ============================================================
# Kernel smoothing
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance matrix (km): rows = points 1, columns = points 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2[None, :] - lat1[:, None]) / 2) ** 2
         + np.cos(lat1)[:, None] * np.cos(lat2)[None, :]
         * np.sin((lon2[None, :] - lon1[:, None]) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class Grid:
    """Target grid plus a mask of places no drifter has ever visited (land, mostly)."""

    def __init__(self, res, all_obs):
        lat1 = np.arange(REGION["lat"][0] + res / 2, REGION["lat"][1], res)
        lon1 = np.arange(REGION["lon"][0] + res / 2, REGION["lon"][1], res)
        self.LON, self.LAT = np.meshgrid(lon1, lat1)
        self.res = res
        obs = all_obs[["lat", "lon"]].values[::20]          # thinned, enough for a mask
        tl, tn = self.LAT.ravel(), self.LON.ravel()
        never = np.ones(tl.size, bool)
        for i in range(0, tl.size, 2000):
            d = haversine_km(tl[i:i+2000], tn[i:i+2000], obs[:, 0], obs[:, 1])
            never[i:i+2000] = (d > NEVER_SAMPLED_KM).all(axis=1)
        self.never = never.reshape(self.LAT.shape)


def smooth_field(pts, grid, bandwidth_km):
    """Gaussian-kernel mean of ve and vn on the grid.

    Returns ve, vn (NaN where unsupported) and `low`, the ocean locations
    with fewer than MIN_DRIFTERS distinct drifters within SUPPORT_KM.
    """
    tl, tn = grid.LAT.ravel(), grid.LON.ravel()
    codes, _ = pd.factorize(pts.drifter_id)
    onehot = np.zeros((len(pts), codes.max() + 1))
    onehot[np.arange(len(pts)), codes] = 1.0
    ve = np.empty(tl.size); vn = np.empty(tl.size); nn = np.empty(tl.size)
    for i in range(0, tl.size, 2000):
        d = haversine_km(tl[i:i+2000], tn[i:i+2000], pts.lat.values, pts.lon.values)
        w = np.exp(-0.5 * (d / bandwidth_km) ** 2)
        w[d > KERNEL_CUTOFF * bandwidth_km] = 0.0
        ws = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ve[i:i+2000] = w @ pts.ve.values / ws
            vn[i:i+2000] = w @ pts.vn.values / ws
        nn[i:i+2000] = (((d <= SUPPORT_KM).astype(float) @ onehot) > 0).sum(axis=1)
    ve, vn, nn = (a.reshape(grid.LAT.shape) for a in (ve, vn, nn))
    low = (nn < MIN_DRIFTERS) & ~grid.never
    ve[low | grid.never] = np.nan
    vn[low | grid.never] = np.nan
    return ve, vn, low


def direction_change(ve1, vn1, ve2, vn2):
    """Angle (0-180 degrees) between two velocity fields."""
    s1, s2 = np.hypot(ve1, vn1), np.hypot(ve2, vn2)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (ve1 * ve2 + vn1 * vn2) / (s1 * s2)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def area_share(mask, condition, lat):
    """Area-weighted % of `mask` where `condition` holds (cells shrink with latitude)."""
    w = np.cos(np.radians(lat[mask]))
    return (w * condition[mask]).sum() / w.sum() * 100 if w.sum() > 0 else np.nan


# ============================================================
# Shared map drawing
# ============================================================
def style_map(ax, small=False):
    ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal")
    if small:
        ax.set_xticks([80, 85, 90, 95]); ax.set_yticks([5, 10, 15, 20])
        ax.tick_params(labelsize=7)


def draw_low_support(ax, grid, low, hatched=False):
    field = np.where(low, 1.0, np.nan)
    if hatched:
        plt.rcParams["hatch.color"] = "#9a9a9a"; plt.rcParams["hatch.linewidth"] = 0.5
        ax.contourf(grid.LON, grid.LAT, field, levels=[0.5, 1.5], colors=["white"], hatches=["////"])
    else:
        ax.contourf(grid.LON, grid.LAT, field, levels=[0.5, 1.5], colors=["#e3e3e3"])


def draw_arrows(ax, grid, ve, vn, step, white=True, scale=7, width=0.0038):
    s = (slice(step // 2, None, step), slice(step // 2, None, step))
    kw = dict(color="white", edgecolor="black", linewidth=0.4) if white else dict(color="#222222")
    return ax.quiver(grid.LON[s], grid.LAT[s], ve[s], vn[s], scale=scale, width=width, **kw)


def eicc_box(ax):
    ax.plot([80, 85, 85, 80, 80], [5, 5, 23, 23, 5], "r--", lw=0.9)


def drifter_label(which):
    return "drogued drifters" if which == "drogued" else "drogued + undrogued drifters"


# ============================================================
# Figures
# ============================================================
def seasonal_fields(df, which, grid):
    pts = per_drifter_means(select_drifters(df, which), "season")
    return {s: smooth_field(pts[pts.season == s], grid, BANDWIDTH_SEASON_KM) for s in SEASON_MONTHS}, pts


def fig_seasonal_speed(fields, pts, grid, which, out):
    levels = np.linspace(0, 0.6, 25)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.6))
    fig.subplots_adjust(left=0.05, right=0.88, top=0.84, bottom=0.12, wspace=0.08)
    step = int(round(1 / grid.res))
    for ax, s, lab in zip(axes, SEASON_MONTHS, "ab"):
        ve, vn, low = fields[s]
        draw_low_support(ax, grid, low)
        cf = ax.contourf(grid.LON, grid.LAT, np.hypot(ve, vn), levels=levels, cmap="viridis", extend="max")
        q = draw_arrows(ax, grid, ve, vn, step)
        eicc_box(ax); style_map(ax)
        ax.set_title(f"({lab}) {s} monsoon ({pts[pts.season == s].drifter_id.nunique()} drifters)", fontsize=13)
        ax.set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[1].quiverkey(q, 0.82, -0.11, 0.5, "0.5 m/s", labelpos="E", coordinates="axes")
    cb = fig.colorbar(cf, cax=fig.add_axes([0.90, 0.16, 0.016, 0.64]))
    cb.set_label("Current speed (m/s)", fontsize=12); cb.set_ticks(np.arange(0, 0.61, 0.1))
    fig.suptitle(f"Kernel-smoothed surface circulation, {drifter_label(which)} "
                 f"({BANDWIDTH_SEASON_KM} km bandwidth)\nLight grey: fewer than {MIN_DRIFTERS} drifters "
                 f"within {SUPPORT_KM} km. White: land or no data. Red box: approximate EICC band.", fontsize=13)
    fig.savefig(out, dpi=140); plt.close(fig)


def fig_seasonal_eastwest(fields, grid, which, out):
    levels = np.linspace(-0.5, 0.5, 25)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.2))
    fig.subplots_adjust(left=0.05, right=0.86, top=0.86, bottom=0.13, wspace=0.08)
    step = int(round(1 / grid.res))
    for ax, s, lab in zip(axes, SEASON_MONTHS, "ab"):
        ve, vn, low = fields[s]
        draw_low_support(ax, grid, low, hatched=True)
        cf = ax.contourf(grid.LON, grid.LAT, ve, levels=levels, cmap="coolwarm", extend="both")
        q = draw_arrows(ax, grid, ve, vn, step, white=False)
        ok = np.isfinite(ve)
        east = area_share(ok, ve > 0, grid.LAT)
        style_map(ax)
        ax.set_title(f"({lab}) {s} monsoon: {east:.0f}% of mapped area flows east", fontsize=13)
        ax.set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[1].quiverkey(q, 0.80, -0.11, 0.5, "0.5 m/s", labelpos="E", coordinates="axes")
    cax = fig.add_axes([0.88, 0.16, 0.016, 0.62])
    cb = fig.colorbar(cf, cax=cax, ticks=[-0.5, -0.25, 0, 0.25, 0.5])
    cb.set_label("East-west current (m/s)", fontsize=11)
    cax.text(0.5, 1.03, "Eastward", transform=cax.transAxes, ha="center", va="bottom", color="#b40426", fontweight="bold")
    cax.text(0.5, -0.03, "Westward", transform=cax.transAxes, ha="center", va="top", color="#3b4cc0", fontweight="bold")
    fig.text(0.05, 0.02, f"Arrows: current direction and speed. Hatched: fewer than {MIN_DRIFTERS} drifters "
             f"within {SUPPORT_KM} km (no estimate).", fontsize=9)
    fig.suptitle(f"East-west surface currents reverse between monsoons "
                 f"(kernel-smoothed, {drifter_label(which)}, {BANDWIDTH_SEASON_KM} km)", fontsize=14)
    fig.savefig(out, dpi=140); plt.close(fig)


def seasonal_angle(fields, grid):
    """Direction change between seasons, plus masks for slow and missing areas."""
    (ve1, vn1, _), (ve2, vn2, _) = fields["Southwest"], fields["Northeast"]
    s1, s2 = np.hypot(ve1, vn1), np.hypot(ve2, vn2)
    both = np.isfinite(s1) & np.isfinite(s2)
    fast = both & (s1 >= MIN_SPEED_DIRECTION) & (s2 >= MIN_SPEED_DIRECTION)
    angle = direction_change(ve1, vn1, ve2, vn2)
    either = ~grid.never                      # every ocean location ever sampled
    return dict(angle=angle, both=both, fast=fast, either=either, s1=s1, s2=s2)


def fig_direction_change_map(A, grid, which, out, annotate=False):
    fig, ax = plt.subplots(figsize=(13.5 if annotate else 10.5, 8.2 if annotate else 7.6))
    fig.subplots_adjust(left=0.06, right=0.68 if annotate else 0.74, top=0.88, bottom=0.08)
    ax.contourf(grid.LON, grid.LAT, np.where(A["either"], 1.0, np.nan), levels=[0.5, 1.5], colors=["#ececec"])
    ax.contourf(grid.LON, grid.LAT, np.where(A["both"], 1.0, np.nan), levels=[0.5, 1.5], colors=["#bdbdbd"])
    im = ax.contourf(grid.LON, grid.LAT, np.where(A["fast"], A["angle"], np.nan),
                     levels=[0, 45, 90, 135, 180], colors=REVERSAL_COLOURS)
    style_map(ax); ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    if annotate:
        box = dict(boxstyle="round,pad=0.3", fc="white", ec="#444444", lw=0.8, alpha=0.92)
        for lo, hi in [(5, 10), (10, 15), (15, 20), (20, 23)]:
            m = A["fast"] & (grid.LAT >= lo) & (grid.LAT < hi)
            if hi < 23:
                ax.axhline(hi, color="#333333", lw=0.9, ls="--", zorder=5)
            if m.sum() == 0:
                continue
            note = "\n(few data)" if m.sum() < 100 else ""
            share = area_share(m, A["angle"] > 90, grid.LAT)
            ax.text(99.3, (lo + hi) / 2, f"{lo}–{hi}°N\n{share:.0f}% reversed{note}", va="center",
                    ha="left", fontsize=10, bbox=box, zorder=6, clip_on=False)
        ax.add_patch(Rectangle((80, 5), 5, 18, fill=False, ec="#1f4e9c", lw=1.4, ls="-.", zorder=5))
        mc = A["fast"] & (grid.LON >= 80) & (grid.LON < 85)
        ax.text(82.5, 22.4, f"Coastal band 80–85°E: {area_share(mc, A['angle'] > 90, grid.LAT):.0f}% reversed",
                ha="center", va="center", fontsize=9.5, color="#1f4e9c", bbox=box, zorder=6)
    cax = fig.add_axes([0.845 if annotate else 0.76, 0.36 if annotate else 0.32, 0.02, 0.42])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([22.5, 67.5, 112.5, 157.5])
    cb.ax.set_yticklabels(["0–45°\nsame direction", "45–90°\nturned", "90–135°\nmostly reversed",
                           "135–180°\nfully reversed"], fontsize=9)
    cax.set_title("Change in direction\n(SW vs NE monsoon)", fontsize=10, loc="left", pad=10)
    fig.text(0.845 if annotate else 0.76, 0.30 if annotate else 0.26,
             f"Dark grey: current slower than\n{MIN_SPEED_DIRECTION} m/s in a season (direction\nnot meaningful)\n"
             "Light grey: too few drifters in\none or both seasons", fontsize=8, va="top")
    fig.suptitle(f"Where do the surface currents reverse between monsoons?\n"
                 f"{drifter_label(which).capitalize()}, kernel-smoothed ({BANDWIDTH_SEASON_KM} km)", fontsize=14)
    fig.savefig(out, dpi=140); plt.close(fig)
    return area_share(A["fast"], A["angle"] > 90, grid.LAT)


def fig_direction_change_chart(A, grid, which, out):
    fast, angle, LAT, LON = A["fast"], A["angle"], grid.LAT, grid.LON
    ang, lat_p, lon_p = angle[fast], LAT[fast], LON[fast]
    w = np.cos(np.radians(lat_p))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.14, wspace=0.28)
    ax = axes[0]
    bins = np.arange(0, 181, 10)
    hist, _ = np.histogram(ang, bins=bins, weights=w)
    hist = hist / hist.sum() * 100
    ax.bar(bins[:-1] + 5, hist, width=9.2, edgecolor="#555555", linewidth=0.5,
           color=[REVERSAL_COLOURS[min(int(b // 45), 3)] for b in bins[:-1]])
    ax.axhline(100 / 18, color="grey", ls=":", lw=1.2)
    ax.text(2, 100 / 18 + 0.35, "if directions were random", fontsize=8.5, color="grey")
    for x in (45, 90, 135):
        ax.axvline(x, color="#777777", lw=0.6, ls="--")
    ax.set_ylim(0, hist.max() * 1.22)
    for a, b in [(0, 45), (45, 90), (90, 135), (135, 181)]:
        sh = (w * ((ang >= a) & (ang < b))).sum() / w.sum() * 100
        ax.text((a + min(b, 180)) / 2, hist.max() * 1.12, f"{sh:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks([0, 45, 90, 135, 180]); ax.set_xlim(0, 180)
    ax.set_xlabel("Change in direction between monsoons (degrees)"); ax.set_ylabel("% of mapped Bay")
    ax.set_title("(a) How far the current turns between monsoons", fontsize=12)

    ax = axes[1]
    regions = [("5–10°N", (lat_p >= 5) & (lat_p < 10)), ("10–15°N", (lat_p >= 10) & (lat_p < 15)),
               ("15–20°N", (lat_p >= 15) & (lat_p < 20)), ("North of 20°N\n(few data)", lat_p >= 20),
               ("Coastal band\n80–85°E", (lon_p >= 80) & (lon_p < 85)), ("Whole Bay", np.ones_like(lat_p, bool))]
    names, vals = [], []
    for nm, m in regions:
        if m.sum() >= 30:
            names.append(nm); vals.append((w[m] * (ang[m] > 90)).sum() / w[m].sum() * 100)
    y = np.arange(len(names))[::-1]
    cols = ["#b30000" if v >= 75 else "#fc8d59" for v in vals]; cols[-1] = "#555555"
    ax.barh(y, vals, color=cols, height=0.6)
    ax.axvline(50, color="grey", ls=":", lw=1.2)
    ax.text(51, -0.75, "random = 50%", fontsize=8.5, color="grey"); ax.set_ylim(-0.95, len(names) - 0.4)
    for yy, v in zip(y, vals):
        ax.text(min(v + 1.5, 88), yy, f"{v:.0f}%", va="center", fontsize=10, color="white" if v > 90 else "black")
    ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 100)
    ax.set_xlabel("% of area turning more than 90°"); ax.set_title("(b) Where the reversal is strongest", fontsize=12)
    fig.suptitle(f"Change in surface current direction between the Southwest and Northeast monsoons\n"
                 f"{drifter_label(which).capitalize()}, kernel-smoothed ({BANDWIDTH_SEASON_KM} km); "
                 f"currents slower than {MIN_SPEED_DIRECTION} m/s excluded", fontsize=13)
    fig.savefig(out, dpi=140); plt.close(fig)


def fig_reversal_vs_speed(A, grid, which, out):
    ok = A["both"]
    ang = A["angle"][ok]; spd = ((A["s1"] + A["s2"]) / 2)[ok]; w = np.cos(np.radians(grid.LAT[ok]))
    edges = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 2.0]
    labels = ["<0.05", "0.05–0.10", "0.10–0.15", "0.15–0.20", "0.20–0.25", "0.25–0.30",
              "0.30–0.40", "0.40–0.60", ">0.60"]
    rows, meds = [], []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (spd >= lo) & (spd < hi)
        if m.sum() < 20:
            continue
        rows.append((lab, (w[m] * (ang[m] > 90)).sum() / w[m].sum() * 100, int(m.sum())))
        meds.append(((lo + hi) / 2 if hi < 2 else 0.7, np.median(ang[m])))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.17, wspace=0.25)
    ax = axes[0]; x = np.arange(len(rows)); vals = [r[1] for r in rows]
    ax.bar(x, vals, edgecolor="#555555", linewidth=0.5,
           color=["#fdcc8a" if v < 60 else "#fc8d59" if v < 80 else "#b30000" for v in vals])
    ax.axhline(50, color="grey", ls=":", lw=1.2); ax.text(-0.45, 46, "random = 50%", fontsize=8.5, color="grey")
    for xi, (lab, v, n) in zip(x, rows):
        ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
        ax.text(xi, 3, f"n={n}", ha="center", fontsize=7.5, color="white" if v >= 80 else "#333333")
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right"); ax.set_ylim(0, 105)
    ax.set_xlabel("Average current speed of the two monsoons (m/s)")
    ax.set_ylabel("% of locations turning more than 90°")
    ax.set_title("(a) Faster currents reverse more often", fontsize=12)
    ax = axes[1]
    ax.scatter(spd, ang, s=3, alpha=0.15, color="#555555", rasterized=True)
    ax.plot(*zip(*meds), "o-", color="#b30000", lw=2.2, label="median per speed class")
    ax.axhline(90, color="grey", ls=":", lw=1)
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 180); ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Average current speed of the two monsoons (m/s)"); ax.set_ylabel("Change in direction (degrees)")
    ax.set_title("(b) Each map location, with the median trend", fontsize=12); ax.legend(frameon=False, loc="lower right")
    fig.suptitle(f"Is the reversal stronger where currents are stronger?\n{drifter_label(which).capitalize()}, "
                 f"kernel-smoothed ({BANDWIDTH_SEASON_KM} km), every location mapped in both monsoons", fontsize=13)
    fig.savefig(out, dpi=140); plt.close(fig)


def _densify(pts, k=40):
    return np.vstack([np.linspace(pts[i], pts[i + 1], k) for i in range(len(pts) - 1)])


def fig_coast_distance(A, grid, which, out):
    coast = np.vstack([_densify(INDIA_COAST), _densify(LANKA_COAST)])
    ok = A["both"]; lat_p, lon_p = grid.LAT[ok], grid.LON[ok]
    dist = np.concatenate([haversine_km(lat_p[i:i+2000], lon_p[i:i+2000], coast[:, 1], coast[:, 0]).min(1)
                           for i in range(0, lat_p.size, 2000)])
    ang = A["angle"][ok]; spd = ((A["s1"] + A["s2"]) / 2)[ok]; w = np.cos(np.radians(lat_p))
    edges = [0, 50, 100, 150, 200, 300, 500, 2000]
    labs = ["0–50", "50–100", "100–150", "150–200", "200–300", "300–500", ">500"]
    rows = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labs):
        m = (dist >= lo) & (dist < hi)
        if m.sum() < 20:
            continue
        f = m & (spd >= MIN_SPEED_DIRECTION)
        rows.append(dict(band=lab, n=int(m.sum()), speed=np.median(spd[m]),
                         rev=(w[m] * (ang[m] > 90)).sum() / w[m].sum() * 100,
                         rev_fast=(w[f] * (ang[f] > 90)).sum() / w[f].sum() * 100 if f.sum() > 20 else np.nan,
                         med_angle=np.median(ang[m])))
    R = pd.DataFrame(rows); x = np.arange(len(R))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.80, bottom=0.17, wspace=0.28)
    ax = axes[0]
    ax.bar(x, R.speed, color="#3b7ea1", edgecolor="#333333", linewidth=0.5)
    for xi, v, n in zip(x, R.speed, R.n):
        ax.text(xi, v + 0.005, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(xi, 0.006, f"n={n}", ha="center", fontsize=7.5, color="white")
    ax.set_ylabel("Median current speed (m/s)\n(average of the two monsoons)"); ax.set_title("(a) Current speed", fontsize=12)
    ax = axes[1]
    ax.plot(x, R.rev, "o-", color="#b30000", lw=2.2, label="all locations")
    ax.plot(x, R.rev_fast, "s--", color="#fc8d59", lw=1.6, label=f"currents ≥ {MIN_SPEED_DIRECTION} m/s only")
    for xi, v in zip(x, R.rev):
        ax.text(xi, v + 2.5, f"{v:.0f}%", ha="center", fontsize=10, color="#b30000")
    ax.axhline(50, color="grey", ls=":", lw=1.1); ax.set_ylim(0, 105)
    ax.set_ylabel("% of locations turning more than 90°"); ax.set_title("(b) Share that reverses (>90°)", fontsize=12)
    ax.legend(frameon=False, loc="lower left")
    ax = axes[2]
    ax.plot(x, R.med_angle, "o-", color="#7a0177", lw=2.2)
    for xi, v in zip(x, R.med_angle):
        ax.text(xi, v + 4, f"{v:.0f}°", ha="center", fontsize=10, color="#7a0177")
    ax.axhline(180, color="grey", ls=":", lw=1); ax.axhline(90, color="grey", ls=":", lw=1)
    ax.set_ylim(0, 190); ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_ylabel("Median change in direction (degrees)"); ax.set_title("(c) How completely it reverses", fontsize=12)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(R.band)
        ax.set_xlabel("Distance from India's east coast and Sri Lanka (km)")
    fig.suptitle(f"Are currents faster and the reversal stronger near the western boundary?\n"
                 f"{drifter_label(which).capitalize()}, kernel-smoothed ({BANDWIDTH_SEASON_KM} km), "
                 f"locations mapped in both monsoons", fontsize=13)
    fig.savefig(out, dpi=140); plt.close(fig)
    return R


def monthly_fields(df, which, grid):
    pts = per_drifter_means(select_drifters(df, which), "month")
    fields = {m: smooth_field(pts[pts.month == m], grid, BANDWIDTH_MONTH_KM) for m in range(1, 13)}
    return fields, pts.groupby("month").drifter_id.nunique()


def fig_monthly_speed(fields, nd, grid, which, out):
    months = [6, 7, 8, 9, 12, 1, 2]
    levels = np.linspace(0, 0.6, 25)
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.6), constrained_layout=True)
    step = int(round(1 / grid.res))
    for ax, m in zip(axes.ravel(), months + [None]):
        if m is None:
            ax.set_visible(False); continue
        ve, vn, low = fields[m]
        draw_low_support(ax, grid, low)
        cf = ax.contourf(grid.LON, grid.LAT, np.hypot(ve, vn), levels=levels, cmap="viridis", extend="max")
        q = draw_arrows(ax, grid, ve, vn, step, scale=6, width=0.006)
        eicc_box(ax); style_map(ax)
        ax.set_title(f"{MONTH_NAMES[m-1]}  ({int(nd.get(m, 0))} drifters)", fontsize=13); ax.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("Southwest monsoon\nLatitude (°N)", fontsize=12)
    axes[1, 0].set_ylabel("Northeast monsoon\nLatitude (°N)", fontsize=12)
    axes[0, 3].quiverkey(q, 0.75, -0.09, 0.5, "0.5 m/s", labelpos="E", coordinates="axes")
    cb = fig.colorbar(cf, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("Current speed (m/s)", fontsize=12); cb.set_ticks(np.arange(0, 0.61, 0.1))
    fig.suptitle(f"Monthly kernel-smoothed surface currents, {drifter_label(which)} "
                 f"({BANDWIDTH_MONTH_KM} km bandwidth)\nLight grey: fewer than {MIN_DRIFTERS} drifters within "
                 f"{SUPPORT_KM} km. White: land or no data. Red box: approximate EICC band.", fontsize=13)
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_monthly_eastwest(fields, nd, grid, which, out):
    order = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
    season_tag = {m: "SW" for m in SEASON_MONTHS["Southwest"]} | {m: "NE" for m in SEASON_MONTHS["Northeast"]}
    levels = np.linspace(-0.5, 0.5, 25)
    fig, axes = plt.subplots(3, 4, figsize=(17, 13.2))
    fig.subplots_adjust(left=0.04, right=0.88, top=0.90, bottom=0.05, hspace=0.22, wspace=0.06)
    step = int(round(1 / grid.res))
    for ax, m in zip(axes.ravel(), order):
        ve, vn, low = fields[m]
        draw_low_support(ax, grid, low, hatched=True)
        cf = ax.contourf(grid.LON, grid.LAT, ve, levels=levels, cmap="coolwarm", extend="both")
        q = draw_arrows(ax, grid, ve, vn, step, white=False, scale=6, width=0.0045)
        ok = np.isfinite(ve)
        east = area_share(ok, ve > 0, grid.LAT) if ok.sum() > 50 else np.nan
        style_map(ax, small=True)
        tag = f" [{season_tag[m]}]" if m in season_tag else ""
        ax.set_title(f"{MONTH_NAMES[m-1]}{tag}: {east:.0f}% flows east ({int(nd.get(m, 0))} drifters)", fontsize=10.5)
    axes[0, 3].quiverkey(q, 0.80, -0.12, 0.5, "0.5 m/s", labelpos="E", coordinates="axes")
    cax = fig.add_axes([0.90, 0.30, 0.016, 0.40])
    cb = fig.colorbar(cf, cax=cax, ticks=[-0.5, -0.25, 0, 0.25, 0.5]); cb.set_label("East-west current (m/s)", fontsize=11)
    cax.text(0.5, 1.03, "Eastward", transform=cax.transAxes, ha="center", va="bottom", color="#b40426", fontweight="bold")
    cax.text(0.5, -0.03, "Westward", transform=cax.transAxes, ha="center", va="top", color="#3b4cc0", fontweight="bold")
    fig.suptitle(f"Monthly east-west surface currents, starting from the Southwest monsoon (June)\n"
                 f"{drifter_label(which).capitalize()}, kernel-smoothed ({BANDWIDTH_MONTH_KM} km). "
                 f"SW = Southwest monsoon, NE = Northeast monsoon. Hatched: fewer than {MIN_DRIFTERS} drifters nearby.",
                 fontsize=13)
    fig.savefig(out, dpi=120); plt.close(fig)


def fig_monthly_reversal(fields, nd, grid, which, out_maps, out_line, ref_month=7):
    """Direction change of every month relative to July (peak Southwest monsoon)."""
    ve0, vn0, _ = fields[ref_month]; s0 = np.hypot(ve0, vn0)
    angle, summary = {}, []
    for m in range(1, 13):
        ve, vn, _ = fields[m]; s = np.hypot(ve, vn)
        ok = np.isfinite(s) & np.isfinite(s0) & (s >= MIN_SPEED_DIRECTION) & (s0 >= MIN_SPEED_DIRECTION)
        ang = direction_change(ve, vn, ve0, vn0)
        angle[m] = (np.where(ok, ang, np.nan), np.isfinite(s) & np.isfinite(s0),
                    (np.isfinite(s) | np.isfinite(s0)) & ~grid.never)
        summary.append((m, area_share(ok, ang > 90, grid.LAT) if ok.sum() > 50 else np.nan, int(nd.get(m, 0))))
    S = pd.DataFrame(summary, columns=["month", "share_gt90", "drifters"])

    fig, axes = plt.subplots(3, 4, figsize=(17, 13.2))
    fig.subplots_adjust(left=0.04, right=0.86, top=0.90, bottom=0.04, hspace=0.22, wspace=0.06)
    for ax, m in zip(axes.ravel(), range(1, 13)):
        ang, both, either = angle[m]
        ax.contourf(grid.LON, grid.LAT, np.where(either, 1.0, np.nan), levels=[0.5, 1.5], colors=["#ececec"])
        ax.contourf(grid.LON, grid.LAT, np.where(both, 1.0, np.nan), levels=[0.5, 1.5], colors=["#bdbdbd"])
        im = ax.contourf(grid.LON, grid.LAT, ang, levels=[0, 45, 90, 135, 180], colors=REVERSAL_COLOURS)
        style_map(ax, small=True)
        sh = S.loc[S.month == m, "share_gt90"].item()
        lab = "reference" if m == ref_month else (f"{sh:.0f}% reversed" if np.isfinite(sh) else "too little data")
        ax.set_title(f"{MONTH_NAMES[m-1]} ({int(nd.get(m, 0))} drifters): {lab}", fontsize=10.5)
    cax = fig.add_axes([0.88, 0.35, 0.018, 0.35])
    cb = fig.colorbar(im, cax=cax); cb.set_ticks([22.5, 67.5, 112.5, 157.5])
    cb.ax.set_yticklabels([f"0–45°\nsame as {MONTH_NAMES[ref_month-1]}", "45–90°\nturned",
                           "90–135°\nmostly reversed", "135–180°\nfully reversed"], fontsize=9)
    fig.text(0.875, 0.30, f"Dark grey: slower than {MIN_SPEED_DIRECTION} m/s\nin this month or "
             f"{MONTH_NAMES[ref_month-1]}\nLight grey: data in only one\nof the two months", fontsize=8.5, va="top")
    fig.suptitle(f"How far has the current turned from its {MONTH_NAMES[ref_month-1]} (peak Southwest monsoon) "
                 f"direction?\n{drifter_label(which).capitalize()}, kernel-smoothed ({BANDWIDTH_MONTH_KM} km)", fontsize=14)
    fig.savefig(out_maps, dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.3), constrained_layout=True)
    ax.axvspan(5.5, 9.5, color="#d95f02", alpha=0.08)
    ax.axvspan(0.5, 2.5, color="#1b9e77", alpha=0.08); ax.axvspan(11.5, 12.5, color="#1b9e77", alpha=0.08)
    ax.axhline(50, color="grey", ls=":", lw=1)
    ax.text(12.4, 51.5, f"50% = no relation to {MONTH_NAMES[ref_month-1]}", ha="right", fontsize=8, color="grey")
    ax.plot(S.month, S.share_gt90, "o-", color="#b30000", lw=2)
    for _, r in S.iterrows():
        if np.isfinite(r.share_gt90):
            ax.text(r.month, r.share_gt90 + 3, f"{r.share_gt90:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_NAMES); ax.set_ylim(0, 100)
    ax.set_ylabel(f"% of mapped Bay turned >90°\nfrom the {MONTH_NAMES[ref_month-1]} direction")
    ax.set_title(f"When does the circulation turn away from the Southwest monsoon pattern? ({drifter_label(which)})",
                 fontsize=10.5)
    ax.text(7.5, 93, "SW monsoon", ha="center", color="#d95f02", fontsize=9)
    ax.text(1.5, 93, "NE monsoon", ha="center", color="#1b9e77", fontsize=9)
    fig.savefig(out_line, dpi=140); plt.close(fig)
    return S


def fig_eicc_timeline(df, which, out, band_km=150, n_boot=2000, seed=42):
    """Monthly along-coast velocity within band_km of India's east coast, with drifter bootstrap."""
    rng = np.random.default_rng(seed)
    d = select_drifters(df, which)
    near = d[d.lat.between(10, 21.5) & (d.lon < 90)].copy()
    coast = _densify(INDIA_COAST[::-1])
    dist = np.concatenate([haversine_km(near.lat.values[i:i+50000], near.lon.values[i:i+50000],
                                        coast[:, 1], coast[:, 0]).min(1) for i in range(0, len(near), 50000)])
    band = near[dist <= band_km].copy()
    ang = np.radians(40)                                   # coast runs ~40 degrees east of north
    band["along"] = band.ve * np.sin(ang) + band.vn * np.cos(ang)   # + = poleward
    band["segment"] = np.where(band.lat < 16, "South (10–16°N)", "North (16–21°N)")

    def monthly(sub):
        out = []
        for m in range(1, 13):
            g = sub[sub.month == m].groupby("drifter_id").agg(v=("along", "mean"), n=("along", "size"))
            v = g[g.n >= MIN_OBS_PER_DRIFTER].v.values
            if len(v) < 3:
                out.append((m, np.nan, np.nan, np.nan, len(v))); continue
            boot = rng.choice(v, (n_boot, len(v)), replace=True).mean(1)
            out.append((m, v.mean(), *np.percentile(boot, [2.5, 97.5]), len(v)))
        return pd.DataFrame(out, columns=["month", "mean", "lo", "hi", "n_drifters"])

    whole = monthly(band)
    segs = {s: monthly(band[band.segment == s]) for s in ["South (10–16°N)", "North (16–21°N)"]}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True, sharey=True)
    for ax, data, title in [(axes[0], {"Whole coastal band": whole}, "(a) East India Coastal Current band"),
                            (axes[1], segs, "(b) South vs north of 16°N")]:
        ax.axvspan(5.5, 9.5, color="#d95f02", alpha=0.08)
        ax.axvspan(11.5, 12.5, color="#1b9e77", alpha=0.08); ax.axvspan(0.5, 2.5, color="#1b9e77", alpha=0.08)
        ax.axhline(0, color="black", lw=0.8)
        colours = ["black"] if len(data) == 1 else ["#7570b3", "#e7298a"]
        for (lab, t), c in zip(data.items(), colours):
            ax.fill_between(t.month, t.lo, t.hi, color=c, alpha=0.18, lw=0)
            ax.plot(t.month, t["mean"], "o-", color=c, lw=2, label=lab)
        if len(data) == 1:
            for _, r in whole.iterrows():
                ax.text(r.month, -0.62, f"n={int(r.n_drifters)}", ha="center", fontsize=7, color="grey")
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_NAMES)
        ax.set_title(title, fontsize=12); ax.legend(frameon=False, loc="upper right", fontsize=9)
    axes[0].set_ylim(-0.65, 0.65)
    axes[0].set_ylabel("Along-coast velocity (m/s)\n+ poleward (NE)   − equatorward (SW)")
    fig.suptitle(f"Monthly along-coast current within {band_km} km of India's east coast "
                 f"({drifter_label(which)}, drifter-weighted mean, 95% bootstrap interval)", fontsize=13)
    fig.savefig(out, dpi=140); plt.close(fig)
    return whole


# ============================================================
# Run everything
# ============================================================
def make_all(df, outdir, which_list):
    os.makedirs(outdir, exist_ok=True)
    grid_season = Grid(0.2, df)
    grid_month = Grid(0.25, df)
    for which in which_list:
        tag = "drogued" if which == "drogued" else "all_drifters"
        print(f"\n=== {drifter_label(which)} ===")

        fields, pts = seasonal_fields(df, which, grid_season)
        fig_seasonal_speed(fields, pts, grid_season, which, os.path.join(outdir, f"seasonal_speed_{tag}.png"))
        fig_seasonal_eastwest(fields, grid_season, which, os.path.join(outdir, f"seasonal_east_west_{tag}.png"))
        A = seasonal_angle(fields, grid_season)
        share = fig_direction_change_map(A, grid_season, which, os.path.join(outdir, f"direction_change_map_{tag}.png"))
        fig_direction_change_map(A, grid_season, which,
                                 os.path.join(outdir, f"direction_change_map_annotated_{tag}.png"), annotate=True)
        fig_direction_change_chart(A, grid_season, which, os.path.join(outdir, f"direction_change_chart_{tag}.png"))
        fig_reversal_vs_speed(A, grid_season, which, os.path.join(outdir, f"reversal_vs_speed_{tag}.png"))
        R = fig_coast_distance(A, grid_season, which, os.path.join(outdir, f"reversal_vs_coast_distance_{tag}.png"))
        print(f"Share of mapped Bay turning >90° (smoothed, currents >= {MIN_SPEED_DIRECTION} m/s): {share:.1f}%")
        print("By distance from the western boundary:\n", R.round(2).to_string(index=False))

        mfields, nd = monthly_fields(df, which, grid_month)
        fig_monthly_speed(mfields, nd, grid_month, which, os.path.join(outdir, f"monthly_speed_{tag}.png"))
        fig_monthly_eastwest(mfields, nd, grid_month, which, os.path.join(outdir, f"monthly_east_west_{tag}.png"))
        S = fig_monthly_reversal(mfields, nd, grid_month, which,
                                 os.path.join(outdir, f"monthly_reversal_maps_{tag}.png"),
                                 os.path.join(outdir, f"monthly_reversal_line_{tag}.png"))
        print("Monthly share turned >90° from July:\n", S.round(1).to_string(index=False))

        W = fig_eicc_timeline(df, which, os.path.join(outdir, f"eicc_timeline_{tag}.png"))
        print("Coastal (EICC) monthly along-coast velocity:\n", W.round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Bay of Bengal monsoon reversal figures")
    p.add_argument("--data", default="bob_subset.nc", help="cached GDP subset (netCDF)")
    p.add_argument("--out", default="figures", help="output folder for PNGs")
    p.add_argument("--drifters", choices=["drogued", "all", "both"], default="both",
                   help="drogued only (main), drogued + undrogued, or both sets")
    a = p.parse_args()
    df = load_observations(a.data)
    print(f"Loaded {len(df):,} observations from {df.drifter_id.nunique()} drifters "
          f"({df.time.min():%d %b %Y} to {df.time.max():%d %b %Y})")
    which = ["drogued", "all"] if a.drifters == "both" else [a.drifters]
    make_all(df, a.out, which)


if __name__ == "__main__":
    main()
