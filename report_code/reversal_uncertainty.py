"""
How reliable is the smoothed reversal map? Drifter bootstrap + sensitivity tests.

Uses exactly the same map as reversal_figures.py (drifter-level kernel
smoothing, 75 km, 0.2 degree grid, coverage rule >= 3 drifters within 100 km,
reversal = direction change > 90 degrees where both seasons are >= 0.1 m/s,
area-weighted).

1. Drifter bootstrap (N_BOOT runs): whole drifters are resampled with
   replacement (a drifter keeps all its points, in both seasons), and both
   seasonal maps are rebuilt. Gives:
     - 95% intervals for the Bay-wide, coastal-band and latitude-band shares
     - an agreement map: % of runs in which each location reverses
     - direction stability: how far each run's arrow turns from the full-data arrow
2. Sensitivity tests (one run each): bandwidth, reversal cut-off, minimum
   speed, leaving out 2015 and 2019, and all drifters.

Usage:  python reversal_uncertainty.py --data bob_subset.nc --out figures
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reversal_figures as rf          # same folder

N_BOOT = 500
GRID_RES = 0.2
SEED = 3001
BANDS = [(5, 10), (10, 15), (15, 20), (20, 23)]


class SeasonWeights:
    """Sparse kernel weights grid -> points for one season, computed once and reused."""

    def __init__(self, pts, grid, bw, drifter_codes):
        tl, tn = grid.LAT.ravel(), grid.LON.ravel()
        cells, idx, wts, close = [], [], [], []
        for i in range(0, tl.size, 1000):
            d = rf.haversine_km(tl[i:i+1000], tn[i:i+1000], pts.lat.values, pts.lon.values)
            r, c = np.nonzero(d <= max(rf.KERNEL_CUTOFF * bw, rf.SUPPORT_KM))
            dd = d[r, c]
            w = np.exp(-0.5 * (dd / bw) ** 2); w[dd > rf.KERNEL_CUTOFF * bw] = 0.0
            cells.append(r + i); idx.append(c); wts.append(w); close.append(dd <= rf.SUPPORT_KM)
        self.cell = np.concatenate(cells); self.pt = np.concatenate(idx)
        self.w = np.concatenate(wts); close = np.concatenate(close)
        self.ncell, self.shape, self.never = tl.size, grid.LAT.shape, grid.never
        self.ve, self.vn = pts.ve.values, pts.vn.values
        self.code = drifter_codes                       # drifter index of each point
        self.ndr = drifter_codes.max() + 1 if len(drifter_codes) else 1
        self.close_pt = self.pt[close]
        self.close_key = self.cell[close].astype(np.int64) * (self.code.max() + 1) + self.code[self.close_pt]
        self.nkey = self.code.max() + 1

    def field(self, drifter_count):
        """Smoothed ve, vn for a given number of copies of each drifter (1 = original data)."""
        c = drifter_count[self.code]
        w = self.w * c[self.pt]
        ws = np.bincount(self.cell, w, self.ncell)
        with np.errstate(invalid="ignore", divide="ignore"):
            ve = np.bincount(self.cell, w * self.ve[self.pt], self.ncell) / ws
            vn = np.bincount(self.cell, w * self.vn[self.pt], self.ncell) / ws
        keys = np.unique(self.close_key[c[self.close_pt] > 0])
        nn = np.bincount(keys // self.nkey, minlength=self.ncell)
        ve, vn, nn = (a.reshape(self.shape) for a in (ve, vn, nn))
        hide = (nn < rf.MIN_DRIFTERS) | self.never
        ve[hide] = np.nan; vn[hide] = np.nan
        return ve, vn


def summarise(f_sw, f_ne, grid, min_speed=rf.MIN_SPEED_DIRECTION, cutoff=90):
    (ve1, vn1), (ve2, vn2) = f_sw, f_ne
    s1, s2 = np.hypot(ve1, vn1), np.hypot(ve2, vn2)
    fast = np.isfinite(s1) & np.isfinite(s2) & (s1 >= min_speed) & (s2 >= min_speed)
    ang = rf.direction_change(ve1, vn1, ve2, vn2)
    rev = ang > cutoff
    out = {"Bay": rf.area_share(fast, rev, grid.LAT),
           "Coastal band 80-85E": rf.area_share(fast & (grid.LON >= 80) & (grid.LON < 85), rev, grid.LAT)}
    for lo, hi in BANDS:
        out[f"{lo}-{hi}N"] = rf.area_share(fast & (grid.LAT >= lo) & (grid.LAT < hi), rev, grid.LAT)
    out["fully reversed (>135)"] = rf.area_share(fast, ang > 135, grid.LAT)
    return out, fast, rev


def build(df, which, grid, bw=rf.BANDWIDTH_SEASON_KM):
    pts = rf.per_drifter_means(rf.select_drifters(df, which), "season")
    pts = pts[pts.season.isin(["Southwest", "Northeast"])].reset_index(drop=True)
    codes, ids = pd.factorize(pts.drifter_id)
    W = {s: SeasonWeights(pts[pts.season == s].reset_index(drop=True), grid, bw, codes[pts.season.values == s])
         for s in ["Southwest", "Northeast"]}
    return W, len(ids)


def turn(ve_a, vn_a, ve_b, vn_b):
    return rf.direction_change(ve_a, vn_a, ve_b, vn_b)


def make(df, outdir, n_boot=N_BOOT):
    os.makedirs(outdir, exist_ok=True)
    grid = rf.Grid(GRID_RES, df)
    W, nd = build(df, "drogued", grid)
    ones = np.ones(nd)
    full = {s: W[s].field(ones) for s in W}
    base, fast0, rev0 = summarise(full["Southwest"], full["Northeast"], grid)

    rng = np.random.default_rng(SEED)
    rows, agree, counted = [], np.zeros(grid.LAT.shape), np.zeros(grid.LAT.shape)
    turns = {s: [] for s in W}
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, nd, nd), minlength=nd).astype(float)
        f = {s: W[s].field(cnt) for s in W}
        res, fast, rev = summarise(f["Southwest"], f["Northeast"], grid)
        rows.append(res)
        agree += (fast & rev); counted += fast
        for s in W:
            ok = np.isfinite(f[s][0]) & np.isfinite(full[s][0]) & (np.hypot(*full[s]) >= rf.MIN_SPEED_DIRECTION)
            t = np.full(grid.LAT.shape, np.nan); t[ok] = turn(*full[s], *f[s])[ok]
            turns[s].append(t)
    B = pd.DataFrame(rows)
    ci = B.quantile([0.025, 0.975]).T
    table = pd.DataFrame({"estimate": pd.Series(base), "lower 95%": ci[0.025], "upper 95%": ci[0.975]}).round(1)
    print(f"Drifter bootstrap ({n_boot} runs, drogued drifters):\n", table.to_string())

    stab = {}
    for s in W:
        T = np.stack(turns[s]); med = np.nanmedian(T, axis=0)
        ok = fast0 & np.isfinite(med)
        stab[s] = med
        print(f"{s}: median turn of bootstrap arrows from full-data arrow = {np.nanmedian(med[ok]):.0f} deg; "
              f"locations with median turn > 45 deg: {rf.area_share(ok, med > 45, grid.LAT):.0f}% of mapped area")
    with np.errstate(invalid="ignore", divide="ignore"):
        agree_pct = np.where(fast0, agree / np.maximum(counted, 1) * 100, np.nan)
    solid = rf.area_share(fast0 & np.isfinite(agree_pct), agree_pct >= 95, grid.LAT)
    unsure = rf.area_share(fast0 & np.isfinite(agree_pct), (agree_pct > 25) & (agree_pct < 75), grid.LAT)
    print(f"Locations reversing in >= 95% of runs: {solid:.0f}% of mapped area; uncertain (25-75%): {unsure:.0f}%")

    # ---------- sensitivity tests ----------
    sens = [("Main result (75 km, >90 deg, >=0.1 m/s, drogued)", base["Bay"], base["Coastal band 80-85E"])]
    for bw in [50, 100, 150]:
        Wb, n = build(df, "drogued", grid, bw); f = {s: Wb[s].field(np.ones(n)) for s in Wb}
        r, _, _ = summarise(f["Southwest"], f["Northeast"], grid); sens.append((f"Bandwidth {bw} km", r["Bay"], r["Coastal band 80-85E"]))
    for ms in [0.05, 0.2]:
        r, _, _ = summarise(full["Southwest"], full["Northeast"], grid, min_speed=ms)
        sens.append((f"Minimum speed {ms} m/s", r["Bay"], r["Coastal band 80-85E"]))
    r, _, _ = summarise(full["Southwest"], full["Northeast"], grid, cutoff=120)
    sens.append(("Stricter cut-off: >120 deg", r["Bay"], r["Coastal band 80-85E"]))
    d2 = df[~pd.to_datetime(df.time).dt.year.isin([2015, 2019])]
    Wy, n = build(d2, "drogued", grid); f = {s: Wy[s].field(np.ones(n)) for s in Wy}
    r, _, _ = summarise(f["Southwest"], f["Northeast"], grid); sens.append(("Leave out 2015 and 2019", r["Bay"], r["Coastal band 80-85E"]))
    Wa, n = build(df, "all", grid); f = {s: Wa[s].field(np.ones(n)) for s in Wa}
    r, _, _ = summarise(f["Southwest"], f["Northeast"], grid); sens.append(("All drifters (drogued + undrogued)", r["Bay"], r["Coastal band 80-85E"]))
    S = pd.DataFrame(sens, columns=["test", "Bay reversing (%)", "coastal band reversing (%)"]).round(0)
    print("Sensitivity tests:\n", S.to_string(index=False))
    S.to_csv(os.path.join(outdir, "reversal_sensitivity.csv"), index=False)
    table.to_csv(os.path.join(outdir, "reversal_bootstrap_intervals.csv"))

    # ---------- figure ----------
    fig = plt.figure(figsize=(19, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.85], wspace=0.42, left=0.04, right=0.97, top=0.83, bottom=0.12)
    ax = fig.add_subplot(gs[0])
    ax.contourf(grid.LON, grid.LAT, np.where(~grid.never, 1.0, np.nan), levels=[0.5, 1.5], colors=["#ececec"])
    cf = ax.contourf(grid.LON, grid.LAT, agree_pct, levels=[0, 25, 50, 75, 95, 100.01],
                     colors=["#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#8b0000"])
    cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02, ticks=[0, 25, 50, 75, 95, 100]); cb.set_label("% of bootstrap runs reversing")
    rf.style_map(ax); ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"(a) How often each location reverses\n{solid:.0f}% of the area reverses in ≥95% of runs", fontsize=12)

    ax = fig.add_subplot(gs[1])
    ax.contourf(grid.LON, grid.LAT, np.where(~grid.never, 1.0, np.nan), levels=[0.5, 1.5], colors=["#ececec"])
    med = stab["Northeast"]
    cf = ax.contourf(grid.LON, grid.LAT, np.where(fast0, med, np.nan), levels=[0, 15, 30, 45, 90, 180],
                     colors=["#f1eef6", "#bdc9e1", "#74a9cf", "#2b8cbe", "#045a8d"])
    cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Median arrow turn between runs (°)")
    rf.style_map(ax); ax.set_xlabel("Longitude (°E)")
    ax.set_title("(b) Northeast monsoon: how stable the arrows are\n(darker = arrow direction less certain)", fontsize=12)

    ax = fig.add_subplot(gs[2])
    keys = ["Bay", "Coastal band 80-85E"] + [f"{lo}-{hi}N" for lo, hi in BANDS]
    labels = ["Whole Bay", "Coastal band\n80–85°E", "5–10°N", "10–15°N", "15–20°N", "20–23°N\n(few data)"]
    y = np.arange(len(keys))[::-1]
    for yi, k in zip(y, keys):
        lo, hi = table.loc[k, "lower 95%"], table.loc[k, "upper 95%"]
        ax.plot([lo, hi], [yi, yi], color="#444", lw=3, solid_capstyle="round")
        ax.plot(table.loc[k, "estimate"], yi, "o", color="#b2182b", ms=9)
        ax.text(102, yi, f"{table.loc[k, 'estimate']:.0f}% ({lo:.0f}–{hi:.0f})", va="center", fontsize=10)
    ax.axvline(50, color="#888", ls=":", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, 135); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of mapped area reversing (>90°)"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(c) Reversal share with 95% bootstrap interval", fontsize=12)
    fig.suptitle(f"How reliable is the reversal map? Drifter bootstrap ({n_boot} runs, drogued drifters, 75 km smoothing)",
                 fontsize=14)
    fig.savefig(os.path.join(outdir, "reversal_uncertainty.png"), dpi=140); plt.close(fig)
    return table, S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bob_subset.nc")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    make(rf.load_observations(a.data), a.out, a.n_boot)
