"""Monthly switch (Section 4.2, Figure Z): share of the measured area whose monthly current
points more than 90 deg away from the Southwest-monsoon direction, near the coast
(< 150 km from India's east coast or Sri Lanka) and offshore, with 95% ranges.

Monthly maps use the 100 km bandwidth; the Southwest reference uses 75 km.
Outputs: monthly_switch.png, monthly_turn.csv
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import reversal_figures as rf, reversal_uncertainty as ru
from common import load, OUT, NB, SEED, NAVY, coast_distance_km, redraw

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.labelsize": 24})
ORDER = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]


def monthly_weights(df, grid):
    """Kernel weights for the Southwest reference and for each month, sharing one drifter index."""
    d = rf.select_drifters(df, "drogued")
    pm = rf.per_drifter_means(d, "month")
    ps = rf.per_drifter_means(d, "season"); ps = ps[ps.season == "Southwest"]
    allid = pd.Index(pd.unique(pd.concat([pm.drifter_id, ps.drifter_id])))
    code = lambda p: allid.get_indexer(p.drifter_id.values)
    Wsw = ru.SeasonWeights(ps.reset_index(drop=True), grid, rf.BANDWIDTH_SEASON_KM, code(ps))
    Wm = {m: ru.SeasonWeights(pm[pm.month == m].reset_index(drop=True), grid, rf.BANDWIDTH_MONTH_KM,
                              code(pm[pm.month == m])) for m in range(1, 13)}
    return Wsw, Wm, pm, allid


def monthly_shares(Wsw, Wm, cnt, masks, grid, min_cells=30):
    """(12 months x len(masks)) % of each mask's measured area turned > 90 deg."""
    ve0, vn0 = Wsw.field(cnt); s0 = np.hypot(ve0, vn0); out = []
    for m in ORDER:
        ve, vn = Wm[m].field(cnt); s = np.hypot(ve, vn)
        ok = np.isfinite(s) & np.isfinite(s0) & (s >= rf.MIN_SPEED_DIRECTION) & (s0 >= rf.MIN_SPEED_DIRECTION)
        a = rf.direction_change(ve, vn, ve0, vn0) > 90
        out.append([rf.area_share(ok & r, a, grid.LAT) if (ok & r).sum() >= min_cells else np.nan for r in masks])
    return np.array(out)


if __name__ == "__main__":
    df = load(); grid = rf.Grid(0.2, df)
    Wsw, Wm, pm, allid = monthly_weights(df, grid)
    near = coast_distance_km(grid) < 150; masks = [near, ~near]
    nd = pm.groupby("month").drifter_id.nunique()
    full = monthly_shares(Wsw, Wm, np.ones(len(allid)), masks, grid)
    rng = np.random.default_rng(SEED)
    B = np.array([monthly_shares(Wsw, Wm, redraw(rng, len(allid)), masks, grid) for _ in range(NB)])
    lo, hi = np.nanpercentile(B, 2.5, axis=0), np.nanpercentile(B, 97.5, axis=0)
    pd.DataFrame({"month": [rf.MONTH_NAMES[m-1] for m in ORDER], "drifters": [int(nd.get(m, 0)) for m in ORDER],
                  "coast": full[:, 0], "coast_lo": lo[:, 0], "coast_hi": hi[:, 0],
                  "offshore": full[:, 1], "off_lo": lo[:, 1], "off_hi": hi[:, 1]}).round(1).to_csv(OUT + "monthly_turn.csv", index=False)
    print(pd.read_csv(OUT + "monthly_turn.csv").to_string(index=False))

    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(15, 8.6)); fig.subplots_adjust(left=.13, right=.97, top=.84, bottom=.2)
    ax.axvspan(-.5, 3.5, color="#fdf1e7", zorder=0); ax.axvspan(5.5, 8.5, color="#eaf1f9", zorder=0)
    ax.text(1.5, 103, "Southwest monsoon", ha="center", fontsize=19, color="#a04000", fontweight="bold")
    ax.text(7, 103, "Northeast monsoon", ha="center", fontsize=19, color="#1f4e8c", fontweight="bold")
    for k, col, lab, mk in [(0, "#7a0177", "Near coast (<150 km)", "o"), (1, "#3b7ea1", "Offshore (≥150 km)", "s")]:
        ax.fill_between(x, lo[:, k], hi[:, k], color=col, alpha=.15, lw=0)
        ax.plot(x, full[:, k], marker=mk, color=col, lw=3.5, ms=11, label=lab)
    ax.axhline(50, color="grey", ls=":", lw=1.5)
    ax.set_xticks(x); ax.set_xticklabels([rf.MONTH_NAMES[m-1] for m in ORDER]); ax.set_xlim(-.5, 11.5)
    ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("% turned more than 90°")
    for xi, m in zip(x, ORDER):
        ax.text(xi, -17, f"{int(nd.get(m, 0))}", ha="center", fontsize=15, color="#555", clip_on=False)
    ax.text(-.65, -17, "Drifters:", ha="right", fontsize=15, color="#555", clip_on=False)
    ax.legend(loc="center left", frameon=True, fontsize=19)
    ax.set_title("Monthly switch", fontsize=30, fontweight="bold", color=NAVY, pad=40)
    fig.savefig(OUT + "monthly_switch.png", dpi=150, facecolor="white"); print("saved", OUT + "monthly_switch.png")
