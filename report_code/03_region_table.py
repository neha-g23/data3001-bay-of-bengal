"""Region table (Q1, Table X): reversal by region A-F with 95% ranges.

Regions: A = within 150 km of India's east coast, B = within 150 km of Sri Lanka,
then offshore C = south of 10N, D = 10-15N west of 88E, E = 10-15N east of 88E,
F = north of 15N.

Columns: drifters, reverses >90 deg (95% range), consistently reverses
(share of the region reversing in >= 95% of redraws; needs agreement_pct.npy
from 01_headline_bootstrap.py), reverses >135 deg (95% range), median turn
(95% range), too slow (share excluded for < 0.1 m/s in either monsoon).

Outputs: region_table.csv, regions.npy (region masks, used by 04).
"""
import os
import numpy as np, pandas as pd
import reversal_figures as rf, reversal_uncertainty as ru
from common import load, OUT, NB, SEED, coast_distance_km, redraw

NAMES = ["A. Near India's east coast", "B. Near Sri Lanka", "C. Offshore south", "D. Offshore central-west",
         "E. Offshore central-east", "F. Offshore north"]


def region_masks(grid):
    dI = coast_distance_km(grid, [rf.INDIA_COAST]); dL = coast_distance_km(grid, [rf.LANKA_COAST])
    LAT, LON = grid.LAT, grid.LON
    india = dI < 150; lanka = (dL < 150) & ~india; off = ~india & ~lanka
    return np.stack([india, lanka, off & (LAT < 10),
                     off & (LAT >= 10) & (LAT < 15) & (LON < 88),
                     off & (LAT >= 10) & (LAT < 15) & (LON >= 88),
                     off & (LAT >= 15)])


if __name__ == "__main__":
    df = load(); grid = rf.Grid(0.2, df); LAT = grid.LAT; w = np.cos(np.radians(LAT))
    M = region_masks(grid); np.save(OUT + "regions.npy", M)
    masks = list(M) + [np.ones_like(M[0])]
    W, nd = ru.build(df, "drogued", grid)

    def stats(cnt):
        f = {s: W[s].field(cnt) for s in W}
        (a, b), (c, d) = f["Southwest"], f["Northeast"]
        s1, s2 = np.hypot(a, b), np.hypot(c, d)
        both = np.isfinite(s1) & np.isfinite(s2); fast = both & (s1 >= .1) & (s2 >= .1)
        ang = rf.direction_change(a, b, c, d)
        out = []
        for m in masks:
            fm = fast & m
            out.append([rf.area_share(fm, ang > 90, LAT), rf.area_share(fm, ang > 135, LAT), np.median(ang[fm])]
                       if fm.sum() >= 20 else [np.nan] * 3)
        return np.array(out), both, fast

    full, both, fast = stats(np.ones(nd))
    rng = np.random.default_rng(SEED)
    B = np.array([stats(redraw(rng, nd))[0] for _ in range(NB)])
    lo, hi = np.nanpercentile(B, 2.5, 0), np.nanpercentile(B, 97.5, 0)

    # drifters with seasonal data in each region
    p = rf.per_drifter_means(rf.select_drifters(df, "drogued"), "season")
    p = p[p.season.isin(["Southwest", "Northeast"])]
    i = np.clip(np.round((p.lat.values - LAT[0, 0]) / 0.2).astype(int), 0, LAT.shape[0] - 1)
    j = np.clip(np.round((p.lon.values - grid.LON[0, 0]) / 0.2).astype(int), 0, LAT.shape[1] - 1)
    pct_file = OUT + "agreement_pct.npy"
    pct = np.load(pct_file) if os.path.exists(pct_file) else None

    rows = []
    for k, (name, m) in enumerate(zip(NAMES + ["Whole measured Bay"], masks)):
        drifters = p[m[i, j]].drifter_id.nunique() if k < 6 else p.drifter_id.nunique()
        fm = fast & m
        cons = (w[fm] * (pct[fm] >= 95)).sum() / w[fm].sum() * 100 if pct is not None else np.nan
        rows.append({"Region": name, "Drifters": drifters,
                     "Reverses >90° (95% CI)": f"{full[k,0]:.0f}% ({lo[k,0]:.0f}–{hi[k,0]:.0f})",
                     "Consistently reverses": f"{cons:.0f}%",
                     "Reverses >135° (95% CI)": f"{full[k,1]:.0f}% ({lo[k,1]:.0f}–{hi[k,1]:.0f})",
                     "Median turn (95% CI)": f"{full[k,2]:.0f}° ({lo[k,2]:.0f}–{hi[k,2]:.0f})",
                     "Too slow, excluded": f"{w[both & m & ~fast].sum() / w[both & m].sum() * 100:.0f}%",
                     "reverses_pct": round(full[k, 0])})
    T = pd.DataFrame(rows); T.to_csv(OUT + "region_table.csv", index=False)
    print(T.drop(columns="reverses_pct").to_string(index=False))
    print("Note: the whole-Bay range in the report (72-88%) uses 500 redraws from 01_headline_bootstrap.py.")
