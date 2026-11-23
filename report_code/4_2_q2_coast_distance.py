"""Q2: is the reversal stronger near the coast? (Table Y, Figure W)

For distance bands from India's east coast and Sri Lanka: median speed, share
reversing (>90 deg), share near-fully reversing (>135 deg), median turn and
share too slow, each with 95% ranges from whole-drifter redraws. Also prints
the near-coast minus offshore differences with their 95% ranges.

Outputs: q2_bands.csv, q2_coast_distance.png
"""
import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import reversal_figures as rf, reversal_uncertainty as ru
from common import load, OUT, SEED, NAVY, coast_distance_km, redraw

NB = int(os.environ.get("NB", 300))
BANDS = [("0–150 km (near coast)", 0, 150), ("150–300 km", 150, 300), ("300–500 km", 300, 500),
         (">500 km", 500, 1e9), ("All offshore (≥150 km)", 150, 1e9)]
COLS = ["speed", "reverses", "nearfull", "median_turn", "too_slow"]

df = load(); g = rf.Grid(0.2, df)
ps = rf.per_drifter_means(rf.select_drifters(df, "drogued"), "season")
ids = pd.Index(ps.drifter_id.unique()); nd = len(ids)
W = {s: ru.SeasonWeights(ps[ps.season == s].reset_index(drop=True), g, rf.BANDWIDTH_SEASON_KM,
                         ids.get_indexer(ps[ps.season == s].drifter_id.values)) for s in ["Southwest", "Northeast"]}
D = coast_distance_km(g); wlat = np.cos(np.radians(g.LAT))


def stats(cnt):
    (ve1, vn1), (ve2, vn2) = W["Southwest"].field(cnt), W["Northeast"].field(cnt)
    s1, s2 = np.hypot(ve1, vn1), np.hypot(ve2, vn2); both = np.isfinite(s1) & np.isfinite(s2)
    fast = both & (s1 >= .1) & (s2 >= .1); ang = rf.direction_change(ve1, vn1, ve2, vn2); spd = (s1 + s2) / 2
    out = []
    for _, a, b in BANDS:
        m = (D >= a) & (D < b); mb, mf = m & both, m & fast
        if mf.sum() < 20: out.append([np.nan] * 5); continue
        w = wlat[mf]
        out.append([np.median(spd[mb]), (w * (ang[mf] > 90)).sum() / w.sum() * 100,
                    (w * (ang[mf] > 135)).sum() / w.sum() * 100, np.median(ang[mf]),
                    wlat[mb & ~fast].sum() / wlat[mb].sum() * 100])
    return np.array(out)


P = stats(np.ones(nd))
rng = np.random.default_rng(SEED)
B = np.array([stats(redraw(rng, nd)) for _ in range(NB)])
lo, hi = np.nanpercentile(B, 2.5, axis=0), np.nanpercentile(B, 97.5, axis=0)
T = pd.DataFrame([{"band": lab, **{c: P[k, j] for j, c in enumerate(COLS)},
                   **{c + "_lo": lo[k, j] for j, c in enumerate(COLS)}, **{c + "_hi": hi[k, j] for j, c in enumerate(COLS)}}
                  for k, (lab, _, _) in enumerate(BANDS)])
T.round(2).to_csv(OUT + "q2_bands.csv", index=False)
print(T[["band"] + COLS].round(2).to_string(index=False))
dif = B[:, 0, :] - B[:, 4, :]
for j, c in enumerate(COLS):
    print(f"near coast minus offshore, {c}: {P[0, j] - P[4, j]:.2f} (95% range {np.nanpercentile(dif[:, j], 2.5):.2f} to {np.nanpercentile(dif[:, j], 97.5):.2f})")

# ---- figure: first four bands
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(20, 7)); fig.subplots_adjust(left=.06, right=.98, top=.8, bottom=.2, wspace=.32)
x = np.arange(4); labs = ["0–150", "150–300", "300–500", ">500"]; colours = ["#7a0177"] + ["#3b7ea1"] * 3
for ax, j, ttl, yl, fmt in [(axes[0], 0, "Current speed", "Median speed (m/s)", "{:.2f}"),
                            (axes[1], 2, "Near-full reversal (>135°)", "% of measured area", "{:.0f}%"),
                            (axes[2], 3, "Median turn", "Degrees", "{:.0f}°")]:
    v, l, h = P[:4, j], lo[:4, j], hi[:4, j]
    ax.bar(x, v, color=colours, width=.62)
    ax.errorbar(x, v, yerr=[np.clip(v - l, 0, None), np.clip(h - v, 0, None)], fmt="none", ecolor="#222", capsize=7, lw=2)
    for xi, val, top in zip(x, v, h):
        ax.text(xi + .05, top + (0.012 if j == 0 else 3), fmt.format(val), ha="center", fontsize=16, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_xlabel("Distance from coast (km)"); ax.set_ylabel(yl)
    ax.set_title(ttl, fontsize=21, fontweight="bold", color=NAVY, pad=12)
axes[0].set_ylim(0, .56); axes[1].set_ylim(0, 100); axes[2].set_ylim(0, 190); axes[2].set_yticks([0, 45, 90, 135, 180])
axes[2].axhline(90, color="grey", ls=":", lw=1.5)
fig.suptitle("Reversal and distance from the coast", fontsize=27, fontweight="bold", color=NAVY, y=.97)
fig.text(.5, .03, f"Bars: drogued drifters, kernel-smoothed (75 km). Error bars: 95% range from {NB} drifter redraws. "
         "Coast = India's east coast and Sri Lanka.", ha="center", fontsize=14, color="#555")
fig.savefig(OUT + "q2_coast_distance.png", dpi=150, facecolor="white"); print("saved", OUT + "q2_coast_distance.png")
