"""Q2 figure: median turn against current speed, near the coast (< 150 km) and offshore.

Locations mapped in both monsoons are split into five equal-sized speed groups
(quintiles) within each zone; each point is the group's median speed and median
turn. Same locations as the regression in 08_q2_speed_turn.py (including currents
under 0.1 m/s, whose turn is noisy).
Output: speed_vs_turn.png
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import reversal_figures as rf
from common import load, OUT, NAVY, coast_distance_km

df = load(); g = rf.Grid(0.2, df)
fields, _ = rf.seasonal_fields(df, "drogued", g)
A = rf.seasonal_angle(fields, g)
ok = A["both"]; ang = A["angle"][ok]; spd = ((A["s1"] + A["s2"]) / 2)[ok]; d = coast_distance_km(g)[ok]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(12, 7)); fig.subplots_adjust(left=.1, right=.97, top=.88, bottom=.13)
for m, lab, c, mk in [(d < 150, "Near the coast (< 150 km)", "#7a0177", "o"), (d >= 150, "Offshore (≥ 150 km)", "#3b7ea1", "s")]:
    q = np.quantile(spd[m], np.linspace(0, 1, 6)); xs, ys = [], []
    for a, b in zip(q[:-1], q[1:]):
        k = m & (spd >= a) & (spd <= b); xs.append(np.median(spd[k])); ys.append(np.median(ang[k]))
    ax.plot(xs, ys, marker=mk, color=c, lw=3.5, ms=11, label=lab)
    print(lab, [(round(x, 2), round(y)) for x, y in zip(xs, ys)])
ax.axhline(90, color="grey", ls=":", lw=1.5); ax.text(0.74, 82, "90° = reversal", ha="right", color="grey", fontsize=15)
ax.set_ylim(0, 190); ax.set_yticks([0, 45, 90, 135, 180]); ax.set_xlim(0.05, 0.75)
ax.set_xlabel("Current speed (m/s, average of both monsoons)"); ax.set_ylabel("Median turn (degrees)")
ax.set_title("Speed and turn", fontsize=26, fontweight="bold", color=NAVY, pad=14)
ax.legend(frameon=False, loc="lower right")
fig.savefig(OUT + "speed_vs_turn.png", dpi=150, facecolor="white"); print("saved", OUT + "speed_vs_turn.png")
