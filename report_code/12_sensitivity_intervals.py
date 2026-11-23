"""Q3 sensitivity table (Table Z) with 95% intervals.

Changes one analysis choice at a time and reports the share of the measured Bay
that reverses, with a 95% interval from whole-drifter redraws for every row.
Rows: main result; smoothing distance 50/60/100/150 km; minimum speed 0.05/0.2 m/s;
cut-off 120 and 135 deg; leaving out 2015 and 2019; all drifters (drogued and
undrogued); coverage rule with 2 or 5 drifters, or a 75 or 150 km radius.
Output: sensitivity_intervals.csv. Takes about 5-10 minutes with NB=500.
"""
import os
import numpy as np, pandas as pd
import reversal_figures as rf, reversal_uncertainty as ru
from common import load, OUT, SEED, redraw

NB = int(os.environ.get("NB", 500))
df = load(); grid = rf.Grid(0.2, df)
no_busy_years = df[~pd.to_datetime(df.time).dt.year.isin([2015, 2019])]
# (label, data, drifters, bandwidth, min speed, cut-off, min drifters, coverage radius)
TESTS = [("Main result (75 km, >90°, ≥0.1 m/s, 3 drifters within 100 km, drogued)", df, "drogued", 75, .1, 90, 3, 100),
         ("Smoothing distance 50 km", df, "drogued", 50, .1, 90, 3, 100),
         ("Smoothing distance 60 km (one-standard-error choice)", df, "drogued", 60, .1, 90, 3, 100),
         ("Smoothing distance 100 km", df, "drogued", 100, .1, 90, 3, 100),
         ("Smoothing distance 150 km", df, "drogued", 150, .1, 90, 3, 100),
         ("Minimum speed 0.05 m/s", df, "drogued", 75, .05, 90, 3, 100),
         ("Minimum speed 0.2 m/s", df, "drogued", 75, .2, 90, 3, 100),
         ("Stricter cut-off: turn of more than 120°", df, "drogued", 75, .1, 120, 3, 100),
         ("Stricter cut-off: turn of more than 135°", df, "drogued", 75, .1, 135, 3, 100),
         ("Leave out 2015 and 2019", no_busy_years, "drogued", 75, .1, 90, 3, 100),
         ("All drifters (drogued and undrogued)", df, "all", 75, .1, 90, 3, 100),
         ("Coverage: at least 2 drifters within 100 km", df, "drogued", 75, .1, 90, 2, 100),
         ("Coverage: at least 5 drifters within 100 km", df, "drogued", 75, .1, 90, 5, 100),
         ("Coverage: at least 3 drifters within 75 km", df, "drogued", 75, .1, 90, 3, 75),
         ("Coverage: at least 3 drifters within 150 km", df, "drogued", 75, .1, 90, 3, 150)]
rows = []
for lab, data, which, bw, ms, cut, md, sk in TESTS:
    rf.MIN_DRIFTERS, rf.SUPPORT_KM = md, sk
    W, nd = ru.build(data, which, grid, bw)

    def share(cnt):
        r, _, _ = ru.summarise(W["Southwest"].field(cnt), W["Northeast"].field(cnt), grid, min_speed=ms, cutoff=cut)
        return r["Bay"]
    est = share(np.ones(nd))
    rng = np.random.default_rng(SEED)
    B = np.array([share(redraw(rng, nd)) for _ in range(NB)])
    rows.append({"test": lab, "reverses_pct": round(est), "lower": round(np.nanpercentile(B, 2.5)), "upper": round(np.nanpercentile(B, 97.5))})
    print(rows[-1], flush=True)
rf.MIN_DRIFTERS, rf.SUPPORT_KM = 3, 100
pd.DataFrame(rows).to_csv(OUT + "sensitivity_intervals.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
