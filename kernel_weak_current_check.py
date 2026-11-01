
import numpy as np
import pandas as pd

df = pd.read_csv("outputs/tables/kernel_smoothed_grid_drogued.csv")

sw = df[df["monsoon"] == "Southwest"]
ne = df[df["monsoon"] == "Northeast"]

m = sw.merge(
    ne, on=["lon0", "lat0"], suffixes=("_sw", "_ne")
).dropna(subset=[
    "ve_sw", "vn_sw", "ve_ne", "vn_ne",
    "kernel_ve_sw", "kernel_vn_sw",
    "kernel_ve_ne", "kernel_vn_ne"
]).copy()


def angle(u1, v1, u2, v2):
    u1, v1, u2, v2 = [
        np.asarray(x, dtype=float)
        for x in (u1, v1, u2, v2)
    ]
    dot = u1 * u2 + v1 * v2
    norm = np.hypot(u1, v1) * np.hypot(u2, v2)
    cosine = np.divide(
        dot, norm,
        out=np.full(dot.shape, np.nan),
        where=norm > 1e-12
    )
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


m["raw_angle"] = angle(
    m.ve_sw, m.vn_sw, m.ve_ne, m.vn_ne
)

m["smooth_angle"] = angle(
    m.kernel_ve_sw, m.kernel_vn_sw,
    m.kernel_ve_ne, m.kernel_vn_ne
)

m["min_speed"] = m[
    ["speed_sw", "speed_ne",
     "kernel_speed_sw", "kernel_speed_ne"]
].min(axis=1)

m = m.dropna(subset=["raw_angle", "smooth_angle"])

print("========== WEAK-CURRENT SENSITIVITY ==========")

for cutoff in [0.0, 0.05, 0.10]:
    s = m[m["min_speed"] >= cutoff]
    print(f"\nMinimum speed: {cutoff:.2f} m/s")
    print(f"Comparable cells: {len(s)}")

    for threshold in [90, 135]:
        raw = s["raw_angle"] > threshold
        smooth = s["smooth_angle"] > threshold

        print(
            f"  >{threshold} degrees: "
            f"original={raw.mean()*100:.1f}%, "
            f"smoothed={smooth.mean()*100:.1f}%, "
            f"flipped={(raw != smooth).sum()}"
        )

print("\n========== RECLASSIFIED CELLS ==========")

for threshold in [90, 135]:
    changed = m[
        (m["raw_angle"] > threshold) !=
        (m["smooth_angle"] > threshold)
    ]

    print(f"\nThreshold >{threshold} degrees")
    print(f"Changed cells: {len(changed)}")
    print(f"Below 0.05 m/s: {(changed.min_speed < 0.05).sum()}")
    print(f"Below 0.10 m/s: {(changed.min_speed < 0.10).sum()}")

    changed.to_csv(
        f"outputs/tables/kernel_reclassified_{threshold}.csv",
        index=False
    )

print("\nWEAK-CURRENT DIAGNOSTIC COMPLETE")
