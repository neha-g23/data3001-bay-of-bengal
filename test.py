import clouddrift as cd
from clouddrift.ragged import subset

ds = cd.datasets.gdp1h()

ds_bob = subset(
    ds, 
    {"lat": (5.0, 23.0), "lon": (78.0, 99.0)}, 
    row_dim_name="traj"
)

print(f"Original global observations: {ds.sizes['obs']}")
print(f"Bay of Bengal observations: {ds_bob.sizes['obs']}")