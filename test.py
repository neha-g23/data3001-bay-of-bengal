# import clouddrift as cd
# from clouddrift.ragged import subset

# ds = cd.datasets.gdp1h()

# ds_bob = subset(
#     ds, 
#     {"lat": (5.0, 23.0), "lon": (78.0, 99.0)}, 
#     row_dim_name="traj"
# )

# print(f"Original global observations: {ds.sizes['obs']}")
# print(f"Bay of Bengal observations: {ds_bob.sizes['obs']}")

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
print(f"Bay of Bengal distinct drifters: {ds_bob.sizes['traj']}")

import matplotlib.pyplot as plt

# Extract the longitude and latitude arrays from your filtered dataset
# Calling .values pulls just these specific coordinates into memory
lon = ds_bob.lon.values
lat = ds_bob.lat.values

# Create the density map using a 2D histogram
plt.figure(figsize=(10, 8))
# bins=50 divides the region into a 50x50 grid. cmin=1 hides empty cells.
counts, xedges, yedges, im = plt.hist2d(lon, lat, bins=50, cmap='YlOrBr', cmin=1)

# Add formatting
plt.colorbar(im, label='Number of Observations per Cell')
plt.title('Drifter Observation Density: Bay of Bengal')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)

# Display the plot
plt.show()
