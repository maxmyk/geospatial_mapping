import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt

# Caching for OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True

# place_name = "Manhattan, New York, USA"
# place_name = "Boston, Massachusetts, USA"
place_name = "Lviv, Lviv Oblast, Ukraine"

graph = ox.graph_from_place(place_name, network_type="drive")
nodes, edges = ox.graph_to_gdfs(graph)
buildings = ox.features_from_place(place_name, tags={"building": True})

# Determine UTM CRS
utm_crs = ox.projection.project_graph(graph).graph["crs"]

# Reproject data
edges = edges.to_crs(utm_crs)
nodes = nodes.to_crs(utm_crs)
buildings = buildings.to_crs(utm_crs)

# Calculate convex hull area as fallback
convex_hull = edges.geometry.unary_union.convex_hull
area_km2 = convex_hull.area / 1e6  # Approximate area in km²
print(f"Calculated area: {area_km2:.2f} km²")

# 1. Intersection Density
# Filter nodes for valid intersections
intersections = nodes[nodes['street_count'] > 2]  # More than two connecting streets
intersection_density = len(intersections) / area_km2
print(f"Intersection Density: {intersection_density:.2f} intersections per km²")

# 2. Building Shape Complexity
buildings['area'] = buildings.geometry.area
buildings['perimeter'] = buildings.geometry.length
buildings['compactness'] = (buildings['perimeter'] ** 2) / buildings['area']
mean_compactness = buildings['compactness'].mean()
print(f"Average Building Compactness: {mean_compactness:.2f}")

# 3. Street Orientation Entropy
def calculate_entropy(angles):
    counts, _ = np.histogram(angles, bins=36, range=(0, 360))
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]  # Remove zero entries
    return -np.sum(probabilities * np.log2(probabilities)), counts

edges['angle'] = edges.geometry.apply(lambda geom: np.arctan2(
    geom.coords[-1][1] - geom.coords[0][1],
    geom.coords[-1][0] - geom.coords[0][0]
) * 180 / np.pi % 360)

street_entropy, angle_counts = calculate_entropy(edges['angle'])
print(f"Street Orientation Entropy: {street_entropy:.2f} bits")


# ======================== Plotting ========================

fig, axs = plt.subplots(1, 2, figsize=(15, 8))

edges.plot(ax=axs[0], linewidth=0.5, color="gray", label="Streets")
intersections.plot(ax=axs[0], color="red", markersize=5, label="Intersections")
axs[0].set_title(f"{place_name} Intersection Density: {intersection_density:.2f}/km²")
axs[0].legend()
axs[0].set_axis_off()

buildings.plot(ax=axs[1], column='compactness', cmap='viridis', legend=True, alpha=0.7)
axs[1].set_title(f"{place_name} Building Compactness")
axs[1].set_axis_off()

filename_friendly_placename = "".join(x for x in place_name if x.isalnum())

plt.tight_layout()
plt.savefig(f"urban_analysis_metrics_{filename_friendly_placename}.png")
plt.show()

bin_centers = np.linspace(0, 360, 36, endpoint=False) + 5  # Center bins
ax = plt.subplot(111, polar=True)
ax.bar(np.radians(bin_centers), angle_counts, width=np.radians(10))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_title(f"{place_name} Street Orientation Entropy")
plt.savefig(f"street_orientation_entropy_{filename_friendly_placename}.png")
plt.show()
