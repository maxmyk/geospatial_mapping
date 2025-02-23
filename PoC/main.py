import osmnx as ox
from metrics.buildings import calculate_building_metrics, calculate_building_wall_entropy
from metrics.intersections import calculate_intersection_complexity, calculate_intersection_density
from metrics.streets import calculate_dead_end_density
from metrics.street_entropy import calculate_street_orientation_entropy, plot_polar_entropy

ox.settings.use_cache = True

place_name = "Manhattan, New York, USA"
place_name = "Vancouver, Canada"
place_name = "Mexico City, Mexico"
# place_name = "Buenos Aires, Argentina"
# place_name = "San Francisco, USA"

print("Downloading data...")
graph = ox.graph_from_place(place_name, network_type="drive")
print("Data downloaded successfully!")
nodes, edges = ox.graph_to_gdfs(graph)
# print("Data converted to GeoDataFrames!")
# buildings = ox.features_from_place(place_name, tags={"building": True})
# print("Buildings downloaded successfully!")
# parks = ox.features_from_place(place_name, tags={"leisure": "park"})
# water_bodies = ox.features_from_place(place_name, tags={"natural": "water"})
# roundabouts = nodes[nodes['highway'] == 'mini_roundabout']

utm_crs = ox.projection.project_graph(graph).graph["crs"]
edges = edges.to_crs(utm_crs)
nodes = nodes.to_crs(utm_crs)
# buildings = buildings.to_crs(utm_crs)
# # parks = parks.to_crs(utm_crs)
# # water_bodies = water_bodies.to_crs(utm_crs)
# # roundabouts = roundabouts.to_crs(utm_crs)
# convex_hull = edges.geometry.union_all().convex_hull
# area_km2 = convex_hull.area / 1e6
# print(f"Study area: {area_km2:.2f} km²")

# print("\nCalculating metrics...")
# intersection_density, intersections = calculate_intersection_density(nodes, area_km2)
# print(f"Intersection Density: {intersection_density:.2f} intersections per km²")

# buildings_metrics = calculate_building_metrics(buildings)
# print(f"Mean Building Compactness: {buildings_metrics['mean_compactness']:.2f}")
# print(f"Mean Building Aspect Ratio: {buildings_metrics['mean_aspect_ratio']:.2f}")
# print(f"Building Vertex Histogram: {buildings_metrics['vertex_histogram']}")
# print(f"Building Angle Histogram: {buildings_metrics['angle_histogram']}")

# result = calculate_building_wall_entropy(buildings)
# print(result.head())

# intersection_complexity = calculate_intersection_complexity(nodes)
# print(f"Intersection Complexity (proportion with 4+ streets): {intersection_complexity:.2%}")

# dead_end_density = calculate_dead_end_density(nodes, area_km2)
# print(f"Dead-End Density: {dead_end_density:.2f} dead-ends per km²")

street_entropy, angle_counts = calculate_street_orientation_entropy(edges)
print(f"Street Orientation Entropy: {street_entropy:.2f} bits")
plot_polar_entropy(angle_counts)
