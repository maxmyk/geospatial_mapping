import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

def calculate_building_metrics(buildings):
    buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])]
    buildings = buildings.copy()
    
    # if buildings.crs and buildings.crs.to_epsg() != 4326:
    #     buildings = buildings.to_crs(epsg=4326)  # Convert to geographic coordinates

    buildings['area'] = buildings.geometry.area
    buildings = buildings[buildings['area'] > 0]

    buildings['perimeter'] = buildings.geometry.length
    buildings['compactness'] = (buildings['perimeter'] ** 2) / buildings['area']

    buildings['aspect_ratio'] = buildings.geometry.apply(
        lambda geom: (geom.bounds[2] - geom.bounds[0]) / (geom.bounds[3] - geom.bounds[1])
        if (geom.bounds[3] - geom.bounds[1]) > 0 else np.nan
    )

    def count_vertices(geom):
        if geom.geom_type == "Polygon":
            return len(geom.exterior.coords[:-1])
        elif geom.geom_type == "MultiPolygon":
            return sum(len(p.exterior.coords[:-1]) for p in geom.geoms)
        return 0

    buildings['num_vertices'] = buildings.geometry.apply(count_vertices)

    def calculate_angles(geom):
        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords[:-1])
        elif geom.geom_type == "MultiPolygon":
            coords = np.array(max(geom.geoms, key=lambda g: g.area).exterior.coords[:-1])
        else:
            return []

        angles = []
        for i in range(len(coords)):
            v1 = coords[i - 1] - coords[i]
            v2 = coords[(i + 1) % len(coords)] - coords[i]
            dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi  # Convert to degrees
            if angle > 5:
                angles.append(angle)
        return angles

    buildings['angles'] = buildings.geometry.apply(calculate_angles)
    all_angles = np.concatenate(buildings['angles'].values)

    vertex_counts = buildings['num_vertices'].dropna().astype(int)
    hist_vertices, bin_edges_vertices = np.histogram(vertex_counts, bins=range(min(vertex_counts), max(vertex_counts) + 2, 1))

    hist_angles, bin_edges_angles = np.histogram(all_angles, bins=np.linspace(0, 180, 20))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].bar(bin_edges_vertices[:-1], hist_vertices, width=1, edgecolor="black", alpha=0.7)
    axs[0].set_xlabel("Number of Vertices per Building")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Histogram of Building Vertex Counts (Corrected)")
    axs[0].grid(axis="y", linestyle="--", alpha=0.7)

    axs[1].bar(bin_edges_angles[:-1], hist_angles, width=8, edgecolor="black", alpha=0.7)
    axs[1].set_xlabel("Angle (Degrees)")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Histogram of Building Angles (Excluding Collinear)")
    axs[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    max_vertex_building = buildings.loc[buildings['num_vertices'].idxmax()]
    if max_vertex_building.geometry.geom_type == "Polygon":
        coords = list(max_vertex_building.geometry.exterior.coords)
    elif max_vertex_building.geometry.geom_type == "MultiPolygon":
        coords = list(max(max_vertex_building.geometry.geoms, key=lambda g: g.area).exterior.coords)
    else:
        coords = []
    print("Building with the most vertices has", max_vertex_building['num_vertices'], "vertices.")
    print("Coordinates:")
    for coord in coords:
        print(coord)

    return {
        "mean_compactness": buildings['compactness'].mean(),
        "mean_aspect_ratio": buildings['aspect_ratio'].mean(),
        "vertex_histogram": (hist_vertices, bin_edges_vertices),
        "angle_histogram": (hist_angles, bin_edges_angles)
    }
