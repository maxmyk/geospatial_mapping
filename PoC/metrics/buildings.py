import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.stats import entropy

use_geographic = False


def calculate_building_metrics(buildings):
    buildings = buildings[buildings.geometry.type.isin(["Polygon"])]
    buildings = buildings.copy()

    if use_geographic and buildings.crs and buildings.crs.to_epsg() != 4326:
        buildings = buildings.to_crs(epsg=4326) # Convert to geographic coordinates

    buildings["area"] = buildings.geometry.area
    buildings = buildings[buildings["area"] > 0]

    buildings["perimeter"] = buildings.geometry.length
    buildings["compactness"] = (buildings["perimeter"] ** 2) / buildings["area"]

    buildings["aspect_ratio"] = buildings.geometry.apply(
        lambda geom: (
            (geom.bounds[2] - geom.bounds[0]) / (geom.bounds[3] - geom.bounds[1])
            if (geom.bounds[3] - geom.bounds[1]) > 0
            else np.nan
        )
    )

    def count_vertices(geom):
        if geom.geom_type == "Polygon":
            return len(geom.exterior.coords[:-1])
        elif geom.geom_type == "MultiPolygon":
            return sum(len(p.exterior.coords[:-1]) for p in geom.geoms)
        return 0

    buildings["num_vertices"] = buildings.geometry.apply(count_vertices)

    def calculate_angles(geom):
        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords[:-1])
        elif geom.geom_type == "MultiPolygon":
            coords = np.array(
                max(geom.geoms, key=lambda g: g.area).exterior.coords[:-1]
            )
        else:
            return []

        angles = []
        for i in range(len(coords)):
            v1 = coords[i - 1] - coords[i]
            v2 = coords[(i + 1) % len(coords)] - coords[i]
            dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = (
                np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
            )  # Convert to degrees
            if angle > 5:
                angles.append(angle)
        return angles

    buildings["angles"] = buildings.geometry.apply(calculate_angles)
    all_angles = np.concatenate(buildings["angles"].values)

    vertex_counts = buildings["num_vertices"].dropna().astype(int)
    hist_vertices, bin_edges_vertices = np.histogram(
        vertex_counts, bins=range(min(vertex_counts), max(vertex_counts) + 2, 1)
    )

    hist_angles, bin_edges_angles = np.histogram(
        all_angles, bins=np.linspace(0, 180, 20)
    )
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].bar(
        bin_edges_vertices[:-1], hist_vertices, width=1, edgecolor="black", alpha=0.7
    )
    axs[0].set_xlabel("Number of Vertices per Building")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Histogram of Building Vertex Counts (Corrected)")
    axs[0].grid(axis="y", linestyle="--", alpha=0.7)

    axs[1].bar(
        bin_edges_angles[:-1], hist_angles, width=8, edgecolor="black", alpha=0.7
    )
    axs[1].set_xlabel("Angle (Degrees)")
    axs[1].set_ylabel("Frequency")
    axs[1].set_title("Histogram of Building Angles (Excluding Collinear)")
    axs[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    max_vertex_building = buildings.loc[buildings["num_vertices"].idxmax()]
    if max_vertex_building.geometry.geom_type == "Polygon":
        coords = list(max_vertex_building.geometry.exterior.coords)
    elif max_vertex_building.geometry.geom_type == "MultiPolygon":
        coords = list(
            max(
                max_vertex_building.geometry.geoms, key=lambda g: g.area
            ).exterior.coords
        )
    else:
        coords = []
    print(
        "Building with the most vertices has",
        max_vertex_building["num_vertices"],
        "vertices.",
    )
    print("Coordinates:")
    for coord in coords:
        print(coord)

    return {
        "mean_compactness": buildings["compactness"].mean(),
        "mean_aspect_ratio": buildings["aspect_ratio"].mean(),
        "vertex_histogram": (hist_vertices, bin_edges_vertices),
        "angle_histogram": (hist_angles, bin_edges_angles),
    }


def calculate_building_wall_entropy(buildings):

    buildings = buildings[
        buildings.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if use_geographic and buildings.crs and buildings.crs.to_epsg() != 4326:
        buildings = buildings.to_crs(epsg=4326)

    def extract_orientations(geom):
        if geom.geom_type == "Polygon":
            coords = np.array(geom.exterior.coords[:-1])
        elif geom.geom_type == "MultiPolygon":
            coords = np.array(
                max(geom.geoms, key=lambda g: g.area).exterior.coords[:-1]
            )
        else:
            return []

        angles = []
        for i in range(len(coords) - 1):
            dx, dy = coords[i + 1] - coords[i]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            angle = np.abs(angle)
            angles.append(angle)

        return angles

    buildings["wall_orientations"] = buildings.geometry.apply(extract_orientations)

    def compute_entropy(angles):
        if not angles:
            return np.nan
        hist, _ = np.histogram(angles, bins=np.linspace(0, 180, 20), density=True)
        hist = hist[hist > 0]
        return entropy(hist) if len(hist) > 1 else 0

    buildings["wall_entropy"] = buildings["wall_orientations"].apply(compute_entropy)

    plt.figure(figsize=(10, 5))
    plt.hist(buildings["wall_entropy"].dropna(), bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Wall Orientation Entropy")
    plt.ylabel("Frequency")
    plt.title("Histogram of Building Wall Orientation Entropy")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # low_ent_building = buildings.loc[buildings["wall_entropy"].idxmin()]
    # high_ent_building = buildings.loc[buildings["wall_entropy"].idxmax()]

    low_ent_buildings = buildings.nsmallest(5, "wall_entropy")
    high_ent_buildings = buildings.nlargest(5, "wall_entropy")

    fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(12, 6))

    def plot_polar_histogram(orientations, ax, title):
        hist, bin_edges = np.histogram(
            orientations, bins=np.linspace(0, 180, 18), density=True
        )
        theta = np.radians(bin_edges[:-1])
        bars = ax.bar(theta, hist, width=np.radians(10), edgecolor="black", alpha=0.7)
        ax.set_title(title)

    # structured
    # plot_polar_histogram(
    #     low_ent_building["wall_orientations"],
    #     axs[0],
    #     f"Low Entropy Building\nEntropy: {low_ent_building['wall_entropy']:.2f}",
    # )

    for i, row in low_ent_buildings.iterrows():
        plot_polar_histogram(
            row["wall_orientations"],
            axs[0],
            f"Low Entropy Building\nEntropy: {row['wall_entropy']:.2f}",
        )

    # chaotic
    # plot_polar_histogram(
    #     high_ent_building["wall_orientations"],
    #     axs[1],
    #     f"High Entropy Building\nEntropy: {high_ent_building['wall_entropy']:.2f}",
    # )
    for i, row in high_ent_buildings.iterrows():
        plot_polar_histogram(
            row["wall_orientations"],
            axs[1],
            f"High Entropy Building\nEntropy: {row['wall_entropy']:.2f}",
        )

    plt.tight_layout()
    plt.show()

    print("Low entropy building coordinates:")
    # print(low_ent_building.geometry.centroid)
    for i, row in low_ent_buildings.iterrows():
        print(row.geometry.centroid)
    print("High entropy building coordinates:")
    # print(high_ent_building.geometry.centroid)
    for i, row in high_ent_buildings.iterrows():
        print(row.geometry.centroid)

    return buildings[["geometry", "wall_entropy"]]
