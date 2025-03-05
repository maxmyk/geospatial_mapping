import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from tqdm import tqdm
from scipy.spatial import ConvexHull

ox.settings.use_cache = True


def get_city_blocks(place_name):
    G = ox.graph_from_place(place_name, network_type="drive")
    G = ox.project_graph(G)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    blocks = []
    for cycle in tqdm(nx.cycle_basis(nx.Graph(G))):
        coords = [
            gdf_nodes.loc[node].geometry for node in cycle if node in gdf_nodes.index
        ]
        if len(coords) >= 4:
            poly = Polygon(coords)
            if poly.is_valid:
                blocks.append(poly)

    return blocks


def minimum_bounding_circle_area(polygon):
    points = np.array(polygon.exterior.coords)
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    def enclosing_circle(points):
        from scipy.spatial import distance_matrix

        dist_mat = distance_matrix(points, points)
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        center = (points[i] + points[j]) / 2
        radius = np.linalg.norm(points[i] - center)
        return center, radius

    center, radius = enclosing_circle(hull_points)
    return np.pi * radius**2


def compute_shape_factor(blocks):
    block_data = []
    for block in tqdm(blocks):
        if not isinstance(block, Polygon):
            continue
        area = block.area
        # circumscribed_circle_area = (
        # np.pi * (block.length / (2 * np.pi)) ** 2
        # )  # Approximate, a bit faster
        circumscribed_circle_area = minimum_bounding_circle_area(block)

        phi = area / circumscribed_circle_area if circumscribed_circle_area > 0 else 0
        block_data.append((area, phi))

    return np.array(block_data)


def plot_fingerprint(city_name):
    blocks = get_city_blocks(city_name)
    block_data = compute_shape_factor(blocks)

    if len(block_data) == 0:
        print(f"No valid blocks found for {city_name}")
        return

    area_bins = np.logspace(
        np.log10(block_data[:, 0].min()), np.log10(block_data[:, 0].max()), num=6
    )
    bin_indices = np.digitize(block_data[:, 0], area_bins)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.viridis
    colors = [
        cmap(i / max(bin_indices)) for i in bin_indices
    ]  # That's incorrect, but ok for PoC
    # plt.scatter(
        # block_data[:, 1], block_data[:, 0], c=colors, alpha=0.7, edgecolors="k", s=10
    # )
    # point size should correlate with area
    plt.scatter(
        block_data[:, 1], block_data[:, 0], c=colors, alpha=0.7, edgecolors="k", s=block_data[:, 0] / 10000
    )

    plt.xlim(0, 1)
    plt.yscale("log")
    plt.xlabel("Shape Factor (Φ)")
    plt.ylabel("Block Area (log scale)")
    plt.title(f"Fingerprint of {city_name}")
    plt.colorbar(label="Area Category")
    plt.savefig(f"louf_{city_name}_ca_t.png")
    plt.show()


plot_fingerprint("Lviv, Ukraine")
plot_fingerprint("Kyiv, Ukraine")
plot_fingerprint("New York, USA")
