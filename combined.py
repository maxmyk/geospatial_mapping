import glob
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm
from scipy.spatial import ConvexHull
import os
import json


ox.settings.use_cache = True
ox.settings.log_console = False


# Data - Copyright (C) 2025 by Curiosio
# Licensed under ECL (Educational Community License)
file_path = "PoC/ds_pt.tsv"
network_type = "all"  # "all" is recommended for smaller settlements; "drive" can be used for large ones


def calculate_boeing_features(G, Gu, min_length=0):
    """
    features:
        phi_orientation_order
        entropy_simplified
        entropy_weighted
        median_street_segment_length
        avg_circuity
        avg_node_degree
        p_dead_ends
        p_four_way
    """

    entropy_simplified = ox.bearing.orientation_entropy(
        G=Gu, num_bins=36, weight=None, min_length=min_length
    )
    entropy_weighted = ox.bearing.orientation_entropy(
        G=Gu, num_bins=36, weight="length", min_length=min_length
    )
    phi_orientation_order = 1 - (((entropy_simplified - 1.386) / (3.584 - 1.386)) ** 2)
    basic_stats = ox.stats.basic_stats(Gu)
    median_street_segment_length = basic_stats["street_length_avg"]
    avg_circuity = basic_stats["circuity_avg"]
    avg_node_degree = basic_stats["k_avg"]
    p_dead_ends = basic_stats["streets_per_node_proportions"][0]
    p_four_way = basic_stats["streets_per_node_proportions"][3]

    return {
        "selected_features": {
            "phi_orientation_order": phi_orientation_order,
            "entropy_simplified": entropy_simplified,
            "entropy_weighted": entropy_weighted,
            "median_street_segment_length": median_street_segment_length,
            "avg_circuity": avg_circuity,
            "avg_node_degree": avg_node_degree,
            "p_dead_ends": p_dead_ends,
            "p_four_way": p_four_way,
        },
        "basic_stats": basic_stats,
    }


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
    for block, area in tqdm(blocks):
        if not isinstance(block, Polygon):
            continue
        # circumscribed_circle_area = (
        # np.pi * (block.length / (2 * np.pi)) ** 2
        # )  # Approximate, a bit faster
        circumscribed_circle_area = minimum_bounding_circle_area(block)

        phi = area / circumscribed_circle_area if circumscribed_circle_area > 0 else 0
        block_data.append((area, phi))

    return np.array(block_data)


bin_ranges = [
    (0, 1e1, "[0 - 10)"),
    (1e1, 1e2, "[10 - 100)"),
    (1e2, 1e3, "[100 - 10^3)"),
    (1e3, 1e4, "[10^3 - 10^4)"),
    (1e4, 1e5, "[10^4 - 10^5)"),
    (1e5, 1e6, "[10^5 - 10^6)"),
    (1e6, 1e7, "[10^6 - 10^7)"),
    # (1e7, 0, "[10^7 - 0) (unused)"),
]

color_names = [
    "black",
    "gray",
    "brown",
    "#66C2A5",
    "#FC8D62",
    "#8DA0CB",
    "#E78AC3",
    "white",
]


def get_pd(block_data, city_name):
    if len(block_data) == 0:
        print(f"No valid blocks found for {city_name}")
        return

    # fig, ax = plt.subplots(figsize=(8, 5))

    # colors = color_names

    bin_edges = np.linspace(0, 1, 31)  # seems to be different in different plots
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    total_blocks = len(block_data)

    total_hist = np.zeros_like(bin_centers)

    number_of_blocks_by_shape_factor = np.histogram(
        block_data[:, 1], bins=bin_edges, density=True
    )[0]

    weighted_hist_by_bin_ranges = np.zeros((len(bin_ranges), len(bin_centers)))

    for i, (a_min, a_max, b_label) in enumerate(bin_ranges):
        phi_values = block_data[
            (block_data[:, 0] >= a_min) & (block_data[:, 0] < a_max), 1
        ]
        n = len(phi_values)
        if n == 0:
            continue
        hist, _ = np.histogram(phi_values, bins=bin_edges, density=True)
        weighted_hist = hist * (n / total_blocks)
        weighted_hist_by_bin_ranges[i] = weighted_hist
        total_hist += weighted_hist
        # ax.plot(bin_centers, weighted_hist, color=colors[i], label=b_label)

    # ax.plot(
    #     bin_centers,
    #     total_hist,
    #     color="gray",
    #     linewidth=0.5,
    #     linestyle="--",
    # )
    # ax.fill_between(bin_centers, total_hist, color="gray", alpha=0.1)

    # ax.set_xlabel(r"$\Phi$", fontsize=14)
    # ax.set_ylabel(r"$f(\Phi)$", fontsize=14)
    # ax.set_title(f"{city_name}", fontsize=18, weight="bold")
    # ax.legend()
    # plt.tight_layout()
    # plt.show()
    return {
        "selected_features": {
            "total_hist": total_hist,
            "weighted_hist_by_bin_ranges": weighted_hist_by_bin_ranges,
        },
        "our_features": {
            "number_of_blocks": total_blocks,
            "number_of_blocks_by_shape_factor": number_of_blocks_by_shape_factor,
            "blocks_area": np.sum(block_data[:, 0]),
        },
    }


def calculate_lnb_features(G, gdf_nodes, place_name):
    """
    phi_shape_factor0, ..., phi_shape_factor29
    phi_by_log_area_2_3_0, ..., phi_by_log_area_2_3_29
    phi_by_log_area_3_4_0, ..., phi_by_log_area_3_4_29
    phi_by_log_area_4_5_0, ..., phi_by_log_area_4_5_29
    """

    blocks = []
    for cycle in tqdm(nx.cycle_basis(nx.Graph(G))):
        coords = [
            gdf_nodes.loc[node].geometry for node in cycle if node in gdf_nodes.index
        ]
        if len(coords) >= 3:
            poly = Polygon(coords)
            ogpoly = poly
            narea = poly.area
            if poly.is_valid:
                blocks.append((ogpoly, narea))
    # # plotting blocks
    # fig, ax = plt.subplots(figsize=(8, 8))
    # for block, area in blocks:
    #     x, y = block.exterior.xy
    #     ax.fill(x, y, alpha=0.5)
    # ax.set_aspect("equal", "datalim")
    # plt.title(f"Blocks of {place_name}")
    # # plt.savefig(f"louf_LVN_{ncname}_blocks.png")
    # plt.show()

    # calculate features
    return get_pd(compute_shape_factor(blocks), place_name)  # , min_length=0.001)


# Our custom features
def calculate_min_zoom_level(place_name):
    # from OSM wiki on Zoom_levels - https://wiki.openstreetmap.org/wiki/Zoom_levels
    geocode_result = ox.geocode_to_gdf(place_name, by_osmid=True)
    if geocode_result is not None and not geocode_result.empty:
        lon_min, lat_min, lon_max, lat_max = geocode_result.total_bounds
        utm_crs = ox.projection.project_gdf(geocode_result).crs
        geocode_result = geocode_result.to_crs(utm_crs)
        area = geocode_result.area[0] / 10**6
        lat_span_km = (
            geocode_result.bounds.maxy[0] - geocode_result.bounds.miny[0]
        ) / 1000
        lon_span_km = (
            geocode_result.bounds.maxx[0] - geocode_result.bounds.minx[0]
        ) / 1000
        lond = abs(lon_max - lon_min)
        lon_level = 0
        for level in range(21, 0, -1):
            s = 360 / 2**level
            if s > lond:
                lon_level = level
                break
        latd = lat_max - lat_min
        lat_level = 0
        for level in range(21, 0, -1):
            s = 180 / 2**level  # Very Approximate
            if s > latd:
                lat_level = level
                break
        return (
            min(lon_level, lat_level),
            area,
            lat_span_km,
            lon_span_km,
            lat_level,
            lon_level,
            (lat_max + lat_min) / 2,
            (lon_max + lon_min) / 2,
        )
    else:
        raise ValueError("Place not found.")


def calculate_our_features(place_name):
    zoom, area, lat_span_km, lon_span_km, lat_level, lon_level, lat, lon = (
        calculate_min_zoom_level(place_name)
    )
    return {
        "zoom": zoom,
        "area_by_boundary": area,
        "span_lat": lat_span_km,
        "span_lon": lon_span_km,
        "zoom_lat": lat_level,
        "zoom_lon": lon_level,
        "lat": lat,
        "lon": lon,
    }


def extract_coordinates(bbox):
    try:
        coords = bbox[9:-2].split(",")
        coords = [tuple(map(float, coord.split())) for coord in coords]
        return coords
    except Exception as e:
        print(f"Error processing bounding box: {bbox}, Error: {e}")
        return None


def process_tsv(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    dft = pd.read_csv(file_path, sep="\t", skiprows=3)
    expected_columns = ["osm_id", "name", "country", "area", "bounding_box"]
    for col in expected_columns:
        if col not in dft.columns:
            raise ValueError(f"Missing expected column: {col}")
    dft["bounding_box"] = dft["bounding_box"].apply(extract_coordinates)
    return dft



try:
    df_ = process_tsv(file_path)
except Exception as e:
    print(f"Error processing file: {e}")


df_used = df_

dirname = "results_pt_" + network_type

os.makedirs(dirname, exist_ok=True)


def convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


for settlement in tqdm(df_used.itertuples(), total=len(df_used)):
    print(
        f"Calculating features for {settlement.name} {settlement.country} ({settlement.osm_id})"
    )
    try:
        place_name = f"R{settlement.osm_id}"
        polygon = ox.geocode_to_gdf(place_name, by_osmid=True).union_all()
        G = ox.graph_from_polygon(polygon, network_type=network_type)
        Gu = ox.graph_from_polygon(
            polygon,
            network_type=network_type,
            simplify=True,
            # retain_all=True,
        )

        ox.bearing.add_edge_bearings(Gu)
        Gu = ox.convert.to_undirected(Gu)
        G = ox.project_graph(G)

        fig, ax = ox.plot_graph(
            G,
            show=False,
            close=False,
            node_size=0,
            edge_linewidth=0.5,
            edge_color="black",
            bgcolor="white",
            figsize=(8, 8),
        )
        ncname = settlement.osm_id
        plt.savefig(f"{dirname}/{ncname}_network.png")
        # plt.show()
        plt.close(fig)

        gdf_nodes, gdf_edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
        lnb_features = calculate_lnb_features(G, gdf_nodes, place_name)
        boeing_features = calculate_boeing_features(G, Gu)
        our_features = calculate_our_features(place_name)
        combined_features = {
            "lnb": lnb_features,
            "boeing": boeing_features,
            "our": our_features,
        }

        filename = f"{dirname}/{ncname}_features.json"

        flattened_features = [
            boeing_features["selected_features"]["phi_orientation_order"],
            boeing_features["selected_features"]["entropy_simplified"],
            boeing_features["selected_features"]["entropy_weighted"],
            boeing_features["selected_features"]["median_street_segment_length"],
            boeing_features["selected_features"]["avg_circuity"],
            boeing_features["selected_features"]["avg_node_degree"],
            boeing_features["selected_features"]["p_dead_ends"],
            boeing_features["selected_features"]["p_four_way"],
        ]

        for i in lnb_features["selected_features"]["total_hist"]:
            flattened_features.append(i)

        for i in lnb_features["selected_features"]["weighted_hist_by_bin_ranges"][2]:
            flattened_features.append(i)
        for i in lnb_features["selected_features"]["weighted_hist_by_bin_ranges"][3]:
            flattened_features.append(i)
        for i in lnb_features["selected_features"]["weighted_hist_by_bin_ranges"][4]:
            flattened_features.append(i)

        flattened_features.append(our_features["zoom"])
        flattened_features.append(lnb_features["our_features"]["blocks_area"])
        flattened_features.append(our_features["span_lat"])
        flattened_features.append(our_features["span_lon"])
        flattened_features.append(lnb_features["our_features"]["number_of_blocks"])
        for i in lnb_features["our_features"]["number_of_blocks_by_shape_factor"]:
            flattened_features.append(i)

        flattened_features.append(our_features["area_by_boundary"])
        flattened_features.append(our_features["lat"])
        flattened_features.append(our_features["lon"])
        flattened_features.append(our_features["zoom_lat"])
        flattened_features.append(our_features["zoom_lon"])

        with open(filename.replace(".json", ".csv"), "w") as f:
            f.write(",".join([str(i) for i in flattened_features]) + "\n")

        with open(filename, "w") as f:
            # converting to json properly
            json_ready = convert_for_json(combined_features)
            json.dump(json_ready, f, indent=4)
    except Exception as e:
        print(
            f"Error calculating features for {settlement.name}, {settlement.country}: {e}"
        )

# saving to csv
csv_files = glob.glob(f"{dirname}/*.csv")
city_names = [os.path.splitext(os.path.basename(f))[0].split("_")[0] for f in csv_files]
df_list = []
for f, name in zip(csv_files, city_names):
    row = pd.read_csv(f, header=None)
    row["city_name"] = name
    df_list.append(row)
df = pd.concat(df_list, ignore_index=True)

shape_factors = [f"phi_shape_factor{i}" for i in range(30)]
phi_area_bins = []
for bin_label in ["2_3", "3_4", "4_5"]:
    phi_area_bins.extend([f"phi_log_area_{bin_label}_{i}" for i in range(30)])
block_shape_counts = [f"n_blocks_by_shape_factor{i}" for i in range(30)]

feature_columns = [
    "phi_orientation_order",
    "entropy_simplified",
    "entropy_weighted",
    "median_street_segment_length",
    "avg_circuity",
    "avg_node_degree",
    "p_dead_ends",
    "p_four_way",
    *shape_factors,
    *phi_area_bins,
    "zoom",
    "blocks_area",
    "span_lat",
    "span_lon",
    "number_of_blocks",
    *block_shape_counts,
    "area_by_boundary",
    "latitude",
    "longitude",
    "zoom_lat",
    "zoom_lon",
]
feature_columns.append("city_name")
df.columns = feature_columns
# save to csv
df.to_csv(f"{dirname}/all_features.csv", index=False)