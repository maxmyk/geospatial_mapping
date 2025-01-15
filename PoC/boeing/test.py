import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from math import log
import matplotlib.pyplot as plt


def load_osm_data(place_name, network_type="all", cache_file="osm_data_lv.pkl"):
    import os
    import pickle

    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        with open(cache_file, "rb") as f:
            nodes, edges = pickle.load(f)
    else:
        print(f"Fetching data for {place_name}...")
        graph = ox.graph_from_place(place_name, network_type=network_type)
        nodes, edges = ox.graph_to_gdfs(graph)
        with open(cache_file, "wb") as f:
            pickle.dump((nodes, edges), f)
        print(f"Data cached to {cache_file}.")

    return nodes, edges


place_name = "Lviv, Lviv oblast, Ukraine"
nodes, edges = load_osm_data(place_name)

print("Number of nodes:", len(nodes))
print("Number of edges:", len(edges))

if "osmid" not in nodes.columns:
    print("Adding osmid column to nodes...")
    nodes["osmid"] = nodes.index

nodes = nodes.set_index("osmid", drop=False)
edges = edges.reset_index(drop=True)

if nodes.crs != edges.crs:
    edges = edges.to_crs(nodes.crs)

# ========================

valid_street_types = [
    "residential",
    "primary",
    "secondary",
    "tertiary",
    "motorway",
    "trunk",
    "primary_link",
    "secondary_link",
    "tertiary_link",
]
edges = edges[edges["highway"].isin(valid_street_types)]

edges = edges[
    edges.geometry.type == "LineString"
]  # Filter out MultiLineString geometries

if edges.crs.is_geographic:
    # Calculating the centroid of the bounding box
    centroid = nodes.geometry.unary_union.centroid
    utm_zone = int((centroid.x + 180) // 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"
    edges = edges.to_crs(utm_crs)

# ========================

fig, ax = plt.subplots(figsize=(10, 10))
edges.plot(ax=ax, color="blue", linewidth=0.5)
plt.title("Filtered Streets for Analysis")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.show()

# ========================

median_segment_length = edges.geometry.length.median()
print(f"Median Street Segment Length: {median_segment_length:.2f} m")
