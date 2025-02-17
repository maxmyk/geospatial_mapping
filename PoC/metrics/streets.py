from sklearn.cluster import KMeans
import numpy as np


def calculate_street_orientation_clusters(edges, n_clusters=4):
    edges["angle"] = edges.geometry.apply(
        lambda geom: np.arctan2(
            geom.coords[-1][1] - geom.coords[0][1],
            geom.coords[-1][0] - geom.coords[0][0],
        )
        * 180 / np.pi % 360
    )
    angles = edges["angle"].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(angles)
    edges["cluster"] = kmeans.labels_
    return edges["cluster"].value_counts()

def calculate_dead_end_density(nodes, area_km2):
    dead_ends = nodes[nodes['street_count'] == 1]
    density = len(dead_ends) / area_km2
    return density
