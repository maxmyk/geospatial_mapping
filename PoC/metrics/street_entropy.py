from matplotlib import pyplot as plt
import numpy as np


def calculate_street_orientation_entropy(edges):
    edges["angle"] = edges.geometry.apply(
        lambda geom: np.arctan2(
            geom.coords[-1][1] - geom.coords[0][1],
            geom.coords[-1][0] - geom.coords[0][0],
        )
        * 180 / np.pi % 360
    )
    counts, _ = np.histogram(edges["angle"], bins=36, range=(0, 360))
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]  # Remove zero entries
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy, counts


def plot_polar_entropy(angle_counts):
    bin_centers = np.linspace(0, 360, 36, endpoint=False) + 5  # Center bins
    theta = np.radians(bin_centers)  # Convert to radians
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.bar(theta, angle_counts, width=np.radians(10), edgecolor="black", alpha=0.7)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Street Orientation Distribution (Entropy)")
    plt.show()
