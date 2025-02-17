import numpy as np


def calculate_intersection_density(nodes, area_km2):
    intersections = nodes[nodes["street_count"] > 2]  # Still questionable
    intersection_density = len(intersections) / area_km2
    return intersection_density, intersections


def calculate_intersection_complexity(nodes):
    complex_intersections = nodes[nodes["street_count"] > 4]
    complexity_ratio = len(complex_intersections) / len(nodes)
    return complexity_ratio
