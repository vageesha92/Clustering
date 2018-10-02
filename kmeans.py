from generate_data import (Point, get_sampling_func, points_list_to_array, array_to_points_list_after_scaling, random_points)
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import minmax_scale


def current_distance_points_from_centers(clusters):
    _distance = 0
    for center, points in clusters.items():
        for point in points:
            _distance = _distance + euclidean(point, center)
    return _distance


def k_means(k, points):
    points = [point.location for point in points]
    def random_centers():
        # choose k centres randomly
        dim = len(points[0])
        list_of_centers = []
        for cen in range(k):
            list_of_centers.append(tuple(np.random.random_sample(dim)))
        return list_of_centers

    def k_means_helper(centers):
        closest_centers = get_closest_center(centers, points)
        clusters = get_clusters(closest_centers, centers)
        centroids = {average_point(clusters[center]): clusters[center] for center in centers}
        return centroids

    centers = random_centers()
    prev_distance_points_from_centers = None
    new_distance_points_from_centers = -1
    while new_distance_points_from_centers != prev_distance_points_from_centers:
        new_clusters = k_means_helper(centers)
        for cluster_center, cluster_points in new_clusters.items():
            if cluster_center == ():
                continue
        prev_distance_points_from_centers = new_distance_points_from_centers
        new_distance_points_from_centers = current_distance_points_from_centers(new_clusters)
    return new_clusters


def average_point(points):
    def sum_elementwise():
        return tuple(map(sum, zip(*points)))
    return tuple(coord/len(points) for coord in sum_elementwise())


def get_clusters(closest_centers, centers):
    # create list of points for each cluster
    clusters = {}
    for point, center in closest_centers.items():
        if center in clusters:
            clusters[center].append(point)
        else:
            clusters[center] = [point]
    for center in centers:
        if center not in clusters:
            clusters[center] = []
    return clusters


def get_closest_center(centers, points):
    # Assume points and centers are a list of tuples
    closest_center = {}
    for point in points:
        # calculate distances from each center
        distances_from_centers = []
        for center in centers:
            distances_from_centers.append(euclidean(point, center))
        # find the center which is closest using above distances
        min_distance = min(distances_from_centers)
        min_distance_index = distances_from_centers.index(min_distance)
        min_distance_center = centers[min_distance_index]
        closest_center[tuple(point)] = min_distance_center
    return closest_center


if __name__ == '__main__':
    dims = range(2, 102, 2)
    for num_dimension in dims:
        #num_dimension = 3
        actual_num_clusters = 15
        total_points = 2000
        points = random_points(total_points, num_dimension, actual_num_clusters)
        num_clusters_found = 0
        for i in range(5):
            centroids = k_means(actual_num_clusters, points)
            if len(centroids) > num_clusters_found:
                num_clusters_found = len(centroids)
        print(num_clusters_found)
