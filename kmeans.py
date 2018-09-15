from generate_data import (Point, get_sampling_func, points_list_to_array, array_to_points_list_after_scaling)
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import minmax_scale
import matplotlib.pylab as pl


def current_distance_points_from_centers(clusters):
    _distance = 0
    for center, points in clusters.items():
        for point in points:
            _distance = _distance + euclidean(point, center)
    return _distance


def k_means(k, points):
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
        pl.figure()
        for cluster_center, cluster_points in new_clusters.items():
            if cluster_center == ():
                continue
            print("cluster center = {}".format(cluster_center))
            print("Num of points in cluster = {}".format(len(cluster_points)))
            pl.scatter(np.array(cluster_points)[:,0], np.array(cluster_points)[:,1])
            pl.scatter(cluster_center[0], cluster_center[1])
        print()
        prev_distance_points_from_centers = new_distance_points_from_centers
        new_distance_points_from_centers = current_distance_points_from_centers(new_clusters)
    pl.show()
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



def get_closest_center (centers, points):
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
    sample_1000 = get_sampling_func(1000, 2, 3)
    data, cluster_centers = sample_1000()
    data = points_list_to_array(data)
    points = minmax_scale(data)
    points = tuple(map(tuple, points))
    centroids = k_means(3, points)

